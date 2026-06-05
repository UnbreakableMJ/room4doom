#[cfg(feature = "hprof")]
use coarse_prof::profile;

use hud_util::{draw_text_line, hud_scale, measure_text_line};
use level::LevelData;
use pic_data::{ByteOrder, PalLit, PicData, PixelFmt, VoxelManager};
use render_common::wipe::Wipe;
use render_common::{BufferSize, DrawBuffer as _, HealthBleed, PixelTarget, RenderView};
use software3d::{DebugDrawOptions, Software3D};
use software25d::Software25D;
use std::sync::Arc;

const CRT_STRETCH: f32 = 240.0 / 200.0;

/// Cached `PalLit<P>` paired with the palette generation it was built from;
/// rebuilt on gamma change. `None` until first built.
type PalLitCache<P> = Option<(u64, PalLit<P>)>;

/// Build/refresh `cache` for the current palette generation, returning a shared
/// ref. Rebuilds only on gamma change. Borrows the cache field alone (not the
/// whole `FrameCtx`), so callers can hold disjoint `&mut` borrows of the
/// surface/index/bleed/wipe fields alongside the returned `&PalLit`.
fn build_pal_lit<'a, P: PixelFmt>(
    cache: &'a mut PalLitCache<P>,
    pic_data: &PicData,
    order: ByteOrder,
) -> &'a PalLit<P> {
    let pgen = pic_data.palette_generation();
    match cache {
        Some((g, table)) if *g != pgen => {
            table.rebuild(pic_data.palettes());
            *g = pgen;
        }
        Some(_) => {}
        None => *cache = Some((pgen, PalLit::new(pic_data.palettes(), order))),
    }
    &cache.as_ref().expect("pal_lit built above").1
}

#[cfg(feature = "display-sdl2")]
mod sdl2_backend;

#[cfg(feature = "display-softbuffer")]
mod softbuffer_backend;

#[cfg(feature = "display-wgpu")]
mod wgpu_backend;
#[cfg(feature = "display-wgpu")]
pub use wgpu_backend::PostEffect;

#[derive(Debug, Default, PartialEq, PartialOrd, Clone, Copy)]
pub enum RenderType {
    /// Purely software. Typically used with blitting a framebuffer maintained
    /// in memory directly to screen using SDL2
    #[default]
    Software,
    /// Fully 3D software rendering.
    Software3D,
}

/// Scene pixel format.
///
/// `Indexed` uses the 8-bit index plane + `resolve()`; `Rgb888`/`Rgb565` map the
/// palette in the scene store (final pixels, no resolve). `Rgb565` needs an
/// RGB565-capable backend; on u32-only backends it falls back to `Rgb888`.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum PixelMode {
    #[default]
    Indexed,
    Rgb888,
    Rgb565,
}

impl PixelMode {
    /// Direct-pixel modes write final pixels into the surface (no resolve);
    /// `Indexed` uses the index plane + resolve.
    #[inline]
    pub const fn is_direct(self) -> bool {
        matches!(self, Self::Rgb888 | Self::Rgb565)
    }
}

pub enum Renderer {
    /// Purely software. Typically used with blitting a framebuffer maintained
    /// in memory directly to screen using SDL2
    Software(Box<Software25D>),
    /// Fully 3D software rendering.
    Software3D(Box<Software3D>),
}

/// Backend-agnostic display presentation.
pub enum DisplayBackend {
    #[cfg(feature = "display-sdl2")]
    Sdl2(sdl2_backend::Sdl2Display),
    #[cfg(feature = "display-softbuffer")]
    Softbuffer(softbuffer_backend::SoftbufferDisplay),
    #[cfg(feature = "display-wgpu")]
    Wgpu(wgpu_backend::WgpuDisplay),
}

impl DisplayBackend {
    /// Create an SDL2 display backend from a canvas.
    #[cfg(feature = "display-sdl2")]
    pub fn new_sdl2(canvas: sdl2::render::Canvas<sdl2::video::Window>) -> Self {
        Self::Sdl2(sdl2_backend::Sdl2Display::from_canvas(canvas))
    }

    /// Create a softbuffer display backend from a winit window.
    #[cfg(feature = "display-softbuffer")]
    pub fn new_softbuffer(window: Arc<winit::window::Window>) -> Self {
        Self::Softbuffer(softbuffer_backend::SoftbufferDisplay::new(window))
    }

    /// Create a wgpu display backend from a winit window, with an ordered
    /// post-process chain (empty = nearest-neighbour stretch only).
    #[cfg(feature = "display-wgpu")]
    pub fn new_wgpu(
        window: Arc<winit::window::Window>,
        vsync: bool,
        post: Vec<PostEffect>,
    ) -> Self {
        Self::Wgpu(wgpu_backend::WgpuDisplay::new(window, vsync, post))
    }

    /// Present the buffer to the screen.
    /// Acquire the display surface (buffer-sized), run `body` to compose into it
    /// (final `P` pixels + row pitch in `P` elements), then present.
    /// softbuffer/wgpu are u32-only and only accept `P = u32`; sdl2 also
    /// presents `P = u16` (native RGB565).
    fn render_frame<P: PixelFmt>(&mut self, w: u32, h: u32, body: impl FnOnce(&mut [P], usize)) {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(d) => d.render_frame(w, h, body),
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(d) => d.render_frame(w, h, body),
            #[cfg(feature = "display-wgpu")]
            Self::Wgpu(d) => d.render_frame(w, h, body),
        }
    }

    /// Set fullscreen mode: 0=windowed, 1=borderless, 2=exclusive.
    pub fn set_fullscreen(&mut self, mode: u8) {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(d) => d.set_fullscreen(mode),
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(d) => d.set_fullscreen(mode),
            #[cfg(feature = "display-wgpu")]
            Self::Wgpu(d) => d.set_fullscreen(mode),
        }
    }

    /// Query the window/drawable size for buffer sizing.
    fn window_size(&self) -> (u32, u32) {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(d) => d.window_size(),
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(d) => d.window_size(),
            #[cfg(feature = "display-wgpu")]
            Self::Wgpu(d) => d.window_size(),
        }
    }

    /// The byte order this backend's surface consumes — the single source of
    /// truth for the format the `PalLit` is baked in (no per-pixel conversion).
    fn byte_order(&self) -> ByteOrder {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(_) => sdl2_backend::Sdl2Display::byte_order(),
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(_) => softbuffer_backend::SoftbufferDisplay::byte_order(),
            #[cfg(feature = "display-wgpu")]
            Self::Wgpu(_) => wgpu_backend::WgpuDisplay::byte_order(),
        }
    }

    /// Whether the backend can present a native RGB565 surface. Only sdl2
    /// (RGB565 streaming texture); softbuffer and wgpu are u32-only.
    fn supports_rgb565(&self) -> bool {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(_) => true,
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(_) => false,
            #[cfg(feature = "display-wgpu")]
            Self::Wgpu(_) => false,
        }
    }
}

/// Owns the renderer, display backend, and per-frame-persistent state.
///
/// Holds the index plane, bleed, and wipe across frames. Generic over the
/// surface pixel type `P` (`u32` ARGB for softbuffer/wgpu/sdl2-888, `u16`
/// RGB565 for sdl2-565). The surface is borrowed only during a frame.
pub struct RenderTarget<P: PixelFmt> {
    renderer: Renderer,
    display: DisplayBackend,
    size: BufferSize,
    /// 8-bit palette-index plane; the scene rasterizes into it, then it is
    /// resolved into the display surface. Reused across frames.
    index: Vec<u8>,
    bleed: HealthBleed,
    wipe: Wipe<P>,
    /// Scene pixel format. `Rgb888`/`Rgb565` enable the direct-pixel path;
    /// `Indexed` is the classic index + resolve path.
    pixel_mode: PixelMode,
    /// Byte order the display surface consumes; the `PalLit` is baked in it so
    /// the scene store and present need no per-pixel conversion. Sourced from
    /// the backend at construction.
    order: ByteOrder,
    /// Palette block table (`P` pixels), rebuilt on gamma change.
    pal_lit: PalLitCache<P>,
}

impl<P: PixelFmt> RenderTarget<P> {
    pub fn new(
        double: bool,
        debug: bool,
        debug_draw: &DebugDrawOptions,
        display: DisplayBackend,
        render_type: RenderType,
    ) -> Self {
        let win = display.window_size();
        // Buffer height fixed at 200 (or 400 hi-res); width chosen so that the
        // compositor's buf->window scale produces the 1.2x CRT pixel aspect.
        let buf_height = if double { 400u32 } else { 200u32 };
        let buf_width = ((win.0 as f32 * buf_height as f32 * CRT_STRETCH / win.1 as f32).round()
            as u32)
            .max(buf_height);
        let (w, h) = (buf_width as usize, buf_height as usize);
        let order = display.byte_order();

        Self {
            display,
            size: BufferSize::new(w, h),
            index: vec![0u8; w * h],
            bleed: HealthBleed::default(),
            wipe: Wipe::new(buf_width as i32, buf_height as i32),
            pixel_mode: PixelMode::default(),
            order,
            pal_lit: None,
            renderer: match render_type {
                RenderType::Software => Renderer::Software(Box::new(Software25D::new(
                    90f32.to_radians(),
                    buf_width as f32,
                    buf_height as f32,
                    double,
                    debug,
                ))),
                RenderType::Software3D => Renderer::Software3D(Box::new(Software3D::new(
                    buf_width as f32,
                    buf_height as f32,
                    90.0_f32.to_radians(),
                    debug_draw.clone(),
                ))),
            },
        }
    }

    /// Forward a debug overlay line to the active renderer. No-op for non-3D.
    pub fn set_debug_line(&mut self, s: String) {
        if let Renderer::Software3D(r) = &mut self.renderer {
            r.set_debug_line(s);
        }
    }

    /// Set the voxel manager on the Software3D renderer. No-op for non-3D.
    pub fn set_voxel_manager(&mut self, mgr: Arc<VoxelManager>) {
        if let Renderer::Software3D(r) = &mut self.renderer {
            r.set_voxel_manager(mgr);
        }
    }

    pub fn clear_voxel_manager(&mut self) {
        if let Renderer::Software3D(r) = &mut self.renderer {
            r.clear_voxel_manager();
        }
    }

    pub fn set_fullscreen(&mut self, mode: u8) {
        self.display.set_fullscreen(mode);
    }

    pub fn window_size(&self) -> (u32, u32) {
        self.display.window_size()
    }

    /// Rebuild the render target, reusing the display backend.
    pub fn resize(
        self,
        double: bool,
        debug: bool,
        debug_draw: &DebugDrawOptions,
        render_type: RenderType,
    ) -> Self {
        let pixel_mode = self.pixel_mode;
        let mut t = Self::new(double, debug, debug_draw, self.display, render_type);
        t.pixel_mode = pixel_mode;
        t
    }

    /// Select the scene pixel format. `Rgb888`/`Rgb565` enable the direct-pixel path on
    /// both renderers. `Rgb565` on a u32-only backend (softbuffer/wgpu) falls
    /// back to `Rgb888` — native 565 present needs an RGB565 backend (sdl2/DRM).
    pub fn set_pixel_mode(&mut self, mode: PixelMode) {
        let mut mode = mode;
        if mode == PixelMode::Rgb565 && !self.display.supports_rgb565() {
            eprintln!(
                "warning: -pixels 565 needs an RGB565 backend; using 888 on this u32-only backend"
            );
            mode = PixelMode::Rgb888;
        }
        self.pixel_mode = mode;
    }

    /// Update the statusbar height (in OG 200px-space pixels).
    pub fn set_statusbar_height(&mut self, og_height: i32) {
        let scale = self.size.height() / 200;
        self.size.set_statusbar_height(og_height * scale);
        let vh = self.size.view_height();
        match &mut self.renderer {
            Renderer::Software(r) => r.set_view_height(vh as usize),
            Renderer::Software3D(r) => r.set_view_height(vh as f32),
        }
    }
}

impl<P: PixelFmt> RenderTarget<P> {
    pub fn buffer_size(&self) -> &BufferSize {
        &self.size
    }

    pub fn is_wiping(&self) -> bool {
        self.wipe.is_wiping()
    }

    pub fn reset_health_bleed(&mut self) {
        self.bleed.reset();
    }

    pub fn with_frame(&mut self, body: impl FnOnce(&mut FrameCtx<'_, P>)) {
        // Split borrows so the display owns the surface while the closure holds
        // the renderer + index + bleed + wipe (disjoint fields of self).
        let Self {
            renderer,
            display,
            size,
            index,
            bleed,
            wipe,
            pixel_mode,
            order,
            pal_lit,
        } = self;
        let pixel_mode = *pixel_mode;
        let order = *order;
        let (w, h) = (size.width() as u32, size.height() as u32);
        display.render_frame(w, h, |surface, pitch| {
            let mut ctx = FrameCtx {
                renderer,
                size,
                surface,
                pitch,
                index,
                bleed,
                wipe,
                needs_resolve: true,
                pixel_mode,
                order,
                pal_lit,
            };
            body(&mut ctx);
        });
    }
}

/// Per-frame draw context.
///
/// The acquired display surface (final `P` pixels) plus the persistent index
/// plane / bleed / wipe (borrowed from `RenderTarget`). Drives the frame
/// lifecycle and implements [`render_common::DrawBuffer`] (the scene rasterizes
/// into the index plane, UI draws onto the surface).
pub struct FrameCtx<'a, P: PixelFmt> {
    renderer: &'a mut Renderer,
    size: &'a BufferSize,
    /// Acquired display surface, final `P` pixels, row stride `pitch` (`P`
    /// elements).
    surface: &'a mut [P],
    pitch: usize,
    index: &'a mut Vec<u8>,
    bleed: &'a mut HealthBleed,
    wipe: &'a mut Wipe<P>,
    needs_resolve: bool,
    pixel_mode: PixelMode,
    /// Surface byte order the `PalLit` is baked in.
    order: ByteOrder,
    /// Palette block table (owned by `RenderTarget`); built/refreshed here when
    /// the direct-pixel path runs and the palette generation changed.
    pal_lit: &'a mut PalLitCache<P>,
}

impl<P: PixelFmt> FrameCtx<'_, P> {
    pub fn start_wipe(&mut self) {
        if self.wipe.is_wiping() {
            return;
        }
        // Size + clear the snapshot; the caller then draws the old state into
        // it via `wipe_buffer()` / `resolve_index_into_wipe`.
        self.wipe.start();
    }

    pub fn wipe_buffer(&mut self) -> impl render_common::DrawBuffer<Pixel = P> {
        self.wipe_view()
    }

    pub fn resolve_index_into_wipe(&mut self, pic_data: &PicData) {
        // Resolve the (still-present) index plane into the wipe snapshot: the
        // old Level view. No bleed — the old frame is transient under the melt.
        let order = self.order;
        let pal_lit = build_pal_lit(self.pal_lit, pic_data, order);
        let use_palette = pic_data.use_palette();
        let mut view = SurfaceBuf {
            size: self.size,
            surface: self.wipe.snapshot_mut(),
            pitch: self.size.width_usize(),
            index: self.index,
            bleed: self.bleed,
            order,
        };
        view.resolve(pal_lit, use_palette);
    }

    pub fn render_player_view(
        &mut self,
        view: &RenderView,
        level_data: &LevelData,
        pic_data: &mut PicData,
    ) {
        // Direct-pixel path: Rgb888/Rgb565 + no bleed → write final pixels
        // straight into the surface, no resolve. Bleed (index-domain) falls back
        // below.
        if self.pixel_mode.is_direct() && !self.bleed.is_active() {
            let pal_lit = build_pal_lit(self.pal_lit, pic_data, self.order);
            let tint = pic_data.use_palette();
            let mut buf = PixelTarget::new(self.surface, *self.size, self.pitch, pal_lit, tint);
            match self.renderer {
                Renderer::Software(r) => {
                    r.draw_view(view, level_data, pic_data, &mut buf);
                }
                Renderer::Software3D(r) => {
                    r.draw_view(view, level_data, pic_data, &mut buf);
                }
            }
            self.needs_resolve = false;
            return;
        }

        // draw_view fills the index plane (or writes the surface directly for
        // debug colour modes, returning `false` so finish_scene skips resolve).
        let (renderer, mut buf) = self.split();
        let needs_resolve = match renderer {
            Renderer::Software(r) => r.draw_view(view, level_data, pic_data, &mut buf),
            Renderer::Software3D(r) => r.draw_view(view, level_data, pic_data, &mut buf),
        };
        self.needs_resolve = needs_resolve;
    }

    pub fn do_wipe(&mut self) -> bool {
        // Melt the snapshot (width-pitched) over the display surface
        // (`self.pitch`, possibly padded).
        let done = self.wipe.do_melt_pixels(self.surface, self.pitch);
        if done {
            self.wipe.reset();
        }
        done
    }

    pub fn finish_scene(&mut self, pic_data: &PicData) {
        let use_palette = pic_data.use_palette();
        // Build the table only on the resolve path; `build_pal_lit` borrows the
        // `pal_lit` field alone, so the renderer + SurfaceBuf borrows below are
        // disjoint.
        let order = self.order;
        let pal_lit = self
            .needs_resolve
            .then(|| build_pal_lit(self.pal_lit, pic_data, order));
        let renderer = &mut *self.renderer;
        let mut buf = SurfaceBuf {
            size: self.size,
            surface: self.surface,
            pitch: self.pitch,
            index: self.index,
            bleed: self.bleed,
            order,
        };
        if let Some(pal_lit) = pal_lit {
            buf.resolve(pal_lit, use_palette);
        }
        if let Renderer::Software3D(r) = renderer {
            let text = r.take_debug_line();
            r.draw_debug_overlays(&mut buf);
            if !text.is_empty() {
                let (sx, sy) = hud_scale(&buf);
                let palette = pic_data.wad_palette();
                let width = measure_text_line(&text, sx);
                let x = buf.size().width_f32() - width - 4.0 * sx;
                draw_text_line(&text, x, 2.0, sx, sy, palette, &mut buf);
            }
        }
    }

    pub fn draw_buffer(&mut self) -> &mut impl render_common::DrawBuffer<Pixel = P> {
        self
    }

    pub fn set_health_bleed(&mut self, health: i32) {
        let (w, h) = (self.size.width_usize(), self.size.height_usize());
        self.bleed.update(health, w, h);
    }
}

/// A `DrawBuffer` borrowing a `FrameCtx`'s surface + index + bleed *fields*
/// (not the whole `FrameCtx`), so the renderer (`&mut Renderer`, also a field)
/// can be borrowed disjointly alongside it.
struct SurfaceBuf<'b, P: PixelFmt> {
    size: &'b BufferSize,
    surface: &'b mut [P],
    pitch: usize,
    index: &'b mut Vec<u8>,
    bleed: &'b mut HealthBleed,
    order: ByteOrder,
}

impl<P: PixelFmt> FrameCtx<'_, P> {
    /// Split into the renderer and a `SurfaceBuf` over the remaining fields, so
    /// the renderer can draw into the surface/index without a borrow conflict.
    #[inline]
    fn split(&mut self) -> (&mut Renderer, SurfaceBuf<'_, P>) {
        (
            self.renderer,
            SurfaceBuf {
                size: self.size,
                surface: self.surface,
                pitch: self.pitch,
                index: self.index,
                bleed: self.bleed,
                order: self.order,
            },
        )
    }

    /// A `SurfaceBuf` whose surface is the wipe snapshot (width-pitched), for
    /// drawing the old frame. The old state draws `P` here as if it were the
    /// display surface.
    #[inline]
    fn wipe_view(&mut self) -> SurfaceBuf<'_, P> {
        SurfaceBuf {
            size: self.size,
            surface: self.wipe.snapshot_mut(),
            pitch: self.size.width_usize(),
            index: self.index,
            bleed: self.bleed,
            order: self.order,
        }
    }
}

macro_rules! impl_drawbuffer {
    ($ty:ty) => {
        impl<P: PixelFmt> render_common::DrawBuffer for $ty {
            type Pixel = P;

            #[inline]
            fn size(&self) -> &BufferSize {
                self.size
            }

            #[inline]
            fn pitch(&self) -> usize {
                self.size.width_usize()
            }

            #[inline]
            fn get_buf_index(&self, x: usize, y: usize) -> usize {
                y * self.size.width_usize() + x
            }

            #[inline]
            fn set_pixel(&mut self, x: usize, y: usize, colour: u32) {
                // UI passes `0xAARRGGBB`; convert to the surface format `P` in
                // the surface's byte order.
                let pos = y * self.pitch + x;
                unsafe {
                    *self.surface.get_unchecked_mut(pos) = P::from_argb(colour, self.order);
                }
            }

            #[inline]
            fn buf_mut(&mut self) -> &mut [P] {
                self.surface
            }

            fn resolve(&mut self, pal_lit: &PalLit<P>, use_palette: usize) {
                let w = self.size.width_usize();
                let h = self.size.height_usize();
                let sp = self.pitch;
                if !self.bleed.is_active() || use_palette != 0 {
                    let block = pal_lit.block(use_palette);
                    if sp == w {
                        for (out, &idx) in self.surface[..w * h].iter_mut().zip(self.index.iter()) {
                            *out = unsafe { *block.get_unchecked(idx as usize) };
                        }
                    } else {
                        for y in 0..h {
                            let src = &self.index[y * w..y * w + w];
                            let dst = &mut self.surface[y * sp..y * sp + w];
                            for (out, &idx) in dst.iter_mut().zip(src) {
                                *out = unsafe { *block.get_unchecked(idx as usize) };
                            }
                        }
                    }
                    return;
                }
                for y in 0..h {
                    let row = y * sp;
                    let irow = y * w;
                    for x in 0..w {
                        let idx = unsafe { *self.index.get_unchecked(irow + x) } as usize;
                        let off = self.bleed.palette_offset(x, y as u16, use_palette);
                        let c = unsafe { *pal_lit.block(off).get_unchecked(idx) };
                        unsafe {
                            *self.surface.get_unchecked_mut(row + x) = c;
                        }
                    }
                }
            }
        }

        impl<P: PixelFmt> render_common::SceneTarget for $ty {
            type Texel = u8;

            #[inline(always)]
            fn texel(&self, lit: u16) -> u8 {
                lit as u8
            }

            #[inline(always)]
            fn put(&mut self, pos: usize, texel: u8) {
                unsafe {
                    *self.index.get_unchecked_mut(pos) = texel;
                }
            }

            #[inline(always)]
            fn scene_fuzz(&mut self, dst_pos: usize, src_pos: usize, colourmap6: &[usize; 256]) {
                unsafe {
                    let src = *self.index.get_unchecked(src_pos) as usize;
                    *self.index.get_unchecked_mut(dst_pos) = *colourmap6.get_unchecked(src) as u8;
                }
            }
        }
    };
}

// SurfaceBuf and FrameCtx hold the same surface/index/bleed fields, so they
// share one DrawBuffer impl: FrameCtx is a DrawBuffer directly (UI draws via
// `draw_buffer()` -> `self`); SurfaceBuf is the disjoint-borrow split used by
// the renderer (see `FrameCtx::split`).
impl_drawbuffer!(SurfaceBuf<'_, P>);
impl_drawbuffer!(FrameCtx<'_, P>);
