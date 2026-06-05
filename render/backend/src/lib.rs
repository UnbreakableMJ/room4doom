#[cfg(feature = "hprof")]
use coarse_prof::profile;

use hud_util::{draw_text_line, hud_scale, measure_text_line};
use level::LevelData;
use pic_data::{PicData, VoxelManager};
use render_common::wipe::Wipe;
use render_common::{
    BufferSize, DrawBuffer as _, GameRenderer, HealthBleed, RenderView, ScreenEffect as _,
};
use software3d::{DebugDrawOptions, Software3D};
use software25d::Software25D;
use std::sync::Arc;

const CRT_STRETCH: f32 = 240.0 / 200.0;

#[cfg(feature = "display-sdl2")]
mod sdl2_backend;

#[cfg(feature = "display-softbuffer")]
mod softbuffer_backend;

#[cfg(feature = "display-pixels")]
mod pixels_backend;

#[derive(Debug, Default, PartialEq, PartialOrd, Clone, Copy)]
pub enum RenderType {
    /// Purely software. Typically used with blitting a framebuffer maintained
    /// in memory directly to screen using SDL2
    #[default]
    Software,
    /// Fully 3D software rendering.
    Software3D,
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
    #[cfg(feature = "display-pixels")]
    Pixels(pixels_backend::PixelsDisplay),
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

    /// Create a pixels (wgpu) display backend from a winit window.
    #[cfg(feature = "display-pixels")]
    pub fn new_pixels(window: Arc<winit::window::Window>, vsync: bool) -> Self {
        Self::Pixels(pixels_backend::PixelsDisplay::new(window, vsync))
    }

    /// Present the buffer to the screen.
    /// Acquire the display surface (buffer-sized), run `body` to compose into it
    /// (`0xFFRRGGBB` slice + row pitch in u32 elements), then present.
    fn render_frame(&mut self, w: u32, h: u32, body: impl FnOnce(&mut [u32], usize)) {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(d) => d.render_frame(w, h, body),
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(d) => d.render_frame(w, h, body),
            #[cfg(feature = "display-pixels")]
            Self::Pixels(d) => d.render_frame(w, h, body),
        }
    }

    /// Set fullscreen mode: 0=windowed, 1=borderless, 2=exclusive.
    pub fn set_fullscreen(&mut self, mode: u8) {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(d) => d.set_fullscreen(mode),
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(d) => d.set_fullscreen(mode),
            #[cfg(feature = "display-pixels")]
            Self::Pixels(d) => d.set_fullscreen(mode),
        }
    }

    /// Query the window/drawable size for buffer sizing.
    fn window_size(&self) -> (u32, u32) {
        match self {
            #[cfg(feature = "display-sdl2")]
            Self::Sdl2(d) => d.window_size(),
            #[cfg(feature = "display-softbuffer")]
            Self::Softbuffer(d) => d.window_size(),
            #[cfg(feature = "display-pixels")]
            Self::Pixels(d) => d.window_size(),
        }
    }
}

/// Owns the renderer, display backend, and the per-frame-persistent state
/// (index plane, bleed, wipe). The u32 surface is borrowed only during a frame.
pub struct RenderTarget {
    renderer: Renderer,
    display: DisplayBackend,
    size: BufferSize,
    /// 8-bit palette-index plane; the scene rasterizes into it, then it is
    /// resolved into the display surface. Reused across frames.
    index: Vec<u8>,
    bleed: HealthBleed,
    wipe: Wipe,
}

impl RenderTarget {
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

        Self {
            display,
            size: BufferSize::new(w, h),
            index: vec![0u8; w * h],
            bleed: HealthBleed::default(),
            wipe: Wipe::new(buf_width as i32, buf_height as i32),
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
        Self::new(double, debug, debug_draw, self.display, render_type)
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

impl GameRenderer for RenderTarget {
    type Frame<'a> = FrameCtx<'a>;

    fn buffer_size(&self) -> &BufferSize {
        &self.size
    }

    fn is_wiping(&self) -> bool {
        self.wipe.is_wiping()
    }

    fn reset_health_bleed(&mut self) {
        self.bleed.reset();
    }

    fn with_frame(&mut self, body: impl FnOnce(&mut FrameCtx<'_>)) {
        // Split borrows so the display owns the surface while the closure holds
        // the renderer + index + bleed + wipe (disjoint fields of self).
        let RenderTarget {
            renderer,
            display,
            size,
            index,
            bleed,
            wipe,
        } = self;
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
            };
            body(&mut ctx);
        });
    }
}

/// Per-frame draw context: the acquired display surface plus the persistent
/// index plane / bleed / wipe (borrowed from `RenderTarget`). Implements both
/// [`Frame`] (frame lifecycle) and [`render_common::DrawBuffer`] (the scene
/// rasterizes into the index plane, UI draws onto the surface).
pub struct FrameCtx<'a> {
    renderer: &'a mut Renderer,
    size: &'a BufferSize,
    /// Acquired display surface, `0xFFRRGGBB`, row stride `pitch` (u32 elements).
    surface: &'a mut [u32],
    pitch: usize,
    index: &'a mut Vec<u8>,
    bleed: &'a mut HealthBleed,
    wipe: &'a mut Wipe,
    needs_resolve: bool,
}

impl render_common::Frame for FrameCtx<'_> {
    fn start_wipe(&mut self) {
        if self.wipe.is_wiping() {
            return;
        }
        // Size + clear the snapshot; the caller then draws the old state into
        // it via `wipe_buffer()` / `resolve_index_into_wipe`.
        self.wipe.start();
    }

    fn wipe_buffer(&mut self) -> impl render_common::DrawBuffer {
        self.wipe_view()
    }

    fn resolve_index_into_wipe(&mut self, pic_data: &PicData) {
        // Resolve the (still-present) index plane into the wipe snapshot: the
        // old Level view. No bleed — the old frame is transient under the melt.
        self.wipe_view().resolve(
            pic_data.palette(),
            pic_data.palettes_flat(),
            pic_data.use_palette(),
        );
    }

    fn render_player_view(
        &mut self,
        view: &RenderView,
        level_data: &LevelData,
        pic_data: &mut PicData,
    ) {
        // draw_view fills the index plane (or writes the surface directly for
        // debug colour modes, returning `false` so finish_scene skips resolve).
        let (renderer, mut buf) = self.split();
        let needs_resolve = match renderer {
            Renderer::Software(r) => r.draw_view(view, level_data, pic_data, &mut buf),
            Renderer::Software3D(r) => r.draw_view(view, level_data, pic_data, &mut buf),
        };
        self.needs_resolve = needs_resolve;
    }

    fn do_wipe(&mut self) -> bool {
        // Melt the u32 snapshot (width-pitched) over the display surface
        // (`self.pitch`, possibly padded).
        let done = self.wipe.do_melt_pixels(self.surface, self.pitch);
        if done {
            self.wipe.reset();
        }
        done
    }

    fn finish_scene(&mut self, pic_data: &PicData) {
        let needs_resolve = self.needs_resolve;
        let (renderer, mut buf) = self.split();
        if needs_resolve {
            buf.resolve(
                pic_data.palette(),
                pic_data.palettes_flat(),
                pic_data.use_palette(),
            );
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

    fn draw_buffer(&mut self) -> &mut impl render_common::DrawBuffer {
        self
    }

    fn size(&self) -> &BufferSize {
        self.size
    }

    fn set_health_bleed(&mut self, health: i32) {
        let (w, h) = (self.size.width_usize(), self.size.height_usize());
        self.bleed.update(health, w, h);
    }
}

/// A `DrawBuffer` borrowing a `FrameCtx`'s surface + index + bleed *fields*
/// (not the whole `FrameCtx`), so the renderer (`&mut Renderer`, also a field)
/// can be borrowed disjointly alongside it.
struct SurfaceBuf<'b> {
    size: &'b BufferSize,
    surface: &'b mut [u32],
    pitch: usize,
    index: &'b mut Vec<u8>,
    bleed: &'b mut HealthBleed,
}

impl<'a> FrameCtx<'a> {
    /// Split into the renderer and a `SurfaceBuf` over the remaining fields, so
    /// the renderer can draw into the surface/index without a borrow conflict.
    #[inline]
    fn split(&mut self) -> (&mut Renderer, SurfaceBuf<'_>) {
        (
            self.renderer,
            SurfaceBuf {
                size: self.size,
                surface: self.surface,
                pitch: self.pitch,
                index: self.index,
                bleed: self.bleed,
            },
        )
    }

    /// A `SurfaceBuf` whose surface is the wipe snapshot (width-pitched), for
    /// drawing the old frame. The old state draws u32 here as if it were the
    /// display surface.
    #[inline]
    fn wipe_view(&mut self) -> SurfaceBuf<'_> {
        SurfaceBuf {
            size: self.size,
            surface: self.wipe.snapshot_mut(),
            pitch: self.size.width_usize(),
            index: self.index,
            bleed: self.bleed,
        }
    }
}

macro_rules! impl_drawbuffer {
    ($ty:ty) => {
        impl render_common::DrawBuffer for $ty {
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
                let pos = y * self.pitch + x;
                unsafe {
                    *self.surface.get_unchecked_mut(pos) = colour;
                }
            }

            #[inline]
            fn read_pixel(&self, x: usize, y: usize) -> u32 {
                unsafe { *self.surface.get_unchecked(y * self.pitch + x) }
            }

            #[inline]
            fn buf_mut(&mut self) -> &mut [u32] {
                self.surface
            }

            #[inline]
            fn set_index(&mut self, x: usize, y: usize, idx: u8) {
                let pos = y * self.size.width_usize() + x;
                unsafe {
                    *self.index.get_unchecked_mut(pos) = idx;
                }
            }

            #[inline]
            fn index_mut(&mut self) -> &mut [u8] {
                self.index
            }

            fn resolve(&mut self, palette: &[u32], palettes_flat: &[u32], use_palette: usize) {
                let w = self.size.width_usize();
                let h = self.size.height_usize();
                let sp = self.pitch;
                if !self.bleed.is_active() || use_palette != 0 {
                    if sp == w {
                        for (out, &idx) in self.surface[..w * h].iter_mut().zip(self.index.iter()) {
                            *out = unsafe { *palette.get_unchecked(idx as usize) };
                        }
                    } else {
                        for y in 0..h {
                            let src = &self.index[y * w..y * w + w];
                            let dst = &mut self.surface[y * sp..y * sp + w];
                            for (out, &idx) in dst.iter_mut().zip(src) {
                                *out = unsafe { *palette.get_unchecked(idx as usize) };
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
                        let c = unsafe { *palettes_flat.get_unchecked(off * 256 + idx) };
                        unsafe {
                            *self.surface.get_unchecked_mut(row + x) = c;
                        }
                    }
                }
            }
        }
    };
}

// SurfaceBuf and FrameCtx hold the same surface/index/bleed fields, so they
// share one DrawBuffer impl: FrameCtx is a DrawBuffer directly (UI draws via
// `draw_buffer()` -> `self`); SurfaceBuf is the disjoint-borrow split used by
// the renderer (see `FrameCtx::split`).
impl_drawbuffer!(SurfaceBuf<'_>);
impl_drawbuffer!(FrameCtx<'_>);
