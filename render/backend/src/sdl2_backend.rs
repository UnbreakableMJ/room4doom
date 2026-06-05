//! SDL2 display backend — presents pixels via SDL2 Canvas + Texture.

use pic_data::{ByteOrder, PixelFmt};
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{Canvas, TextureCreator};
use sdl2::video::{FullscreenType, Window, WindowContext};

/// SDL2 display: owns the canvas, texture, and texture creator.
pub struct Sdl2Display {
    canvas: Canvas<Window>,
    texture: Option<sdl2::render::Texture>,
    _tc: TextureCreator<WindowContext>,
    crop_rect: Rect,
    tex_size: (u32, u32),
    tex_format: Option<PixelFormatEnum>,
}

impl Sdl2Display {
    /// Create from an SDL2 canvas. The texture is created lazily on first
    /// blit to match the framebuffer dimensions.
    pub fn from_canvas(canvas: Canvas<Window>) -> Self {
        let drawable = canvas.window().drawable_size();
        let tc = canvas.texture_creator();
        Self {
            canvas,
            texture: None,
            _tc: tc,
            crop_rect: Rect::new(0, 0, drawable.0, drawable.1),
            tex_size: (0, 0),
            tex_format: None,
        }
    }

    /// Ensure the streaming texture matches the buffer dimensions and pixel
    /// format. Recreated when either changes.
    fn ensure_texture(&mut self, w: u32, h: u32, format: PixelFormatEnum) {
        if self.tex_size != (w, h) || self.tex_format != Some(format) {
            self.texture = Some(
                self._tc
                    .create_texture_streaming(Some(format), w, h)
                    .expect("failed to create SDL2 streaming texture"),
            );
            self.tex_size = (w, h);
            self.tex_format = Some(format);
            let drawable = self.canvas.window().drawable_size();
            self.crop_rect = Rect::new(0, 0, drawable.0, drawable.1);
        }
    }

    /// Lock the streaming texture (sized to the buffer), hand it to `body` as a
    /// final-`P` slice with its row pitch in `P` elements, then copy it stretched
    /// to the canvas and present. `P = u32` → RGB888 texture; `P = u16` → native
    /// RGB565 texture.
    pub(crate) fn render_frame<P: PixelFmt>(
        &mut self,
        w: u32,
        h: u32,
        body: impl FnOnce(&mut [P], usize),
    ) {
        let format = match size_of::<P>() {
            4 => PixelFormatEnum::RGB888,
            2 => PixelFormatEnum::RGB565,
            n => panic!("SDL2 backend: unsupported pixel size {n}"),
        };
        self.ensure_texture(w, h, format);
        let tex = self.texture.as_mut().unwrap();
        tex.with_lock(None, |bytes, byte_pitch| {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr().cast::<P>(),
                    bytes.len() / size_of::<P>(),
                )
            };
            body(dst, byte_pitch / size_of::<P>());
        })
        .unwrap();
        self.canvas.copy(tex, None, Some(self.crop_rect)).unwrap();
        self.canvas.present();
    }

    /// Window size in screen coordinates.
    pub(crate) fn window_size(&self) -> (u32, u32) {
        self.canvas.window().size()
    }

    /// Both streaming-texture formats (RGB888 u32, RGB565 u16) are red-high,
    /// matching engine-native `0xAARRGGBB` / 565-with-R-in-the-top-bits.
    pub(crate) const fn byte_order() -> ByteOrder {
        ByteOrder::Argb
    }

    pub(crate) fn set_fullscreen(&mut self, mode: u8) {
        let fs = match mode {
            1 => FullscreenType::Desktop,
            2 => FullscreenType::True,
            _ => FullscreenType::Off,
        };
        let _ = self.canvas.window_mut().set_fullscreen(fs);
    }
}
