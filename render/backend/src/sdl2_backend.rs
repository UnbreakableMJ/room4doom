//! SDL2 display backend — presents pixels via SDL2 Canvas + Texture.

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
        }
    }

    /// Ensure the streaming texture matches the buffer dimensions.
    fn ensure_texture(&mut self, w: u32, h: u32) {
        if self.tex_size != (w, h) {
            self.texture = Some(
                self._tc
                    .create_texture_streaming(Some(PixelFormatEnum::RGB888), w, h)
                    .expect("failed to create SDL2 streaming texture"),
            );
            self.tex_size = (w, h);
            let drawable = self.canvas.window().drawable_size();
            self.crop_rect = Rect::new(0, 0, drawable.0, drawable.1);
        }
    }

    /// Lock the streaming texture (sized to the buffer), hand it to `body` as a
    /// `0xFFRRGGBB` slice with its row pitch in u32 elements, then copy it
    /// stretched to the canvas and present.
    pub(crate) fn render_frame(&mut self, w: u32, h: u32, body: impl FnOnce(&mut [u32], usize)) {
        self.ensure_texture(w, h);
        let tex = self.texture.as_mut().unwrap();
        tex.with_lock(None, |bytes, byte_pitch| {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut u32, bytes.len() / 4)
            };
            body(dst, byte_pitch / 4);
        })
        .unwrap();
        self.canvas.copy(tex, None, Some(self.crop_rect)).unwrap();
        self.canvas.present();
    }

    /// Window size in screen coordinates.
    pub(crate) fn window_size(&self) -> (u32, u32) {
        self.canvas.window().size()
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
