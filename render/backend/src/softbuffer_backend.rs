//! Softbuffer display backend — presents pixels via winit + softbuffer.

use std::num::NonZeroU32;
use std::sync::Arc;

#[cfg(feature = "hprof")]
use coarse_prof::profile;
use softbuffer::{Context, Pixel, Surface};
use winit::window::{Fullscreen, Window};

/// Reinterpret a `&mut [Pixel]` as `&mut [u32]` for bulk pixel operations.
///
/// # Safety
/// `Pixel` is `#[repr(C, align(4))]` with four `u8` fields — identical size
/// (4) and alignment (4) to `u32` — so the slice's length and element layout
/// are preserved. This is the cast softbuffer documents for fast blits.
#[inline(always)]
fn pixels_as_u32_mut(pixels: &mut [Pixel]) -> &mut [u32] {
    let len = pixels.len();
    let ptr = pixels.as_mut_ptr().cast::<u32>();
    // SAFETY: layout-identical (see above); `len` elements, same alignment.
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Softbuffer display: owns the surface and a reference to the window.
///
/// The `Context` is leaked (`Box::leak`) to satisfy `Surface`'s borrow of
/// `&Context`. This is fine — there is exactly one display for the lifetime
/// of the process.
pub struct SoftbufferDisplay {
    surface: Surface<Arc<Window>, Arc<Window>>,
    window: Arc<Window>,
}

impl SoftbufferDisplay {
    /// Create from a winit window. The window must be wrapped in `Arc`.
    pub fn new(window: Arc<Window>) -> Self {
        let ctx: &'static Context<Arc<Window>> = Box::leak(Box::new(
            Context::new(window.clone()).expect("failed to create softbuffer context"),
        ));
        let surface: Surface<Arc<Window>, Arc<Window>> =
            Surface::new(ctx, window.clone()).expect("failed to create softbuffer surface");
        Self {
            surface,
            window,
        }
    }

    /// Acquire the display surface (sized to the buffer; the compositor scales
    /// to the window), hand it to `body` as a `0xFFRRGGBB` slice with its row
    /// pitch in u32 elements (IOSurface rows may be padded), then present.
    pub(crate) fn render_frame(&mut self, w: u32, h: u32, body: impl FnOnce(&mut [u32], usize)) {
        #[cfg(feature = "hprof")]
        profile!("softbuffer_frame");
        self.surface
            .resize(
                NonZeroU32::new(w).unwrap_or(NonZeroU32::new(1).unwrap()),
                NonZeroU32::new(h).unwrap_or(NonZeroU32::new(1).unwrap()),
            )
            .expect("failed to resize softbuffer surface");

        let mut sb = self
            .surface
            .next_buffer()
            .expect("failed to get softbuffer buffer");

        let stride_px = sb.byte_stride().get() as usize / size_of::<Pixel>();
        body(pixels_as_u32_mut(sb.pixels()), stride_px);

        {
            #[cfg(feature = "hprof")]
            profile!("softbuffer_present");
            sb.present().expect("failed to present softbuffer");
        }
    }

    /// Window size in logical pixels.
    pub fn window_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    pub(crate) fn set_fullscreen(&self, mode: u8) {
        let fs = match mode {
            1 => Some(Fullscreen::Borderless(None)),
            2 => {
                let monitor = self
                    .window
                    .current_monitor()
                    .or_else(|| self.window.primary_monitor());
                monitor
                    .and_then(|m| m.video_modes().next())
                    .map(Fullscreen::Exclusive)
            }
            _ => None,
        };
        self.window.set_fullscreen(fs);
    }
}
