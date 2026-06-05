//! wgpu display backend — presents the engine framebuffer via winit + wgpu.
//!
//! Owns the wgpu device/queue/surface directly (no `pixels` wrapper).
//!
//! Each frame uploads the engine `Bgra8Unorm` framebuffer into a texture, then
//! runs the [`PostChain`] — an ordered list of full-screen passes (stretch, CRT,
//! …) where the last targets the surface.

use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::Arc;

#[cfg(feature = "hprof")]
use coarse_prof::profile;
use pic_data::{ByteOrder, PixelFmt};
use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
use winit::window::{Fullscreen, Window};

/// Engine framebuffer + surface pixel format. The engine writes `0xFFRRGGBB`,
/// whose little-endian bytes are `[BB,GG,RR,FF]` = BGRA, so `Bgra8Unorm` is a
/// straight upload. Non-sRGB: the palette is already gamma-baked.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// A full-screen post-process effect.
///
/// Each runs as one pass that samples the previous stage's texture and draws a
/// full-screen triangle into the next. Stackable: the chain runs them in order,
/// the last pass targeting the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostEffect {
    /// Nearest-neighbour upscale, sharp pixels.
    Stretch,
    /// CRT emulation (crt-lottes): scanlines, mask, warp, bloom.
    Crt,
}

impl PostEffect {
    /// Linear filtering suits the CRT's sub-texel sampling; Stretch wants
    /// nearest for crisp pixels.
    fn linear(self) -> bool {
        matches!(self, Self::Crt)
    }
}

/// One post-process pass: a pipeline plus the bind group for its input texture.
struct PostPass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

/// Ordered post-process chain. Pass 0 samples the engine framebuffer; each
/// later pass samples the previous pass's intermediate texture; the last pass
/// targets the surface. Intermediates are surface-sized and ping-pong.
struct PostChain {
    effects: Vec<PostEffect>,
    bind_group_layout: wgpu::BindGroupLayout,
    nearest: wgpu::Sampler,
    linear: wgpu::Sampler,
    /// One pipeline per [`PostEffect`] kind, built once.
    pipelines: Vec<(PostEffect, wgpu::RenderPipeline)>,
    /// Surface-sized intermediate targets, one fewer than passes (last pass
    /// writes the surface). Recreated on resize. Empty for a single pass.
    intermediates: Vec<TargetTexture>,
    /// Per-pass bind groups, rebuilt when input textures change (resize).
    passes: Vec<PostPass>,
}

/// A renderable + samplable target texture (intermediate stage output).
struct TargetTexture {
    view: wgpu::TextureView,
}

impl TargetTexture {
    fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post_intermediate"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}

impl PostChain {
    fn new(
        device: &wgpu::Device,
        effects: Vec<PostEffect>,
        frame_view: &wgpu::TextureView,
        surface: (u32, u32),
    ) -> Self {
        let effects = if effects.is_empty() {
            vec![PostEffect::Stretch]
        } else {
            effects
        };

        let bind_group_layout = post_bind_group_layout(device);

        let make_sampler = |linear: bool| {
            let f = if linear {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            };
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("post_sampler"),
                mag_filter: f,
                min_filter: f,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        };
        let nearest = make_sampler(false);
        let linear = make_sampler(true);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let mut pipelines: Vec<(PostEffect, wgpu::RenderPipeline)> = Vec::new();
        for &e in &effects {
            if pipelines.iter().any(|(k, _)| *k == e) {
                continue;
            }
            pipelines.push((e, build_pipeline(device, &layout, e)));
        }

        let mut chain = Self {
            effects,
            bind_group_layout,
            nearest,
            linear,
            pipelines,
            intermediates: Vec::new(),
            passes: Vec::new(),
        };
        chain.resize(device, frame_view, surface);
        chain
    }

    /// Recreate intermediate targets at the surface size and rebuild every
    /// pass's bind group over its input texture. Called on init and resize.
    fn resize(
        &mut self,
        device: &wgpu::Device,
        frame_view: &wgpu::TextureView,
        surface: (u32, u32),
    ) {
        let n = self.effects.len();
        self.intermediates = (0..n.saturating_sub(1))
            .map(|_| TargetTexture::new(device, surface.0, surface.1))
            .collect();

        self.passes = (0..n)
            .map(|i| {
                let effect = self.effects[i];
                let input = if i == 0 {
                    frame_view
                } else {
                    &self.intermediates[i - 1].view
                };
                let sampler = if effect.linear() {
                    &self.linear
                } else {
                    &self.nearest
                };
                let pipeline = self
                    .pipelines
                    .iter()
                    .find(|(k, _)| *k == effect)
                    .map(|(_, p)| p.clone())
                    .expect("pipeline built for effect");
                let bind_group =
                    texture_bind_group(device, &self.bind_group_layout, input, sampler);
                PostPass {
                    pipeline,
                    bind_group,
                }
            })
            .collect();
    }

    /// Run the chain: each pass draws into the next intermediate; the last into
    /// `surface`.
    fn render(&self, encoder: &mut wgpu::CommandEncoder, surface: &wgpu::TextureView) {
        let last = self.passes.len() - 1;
        for (i, pass) in self.passes.iter().enumerate() {
            let target = if i == last {
                surface
            } else {
                &self.intermediates[i].view
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&pass.pipeline);
            rpass.set_bind_group(0, &pass.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}

/// Bind-group layout shared by every post pass: one sampled texture (the
/// previous stage's output) plus its sampler, both fragment-visible.
fn post_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float {
                        filterable: true,
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Bind a sampled texture `view` + `sampler` against [`post_bind_group_layout`].
fn texture_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Build a full-screen-triangle pipeline for `effect`. All effects share the
/// `vs_main`/`fs_main` entry points and the single texture+sampler bind layout.
fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    effect: PostEffect,
) -> wgpu::RenderPipeline {
    let shader = match effect {
        PostEffect::Stretch => device.create_shader_module(wgpu::include_wgsl!("stretch.wgsl")),
        PostEffect::Crt => device.create_shader_module(wgpu::include_wgsl!("lottes-crt.wgsl")),
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("post_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The engine framebuffer texture plus its CPU scratch and dimensions.
struct FrameTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// CPU-side framebuffer the engine composes into; uploaded each frame.
    scratch: Vec<u32>,
    w: u32,
    h: u32,
}

impl FrameTexture {
    fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            scratch: vec![0u32; (w * h) as usize],
            w,
            h,
        }
    }

    /// Upload the CPU scratch into the GPU texture.
    fn upload(&self, queue: &wgpu::Queue) {
        let bytes: &[u8] = bytemuck_cast(&self.scratch);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.w * 4),
                rows_per_image: Some(self.h),
            },
            wgpu::Extent3d {
                width: self.w,
                height: self.h,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Reinterpret a `&[u32]` as a byte slice for upload (no copy).
#[inline]
fn bytemuck_cast(px: &[u32]) -> &[u8] {
    // SAFETY: u32 is 4 contiguous bytes; the resulting slice covers the same
    // bytes with 4x the length and tighter (byte) alignment.
    unsafe { from_raw_parts(px.as_ptr().cast::<u8>(), size_of_val(px)) }
}

/// Create the wgpu instance, surface, adapter and device for `window`. Blocks on
/// the async adapter/device requests (one-time, at startup).
fn init_gpu(window: &Arc<Window>) -> (wgpu::Surface<'static>, wgpu::Device, wgpu::Queue) {
    // Backends from the environment (Metal on macOS), sensible defaults.
    let instance = wgpu::Instance::default();
    // The surface owns an `Arc<Window>` clone, so it borrows nothing with a
    // shorter lifetime — `'static`.
    let surface = instance
        .create_surface(window.clone())
        .expect("failed to create wgpu surface");

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    }))
    .expect("no compatible wgpu adapter");

    // Use the adapter's real limits, not downlevel defaults: a retina surface
    // (e.g. 3024×1898) exceeds the downlevel 2048 max texture size.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("room4doom_device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .expect("failed to request wgpu device");

    (surface, device, queue)
}

/// The swapchain configuration at `w`×`h`.
fn surface_config(w: u32, h: u32, vsync: bool) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: FORMAT,
        width: w,
        height: h,
        present_mode: if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        },
        // 2 frames in flight: lets the CPU run a frame ahead of the GPU. Dropping
        // to 1 serializes CPU+GPU on acquire (measured ~17% slower).
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    }
}

/// wgpu display: owns the device/queue/surface, the framebuffer texture, and the
/// post-process chain.
pub struct WgpuDisplay {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    frame: FrameTexture,
    chain: PostChain,
    window: Arc<Window>,
    /// Last configured surface size; reconfigure only on change (Metal
    /// `surface.configure()` is a pipeline stall).
    surface_size: (u32, u32),
}

impl WgpuDisplay {
    /// Create from a winit window. The window must be wrapped in `Arc`.
    pub fn new(window: Arc<Window>, vsync: bool, post: Vec<PostEffect>) -> Self {
        let size = window.inner_size();
        let (sw, sh) = (size.width.max(1), size.height.max(1));

        let (surface, device, queue) = init_gpu(&window);
        let config = surface_config(sw, sh, vsync);
        surface.configure(&device, &config);

        // The framebuffer texture starts at the window size; `render_frame`
        // resizes it to the engine buffer size on the first frame.
        let frame = FrameTexture::new(&device, sw, sh);
        let chain = PostChain::new(&device, post, &frame.view, (sw, sh));

        Self {
            surface,
            device,
            queue,
            config,
            frame,
            chain,
            window,
            surface_size: (sw, sh),
        }
    }

    /// Hand `body` the engine framebuffer (`0xFFRRGGBB`, tight pitch), then
    /// upload it and draw it stretched to fill the window.
    ///
    /// u32-only (`Bgra8Unorm`): `P` must be `u32` (asserted).
    pub(crate) fn render_frame<P: PixelFmt>(
        &mut self,
        w: u32,
        h: u32,
        body: impl FnOnce(&mut [P], usize),
    ) {
        assert_eq!(
            size_of::<P>(),
            size_of::<u32>(),
            "wgpu backend is u32-only; Rgb565 must fall back to Rgb888"
        );
        #[cfg(feature = "hprof")]
        profile!("wgpu_frame");

        self.sync_sizes(w, h);

        {
            #[cfg(feature = "hprof")]
            profile!("wgpu_compose");
            // SAFETY: P == u32 (asserted); scratch is the engine framebuffer.
            let dst = unsafe {
                from_raw_parts_mut(
                    self.frame.scratch.as_mut_ptr().cast::<P>(),
                    self.frame.scratch.len(),
                )
            };
            body(dst, w as usize);
        }

        {
            #[cfg(feature = "hprof")]
            profile!("wgpu_render");
            self.frame.upload(&self.queue);

            let Some(surface_tex) = self.acquire_surface() else {
                return;
            };
            let view = surface_tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame_encoder"),
                });
            self.chain.render(&mut encoder, &view);
            self.queue.submit(Some(encoder.finish()));
            surface_tex.present();
        }
    }

    /// Resize the framebuffer texture to the engine buffer (`w`×`h`) and the
    /// surface to the window when either changed, rebuilding the post chain over
    /// the new inputs. Reconfigures the surface only on change (Metal
    /// `surface.configure()` is a pipeline stall).
    fn sync_sizes(&mut self, w: u32, h: u32) {
        let frame_changed = self.frame.w != w || self.frame.h != h;
        if frame_changed {
            self.frame = FrameTexture::new(&self.device, w, h);
        }
        let win = self.window.inner_size();
        let win = (win.width.max(1), win.height.max(1));
        let surface_changed = win != self.surface_size;
        if surface_changed {
            self.config.width = win.0;
            self.config.height = win.1;
            self.surface.configure(&self.device, &self.config);
            self.surface_size = win;
        }
        if frame_changed || surface_changed {
            self.chain
                .resize(&self.device, &self.frame.view, self.surface_size);
        }
    }

    /// Acquire the next swapchain texture, reconfiguring and retrying once if the
    /// surface is lost/outdated. `None` drops the frame.
    fn acquire_surface(&self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            Success(t) | Suboptimal(t) => Some(t),
            _ => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    Success(t) | Suboptimal(t) => Some(t),
                    _ => {
                        log::warn!("wgpu: dropped frame, surface unavailable");
                        None
                    }
                }
            }
        }
    }

    /// Window size in logical pixels.
    pub fn window_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    /// The `Bgra8Unorm` surface consumes engine-native `0xAARRGGBB` directly
    /// (LE bytes `[BB,GG,RR,AA]` = BGRA).
    pub(crate) const fn byte_order() -> ByteOrder {
        ByteOrder::Argb
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
