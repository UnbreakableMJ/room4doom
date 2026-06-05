//! Golden cross-path harness.
//!
//! Renders a fixed E1M2 pose through the current index path and captures the
//! resolved u32 frame as the reference. The direct `PixelTarget` path renders the
//! same pose and asserts `direct == widen(index)` bit-for-bit on a frame with no
//! fuzz/bleed/translucency (those are index-domain and break equality by
//! construction). The direct-vs-index assertion is the gate for the scene port.
//!
//! Uses the bundled `data/doom1.wad`; skips cleanly if absent.

use pic_data::{ByteOrder, PalLit, PicData, PixelFmt as _};
use render_common::{BufferSize, DrawBuffer, PixelTarget, RenderPspDef, RenderView, SceneTarget};
use software3d::{DebugDrawOptions, Software3D};
use wad::WadData;

use level::LevelData;
use math::{Angle, Bam, FixedT};
use std::collections::HashSet;

const VIEWHEIGHT: f32 = 41.0;
const FOV: f32 = std::f32::consts::FRAC_PI_2;

/// Minimal headless DrawBuffer mirroring render-backend's resolve, so the test
/// exercises the real index→u32 path without the display backend.
struct HeadlessBuffer {
    size: BufferSize,
    data: Vec<u32>,
    index: Vec<u8>,
    w: usize,
}

impl HeadlessBuffer {
    fn new(w: usize, h: usize) -> Self {
        Self {
            size: BufferSize::new(w, h),
            data: vec![0u32; w * h],
            index: vec![0u8; w * h],
            w,
        }
    }
}

impl DrawBuffer for HeadlessBuffer {
    type Pixel = u32;

    fn size(&self) -> &BufferSize {
        &self.size
    }
    fn set_pixel(&mut self, x: usize, y: usize, colour: u32) {
        self.data[y * self.w + x] = colour;
    }
    fn get_buf_index(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }
    fn pitch(&self) -> usize {
        self.w
    }
    fn buf_mut(&mut self) -> &mut [u32] {
        &mut self.data
    }
    fn resolve(&mut self, pal_lit: &PalLit<u32>, use_palette: usize) {
        let block = pal_lit.block(use_palette);
        for (out, &idx) in self.data.iter_mut().zip(self.index.iter()) {
            *out = block[idx as usize];
        }
    }
}

impl SceneTarget for HeadlessBuffer {
    type Texel = u8;
    fn texel(&self, lit: u16) -> u8 {
        lit as u8
    }
    fn put(&mut self, pos: usize, texel: u8) {
        self.index[pos] = texel;
    }
    fn scene_fuzz(&mut self, dst_pos: usize, src_pos: usize, colourmap6: &[usize; 256]) {
        self.index[dst_pos] = colourmap6[self.index[src_pos] as usize] as u8;
    }
}

fn build_view(level: &mut LevelData) -> RenderView {
    let start = level.things().iter().find(|t| t.kind == 1).copied();
    let (x, y, angle) = match start {
        Some(t) => (t.x as f32, t.y as f32, t.angle as f32),
        None => (0.0, 0.0, 0.0),
    };
    let floor = level
        .point_in_subsector(FixedT::from_f32(x), FixedT::from_f32(y))
        .sector
        .floorheight
        .to_f32();
    let eye = floor + VIEWHEIGHT;
    let fp = FixedT::from_f32;
    RenderView {
        x: fp(x),
        y: fp(y),
        z: fp(eye),
        viewz: fp(eye),
        viewheight: fp(0.0),
        angle: Angle::<Bam>::new(angle.to_radians()),
        lookdir: 0.0,
        fixedcolormap: 0,
        extralight: 0,
        is_shadow: false,
        subsector_id: 0,
        psprites: [RenderPspDef::default(); 2],
        sector_lightlevel: 0,
        player_mobj_id: 0,
        frac: 1.0,
        frac_fp: fp(1.0),
        game_tic: 0,
    }
}

fn load(map: &str) -> Option<(LevelData, PicData)> {
    let path = test_utils::doom1_wad_path();
    if !path.exists() {
        eprintln!("skip golden_cross_path: {} not found", path.display());
        return None;
    }
    let wad = WadData::new(&path);
    let pics = PicData::init(&wad, &["TROO"]);
    let mut level = LevelData::default();
    level.load(map, |n| pics.flat_num_for_name(n), &wad, None, None);
    Some((level, pics))
}

/// Render the fixed pose via the index path and resolve to u32.
fn render_index_u32(w: usize, h: usize) -> Option<Vec<u32>> {
    let (mut level, mut pics) = load("E1M2")?;
    let mut r = Software3D::new(w as f32, h as f32, FOV, DebugDrawOptions::default());
    let view = build_view(&mut level);
    let mut buf = HeadlessBuffer::new(w, h);
    r.draw_view(&view, &level, &mut pics, &mut buf);
    let pal_lit: PalLit<u32> = pics.build_pal_lit(ByteOrder::Argb);
    buf.resolve(&pal_lit, 0);
    Some(buf.data)
}

/// Render the same pose via the direct `PixelTarget` path (final u32 pixels, no
/// resolve). Tight pitch, so the returned buffer is directly comparable.
fn render_direct_u32(w: usize, h: usize) -> Option<Vec<u32>> {
    let (mut level, mut pics) = load("E1M2")?;
    let mut r = Software3D::new(w as f32, h as f32, FOV, DebugDrawOptions::default());
    let view = build_view(&mut level);
    let pal_lit: PalLit<u32> = pics.build_pal_lit(ByteOrder::Argb);
    // The index path starts the plane at 0 and resolve maps 0 -> palette[0];
    // mirror that so undrawn (void) pixels compare equal.
    let void = pal_lit.block(pics.use_palette())[0];
    let mut surface = vec![void; w * h];
    {
        let mut target = PixelTarget::new(
            &mut surface,
            BufferSize::new(w, h),
            w,
            &pal_lit,
            pics.use_palette(),
        );
        r.draw_view(&view, &level, &mut pics, &mut target);
    }
    Some(surface)
}

/// Render the same pose via the direct `PixelTarget` (RGB565) path.
fn render_direct_u16(w: usize, h: usize) -> Option<Vec<u16>> {
    let (mut level, mut pics) = load("E1M2")?;
    let mut r = Software3D::new(w as f32, h as f32, FOV, DebugDrawOptions::default());
    let view = build_view(&mut level);
    let pal_lit: PalLit<u16> = pics.build_pal_lit(ByteOrder::Argb);
    let void = pal_lit.block(pics.use_palette())[0];
    let mut surface = vec![void; w * h];
    {
        let mut target = PixelTarget::new(
            &mut surface,
            BufferSize::new(w, h),
            w,
            &pal_lit,
            pics.use_palette(),
        );
        r.draw_view(&view, &level, &mut pics, &mut target);
    }
    Some(surface)
}

#[test]
fn index_path_renders_deterministic_nonblank_frame() {
    let (w, h) = (320, 200);
    let Some(a) = render_index_u32(w, h) else {
        return;
    };
    let distinct = a.iter().collect::<HashSet<_>>().len();
    assert!(
        distinct > 50,
        "expected a varied frame, got {distinct} colours"
    );
    let b = render_index_u32(w, h).expect("second render");
    assert_eq!(a, b, "index path must be deterministic across renders");
}

/// THE PHASE-3 GATE: the direct u32 scene path produces a frame bit-for-bit
/// equal to widening the index path through the same palette. Proves the scene
/// store rewrite is correct (no fuzz/bleed/translucency in this pose, so the
/// index-domain effects that diverge by construction are absent).
#[test]
fn direct_u32_equals_index_widened() {
    let (w, h) = (320, 200);
    let Some(index) = render_index_u32(w, h) else {
        return;
    };
    let direct = render_direct_u32(w, h).expect("direct render");
    assert_eq!(index.len(), direct.len());
    let mismatches = index
        .iter()
        .zip(direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches,
        0,
        "direct u32 must equal index-widened bit-for-bit ({mismatches}/{} pixels differ)",
        index.len()
    );
}

/// Direct u16 (RGB565) equals the index path 565-quantized: each direct pixel ==
/// `u16::from_argb(index_widened_pixel)`. Same E1M2 pose, no index-domain effects.
#[test]
fn direct_u16_equals_index_quantized() {
    let (w, h) = (320, 200);
    let Some(index) = render_index_u32(w, h) else {
        return;
    };
    let direct = render_direct_u16(w, h).expect("direct u16 render");
    assert_eq!(index.len(), direct.len());
    let mismatches = index
        .iter()
        .zip(direct.iter())
        .filter(|(argb, got)| u16::from_argb(**argb, ByteOrder::Argb) != **got)
        .count();
    assert_eq!(
        mismatches,
        0,
        "direct u16 must equal 565-quantized index ({mismatches}/{} differ)",
        index.len()
    );
}

/// The direct invariant on the data layer: `PalLit<u32>` (ARGB) applied to a lit
/// index reproduces exactly what the index path's resolve produces via the
/// active palette. This is what the direct store must preserve per pixel.
#[test]
fn pal_lit_u32_matches_palette_resolve() {
    let Some((_, pics)) = load("E1M2") else {
        return;
    };
    let pal_lit: PalLit<u32> = pics.build_pal_lit(ByteOrder::Argb);
    let palette = pics.palette(); // active tint (0 here) palette as u32
    let block = pal_lit.block(pics.use_palette());
    for i in 0..256 {
        assert_eq!(
            block[i], palette[i],
            "pal_lit[{i}] must equal palette[{i}] (direct == resolve for tint 0)"
        );
    }
}
