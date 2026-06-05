pub mod wipe;

use level::LevelData;
use math::{Angle, Bam, FixedT, m_random};
use pic_data::PicData;

// =============================================================================
// Constants
// =============================================================================

/// OG Doom's original resolution.
const OG_WIDTH: f32 = 320.0;
const OG_HEIGHT: f32 = 200.0;

/// OG STBAR height in native 200px space.
pub const STBAR_HEIGHT: i32 = 32;

/// Classic Doom fuzz Y-offsets. The table cycles per-pixel to create the
/// spectre shimmer effect.
pub const FUZZ_TABLE: [i32; 50] = [
    1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1,
    -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1,
];

// Health-bleed tuning. The effect: red columns hang from the top, longer the
// lower the health, jagged with occasional tall peaks biased toward the edges.

/// First PLAYPAL red palette; `COUNT` bands run light (top) to dark (count).
const BLEED_PAL_START: usize = 1;
const BLEED_PAL_COUNT: usize = 3;
/// Tallest a peak column can reach (screen height fraction).
const BLEED_MAX_FRAC: f32 = 1.0;
/// Resting length of a non-peak column (screen height fraction).
const BLEED_BASE_FRAC: f32 = 0.03;
/// Baseline roughness amplitude (screen height fraction).
const BLEED_BASE_JITTER: f32 = 0.24;
/// Scales the baseline field, lower = shorter (multiplier, 0..1).
const BLEED_BASE_SCALE: f32 = 0.85;
/// Value-noise control-point spacing, larger = broader curves (pixels).
const BLEED_NOISE_STEP: usize = 14;
/// Depth of one palette band (screen height fraction).
const BLEED_BAND_FRAC: f32 = 0.02;
/// Peaks occur roughly 1 column in N (count).
const BLEED_PEAK_RATE: i32 = 50;
/// Columns each side of a peak that taper it into the baseline hump (pixels).
const BLEED_PEAK_BLEND: usize = 4;
/// Peak height at screen centre vs the edge (multiplier, 0..1).
const BLEED_CENTER_SHORT: f32 = 0.5;
/// Edge→centre shortening curve, >1 stays tall toward edges (exponent).
const BLEED_EDGE_POW: f32 = 2.0;

/// Per-pixel scanout effect selecting which PLAYPAL palette resolves a pixel:
/// `out[i] = palettes_flat[palette_offset(x, y) * 256 + idx[i]]`. 0 = base.
pub trait ScreenEffect {
    fn palette_offset(&self, x: usize, y: u16, use_palette: usize) -> usize;
    fn is_active(&self) -> bool {
        true
    }
}

/// Health bleed: jagged red columns hang from the top of the screen, grown to
/// `target[x] * (100 - health) / 100` and banded top-darkest to bottom-lightest.
/// `target`/`band_off` are built once per size; only `shown` is recomputed when
/// health changes.
pub struct HealthBleed {
    /// Per-column max length, px (the full jagged shape: baseline + peaks).
    target: Vec<u16>,
    /// Per-column current length, px, for the active health.
    shown: Vec<u16>,
    /// Per-column `from_edge` thresholds where each band ends (the last band
    /// runs to the top). Precomputed per health change so the hot path is a
    /// couple of compares — no per-pixel divide, multiply or jitter lookup.
    bound: Vec<[u16; BLEED_PAL_COUNT - 1]>,
    /// Per-column band offset, shifting where the palette boundaries fall so
    /// the transitions are ragged across columns.
    band_off: Vec<i32>,
    /// Depth of one palette band, px.
    band_px: usize,
    /// Health the `shown`/`bound` caches were computed for; `update` is a no-op
    /// when it (and the size) are unchanged.
    last_health: i32,
    width: usize,
    height: usize,
    active: bool,
}

impl Default for HealthBleed {
    fn default() -> Self {
        Self {
            target: Vec::new(),
            shown: Vec::new(),
            bound: Vec::new(),
            band_off: Vec::new(),
            band_px: 1,
            last_health: 100,
            width: 0,
            height: 0,
            active: false,
        }
    }
}

impl HealthBleed {
    pub fn update(&mut self, health: i32, width: usize, height: usize) {
        let health = health.clamp(0, 100);
        if self.target.is_empty() || width != self.width || height != self.height {
            self.rebuild(width, height);
            self.last_health = -1; // force the recompute below
        }
        // Caches only depend on health; skip the recompute when it is unchanged.
        if health == self.last_health {
            return;
        }
        self.last_health = health;
        self.active = health < 100;
        if !self.active {
            self.shown.fill(0);
            return;
        }
        let drop = (100 - health) as u32;
        for x in 0..self.width {
            let shown = (self.target[x] as u32 * drop / 100) as usize;
            self.shown[x] = shown as u16;
            // Band depth: fixed for short columns, stretched for tall peaks so
            // the gradient spans the whole column. Thresholds = band end depths.
            let eff = self.band_px.max(shown / BLEED_PAL_COUNT) as i32;
            let off = self.band_off[x];
            for b in 0..BLEED_PAL_COUNT - 1 {
                self.bound[x][b] = (((b as i32 + 1) * eff) + off).clamp(0, u16::MAX as i32) as u16;
            }
        }
    }

    /// Regenerate the random column pattern on the next `update` — call on a new
    /// game / level load so each level gets a fresh bleed shape.
    pub fn reset(&mut self) {
        self.target.clear();
    }

    /// Build the per-column targets, band depth and jittered boundaries for a
    /// given size (cold path: size change or [`Self::reset`]).
    fn rebuild(&mut self, width: usize, height: usize) {
        self.target = build_column_targets(width, height);
        self.shown = vec![0u16; width];
        self.bound = vec![[0u16; BLEED_PAL_COUNT - 1]; width];
        self.band_px = (height as f32 * BLEED_BAND_FRAC).max(1.0) as usize;
        // Per-column boundary jitter, centred on zero (±band_px/2).
        let amp = self.band_px as i32;
        self.band_off = smooth_noise(width, amp, BLEED_NOISE_STEP)
            .into_iter()
            .map(|v| v - amp / 2)
            .collect();
        self.width = width;
        self.height = height;
    }
}

impl ScreenEffect for HealthBleed {
    #[inline(always)]
    fn palette_offset(&self, x: usize, y: u16, use_palette: usize) -> usize {
        let shown = unsafe { *self.shown.get_unchecked(x) };
        if y >= shown {
            return use_palette;
        }
        // Count how many precomputed band thresholds this depth exceeds — a
        // couple of compares, no arithmetic.
        let from_edge = shown - 1 - y;
        let bounds = unsafe { self.bound.get_unchecked(x) };
        let mut band = 0;

        while band < BLEED_PAL_COUNT - 1
            && from_edge >=
        // SAFETY: the `band < BLEED_PAL_COUNT - 1` guard bounds the index into
        // `bounds: [_; BLEED_PAL_COUNT - 1]`.
        unsafe { *bounds.get_unchecked(band) }
        {
            band += 1;
        }
        BLEED_PAL_START + band
    }

    #[inline(always)]
    fn is_active(&self) -> bool {
        self.active
    }
}

/// Build the full per-column max-length shape: a smooth value-noise baseline,
/// then sharp peaks placed on top — each swelling the baseline into a soft
/// crest so it rises out of a hump, not an arbitrary baseline height.
fn build_column_targets(width: usize, height: usize) -> Vec<u16> {
    let max_len = (height as f32 * BLEED_MAX_FRAC) as i32;
    let base_len = (height as f32 * BLEED_BASE_FRAC) as i32;
    let span = (max_len - base_len).max(1);
    let jitter = (height as f32 * BLEED_BASE_JITTER).max(1.0) as i32;

    let mut len: Vec<i32> = smooth_noise(width, jitter, BLEED_NOISE_STEP)
        .into_iter()
        .map(|n| ((base_len + n) as f32 * BLEED_BASE_SCALE) as i32)
        .collect();

    // Place a peak at column `cx` of height `rise`: a wide soft cosine crest
    // under it, then the sharp feathered spike on top.
    let crest_radius = BLEED_NOISE_STEP.max(1);
    let crest = base_len + jitter;
    let place_peak = |len: &mut [i32], cx: usize, rise: i32| {
        for d in 0..=crest_radius {
            let t = d as f32 / crest_radius as f32;
            let hump = (base_len as f32
                + (crest - base_len) as f32 * (1.0 + (t * std::f32::consts::PI).cos()) * 0.5)
                as i32;
            if cx >= d {
                len[cx - d] = len[cx - d].max(hump);
            }
            if cx + d < width {
                len[cx + d] = len[cx + d].max(hump);
            }
        }
        for d in 0..=BLEED_PEAK_BLEND {
            let frac = 1.0 - 0.5 * d as f32 / BLEED_PEAK_BLEND as f32;
            let lift = base_len + (rise as f32 * frac) as i32;
            if cx >= d {
                len[cx - d] = len[cx - d].max(lift);
            }
            if cx + d < width {
                len[cx + d] = len[cx + d].max(lift);
            }
        }
    };

    // Random peaks, shorter toward the centre (see BLEED_CENTER_SHORT/EDGE_POW).
    let half_w = (width as f32 * 0.5).max(1.0);
    for x in 0..width {
        if m_random() % BLEED_PEAK_RATE != 0 {
            continue;
        }
        let edge = (x as f32 - half_w).abs() / half_w; // 0 centre .. 1 edge
        let scale = BLEED_CENTER_SHORT + (1.0 - BLEED_CENTER_SHORT) * edge.powf(BLEED_EDGE_POW);
        let rise = ((span / 2 + (m_random() % (span / 2).max(1))) as f32 * scale) as i32;
        place_peak(&mut len, x, rise);
    }

    // Fixed full-height peaks anchoring each screen edge.
    for &x in &[1usize, width.wrapping_sub(2)] {
        if x < width {
            place_peak(&mut len, x, max_len - base_len);
        }
    }

    len.iter().map(|&l| l.clamp(0, max_len) as u16).collect()
}

/// Value noise: a random control point in `0..amplitude` every `step` columns,
/// cosine-interpolated between them. Smooth undulation with no per-column
/// alternation; `step` is the feature size (larger = broader curves).
fn smooth_noise(width: usize, amplitude: i32, step: usize) -> Vec<i32> {
    let step = step.max(1);
    let ctrl: Vec<f32> = (0..width / step + 2)
        .map(|_| (m_random() % amplitude.max(1)) as f32)
        .collect();

    (0..width)
        .map(|x| {
            let i = x / step;
            let t = (x % step) as f32 / step as f32;
            let t = (1.0 - (t * std::f32::consts::PI).cos()) * 0.5;
            (ctrl[i] + (ctrl[i + 1] - ctrl[i]) * t) as i32
        })
        .collect()
}

// =============================================================================
// Traits
// =============================================================================

/// Owns the renderer + display backend. A frame is composed inside
/// [`GameRenderer::with_frame`], which acquires the display surface, runs the
/// caller's closure against a [`Frame`], then presents.
pub trait GameRenderer {
    /// Query the size things are drawn at, outside a frame (for layout/setup).
    fn buffer_size(&self) -> &BufferSize;

    /// Whether a wipe transition is in progress.
    fn is_wiping(&self) -> bool;

    /// Regenerate the bleed's random column pattern (call on new game / level
    /// load so each level gets a fresh shape).
    fn reset_health_bleed(&mut self);

    /// Acquire the display surface and compose one frame into it via `body`,
    /// then present. Everything that draws the frame (scene, wipe, UI) runs
    /// inside `body` against the [`Frame`].
    fn with_frame(&mut self, body: impl FnOnce(&mut Self::Frame<'_>));

    /// The per-frame draw context produced by [`Self::with_frame`].
    type Frame<'a>: Frame
    where
        Self: 'a;
}

/// Per-frame draw operations, valid only inside [`GameRenderer::with_frame`].
/// The display surface is acquired for the duration; the scene resolves into it
/// and UI draws onto it directly.
pub trait Frame {
    /// Begin a wipe: size + clear the old-frame snapshot buffer. The caller then
    /// draws the old state into it via [`Self::wipe_buffer`] /
    /// [`Self::resolve_index_into_wipe`].
    fn start_wipe(&mut self);

    /// The old-frame wipe buffer as a [`DrawBuffer`], for the caller to draw the
    /// old state into (`0xAARRGGBB`, width-pitched).
    fn wipe_buffer(&mut self) -> impl DrawBuffer;

    /// Resolve the current index plane into the wipe buffer — the old `Level`
    /// view (still present in the index at wipe start).
    fn resolve_index_into_wipe(&mut self, pic_data: &PicData);

    /// Render the 3D view, filling the 8-bit index plane (the scene is not yet
    /// resolved to the surface — see [`Self::finish_scene`]).
    fn render_player_view(
        &mut self,
        view: &RenderView,
        level_data: &LevelData,
        pic_data: &mut PicData,
    );

    /// Melt the old-frame wipe buffer one step over the display surface. Returns
    /// true when complete. Called after the new frame is composited.
    fn do_wipe(&mut self) -> bool;

    /// Resolve the index plane (+ health bleed) into the display surface, then
    /// draw any debug overlays. No-op resolve when the scene wrote the surface
    /// directly (debug colour modes).
    fn finish_scene(&mut self, pic_data: &PicData);

    /// The surface as a [`DrawBuffer`] for UI subsystems to draw onto.
    fn draw_buffer(&mut self) -> &mut impl DrawBuffer;

    fn size(&self) -> &BufferSize;

    /// Update the health bleed effect for this frame's resolve. Pass 100 to clear.
    fn set_health_bleed(&mut self, health: i32);
}

#[cfg(test)]
mod bleed_tests {
    use super::{BLEED_PAL_COUNT, BLEED_PAL_START, HealthBleed, ScreenEffect as _};

    const W: usize = 64;
    const H: usize = 120;

    /// Lit pixel count over the whole frame.
    fn lit(v: &HealthBleed) -> usize {
        (0..v.height)
            .flat_map(|y| (0..v.width).map(move |x| (x, y as u16)))
            .filter(|&(x, y)| v.palette_offset(x, y, 0) != 0)
            .count()
    }

    /// Full health: inactive, no pixel tinted (columns zero-length).
    #[test]
    fn full_health_inactive() {
        let mut v = HealthBleed::default();
        v.update(100, W, H);
        assert!(!v.is_active());
        assert_eq!(lit(&v), 0);
    }

    /// Damage hangs columns from the top: row 0 of a column tints, the bottom
    /// stays clear.
    #[test]
    fn damage_tints_top_not_bottom() {
        let mut v = HealthBleed::default();
        v.update(20, W, H);
        assert!(v.is_active());
        let top = v.palette_offset(0, 0, 0);
        assert!(top >= BLEED_PAL_START, "top of column tints");
        assert!(top < BLEED_PAL_START + BLEED_PAL_COUNT, "in red range");
        assert_eq!(
            v.palette_offset(0, H as u16 - 1, 0),
            0,
            "bottom stays clear"
        );
    }

    /// Lower health grows the columns (more lit pixels).
    #[test]
    fn lower_health_grows_columns() {
        let mut v = HealthBleed::default();
        v.update(80, W, H);
        let mid = lit(&v);
        v.update(20, W, H);
        let low = lit(&v);
        assert!(low > mid, "lower health => longer columns");
    }

    /// On average the palette lightens top → bottom (leading edge lightest).
    /// Per-boundary jitter can break the order in a single column, so test the
    /// mean palette of the top vs bottom row across all columns.
    #[test]
    fn column_lightens_downward() {
        let mut v = HealthBleed::default();
        v.update(0, W, H);
        let row_mean = |y: u16| -> f32 {
            let sum: usize = (0..v.width).map(|x| v.palette_offset(x, y, 0)).sum();
            sum as f32 / v.width as f32
        };
        assert!(
            row_mean(0) > row_mean(H as u16 - 1),
            "top darker than bottom on average"
        );
    }

    /// Restoring to 100 clears the effect.
    #[test]
    fn clears_when_restored() {
        let mut v = HealthBleed::default();
        v.update(10, W, H);
        assert!(v.is_active());
        v.update(100, W, H);
        assert!(!v.is_active());
        assert_eq!(lit(&v), 0);
    }
}

/// A u32 draw target plus the 8-bit scene index plane behind it. The renderer
/// fills the index plane (`set_index`/`index_mut`) then `resolve`s it to u32;
/// UI draws u32 directly (`set_pixel`/`buf_mut`). For the real backend the u32
/// side is the display surface (possibly strided — always use `pitch`).
pub trait DrawBuffer {
    fn size(&self) -> &BufferSize;
    /// Write a `0xAARRGGBB` pixel to the resolved display surface (UI/overlays).
    fn set_pixel(&mut self, x: usize, y: usize, colour: u32);
    /// Read a `0xAARRGGBB` pixel from the resolved display surface.
    fn read_pixel(&self, x: usize, y: usize) -> u32;
    /// Flat position of `(x, y)` in the 8-bit index plane (`y * pitch() + x`).
    /// Pairs with [`Self::index_mut`] and [`Self::pitch`] for column stepping.
    fn get_buf_index(&self, x: usize, y: usize) -> usize;
    /// Row stride of the index plane in elements (the buffer width; the display
    /// surface may be padded wider, but that padding is hidden behind the pixel
    /// accessors).
    fn pitch(&self) -> usize;
    fn buf_mut(&mut self) -> &mut [u32];

    /// Write a palette index to the 8-bit scene plane (the hot geometry path).
    fn set_index(&mut self, x: usize, y: usize, idx: u8);
    /// The 8-bit palette-index plane, for inner-loop `idx[pos] = colourmap[src]`.
    fn index_mut(&mut self) -> &mut [u8];
    /// Resolve the index plane to u32. No effect: `out[i] = palette[idx[i]]`.
    /// Bleed active: per-pixel select into `palettes_flat` (`PALLETE_LEN*256`).
    fn resolve(&mut self, palette: &[u32], palettes_flat: &[u32], use_palette: usize);
}

// =============================================================================
// Types
// =============================================================================

/// Pre-resolved weapon sprite for rendering. Extracted from gameplay's PspDef
/// so render-common doesn't depend on StateData/SpriteNum.
#[derive(Clone, Copy)]
pub struct RenderPspDef {
    pub active: bool,
    pub sprite: usize,
    pub frame: u32,
    pub sx: f32,
    pub sy: f32,
}

impl Default for RenderPspDef {
    fn default() -> Self {
        Self {
            active: false,
            sprite: 0,
            frame: 0,
            sx: 0.0,
            sy: 0.0,
        }
    }
}

/// Snapshot of player/level state needed for one frame.
pub struct RenderView {
    /// Player world X position.
    pub x: FixedT,
    /// Player world Y position.
    pub y: FixedT,
    /// Player z (floor level of mobj).
    pub z: FixedT,
    /// Eye height (includes bobbing).
    pub viewz: FixedT,
    /// Base eye height above floor.
    pub viewheight: FixedT,
    /// Facing angle (BAM).
    pub angle: Angle<Bam>,
    /// Vertical look pitch in radians.
    pub lookdir: f32,
    /// Fixed colormap index (0 = off, >0 = invulnerability/light goggles).
    pub fixedcolormap: usize,
    /// Extra brightness from gun flash.
    pub extralight: usize,
    /// Whether the player mobj has the Shadow flag (partial invisibility).
    pub is_shadow: bool,
    /// Pre-resolved subsector index into `LevelData::subsectors()`.
    pub subsector_id: usize,
    /// Weapon overlay sprites.
    pub psprites: [RenderPspDef; 2],
    /// Sector light level at player's subsector (for weapon sprite lighting).
    pub sector_lightlevel: usize,
    /// Player mobj pointer for skipping own sprite in rendering.
    pub player_mobj_id: usize,
    /// Sub-tic interpolation fraction (0.0..1.0) for smooth rendering.
    pub frac: f32,
    /// Sub-tic interpolation fraction in fixed-point.
    pub frac_fp: FixedT,
    /// Monotonic game tic counter (for time-based effects like voxel spin).
    pub game_tic: u32,
}

/// Pre-computed buffer dimensions for fast 2.5D rendering.
#[derive(Clone, Copy)]
pub struct BufferSize {
    hi_res: bool,
    width_usize: usize,
    height_usize: usize,
    width: i32,
    height: i32,
    width_f32: f32,
    height_f32: f32,
    statusbar_height: i32,
}

impl BufferSize {
    pub const fn new(width: usize, height: usize) -> Self {
        Self {
            hi_res: height > 200,
            width_usize: width,
            height_usize: height,
            width: width as i32,
            height: height as i32,
            width_f32: width as f32,
            height_f32: height as f32,
            statusbar_height: 0,
        }
    }

    pub const fn hi_res(&self) -> bool {
        self.hi_res
    }
    pub const fn width(&self) -> i32 {
        self.width
    }
    pub const fn height(&self) -> i32 {
        self.height
    }
    pub const fn half_width(&self) -> i32 {
        self.width / 2
    }
    pub const fn half_height(&self) -> i32 {
        self.height / 2
    }
    pub const fn width_usize(&self) -> usize {
        self.width_usize
    }
    pub const fn height_usize(&self) -> usize {
        self.height_usize
    }
    pub const fn width_f32(&self) -> f32 {
        self.width_f32
    }
    pub const fn height_f32(&self) -> f32 {
        self.height_f32
    }
    pub const fn half_width_f32(&self) -> f32 {
        self.half_width() as f32
    }
    pub const fn half_height_f32(&self) -> f32 {
        self.half_height() as f32
    }

    pub fn set_statusbar_height(&mut self, h: i32) {
        self.statusbar_height = h;
    }
    pub const fn statusbar_height(&self) -> i32 {
        self.statusbar_height
    }
    pub const fn view_height(&self) -> i32 {
        self.height - self.statusbar_height
    }
    pub const fn view_height_usize(&self) -> usize {
        (self.height - self.statusbar_height) as u32 as usize
    }
    pub const fn view_height_f32(&self) -> f32 {
        (self.height - self.statusbar_height) as f32
    }
    pub const fn half_view_height(&self) -> i32 {
        self.view_height() / 2
    }
    pub const fn half_view_height_f32(&self) -> f32 {
        self.half_view_height() as f32
    }
}

// =============================================================================
// Projection
// =============================================================================

/// Derive FOV and focal length for a given buffer size, keeping the view
/// proportional to OG Doom's 320x200 at the specified base hfov (90°).
/// Scales with buffer height so hi-res (400p) works correctly.
/// Returns `(hfov, vfov, focal_length)` in radians / pixels.
pub fn og_projection(base_hfov: f32, buf_width: f32, buf_height: f32) -> (f32, f32, f32) {
    let scale = buf_height / OG_HEIGHT;
    let og_half_w = (OG_WIDTH / 2.0) * scale;
    let focal_length = og_half_w / (base_hfov / 2.0).tan();
    let hfov = 2.0 * ((buf_width / 2.0) / focal_length).atan();
    let vfov = 2.0 * ((buf_height / 2.0) / focal_length).atan();
    (hfov, vfov, focal_length)
}
