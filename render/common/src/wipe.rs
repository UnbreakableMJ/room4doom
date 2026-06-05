use std::time::Instant;

use math::m_random;

/// Duration between wipe steps (~60Hz).
const STEP_INTERVAL_MS: u128 = 16;

/// Opaque black (`0xAARRGGBB`), used to clear the snapshot so uncaptured pixels
/// melt opaque (some display backends require fully-opaque surface pixels).
const OPAQUE_BLACK: u32 = 0xFF00_0000;

pub struct Wipe {
    y: Vec<i32>,
    height: i32,
    width: i32,
    /// The old frame, drawn into by the caller when a wipe starts and melted
    /// over the display surface. Width-pitched `0xAARRGGBB`.
    snapshot: Vec<u32>,
    /// Time of the last wipe step, used to gate advancement.
    last_step: Instant,
}

impl Wipe {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            y: Self::init_offsets(width),
            height,
            width,
            snapshot: Vec::new(),
            last_step: Instant::now(),
        }
    }

    /// Generate the random jagged column offsets for a new wipe.
    fn init_offsets(width: i32) -> Vec<i32> {
        let mut y = Vec::with_capacity(width as usize);
        y.push(-(m_random() % 16));

        for i in 1..width as usize {
            let r = (m_random() % 3) - 1;
            y.push(y[i - 1] + r);
            if y[i] > 0 {
                y[i] = 0;
            } else if y[i] <= -16 {
                y[i] = -15;
            }
        }
        y
    }

    pub fn reset(&mut self) {
        self.y = Self::init_offsets(self.width);
        self.snapshot.clear();
    }

    /// Begin a wipe: size the snapshot buffer and clear it to opaque black for
    /// the caller to draw the old frame into (via [`Self::snapshot_mut`]).
    /// Opaque (`0xFF` alpha) because the snapshot is melted onto the display
    /// surface, which some backends require to be fully opaque.
    pub fn start(&mut self) {
        let n = self.width as usize * self.height as usize;
        self.snapshot.clear();
        self.snapshot.resize(n, OPAQUE_BLACK);
        self.last_step = Instant::now();
    }

    /// The old-frame buffer, width-pitched `0xAARRGGBB`, for the caller to draw
    /// the old state into after [`Self::start`].
    pub fn snapshot_mut(&mut self) -> &mut [u32] {
        &mut self.snapshot
    }

    /// Returns true if a wipe is in progress (snapshot captured).
    pub fn is_wiping(&self) -> bool {
        !self.snapshot.is_empty()
    }

    /// Overdraw shifted old-frame columns on top of the display surface.
    ///
    /// The caller must have already composited the new scene into `buf`. This
    /// paints the old frame's columns shifted down, covering the bottom portion
    /// where the old scene should still be visible.
    ///
    /// `buf` is the display surface (`surface_pitch` elements per row, possibly
    /// padded); the snapshot is width-pitched, so the two are addressed with
    /// their own pitches.
    ///
    /// Only advances the melt when at least `STEP_INTERVAL_MS` has elapsed
    /// since the last step; otherwise it redraws the current state without
    /// advancing.
    ///
    /// Returns true when the melt is complete.
    pub fn do_melt_pixels(&mut self, buf: &mut [u32], surface_pitch: usize) -> bool {
        let elapsed = self.last_step.elapsed().as_millis();
        let should_step = elapsed >= STEP_INTERVAL_MS;
        if should_step {
            self.last_step = Instant::now();
        }

        let mut done = true;
        let stepping = self.height as usize / 100;
        let f = self.height / 200;
        let src_pitch = self.width as usize;

        for x in (0..self.width as usize - stepping).step_by(stepping) {
            if self.y[x] < 0 {
                if should_step {
                    self.y[x] += stepping as i32 / 2;
                }
                // Column hasn't started melting yet — overdraw entire column
                // with old frame pixels.
                for col in x..x + stepping {
                    for row in 0..self.height as usize {
                        buf[row * surface_pitch + col] = self.snapshot[row * src_pitch + col];
                    }
                }
                done = false;
            } else if self.y[x] < self.height {
                let melt_y = self.y[x] as usize;

                // Overdraw: paint old-frame pixels shifted down by melt_y.
                // Old row 0..(height - melt_y) appears at display rows
                // melt_y..height.
                for col in x..x + stepping {
                    for src_y in 0..(self.height as usize - melt_y) {
                        let dst_y = src_y + melt_y;
                        buf[dst_y * surface_pitch + col] = self.snapshot[src_y * src_pitch + col];
                    }
                }

                if should_step {
                    let mut dy = if self.y[x] < (16 * f) {
                        self.y[x] + stepping as i32
                    } else {
                        8 * f
                    };
                    if self.y[x] + dy >= self.height {
                        dy = self.height - self.y[x];
                    }
                    for col in x..x + stepping {
                        if col < self.y.len() {
                            self.y[col] += dy;
                        }
                    }
                }
                done = false;
            }
            // else: column fully melted, new scene shows through
        }
        done
    }
}
