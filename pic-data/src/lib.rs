//! Visual asset data for rendering: textures, sprites, palettes, colourmaps,
//! and voxel models. Loaded from WAD and KVX/PK3 files.

pub mod colour;
pub mod pic;
pub mod voxel;

pub use colour::{ByteOrder, PALETTE_LEN, PalLit, PixelFmt, WadPalette};
pub use pic::sprites::{SpriteDef, SpriteFrame};
pub use pic::{
    CrtGamma, FlatPic, INVERSECOLORMAP, PaletteFade, PicAnimation, PicData, SpritePic, Switches,
    WallPic,
};
pub use voxel::VoxelManager;
pub use voxel::slices::{VoxelColumn, VoxelSliceQuad, VoxelSlices, VoxelSpan};
