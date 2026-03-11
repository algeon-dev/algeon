//! Layout algorithms (Force Atlas 2, etc.).

pub mod force_atlas2;

pub use force_atlas2::{
    force_atlas2, ForceAtlas2Params, ForceAtlas2VertexAttrs, InitialPositions, LayoutResult,
};
