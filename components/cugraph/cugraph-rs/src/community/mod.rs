//! Community-detection algorithms.
//!
//! Louvain is currently exposed here. The module layout leaves room for
//! additional hierarchical community algorithms, such as Leiden and ECG, to
//! share internal result ownership without changing the public surface.

pub mod louvain;
mod result;
pub mod triangle_count;

pub use louvain::LouvainParams;
pub use result::LouvainResult;
pub use triangle_count::{TriangleCountParams, TriangleCountResult};
