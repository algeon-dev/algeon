//! Community-detection algorithms.
//!
//! Louvain is currently exposed here. The module layout leaves room for
//! additional hierarchical community algorithms, such as Leiden and ECG, to
//! share internal result ownership without changing the public surface.

pub mod leiden;
pub mod louvain;
mod result;
pub mod subgraph;
pub mod triangle_count;

pub use leiden::LeidenParams;
pub use louvain::LouvainParams;
pub use result::{LeidenResult, LouvainResult};
pub use subgraph::{ExtractEgoParams, ExtractEgoResult, KTrussSubgraphParams, KTrussSubgraphResult};
pub use triangle_count::{TriangleCountParams, TriangleCountResult};
