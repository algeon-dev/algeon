//! Community-detection algorithms.
//!
//! Louvain is currently exposed here. The module layout leaves room for
//! additional hierarchical community algorithms, such as Leiden and ECG, to
//! share internal result ownership without changing the public surface.

pub mod clustering;
pub mod ecg;
pub mod leiden;
pub mod louvain;
mod result;
pub mod subgraph;
pub mod triangle_count;

#[allow(deprecated)]
pub use clustering::{BalancedCutClusteringParams, BalancedCutClusteringResult};
pub use clustering::{SpectralClusteringParams, SpectralClusteringResult};
pub use ecg::EcgParams;
pub use leiden::LeidenParams;
pub use louvain::LouvainParams;
pub use result::{EcgResult, LeidenResult, LouvainResult};
pub use subgraph::{ExtractEgoParams, ExtractEgoResult, KTrussSubgraphParams, KTrussSubgraphResult};
pub use triangle_count::{TriangleCountParams, TriangleCountResult};
