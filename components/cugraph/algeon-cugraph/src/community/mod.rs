//! Community-detection algorithms.
//!
//! This module exposes hierarchical community detection, clustering analysis,
//! triangle counting, and community-oriented subgraph algorithms.

pub mod clustering;
pub mod ecg;
pub mod leiden;
pub mod louvain;
mod result;
pub mod subgraph;
pub mod triangle_count;

pub use clustering::{SpectralClusteringParams, SpectralClusteringResult};
pub use ecg::EcgParams;
pub use leiden::LeidenParams;
pub use louvain::LouvainParams;
pub use result::{EcgResult, LeidenResult, LouvainResult};
pub use subgraph::{ExtractEgoParams, ExtractEgoResult, KTrussSubgraphParams, KTrussSubgraphResult};
pub use triangle_count::{TriangleCountParams, TriangleCountResult};
