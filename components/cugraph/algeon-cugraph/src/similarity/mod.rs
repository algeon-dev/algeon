//! Similarity algorithms and the owned vertex-pair collection (`VertexPairs`).
//!
//! cuGraph similarity algorithms require symmetric graphs. This module leaves
//! graph-property validation to the native library so callers get the same
//! detailed diagnostics as the C API.

use std::num::NonZeroUsize;

mod algorithms;
mod result;
mod vertex_pairs;

pub use result::{CosineSimilarityResult, JaccardSimilarityResult, OverlapSimilarityResult, SorensenSimilarityResult};
pub use vertex_pairs::VertexPairs;

/// Parameters for pairwise similarity coefficient algorithms.
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct SimilarityParams {
    /// Whether to use edge weights when computing the coefficient.
    pub use_weight: bool,
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

/// Parameters for all-pairs similarity coefficient algorithms.
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct AllPairsSimilarityParams {
    /// Whether to use edge weights when computing the coefficient.
    pub use_weight: bool,
    /// Maximum number of output pairs to return. `None` requests all values.
    pub topk: Option<NonZeroUsize>,
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

pub(crate) use algorithms::{
    all_pairs_cosine_similarity_coefficients, all_pairs_jaccard_coefficients, all_pairs_overlap_coefficients,
    all_pairs_sorensen_coefficients, cosine_similarity_coefficients, create_vertex_pairs, jaccard_coefficients,
    overlap_coefficients, sorensen_coefficients,
};
