//! Safe Rust bindings for NVIDIA cuGraph GPU-accelerated graph analytics.
//!
//! This crate wraps the cuGraph C API in safe Rust types with RAII semantics,
//! zero-copy array views, and structured error handling.
//!
//! # Key concepts
//!
//! - **[`ResourceHandle`]** — manages GPU resources (CUDA stream, RAFT handle).
//!   Required by every cuGraph operation. Not thread-safe.
//! - **[`DeviceArray<T>`]** — owning GPU array with compile-time element type.
//!   Freed on drop. Create views via [`DeviceArray::view()`].
//! - **[`GraphBuilder`]** — fluent builder for constructing a [`Graph`] from
//!   edge lists. Validates inputs before crossing the FFI boundary.
//! - **[`Graph<T>`]** — the central graph data structure, parameterized by
//!   transposition state ([`Forward`] or [`Transposed`]). Algorithms are methods
//!   on `Graph`, constrained by the required layout.
//!
//! # Quick start
//!
//! ```no_run
//! use cugraph::{
//!     DeviceArray, GraphBuilder, PageRankParams, ResourceHandle, Transposed,
//! };
//!
//! # fn main() -> Result<(), cugraph::Error> {
//! let handle = ResourceHandle::new()?;
//!
//! let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2])?;
//! let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0])?;
//!
//! let mut graph = GraphBuilder::new(&src.view(), &dst.view())
//!     .renumber()
//!     .build::<Transposed>(&handle)?;
//!
//! let result = graph.pagerank_simple(&handle, &PageRankParams::default())?;
//! let vertices: Vec<i64> = result.vertex_ids()?.to_vec_i64(&handle)?;
//! let scores: Vec<f64> = result.scores()?.to_vec_f64(&handle)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Type-state encoding
//!
//! The [`Graph<T>`] type parameter encodes whether the graph is stored in
//! forward or transposed form. Algorithms that require a specific layout
//! accept only the correct variant, catching mismatches at compile time:
//!
//! - **[`Forward`]** — required by [`Graph::bfs`]
//! - **[`Transposed`]** — required by [`Graph::pagerank`],
//!   [`Graph::in_degrees`], [`Graph::degrees`]
//! - **Generic** (`T: Transposition`) — [`Graph::betweenness_centrality`],
//!   [`Graph::force_atlas2`], [`Graph::out_degrees`]
//!
//! # Feature flags
//!
//! - **`cudf-interop`** — enables zero-copy interop with the [`cudf`] crate,
//!   including [`graph_from_parquet`](cudf_interop::graph_from_parquet),
//!   [`graph_from_dataframe`](cudf_interop::graph_from_dataframe), and
//!   result-to-DataFrame conversions.

pub mod array;
pub mod centrality;
pub mod error;
pub mod graph;
pub mod handle;
pub mod layout;
pub mod params;
pub mod random;
pub mod structure;
pub mod traversal;
pub mod types;

#[cfg(feature = "cudf-interop")]
pub mod cudf_interop;

// ═══════════════════════════════════════════════════════════════════════════
// Crate-root re-exports — the recommended import path for common types
// ═══════════════════════════════════════════════════════════════════════════

// Arrays
pub use array::{DeviceArray, DeviceArrayView, DeviceBuffer};
// Centrality algorithms
pub use centrality::{
    BetweennessCentralityParams, BetweennessCentralityResult, InitialGuess, PageRankOptions, PageRankParams,
    PageRankResult, PrecomputedWeights,
};
#[cfg(feature = "cudf-interop")]
pub use cudf_interop::{DataFrameGraphOptions, LoadedGraph, ParquetGraphOptions, TableGraphOptions};
// Error handling
pub use error::{Error, ErrorKind, ErrorStatus};
// Graph construction and type-state
pub use graph::{Forward, Graph, GraphBuilder, Transposed, Transposition};
// Infrastructure
pub use handle::ResourceHandle;
// Layout algorithms
pub use layout::{ForceAtlas2Options, ForceAtlas2Params, ForceAtlas2VertexAttrs, InitialPositions, LayoutResult};
// Parameter newtypes
pub use params::{BfsDepthLimit, ClosedUnitF64, IterationCount, NonNegativeF64, PositiveF64, UnitF64};
pub use random::RngState;
// Structure queries
pub use structure::{
    DegreeValues, DegreeVertices, DegreesResult, Distances, InDegreesResult, OutDegreesResult, ScoreValues, VertexIds,
};
// Traversal algorithms
pub use traversal::{BfsParams, BfsResult};
pub use types::{CugraphElement, DataType, GraphProperties};
