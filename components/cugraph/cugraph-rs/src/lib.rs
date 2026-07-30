#![warn(clippy::undocumented_unsafe_blocks)]

//! Safe Rust bindings for NVIDIA cuGraph GPU-accelerated graph analytics.
//!
//! This crate wraps the cuGraph C API in safe Rust types with RAII semantics,
//! zero-copy array views, and structured error handling.
//!
//! For most applications, the intended entry point is
//! [`cudf_interop`]: build graphs from cudf
//! [`Series`](cudf::Series)/[`DataFrame`](cudf::DataFrame), run algorithms,
//! then convert results back to cudf tables. [`DeviceArray`] and
//! [`GraphBuilder`] remain available for lower-level workflows that already own
//! typed device arrays.
//!
//! # Key concepts
//!
//! - **[`cudf_interop`]** — preferred graph-ingestion path for cudf-backed
//!   applications. Includes `graph_from_series`, `graph_from_dataframe`, and
//!   Parquet helpers.
//! - **[`ResourceHandle`]** — long-lived owner of GPU execution provenance
//!   (bound device, retained runtime, cuGraph resource handle). Not
//!   thread-safe.
//! - **[`EnteredContext`]** — short-lived execution session created by
//!   [`ResourceHandle::enter`]. GPU-capable APIs require this token.
//! - **[`DeviceArray<T>`]** — owning GPU array with compile-time element type.
//!   Freed on drop. Create views via [`DeviceArray::view()`].
//! - **[`GraphBuilder`]** — low-level builder for constructing a [`Graph`] from
//!   pre-materialized device edge lists. Validates inputs before crossing the
//!   FFI boundary.
//! - **[`Graph<T>`]** — the central graph data structure, parameterized by
//!   transposition state ([`Forward`] or [`Transposed`]). Algorithms are methods
//!   on `Graph`, constrained by the required layout.
//!
//! # Quick start
//!
//! ```no_run
//! use cudf::Runtime;
//! use cugraph::{
//!     ResourceHandle, Transposed,
//!     centrality::PageRankParams,
//!     cudf_interop::{DataFrameGraphOptions, graph_from_dataframe},
//!     types::GraphProperties,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let runtime = std::sync::Arc::new(Runtime::new()?);
//! let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime))?;
//! let entered = handle.enter("quick_start")?;
//!
//! let src = runtime.series(&[0i32, 1, 2, 1, 2, 0])?;
//! let dst = runtime.series(&[1i32, 2, 0, 0, 1, 2])?;
//! let edges = cudf::DataFrame::from_series([("src", src), ("dst", dst)])?;
//!
//! let opts = DataFrameGraphOptions::new("src", "dst")
//!     .properties(GraphProperties::undirected())
//!     .renumber();
//! let mut graph =
//!     graph_from_dataframe::<Transposed>(&entered, &edges, &opts)?;
//!
//! let result = graph.pagerank(&entered, &PageRankParams::default())?;
//! let df = result.to_dataframe(&runtime)?;
//! let vertices: Vec<i32> = df.series("vertex")?.to_vec_strict()?;
//! let scores: Vec<f32> = df.series("value")?.to_vec_strict()?;
//! let _ = (vertices, scores);
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
//! - **[`Forward`]** — required by [`Graph::bfs`] and [`Graph::louvain`]
//! - **[`Transposed`]** — required by [`Graph::pagerank`],
//!   [`Graph::in_degrees`], [`Graph::degrees`]
//! - **Generic** (`T: Transposition`) — [`Graph::betweenness_centrality`],
//!   [`Graph::force_atlas2`], [`Graph::out_degrees`]
//!
//! # cudf interop
//!
//! The [`cudf_interop`] module provides single-GPU graph construction from
//! cudf [`Series`](cudf::Series) and [`DataFrame`](cudf::DataFrame), plus
//! result-to-DataFrame conversions. Prefer its prepared-export entry points
//! for new shared-contract integrations; borrowed zero-copy entry points remain
//! available as compatibility paths.

pub mod array;
pub mod centrality;
pub mod community;
mod cuda;
pub mod cudf_interop;
pub mod error;
pub mod graph;
pub mod handle;
pub mod labeling;
pub mod layout;
pub mod params;
pub mod random;
pub mod structure;
pub mod traversal;
pub mod tree;
pub mod types;

// ═══════════════════════════════════════════════════════════════════════════
// Crate-root re-exports — foundational cross-cutting types only
// ═══════════════════════════════════════════════════════════════════════════

// Arrays
pub use array::{DeviceArray, DeviceArrayView, DeviceBuffer};
// Error handling
pub use error::{Error, ErrorKind, ErrorStatus};
// Graph construction and type-state
pub use graph::{Forward, Graph, GraphBuilder, Transposed, Transposition};
// Infrastructure
pub use handle::{EnteredContext, ResourceHandle};
// Unified cross-module vertex input model
pub use structure::VertexSelection;
