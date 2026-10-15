#![warn(clippy::undocumented_unsafe_blocks)]

//! Workspace-internal safe libcugraph execution component for Algeon.
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion); workspace code
//! uses this crate for its component contracts. See the
//! [API documentation boundary](https://algeon.dev/design#api-documentation).
//!
//! Algeon integrations enter through a runtime-bound [`ResourceHandle`], use
//! the prepared transfer contracts in [`cudf_interop`], and retain explicit
//! producer readiness and consumer completion ordering. Native failures are
//! preserved as structured [`Error`] values.
//!
//! # Key concepts
//!
//! - **[`cudf_interop`]**: canonical prepared cuDF graph ingress and result
//!   egress for the Algeon engine.
//! - **[`ResourceHandle`]**: long-lived owner of GPU execution provenance
//!   (bound device, retained runtime, cuGraph resource handle). Not
//!   thread-safe.
//! - **[`EnteredContext`]**: short-lived execution session created by
//!   [`ResourceHandle::enter`]. GPU-capable APIs require this token.
//! - **[`DeviceArray<T>`]**: owning GPU array with compile-time element type.
//!   Freed on drop. Create views via [`DeviceArray::view()`].
//! - **[`GraphBuilder`]**: low-level builder for constructing a [`Graph`] from
//!   pre-materialized device edge lists. Validates inputs before crossing the
//!   FFI boundary.
//! - **[`Graph<T>`]**: graph container parameterized by transposition state
//!   ([`Forward`] or [`Transposed`]). Algorithm methods are constrained by the
//!   required layout.
//!
//! # Type-state encoding
//!
//! The [`Graph<T>`] type parameter encodes whether storage is in forward or
//! transposed layout. Algorithms enforce their required layout at compile time:
//!
//! - **[`Forward`]**: required by [`Graph::bfs`] and [`Graph::louvain`]
//! - **[`Transposed`]**: required by [`Graph::pagerank`],
//!   [`Graph::in_degrees`], [`Graph::degrees`]
//! - **Generic** (`T: Transposition`): [`Graph::betweenness_centrality`],
//!   [`Graph::force_atlas2`], [`Graph::out_degrees`]
//!
//! # cudf interop
//!
//! The [`cudf_interop`] module owns single-GPU prepared graph construction
//! from cudf tables and result-to-DataFrame conversion. The prepared export
//! contract is the canonical Algeon ingress path.

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
pub mod sampling;
pub mod similarity;
pub mod structure;
pub mod traversal;
pub mod tree;
pub mod types;

// ═══════════════════════════════════════════════════════════════════════════
// Crate-root re-exports; foundational cross-cutting types only
// ═══════════════════════════════════════════════════════════════════════════

// Arrays
pub use array::{DeviceArray, DeviceArrayView};
// Error handling
pub use error::{AllocationFailureSource, AllocationSource, Error, ErrorKind, ErrorStatus};
// Graph construction and type-state
pub use graph::{Forward, Graph, GraphBuilder, ParkedGraph, Transposed, Transposition};
// Infrastructure
pub use handle::{EnteredContext, ResourceHandle};
// Unified cross-module vertex input model
pub use structure::VertexSelection;
