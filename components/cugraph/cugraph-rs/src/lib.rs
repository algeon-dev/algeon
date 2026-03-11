//! Safe Rust bindings for NVIDIA cuGraph GPU-accelerated graph analytics.
//!
//! This crate wraps the cuGraph C API in safe Rust types with RAII semantics,
//! zero-copy array views, and structured error handling.
//!
//! # Quick start
//!
//! ```no_run
//! use cugraph::{
//!     array::device_array_from_host,
//!     centrality::{pagerank, PageRankParams},
//!     graph::Graph,
//!     handle::ResourceHandle,
//!     types::{DataType, GraphProperties},
//! };
//!
//! let handle = ResourceHandle::new().unwrap();
//!
//! let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).unwrap();
//! let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).unwrap();
//!
//! let props = GraphProperties { is_symmetric: false, is_multigraph: false };
//! let mut graph = Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true).unwrap();
//!
//! let result = pagerank(&handle, &mut graph, &PageRankParams::default()).unwrap();
//! println!(
//!     "converged: {}, iterations: {}",
//!     result.converged().unwrap(),
//!     result.num_iterations()
//! );
//! ```

pub mod array;
pub mod centrality;
pub mod error;
pub mod graph;
pub mod handle;
pub mod traversal;
pub mod types;

#[cfg(feature = "cudf-interop")]
pub mod cudf_interop;
