//! Graph construction and management.
//!
//! A [`Graph`] is the central data structure for cuGraph algorithms. It is
//! constructed from an edge list and freed on drop.

use std::ptr;

use cugraph_sys::{
    cugraph_error_t, cugraph_graph_create_with_times_sg, cugraph_graph_free, cugraph_graph_t,
};

use crate::array::DeviceArrayView;
use crate::error::{check_error, Error, ErrorKind, Result};
use crate::handle::ResourceHandle;
use crate::types::{to_bool_t, GraphProperties};

/// RAII wrapper for a cuGraph graph object.
///
/// Constructed from an edge list via [`Graph::from_edge_list`], and freed
/// automatically on drop.
pub struct Graph {
    pub(crate) raw: *mut cugraph_graph_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl Graph {
    /// Create a single-GPU graph from an edge list.
    ///
    /// # Arguments
    ///
    /// - `handle` — resource handle for GPU operations
    /// - `properties` — whether the graph is symmetric / multigraph
    /// - `src` — source vertex ids (device array view)
    /// - `dst` — destination vertex ids (device array view)
    /// - `weights` — optional edge weights (device array view)
    /// - `store_transposed` — whether to store the graph transposed (needed by
    ///   some algorithms like PageRank)
    /// - `renumber` — whether to renumber vertices (strongly recommended when
    ///   vertex IDs are not sequential from 0)
    pub fn from_edge_list(
        handle: &ResourceHandle,
        properties: GraphProperties,
        src: &DeviceArrayView<'_>,
        dst: &DeviceArrayView<'_>,
        weights: Option<&DeviceArrayView<'_>>,
        store_transposed: bool,
        renumber: bool,
    ) -> Result<Self> {
        if src.len() != dst.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "Graph::from_edge_list")
                .with_context(
                    "message",
                    "src and dst must have the same number of elements",
                )
                .with_context("src_len", src.len().to_string())
                .with_context("dst_len", dst.len().to_string()));
        }

        if let Some(w) = weights {
            if w.len() != src.len() {
                return Err(Error::new(ErrorKind::InvalidInput, "Graph::from_edge_list")
                    .with_context(
                        "message",
                        "weights must have the same number of elements as src/dst",
                    )
                    .with_context("src_len", src.len().to_string())
                    .with_context("weights_len", w.len().to_string()));
            }
        }

        let c_props = properties.into();
        let weights_ptr = weights
            .map(|w| w.as_ptr() as *const _)
            .unwrap_or(ptr::null());

        let mut graph: *mut cugraph_graph_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        unsafe {
            let code = cugraph_graph_create_with_times_sg(
                handle.as_ptr(),
                &c_props,
                ptr::null(), // vertices (auto-detect)
                src.as_ptr(),
                dst.as_ptr(),
                weights_ptr,
                ptr::null(), // edge_ids
                ptr::null(), // edge_type_ids
                ptr::null(), // edge_start_time_ids
                ptr::null(), // edge_end_time_ids
                to_bool_t(store_transposed),
                to_bool_t(renumber),
                to_bool_t(false), // drop_self_loops
                to_bool_t(false), // drop_multi_edges
                to_bool_t(false), // symmetrize
                to_bool_t(false), // do_expensive_check
                &mut graph,
                &mut error,
            );
            check_error(code, error, "Graph::from_edge_list")?;
        }

        Ok(Self {
            raw: graph,
            _nosend_nosync: std::marker::PhantomData,
        })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_graph_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::device_array_from_host;
    use crate::error::ErrorKind;
    use crate::handle::ResourceHandle;
    use crate::types::DataType;

    #[test]
    fn test_graph_creation_triangle() {
        let handle = ResourceHandle::new().expect("handle");

        // Triangle: 0->1, 1->2, 2->0
        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties {
            is_symmetric: false,
            is_multigraph: false,
        };

        let graph =
            Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, false, false);
        assert!(graph.is_ok(), "graph creation failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_src_dst_lengths() {
        let handle = ResourceHandle::new().expect("handle");

        let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &[1i32, 2], DataType::Int32).expect("dst");

        let props = GraphProperties {
            is_symmetric: false,
            is_multigraph: false,
        };

        let err = match Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            false,
            false,
        ) {
            Ok(_) => panic!("should reject mismatched edge-list lengths"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "Graph::from_edge_list");
    }
}
