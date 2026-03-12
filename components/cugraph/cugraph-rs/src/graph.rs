//! Graph construction and management.
//!
//! A [`Graph`] is the central data structure for cuGraph algorithms. It is
//! constructed from an edge list and freed on drop.

use std::ptr;

use cugraph_sys::{cugraph_error_t, cugraph_graph_create_with_times_sg, cugraph_graph_free, cugraph_graph_t};

use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error},
    handle::ResourceHandle,
    types::{GraphProperties, to_bool_t},
};

/// RAII wrapper for a cuGraph graph object.
///
/// Constructed from an edge list via [`Graph::from_edge_list`], and freed
/// automatically on drop.
pub struct Graph {
    pub(crate) raw: *mut cugraph_graph_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph").field("raw", &self.raw).finish()
    }
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
    /// - `edge_ids` — optional edge ids for each edge (device array view).
    ///   Must have the same length as `src`/`dst` if provided.
    /// - `edge_type_ids` — optional edge type ids for each edge (device array
    ///   view). Must have the same length as `src`/`dst` if provided.
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
        edge_ids: Option<&DeviceArrayView<'_>>,
        edge_type_ids: Option<&DeviceArrayView<'_>>,
        store_transposed: bool,
        renumber: bool,
    ) -> Result<Self> {
        if src.len() != dst.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "Graph::from_edge_list")
                .with_message("src and dst must have the same number of elements")
                .with_context("src_len", src.len().to_string())
                .with_context("dst_len", dst.len().to_string()));
        }

        if let Some(w) = weights {
            if w.len() != src.len() {
                return Err(Error::new(ErrorKind::InvalidInput, "Graph::from_edge_list")
                    .with_message("weights must have the same number of elements as src/dst")
                    .with_context("src_len", src.len().to_string())
                    .with_context("weights_len", w.len().to_string()));
            }
        }

        if let Some(eids) = edge_ids {
            if eids.len() != src.len() {
                return Err(Error::new(ErrorKind::InvalidInput, "Graph::from_edge_list")
                    .with_message("edge_ids must have the same number of elements as src/dst")
                    .with_context("src_len", src.len().to_string())
                    .with_context("edge_ids_len", eids.len().to_string()));
            }
        }

        if let Some(etids) = edge_type_ids {
            if etids.len() != src.len() {
                return Err(Error::new(ErrorKind::InvalidInput, "Graph::from_edge_list")
                    .with_message("edge_type_ids must have the same number of elements as src/dst")
                    .with_context("src_len", src.len().to_string())
                    .with_context("edge_type_ids_len", etids.len().to_string()));
            }
        }

        let c_props = properties.into();
        let weights_ptr = weights.map(|w| w.as_ptr() as *const _).unwrap_or(ptr::null());
        let edge_ids_ptr = edge_ids.map(|e| e.as_ptr() as *const _).unwrap_or(ptr::null());
        let edge_type_ids_ptr = edge_type_ids.map(|e| e.as_ptr() as *const _).unwrap_or(ptr::null());

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
                edge_ids_ptr,
                edge_type_ids_ptr,
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

        Ok(Self { raw: graph, _nosend_nosync: std::marker::PhantomData })
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
    use crate::{array::device_array_from_host, error::ErrorKind, handle::ResourceHandle, types::DataType};

    #[test]
    fn test_graph_creation_triangle() {
        let handle = ResourceHandle::new().expect("handle");

        // Triangle: 0->1, 1->2, 2->0
        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, false);
        assert!(graph.is_ok(), "graph creation failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_src_dst_lengths() {
        let handle = ResourceHandle::new().expect("handle");

        let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &[1i32, 2], DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let err = match Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, false)
        {
            Ok(_) => panic!("should reject mismatched edge-list lengths"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "Graph::from_edge_list");
    }

    #[test]
    fn test_graph_creation_with_edge_ids_and_edge_type_ids() {
        let handle = ResourceHandle::new().expect("handle");

        // Triangle: 0->1, 1->2, 2->0 with edge IDs and edge type IDs
        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];
        let edge_id_data: Vec<i32> = vec![100, 101, 102];
        let edge_type_data: Vec<i32> = vec![0, 1, 0];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let edge_ids = device_array_from_host(&handle, &edge_id_data, DataType::Int32).expect("edge_ids");
        let edge_types = device_array_from_host(&handle, &edge_type_data, DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            Some(&edge_ids.view()),
            Some(&edge_types.view()),
            false,
            true, // renumber required when using edge properties
        );
        assert!(graph.is_ok(), "graph creation with edge_ids and edge_type_ids failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_edge_ids_only() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];
        let edge_id_data: Vec<i32> = vec![10, 20, 30];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let edge_ids = device_array_from_host(&handle, &edge_id_data, DataType::Int32).expect("edge_ids");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            Some(&edge_ids.view()),
            None,
            false,
            true,
        );
        assert!(graph.is_ok(), "graph creation with edge_ids only failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_edge_type_ids_only() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];
        let edge_type_data: Vec<i32> = vec![0, 1, 2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let edge_types = device_array_from_host(&handle, &edge_type_data, DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            None,
            Some(&edge_types.view()),
            false,
            true,
        );
        assert!(graph.is_ok(), "graph creation with edge_type_ids only failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_weights_and_edge_properties() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0];
        let wgt_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let edge_id_data: Vec<i32> = vec![0, 1, 2, 3];
        let edge_type_data: Vec<i32> = vec![0, 0, 1, 1];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let wgt = device_array_from_host(&handle, &wgt_data, DataType::Float32).expect("wgt");
        let edge_ids = device_array_from_host(&handle, &edge_id_data, DataType::Int32).expect("edge_ids");
        let edge_types = device_array_from_host(&handle, &edge_type_data, DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            Some(&wgt.view()),
            Some(&edge_ids.view()),
            Some(&edge_types.view()),
            false,
            true,
        );
        assert!(graph.is_ok(), "graph creation with weights + edge properties failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_ids_length() {
        let handle = ResourceHandle::new().expect("handle");

        let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
        let edge_ids = device_array_from_host(&handle, &[10i32, 20], DataType::Int32).expect("edge_ids");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let err = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            Some(&edge_ids.view()),
            None,
            false,
            false,
        )
        .expect_err("should reject mismatched edge_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_ids"));
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_type_ids_length() {
        let handle = ResourceHandle::new().expect("handle");

        let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
        let edge_types = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let err = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            None,
            Some(&edge_types.view()),
            false,
            false,
        )
        .expect_err("should reject mismatched edge_type_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_type_ids"));
    }
}
