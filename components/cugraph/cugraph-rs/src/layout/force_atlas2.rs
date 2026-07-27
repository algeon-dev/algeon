//! Force Atlas 2 — force-directed graph layout algorithm.
//!
//! Force Atlas 2 (FA2) computes a 2D layout for graph visualization by
//! simulating physical forces (attraction along edges, repulsion between all
//! vertex pairs, gravity toward center).
//!
//! The algorithm supports an optional initial position for refinement passes
//! (the Python workload calls FA2 twice: first with random positions, then
//! with the previous result as `pos_list`).
//!
//! # Example
//!
//! ```no_run
//! use cugraph::{
//!     array::device_array_from_host,
//!     graph::GraphBuilder,
//!     handle::ResourceHandle,
//!     layout::{ForceAtlas2Params, force_atlas2},
//!     params::IterationCount,
//!     random::RngState,
//!     types::{DataType, GraphProperties},
//! };
//!
//! let handle = ResourceHandle::new().unwrap();
//! let mut rng = RngState::new(&handle, 42).unwrap();
//!
//! let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32)
//!     .unwrap();
//! let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32)
//!     .unwrap();
//!
//! let props = GraphProperties { is_symmetric: false, is_multigraph: false };
//! let src_view = src.view();
//! let dst_view = dst.view();
//! let mut graph = GraphBuilder::new(props, &src_view, &dst_view)
//!     .renumber(true)
//!     .build_forward(&handle)
//!     .unwrap();
//!
//! let params = ForceAtlas2Params {
//!     max_iter: IterationCount::new(100).unwrap(),
//!     ..Default::default()
//! };
//! let result =
//!     force_atlas2(&handle, &mut rng, &mut graph, &params, None, None)
//!         .unwrap();
//!
//! assert_eq!(result.vertices().len(), 3);
//! assert_eq!(result.x().len(), 3);
//! assert_eq!(result.y().len(), 3);
//! ```

use std::ptr;

use cugraph_sys::{
    bool_t, cugraph_error_t, cugraph_force_atlas2, cugraph_layout_result_free, cugraph_layout_result_get_vertices,
    cugraph_layout_result_get_x_axis, cugraph_layout_result_get_y_axis, cugraph_layout_result_t,
};

use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error},
    graph::Graph,
    handle::ResourceHandle,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64, PositiveF64},
    random::RngState,
    types::to_bool_t,
};

/// Parameters for the Force Atlas 2 layout algorithm.
///
/// All boolean and floating-point parameters map directly to the C API.
/// Numeric parameters use strongly-typed wrappers that enforce invariants:
///
/// - `max_iter` ([`IterationCount`]): must be `>= 1`
/// - `overlap_scaling_ratio`, `scaling_ratio` ([`PositiveF64`]):
///   must be `> 0.0`
/// - `gravity`, `edge_weight_influence`, `jitter_tolerance`
///   ([`NonNegativeF64`]): must be `>= 0.0`
/// - `barnes_hut_theta` ([`ClosedUnitF64`]): must be in `[0.0, 1.0]`
///
/// See the [Force Atlas 2 paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679)
/// for detailed descriptions.
#[derive(Debug, Clone, Copy)]
pub struct ForceAtlas2Params {
    /// Maximum number of iterations (default: 500). Must be at least 1.
    pub max_iter: IterationCount,
    /// Distributes attraction along outbound edges. Hubs attract less and are
    /// pushed to borders (default: `false`).
    pub outbound_attraction_distribution: bool,
    /// Switch from lin-lin to lin-log model. Makes clusters tighter
    /// (default: `false`).
    pub lin_log_mode: bool,
    /// Prevent nodes from overlapping (default: `false`).
    pub prevent_overlapping: bool,
    /// When `prevent_overlapping` is set, scales the repulsion force between
    /// overlapping nodes (default: 2.0). Must be strictly positive.
    pub overlap_scaling_ratio: PositiveF64,
    /// Influence of edge weights. 0 = no influence, 1 = normal (default: 1.0).
    /// Must be non-negative.
    pub edge_weight_influence: NonNegativeF64,
    /// How much swinging is allowed. Above 1 is discouraged. Lower gives less
    /// speed and more precision (default: 1.0). Must be non-negative.
    pub jitter_tolerance: NonNegativeF64,
    /// Use Barnes-Hut approximation for faster computation (default: `false`).
    pub barnes_hut_optimize: bool,
    /// Trade-off for Barnes-Hut: speed (1.0) vs accuracy (0.0)
    /// (default: 0.5). Must be in `[0.0, 1.0]`.
    pub barnes_hut_theta: ClosedUnitF64,
    /// How much repulsion. More makes a sparser graph (default: 2.0). Must be
    /// strictly positive.
    pub scaling_ratio: PositiveF64,
    /// Attract distant nodes more strongly to the center (default: `false`).
    pub strong_gravity_mode: bool,
    /// Gravity — attracts nodes to center, prevents islands drifting
    /// (default: 1.0). Zero disables gravity. Must be non-negative.
    pub gravity: NonNegativeF64,
    /// Print convergence info at each iteration (default: `false`).
    pub verbose: bool,
}

impl Default for ForceAtlas2Params {
    fn default() -> Self {
        Self {
            // SAFETY: these literals are known-valid constants.
            max_iter: IterationCount::new(500).unwrap(),
            outbound_attraction_distribution: false,
            lin_log_mode: false,
            prevent_overlapping: false,
            overlap_scaling_ratio: PositiveF64::new(2.0).unwrap(),
            edge_weight_influence: NonNegativeF64::new(1.0).unwrap(),
            jitter_tolerance: NonNegativeF64::new(1.0).unwrap(),
            barnes_hut_optimize: false,
            barnes_hut_theta: ClosedUnitF64::new(0.5).unwrap(),
            scaling_ratio: PositiveF64::new(2.0).unwrap(),
            strong_gravity_mode: false,
            gravity: NonNegativeF64::new(1.0).unwrap(),
            verbose: false,
        }
    }
}

/// Optional per-vertex attributes for Force Atlas 2.
///
/// Each attribute is a pair of device arrays: one for vertex ids and one for
/// the corresponding values. Both arrays must have the same length.
impl std::fmt::Debug for ForceAtlas2VertexAttrs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForceAtlas2VertexAttrs")
            .field("radius", &self.radius.is_some())
            .field("mobility", &self.mobility.is_some())
            .field("mass", &self.mass.is_some())
            .finish()
    }
}

#[derive(Default)]
pub struct ForceAtlas2VertexAttrs<'a> {
    /// Per-vertex radius (used when `prevent_overlapping` is set).
    /// `(vertex_ids, radius_values)`.
    pub radius: Option<(&'a DeviceArrayView<'a>, &'a DeviceArrayView<'a>)>,
    /// Per-vertex mobility, scales displacement at each iteration.
    /// `(vertex_ids, mobility_values)`.
    pub mobility: Option<(&'a DeviceArrayView<'a>, &'a DeviceArrayView<'a>)>,
    /// Per-vertex mass, controls attraction to other vertices. If not provided,
    /// mass defaults to degree + 1. `(vertex_ids, mass_values)`.
    pub mass: Option<(&'a DeviceArrayView<'a>, &'a DeviceArrayView<'a>)>,
}

/// Optional initial positions for Force Atlas 2.
///
/// Used for refinement passes where a previous layout result is fed back as
/// starting positions.
impl std::fmt::Debug for InitialPositions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InitialPositions").finish_non_exhaustive()
    }
}

pub struct InitialPositions<'a> {
    /// Vertex ids for the initial positions.
    pub vertices: &'a DeviceArrayView<'a>,
    /// X-axis initial positions.
    pub x: &'a DeviceArrayView<'a>,
    /// Y-axis initial positions.
    pub y: &'a DeviceArrayView<'a>,
}

/// Result of a Force Atlas 2 layout computation.
///
/// Provides access to vertex ids and their (x, y) coordinates. The result
/// owns the GPU memory and frees it on drop.
pub struct LayoutResult {
    raw: *mut cugraph_layout_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for LayoutResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutResult").finish_non_exhaustive()
    }
}

impl LayoutResult {
    /// Wrap a raw layout result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by the cuGraph layout
    /// C API. Ownership is transferred to this struct.
    unsafe fn from_raw(raw: *mut cugraph_layout_result_t) -> Self {
        Self { raw, _nosend_nosync: std::marker::PhantomData }
    }

    /// Get the vertex ids as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn vertices(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_layout_result_get_vertices(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Get the x-axis coordinates as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn x(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_layout_result_get_x_axis(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Get the y-axis coordinates as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn y(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_layout_result_get_y_axis(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }
}

#[cfg(feature = "cudf-interop")]
impl LayoutResult {
    /// Convert this layout result to a cudf DataFrame.
    ///
    /// Returns a 3-column DataFrame:
    /// - `"vertex"` — vertex ids
    /// - `"x"` — x-axis coordinates (Float32)
    /// - `"y"` — y-axis coordinates (Float32)
    ///
    /// Performs a device-to-device copy.
    pub fn to_dataframe(&self) -> crate::error::Result<cudf::DataFrame> {
        use crate::{
            cudf_interop::DeviceArrayViewExt,
            error::{Error, ErrorKind},
        };

        let vert_col = self.vertices().to_cudf_column().map_err(|e| e.with_operation("LayoutResult::to_dataframe"))?;
        let x_col = self.x().to_cudf_column().map_err(|e| e.with_operation("LayoutResult::to_dataframe"))?;
        let y_col = self.y().to_cudf_column().map_err(|e| e.with_operation("LayoutResult::to_dataframe"))?;

        cudf::DataFrame::new(vec!["vertex".into(), "x".into(), "y".into()], vec![vert_col, x_col, y_col]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "LayoutResult::to_dataframe")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl Drop for LayoutResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_layout_result_free(self.raw) };
        }
    }
}

/// Compute a Force Atlas 2 layout for a graph.
///
/// Works with either [`Forward`](crate::graph::Forward) or
/// [`Transposed`](crate::graph::Transposed) graphs.
///
/// Returns a [`LayoutResult`] containing vertex ids and (x, y) coordinates.
///
/// # Arguments
///
/// - `handle` — resource handle for GPU operations
/// - `rng_state` — random number generator state (mutated by the call)
/// - `graph` — the graph to lay out (may be mutated for storage transposition)
/// - `params` — algorithm parameters
/// - `vertex_attrs` — optional per-vertex radius, mobility, and mass
/// - `initial_pos` — optional initial positions for refinement passes
///
/// # Errors
///
/// Returns an error if the C API call fails (e.g., invalid input, GPU memory
/// allocation failure).
pub fn force_atlas2<T>(
    handle: &ResourceHandle,
    rng_state: &mut RngState,
    graph: &mut Graph<T>,
    params: &ForceAtlas2Params,
    vertex_attrs: Option<&ForceAtlas2VertexAttrs<'_>>,
    initial_pos: Option<&InitialPositions<'_>>,
) -> Result<LayoutResult> {
    // Validate initial position arrays have consistent lengths.
    if let Some(pos) = initial_pos {
        let n = pos.vertices.len();
        if pos.x.len() != n {
            return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                .with_message("initial positions: x length must match vertices length")
                .with_context("vertices_len", n.to_string())
                .with_context("x_len", pos.x.len().to_string()));
        }
        if pos.y.len() != n {
            return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                .with_message("initial positions: y length must match vertices length")
                .with_context("vertices_len", n.to_string())
                .with_context("y_len", pos.y.len().to_string()));
        }
    }

    // Validate vertex attribute array pairs have consistent lengths.
    if let Some(attrs) = vertex_attrs {
        if let Some((verts, vals)) = attrs.radius {
            if verts.len() != vals.len() {
                return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                    .with_message("vertex radius: vertices and values must have the same length")
                    .with_context("vertices_len", verts.len().to_string())
                    .with_context("values_len", vals.len().to_string()));
            }
        }
        if let Some((verts, vals)) = attrs.mobility {
            if verts.len() != vals.len() {
                return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                    .with_message("vertex mobility: vertices and values must have the same length")
                    .with_context("vertices_len", verts.len().to_string())
                    .with_context("values_len", vals.len().to_string()));
            }
        }
        if let Some((verts, vals)) = attrs.mass {
            if verts.len() != vals.len() {
                return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                    .with_message("vertex mass: vertices and values must have the same length")
                    .with_context("vertices_len", verts.len().to_string())
                    .with_context("values_len", vals.len().to_string()));
            }
        }
    }

    // Extract raw pointers for initial positions.
    let (start_verts_ptr, x_start_ptr, y_start_ptr) = match initial_pos {
        Some(pos) => (pos.vertices.as_ptr(), pos.x.as_ptr(), pos.y.as_ptr()),
        None => (ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
    };

    // Extract raw pointers for vertex attributes.
    let empty = ForceAtlas2VertexAttrs::default();
    let attrs = vertex_attrs.unwrap_or(&empty);

    let (radius_verts_ptr, radius_vals_ptr) = match attrs.radius {
        Some((v, val)) => (v.as_ptr(), val.as_ptr()),
        None => (ptr::null_mut(), ptr::null_mut()),
    };

    let (mobility_verts_ptr, mobility_vals_ptr) = match attrs.mobility {
        Some((v, val)) => (v.as_ptr(), val.as_ptr()),
        None => (ptr::null_mut(), ptr::null_mut()),
    };

    let (mass_verts_ptr, mass_vals_ptr) = match attrs.mass {
        Some((v, val)) => (v.as_ptr(), val.as_ptr()),
        None => (ptr::null_mut(), ptr::null_mut()),
    };

    let mut result: *mut cugraph_layout_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    unsafe {
        let code = cugraph_force_atlas2(
            handle.as_ptr(),
            rng_state.as_ptr(),
            graph.raw,
            i32::from(params.max_iter),
            start_verts_ptr,
            x_start_ptr,
            y_start_ptr,
            to_bool_t(params.outbound_attraction_distribution),
            to_bool_t(params.lin_log_mode),
            to_bool_t(params.prevent_overlapping),
            radius_verts_ptr,
            radius_vals_ptr,
            params.overlap_scaling_ratio.get(),
            params.edge_weight_influence.get(),
            params.jitter_tolerance.get(),
            to_bool_t(params.barnes_hut_optimize),
            params.barnes_hut_theta.get(),
            params.scaling_ratio.get(),
            to_bool_t(params.strong_gravity_mode),
            params.gravity.get(),
            mobility_verts_ptr,
            mobility_vals_ptr,
            mass_verts_ptr,
            mass_vals_ptr,
            to_bool_t(params.verbose),
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "force_atlas2")?;
    }

    Ok(unsafe { LayoutResult::from_raw(result) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::device_array_from_host,
        graph::GraphBuilder,
        params::{ClosedUnitF64, IterationCount, NonNegativeF64},
        types::{DataType, GraphProperties},
    };

    /// Helper: decode device array bytes into Vec<i32>.
    fn to_i32_vec(bytes: &[u8]) -> Vec<i32> {
        bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Helper: decode device array bytes into Vec<f32>.
    fn to_f32_vec(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn test_force_atlas2_triangle_produces_coordinates() {
        // Triangle: 0->1, 1->2, 2->0 (symmetric for layout purposes)
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 42).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
        let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph =
            GraphBuilder::new(props, &src.view(), &dst.view()).renumber(true).build_forward(&handle).expect("graph");

        let params = ForceAtlas2Params { max_iter: IterationCount::new(50).unwrap(), ..Default::default() };

        let result = force_atlas2(&handle, &mut rng, &mut graph, &params, None, None).expect("fa2");

        // Should have 3 vertices with x,y coordinates
        assert_eq!(result.vertices().len(), 3, "should have 3 vertices");
        assert_eq!(result.x().len(), 3, "should have 3 x coordinates");
        assert_eq!(result.y().len(), 3, "should have 3 y coordinates");

        // Verify coordinates are finite (not NaN or infinity)
        let x_bytes = result.x().to_host_vec(&handle).expect("x copy");
        let y_bytes = result.y().to_host_vec(&handle).expect("y copy");
        let x_vals = to_f32_vec(&x_bytes);
        let y_vals = to_f32_vec(&y_bytes);

        for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
            assert!(x.is_finite(), "x[{}] = {} should be finite", i, x);
            assert!(y.is_finite(), "y[{}] = {} should be finite", i, y);
        }
    }

    #[test]
    fn test_force_atlas2_refinement_with_initial_positions() {
        // Two-pass layout: first pass with random positions, second with initial positions.
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 123).expect("rng");

        // Square: 0-1-2-3-0
        let src_data: Vec<i32> = vec![0, 1, 2, 3, 1, 2, 3, 0];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0, 0, 1, 2, 3];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph =
            GraphBuilder::new(props, &src.view(), &dst.view()).renumber(true).build_forward(&handle).expect("graph");

        // First pass: 50 iterations, random initial positions
        let params1 = ForceAtlas2Params { max_iter: IterationCount::new(50).unwrap(), ..Default::default() };
        let result1 = force_atlas2(&handle, &mut rng, &mut graph, &params1, None, None).expect("fa2 pass 1");

        assert_eq!(result1.vertices().len(), 4);

        // Second pass: 10 iterations, using first pass result as initial positions
        let params2 = ForceAtlas2Params { max_iter: IterationCount::new(10).unwrap(), ..Default::default() };
        let init_pos = InitialPositions { vertices: &result1.vertices(), x: &result1.x(), y: &result1.y() };
        let result2 = force_atlas2(&handle, &mut rng, &mut graph, &params2, None, Some(&init_pos)).expect("fa2 pass 2");

        assert_eq!(result2.vertices().len(), 4);
        assert_eq!(result2.x().len(), 4);
        assert_eq!(result2.y().len(), 4);

        // Verify coordinates are finite
        let x_bytes = result2.x().to_host_vec(&handle).expect("x copy");
        let y_bytes = result2.y().to_host_vec(&handle).expect("y copy");
        let x_vals = to_f32_vec(&x_bytes);
        let y_vals = to_f32_vec(&y_bytes);

        for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
            assert!(x.is_finite(), "x[{}] = {} should be finite", i, x);
            assert!(y.is_finite(), "y[{}] = {} should be finite", i, y);
        }
    }

    #[test]
    fn test_force_atlas2_vertex_count_matches_graph() {
        // Verify result has correct number of vertices for different graph sizes.
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 99).expect("rng");

        // 5-vertex line: 0-1-2-3-4
        let src_data: Vec<i32> = vec![0, 1, 2, 3, 1, 2, 3, 4];
        let dst_data: Vec<i32> = vec![1, 2, 3, 4, 0, 1, 2, 3];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph =
            GraphBuilder::new(props, &src.view(), &dst.view()).renumber(true).build_forward(&handle).expect("graph");

        let params = ForceAtlas2Params { max_iter: IterationCount::new(10).unwrap(), ..Default::default() };

        let result = force_atlas2(&handle, &mut rng, &mut graph, &params, None, None).expect("fa2");

        assert_eq!(result.vertices().len(), 5, "should have 5 vertices");
        assert_eq!(result.x().len(), 5, "should have 5 x coords");
        assert_eq!(result.y().len(), 5, "should have 5 y coords");

        // Verify all 5 vertex ids are present
        let vert_bytes = result.vertices().to_host_vec(&handle).expect("vert copy");
        let vertices = to_i32_vec(&vert_bytes);
        let mut sorted = vertices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4], "should contain vertices 0-4");
    }

    #[test]
    fn test_force_atlas2_with_weights() {
        // Test FA2 with edge weights.
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 7).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
        let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];
        let wgt_data: Vec<f32> = vec![1.0, 2.0, 0.5, 1.0, 2.0, 0.5];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let wgt = device_array_from_host(&handle, &wgt_data, DataType::Float32).expect("wgt");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph = GraphBuilder::new(props, &src.view(), &dst.view())
            .weights(&wgt.view())
            .renumber(true)
            .build_forward(&handle)
            .expect("graph");

        let params = ForceAtlas2Params {
            max_iter: IterationCount::new(50).unwrap(),
            edge_weight_influence: NonNegativeF64::new(1.0).unwrap(),
            ..Default::default()
        };

        let result = force_atlas2(&handle, &mut rng, &mut graph, &params, None, None).expect("fa2");

        assert_eq!(result.vertices().len(), 3);
        assert_eq!(result.x().len(), 3);
        assert_eq!(result.y().len(), 3);
    }

    #[test]
    fn test_force_atlas2_barnes_hut_mode() {
        // Test Barnes-Hut approximation mode.
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 55).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
        let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph =
            GraphBuilder::new(props, &src.view(), &dst.view()).renumber(true).build_forward(&handle).expect("graph");

        let params = ForceAtlas2Params {
            max_iter: IterationCount::new(50).unwrap(),
            barnes_hut_optimize: true,
            barnes_hut_theta: ClosedUnitF64::new(0.5).unwrap(),
            ..Default::default()
        };

        let result = force_atlas2(&handle, &mut rng, &mut graph, &params, None, None).expect("fa2 BH");

        assert_eq!(result.vertices().len(), 3);

        let x_bytes = result.x().to_host_vec(&handle).expect("x copy");
        let y_bytes = result.y().to_host_vec(&handle).expect("y copy");
        let x_vals = to_f32_vec(&x_bytes);
        let y_vals = to_f32_vec(&y_bytes);

        for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
            assert!(x.is_finite(), "BH x[{}] = {} should be finite", i, x);
            assert!(y.is_finite(), "BH y[{}] = {} should be finite", i, y);
        }
    }

    #[test]
    fn test_force_atlas2_rejects_mismatched_initial_positions() {
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 0).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 1, 0];
        let dst_data: Vec<i32> = vec![1, 0, 0, 1];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph =
            GraphBuilder::new(props, &src.view(), &dst.view()).renumber(true).build_forward(&handle).expect("graph");

        // Create mismatched initial position arrays
        let verts = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("verts");
        let x_pos = device_array_from_host(&handle, &[1.0f32, 2.0], DataType::Float32).expect("x");
        let y_pos = device_array_from_host(&handle, &[3.0f32], DataType::Float32).expect("y"); // wrong length

        let params = ForceAtlas2Params::default();
        let init = InitialPositions { vertices: &verts.view(), x: &x_pos.view(), y: &y_pos.view() };

        let err = force_atlas2(&handle, &mut rng, &mut graph, &params, None, Some(&init))
            .expect_err("should reject mismatched y length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("y length"));
    }

    #[test]
    fn test_force_atlas2_rejects_mismatched_vertex_attrs() {
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 0).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 1, 0];
        let dst_data: Vec<i32> = vec![1, 0, 0, 1];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph =
            GraphBuilder::new(props, &src.view(), &dst.view()).renumber(true).build_forward(&handle).expect("graph");

        // Create mismatched mass arrays
        let mass_verts = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("mass_verts");
        let mass_vals = device_array_from_host(&handle, &[1.0f32], DataType::Float32).expect("mass_vals"); // wrong length

        let params = ForceAtlas2Params::default();
        let attrs =
            ForceAtlas2VertexAttrs { mass: Some((&mass_verts.view(), &mass_vals.view())), ..Default::default() };

        let err = force_atlas2(&handle, &mut rng, &mut graph, &params, Some(&attrs), None)
            .expect_err("should reject mismatched mass array lengths");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("mass"));
    }

    #[test]
    fn test_force_atlas2_lin_log_and_strong_gravity() {
        // Test with lin_log_mode and strong_gravity_mode enabled.
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 77).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 2, 3, 1, 2, 3, 0];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0, 0, 1, 2, 3];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: true, is_multigraph: false };

        let mut graph =
            GraphBuilder::new(props, &src.view(), &dst.view()).renumber(true).build_forward(&handle).expect("graph");

        let params = ForceAtlas2Params {
            max_iter: IterationCount::new(30).unwrap(),
            lin_log_mode: true,
            strong_gravity_mode: true,
            gravity: NonNegativeF64::new(2.0).unwrap(),
            ..Default::default()
        };

        let result = force_atlas2(&handle, &mut rng, &mut graph, &params, None, None).expect("fa2");

        assert_eq!(result.vertices().len(), 4);
        assert_eq!(result.x().len(), 4);
        assert_eq!(result.y().len(), 4);
    }
}
