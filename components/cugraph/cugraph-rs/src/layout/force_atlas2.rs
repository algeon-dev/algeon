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
//!     DeviceArray, ForceAtlas2Options, ForceAtlas2Params, Forward,
//!     GraphBuilder, IterationCount, ResourceHandle, RngState,
//! };
//!
//! let handle = ResourceHandle::for_current_device().unwrap();
//! let mut rng = RngState::new(&handle, 42).unwrap();
//!
//! let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).unwrap();
//! let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).unwrap();
//!
//! let src_view = src.view();
//! let dst_view = dst.view();
//! let mut graph = GraphBuilder::new(&src_view, &dst_view)
//!     .renumber()
//!     .build::<Forward>(&handle)
//!     .unwrap();
//!
//! let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
//!     params: ForceAtlas2Params {
//!         max_iter: IterationCount::new(100).unwrap(),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! let result = graph.force_atlas2(&handle, &mut rng, &options).unwrap();
//!
//! assert_eq!(result.vertex_ids().unwrap().len(), 3);
//! assert_eq!(result.x().unwrap().len(), 3);
//! assert_eq!(result.y().unwrap().len(), 3);
//! ```

use std::ptr;

use cugraph_sys::{
    cugraph_error_t, cugraph_force_atlas2, cugraph_layout_result_free, cugraph_layout_result_get_vertices,
    cugraph_layout_result_get_x_axis, cugraph_layout_result_get_y_axis, cugraph_layout_result_t,
};

use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error},
    graph::Graph,
    handle::ResourceHandle,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64, PositiveF64},
    random::RngState,
    structure::typed_results::{ScoreValues, VertexIds},
    types::{CugraphElement, DataType, to_bool_t},
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
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
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
            do_expensive_check: false,
        }
    }
}

/// Optional per-vertex attributes for Force Atlas 2.
///
/// Each attribute is a pair of device arrays: one for vertex ids and one for
/// the corresponding values. Both arrays must have the same length.
impl<V: CugraphElement, W: CugraphElement> std::fmt::Debug for ForceAtlas2VertexAttrs<'_, V, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForceAtlas2VertexAttrs")
            .field("radius", &self.radius.is_some())
            .field("mobility", &self.mobility.is_some())
            .field("mass", &self.mass.is_some())
            .finish()
    }
}

pub struct ForceAtlas2VertexAttrs<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Per-vertex radius (used when `prevent_overlapping` is set).
    /// `(vertex_ids, radius_values)`.
    pub radius: Option<(&'a DeviceArrayView<'a, V>, &'a DeviceArrayView<'a, W>)>,
    /// Per-vertex mobility, scales displacement at each iteration.
    /// `(vertex_ids, mobility_values)`.
    pub mobility: Option<(&'a DeviceArrayView<'a, V>, &'a DeviceArrayView<'a, W>)>,
    /// Per-vertex mass, controls attraction to other vertices. If not provided,
    /// mass defaults to degree + 1. `(vertex_ids, mass_values)`.
    pub mass: Option<(&'a DeviceArrayView<'a, V>, &'a DeviceArrayView<'a, W>)>,
}

impl<'a, V: CugraphElement, W: CugraphElement> Default for ForceAtlas2VertexAttrs<'a, V, W> {
    fn default() -> Self {
        Self { radius: None, mobility: None, mass: None }
    }
}

/// Optional initial positions for Force Atlas 2.
///
/// Used for refinement passes where a previous layout result is fed back as
/// starting positions.
impl<V: CugraphElement, W: CugraphElement> std::fmt::Debug for InitialPositions<'_, V, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InitialPositions").finish_non_exhaustive()
    }
}

pub struct InitialPositions<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Vertex ids for the initial positions.
    pub vertices: &'a DeviceArrayView<'a, V>,
    /// X-axis initial positions.
    pub x: &'a DeviceArrayView<'a, W>,
    /// Y-axis initial positions.
    pub y: &'a DeviceArrayView<'a, W>,
}

/// Consolidated options for a Force Atlas 2 layout computation.
///
/// Bundles [`ForceAtlas2Params`] (algorithm parameters) with optional
/// per-vertex attributes and initial positions. This is the single
/// argument passed to [`Graph::force_atlas2`](crate::graph::Graph::force_atlas2).
///
/// # Example
///
/// ```no_run
/// use cugraph::{ForceAtlas2Options, ForceAtlas2Params, IterationCount};
///
/// let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
///     params: ForceAtlas2Params {
///         max_iter: IterationCount::new(100).unwrap(),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Default)]
pub struct ForceAtlas2Options<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Algorithm parameters (iterations, gravity, scaling, etc.).
    pub params: ForceAtlas2Params,
    /// Optional per-vertex attributes (radius, mobility, mass).
    pub vertex_attrs: Option<ForceAtlas2VertexAttrs<'a, V, W>>,
    /// Optional initial vertex positions for refinement passes.
    pub initial_positions: Option<InitialPositions<'a, V, W>>,
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

    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        let view_ptr = unsafe { cugraph_layout_result_get_vertices(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "LayoutResult::vertex_ids")
                .with_message("C API returned null pointer for vertices"));
        }
        let raw = unsafe { crate::array::RawDeviceArrayView::from_borrowed(view_ptr) };
        VertexIds::from_raw(raw)
    }

    /// Get the x-axis coordinates as a runtime-typed device array view.
    ///
    /// Returns layout x-axis coordinates.
    ///
    /// The current cuGraph ForceAtlas2 C API returns Float32 coordinates.
    pub fn x(&self) -> Result<ScoreValues<'_>> {
        let view_ptr = unsafe { cugraph_layout_result_get_x_axis(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "LayoutResult::x")
                .with_message("C API returned null pointer for x-axis coordinates"));
        }
        let raw = unsafe { crate::array::RawDeviceArrayView::from_borrowed(view_ptr) };
        ScoreValues::from_raw(raw)
    }

    /// Get the y-axis coordinates as a runtime-typed device array view.
    ///
    /// Returns layout y-axis coordinates.
    ///
    /// The current cuGraph ForceAtlas2 C API returns Float32 coordinates.
    pub fn y(&self) -> Result<ScoreValues<'_>> {
        let view_ptr = unsafe { cugraph_layout_result_get_y_axis(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "LayoutResult::y")
                .with_message("C API returned null pointer for y-axis coordinates"));
        }
        let raw = unsafe { crate::array::RawDeviceArrayView::from_borrowed(view_ptr) };
        ScoreValues::from_raw(raw)
    }
}

impl LayoutResult {
    /// Get the raw vertices view (for cudf interop).
    pub(crate) fn vertices_raw(&self) -> crate::error::Result<crate::array::RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_layout_result_get_vertices(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "LayoutResult::vertices_raw")
                .with_message("C API returned null pointer for vertices"));
        }
        Ok(unsafe { crate::array::RawDeviceArrayView::from_borrowed(view_ptr) })
    }

    /// Get the raw x-axis view (for cudf interop).
    pub(crate) fn x_raw(&self) -> crate::error::Result<crate::array::RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_layout_result_get_x_axis(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "LayoutResult::x_raw")
                .with_message("C API returned null pointer for x-axis coordinates"));
        }
        Ok(unsafe { crate::array::RawDeviceArrayView::from_borrowed(view_ptr) })
    }

    /// Get the raw y-axis view (for cudf interop).
    pub(crate) fn y_raw(&self) -> crate::error::Result<crate::array::RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_layout_result_get_y_axis(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "LayoutResult::y_raw")
                .with_message("C API returned null pointer for y-axis coordinates"));
        }
        Ok(unsafe { crate::array::RawDeviceArrayView::from_borrowed(view_ptr) })
    }

    /// Convert this layout result to a cudf DataFrame.
    ///
    /// Returns a 3-column DataFrame:
    /// - `"vertex"` — vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"x"` — x-axis coordinates as `Float32`
    /// - `"y"` — y-axis coordinates as `Float32`
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_dataframe(&self) -> crate::error::Result<cudf::DataFrame> {
        cudf::DataFrame::try_from(self).map_err(|e: crate::error::Error| e.with_operation("LayoutResult::to_dataframe"))
    }
}

impl Drop for LayoutResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_layout_result_free(self.raw) };
        }
    }
}

fn validate_force_atlas2_vertex_helper_dtype<T, V: CugraphElement>(
    graph: &Graph<T>,
    helper_name: &'static str,
) -> Result<()> {
    let dtype = V::data_type();
    if !dtype.is_integer() {
        return Err(Error::new(ErrorKind::UnsupportedType, "force_atlas2")
            .with_message(format!("{helper_name} require Int32 or Int64 vertex ids, got {dtype}")));
    }

    if dtype != graph.vertex_dtype() {
        return Err(Error::new(ErrorKind::UnsupportedType, "force_atlas2").with_message(format!(
            "{helper_name} vertex dtype {dtype} does not match graph vertex dtype {}",
            graph.vertex_dtype()
        )));
    }

    Ok(())
}

fn validate_force_atlas2_value_helper_dtype<T, W: CugraphElement>(
    _graph: &Graph<T>,
    helper_name: &'static str,
) -> Result<()> {
    let dtype = W::data_type();
    if dtype != DataType::Float32 {
        return Err(Error::new(ErrorKind::UnsupportedType, "force_atlas2")
            .with_message(format!("{helper_name} require Float32 values, got {dtype}")));
    }

    Ok(())
}

/// Compute a Force Atlas 2 layout for a graph.
///
/// Prefer using [`Graph::force_atlas2`](crate::graph::Graph::force_atlas2)
/// instead of calling this function directly.
pub(crate) fn force_atlas2<T, V: CugraphElement, W: CugraphElement>(
    handle: &ResourceHandle,
    rng_state: &mut RngState,
    graph: &mut Graph<T>,
    options: &ForceAtlas2Options<'_, V, W>,
) -> Result<LayoutResult> {
    let params = &options.params;
    let vertex_attrs = options.vertex_attrs.as_ref();
    let initial_pos = options.initial_positions.as_ref();

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

        validate_force_atlas2_vertex_helper_dtype::<_, V>(graph, "initial positions")?;
        validate_force_atlas2_value_helper_dtype::<_, W>(graph, "initial positions")?;
    }

    // Validate vertex attribute array pairs have consistent lengths.
    if let Some(attrs) = vertex_attrs {
        let has_any_attrs = attrs.radius.is_some() || attrs.mobility.is_some() || attrs.mass.is_some();
        if has_any_attrs {
            validate_force_atlas2_vertex_helper_dtype::<_, V>(graph, "vertex attributes")?;
            validate_force_atlas2_value_helper_dtype::<_, W>(graph, "vertex attributes")?;
        }

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
    let empty = ForceAtlas2VertexAttrs::<V, W>::default();
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
            i32::try_from(params.max_iter)?,
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
            to_bool_t(params.do_expensive_check),
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
        array::DeviceArray,
        graph::{Forward, GraphBuilder},
        params::{ClosedUnitF64, IterationCount, NonNegativeF64},
        types::GraphProperties,
    };

    #[test]
    fn test_force_atlas2_triangle_produces_coordinates() {
        // Triangle: 0->1, 1->2, 2->0 (symmetric for layout purposes)
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 42).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
        let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
            params: ForceAtlas2Params { max_iter: IterationCount::new(50).unwrap(), ..Default::default() },
            ..Default::default()
        };

        let result = graph.force_atlas2(&handle, &mut rng, &options).expect("fa2");

        // Should have 3 vertices with x,y coordinates
        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 3, "should have 3 vertices");
        assert_eq!(result.x().expect("x").len(), 3, "should have 3 x coordinates");
        assert_eq!(result.y().expect("y").len(), 3, "should have 3 y coordinates");

        // Verify coordinates are finite (not NaN or infinity)
        let x_vals: Vec<f32> = result.x().expect("x").to_vec_f32(&handle).expect("x copy");
        let y_vals: Vec<f32> = result.y().expect("y").to_vec_f32(&handle).expect("y copy");

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

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        // First pass: 50 iterations, random initial positions
        let options1: ForceAtlas2Options<'_> = ForceAtlas2Options {
            params: ForceAtlas2Params { max_iter: IterationCount::new(50).unwrap(), ..Default::default() },
            ..Default::default()
        };
        let result1 = graph.force_atlas2(&handle, &mut rng, &options1).expect("fa2 pass 1");

        let vids1 = result1.vertex_ids().expect("vertex_ids");
        assert_eq!(vids1.len(), 4);

        // Second pass: 10 iterations, using first pass result as initial positions
        let vids1_view = vids1.as_i32().expect("i32 vertex_ids");
        let x1_scores = result1.x().expect("x");
        let x1_view = x1_scores.as_f32().expect("f32 x");
        let y1_scores = result1.y().expect("y");
        let y1_view = y1_scores.as_f32().expect("f32 y");
        let init_pos = InitialPositions { vertices: vids1_view, x: x1_view, y: y1_view };
        let options2: ForceAtlas2Options<'_> = ForceAtlas2Options {
            params: ForceAtlas2Params { max_iter: IterationCount::new(10).unwrap(), ..Default::default() },
            initial_positions: Some(init_pos),
            ..Default::default()
        };
        let result2 = graph.force_atlas2(&handle, &mut rng, &options2).expect("fa2 pass 2");

        assert_eq!(result2.vertex_ids().expect("vertex_ids").len(), 4);
        assert_eq!(result2.x().expect("x").len(), 4);
        assert_eq!(result2.y().expect("y").len(), 4);

        // Verify coordinates are finite
        let x_vals: Vec<f32> = result2.x().expect("x").to_vec_f32(&handle).expect("x copy");
        let y_vals: Vec<f32> = result2.y().expect("y").to_vec_f32(&handle).expect("y copy");

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

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
            params: ForceAtlas2Params { max_iter: IterationCount::new(10).unwrap(), ..Default::default() },
            ..Default::default()
        };

        let result = graph.force_atlas2(&handle, &mut rng, &options).expect("fa2");

        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 5, "should have 5 vertices");
        assert_eq!(result.x().expect("x").len(), 5, "should have 5 x coords");
        assert_eq!(result.y().expect("y").len(), 5, "should have 5 y coords");

        // Verify all 5 vertex ids are present
        let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vert copy");
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

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
        let wgt = DeviceArray::from_slice(&handle, &wgt_data).expect("wgt");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .weights(&wgt.view())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
            params: ForceAtlas2Params {
                max_iter: IterationCount::new(50).unwrap(),
                edge_weight_influence: NonNegativeF64::new(1.0).unwrap(),
                ..Default::default()
            },
            ..Default::default()
        };

        let result = graph.force_atlas2(&handle, &mut rng, &options).expect("fa2");

        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 3);
        assert_eq!(result.x().expect("x").len(), 3);
        assert_eq!(result.y().expect("y").len(), 3);
    }

    #[test]
    fn test_force_atlas2_barnes_hut_mode() {
        // Test Barnes-Hut approximation mode.
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 55).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 2, 1, 2, 0];
        let dst_data: Vec<i32> = vec![1, 2, 0, 0, 1, 2];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
            params: ForceAtlas2Params {
                max_iter: IterationCount::new(50).unwrap(),
                barnes_hut_optimize: true,
                barnes_hut_theta: ClosedUnitF64::new(0.5).unwrap(),
                ..Default::default()
            },
            ..Default::default()
        };

        let result = graph.force_atlas2(&handle, &mut rng, &options).expect("fa2 BH");

        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 3);

        let x_vals: Vec<f32> = result.x().expect("x").to_vec_f32(&handle).expect("x copy");
        let y_vals: Vec<f32> = result.y().expect("y").to_vec_f32(&handle).expect("y copy");

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

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        // Create mismatched initial position arrays
        let verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("verts");
        let x_pos = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("x");
        let y_pos = DeviceArray::from_slice(&handle, &[3.0f32]).expect("y"); // wrong length

        let init = InitialPositions { vertices: &verts.view(), x: &x_pos.view(), y: &y_pos.view() };
        let options = ForceAtlas2Options { initial_positions: Some(init), ..Default::default() };

        let err = graph.force_atlas2(&handle, &mut rng, &options).expect_err("should reject mismatched y length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("y length"));
    }

    #[test]
    fn test_force_atlas2_rejects_mismatched_vertex_attrs() {
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 0).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 1, 0];
        let dst_data: Vec<i32> = vec![1, 0, 0, 1];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        // Create mismatched mass arrays
        let mass_verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("mass_verts");
        let mass_vals = DeviceArray::from_slice(&handle, &[1.0f32]).expect("mass_vals"); // wrong length

        let attrs =
            ForceAtlas2VertexAttrs { mass: Some((&mass_verts.view(), &mass_vals.view())), ..Default::default() };
        let options = ForceAtlas2Options { vertex_attrs: Some(attrs), ..Default::default() };

        let err =
            graph.force_atlas2(&handle, &mut rng, &options).expect_err("should reject mismatched mass array lengths");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("mass"));
    }

    #[test]
    fn test_force_atlas2_rejects_initial_positions_for_i64_graphs() {
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 0).expect("rng");

        let src = DeviceArray::from_slice(&handle, &[0i64, 1, 1, 0]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i64, 0, 0, 1]).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        let verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("verts");
        let x_pos = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("x");
        let y_pos = DeviceArray::from_slice(&handle, &[3.0f32, 4.0]).expect("y");

        let init = InitialPositions { vertices: &verts.view(), x: &x_pos.view(), y: &y_pos.view() };
        let options = ForceAtlas2Options { initial_positions: Some(init), ..Default::default() };

        let err = graph
            .force_atlas2(&handle, &mut rng, &options)
            .expect_err("should reject i64 graphs with i32 initial position vertices");
        assert_eq!(err.kind(), ErrorKind::UnsupportedType);
        assert!(err.message().contains("initial positions vertex dtype Int32 does not match graph vertex dtype Int64"));
        assert!(err.message().contains("Int64"));
    }

    #[test]
    fn test_force_atlas2_accepts_f32_vertex_attrs_for_f64_graphs() {
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 0).expect("rng");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 1, 0]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 0, 0, 1]).expect("dst");
        let wgt = DeviceArray::from_slice(&handle, &[1.0f64, 1.0, 1.0, 1.0]).expect("wgt");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .weights(&wgt.view())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        let mass_verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("mass_verts");
        let mass_vals = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("mass_vals");

        let attrs =
            ForceAtlas2VertexAttrs { mass: Some((&mass_verts.view(), &mass_vals.view())), ..Default::default() };
        let options = ForceAtlas2Options { vertex_attrs: Some(attrs), ..Default::default() };

        let result = graph.force_atlas2(&handle, &mut rng, &options).expect("force_atlas2");
        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 2);
        assert!(matches!(result.x().expect("x"), ScoreValues::F32(_)));
        assert!(matches!(result.y().expect("y"), ScoreValues::F32(_)));
    }

    #[test]
    fn test_force_atlas2_lin_log_and_strong_gravity() {
        // Test with lin_log_mode and strong_gravity_mode enabled.
        let handle = ResourceHandle::new().expect("handle");
        let mut rng = RngState::new(&handle, 77).expect("rng");

        let src_data: Vec<i32> = vec![0, 1, 2, 3, 1, 2, 3, 0];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0, 0, 1, 2, 3];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
            params: ForceAtlas2Params {
                max_iter: IterationCount::new(30).unwrap(),
                lin_log_mode: true,
                strong_gravity_mode: true,
                gravity: NonNegativeF64::new(2.0).unwrap(),
                ..Default::default()
            },
            ..Default::default()
        };

        let result = graph.force_atlas2(&handle, &mut rng, &options).expect("fa2");

        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 4);
        assert_eq!(result.x().expect("x").len(), 4);
        assert_eq!(result.y().expect("y").len(), 4);
    }
}
