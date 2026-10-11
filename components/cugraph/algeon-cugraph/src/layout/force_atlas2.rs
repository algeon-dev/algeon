//! Force Atlas 2; force-directed graph layout algorithm.
//!
//! Force Atlas 2 (FA2) computes a 2D layout for graph visualization by
//! simulating physical forces (attraction along edges, repulsion between all
//! vertex pairs, gravity toward center).
//!
//! The algorithm supports an optional initial position for refinement passes
//! (the Python workload calls FA2 twice: first with random positions, then
//! with the previous result as `pos_list`).

use std::{marker::PhantomData, ptr};

use cugraph_sys::{
    cugraph_error_t, cugraph_force_atlas2, cugraph_layout_result_get_vertices, cugraph_layout_result_get_x_axis,
    cugraph_layout_result_get_y_axis, cugraph_layout_result_t,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    cuda::{DeviceResourceKind, drop_device_resource},
    cudf_interop::layout_result_to_dataframe,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Graph, Transposition},
    handle::{EnteredContext, ResourceHandleLease},
    params::{ClosedUnitF64, IterationCount, NonNegativeF64, PositiveF64, defaults},
    random::RngState,
    structure::{
        VertexSelection,
        typed_results::{ScoreValues, VertexIds},
    },
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
#[must_use]
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
    /// Gravity; attracts nodes to center, prevents islands drifting
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
            max_iter: defaults::FORCE_ATLAS2_MAX_ITER,
            outbound_attraction_distribution: false,
            lin_log_mode: false,
            prevent_overlapping: false,
            overlap_scaling_ratio: defaults::FORCE_ATLAS2_OVERLAP_SCALING_RATIO,
            edge_weight_influence: defaults::FORCE_ATLAS2_EDGE_WEIGHT_INFLUENCE,
            jitter_tolerance: defaults::FORCE_ATLAS2_JITTER_TOLERANCE,
            barnes_hut_optimize: false,
            barnes_hut_theta: defaults::FORCE_ATLAS2_BARNES_HUT_THETA,
            scaling_ratio: defaults::FORCE_ATLAS2_SCALING_RATIO,
            strong_gravity_mode: false,
            gravity: defaults::FORCE_ATLAS2_GRAVITY,
            verbose: false,
            do_expensive_check: false,
        }
    }
}

impl<V: CugraphElement, W: CugraphElement> std::fmt::Debug for ForceAtlas2VertexAttrs<'_, V, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForceAtlas2VertexAttrs")
            .field("radius", &self.radius.is_some())
            .field("mobility", &self.mobility.is_some())
            .field("mass", &self.mass.is_some())
            .finish()
    }
}

/// Optional per-vertex attributes for Force Atlas 2.
///
/// Each attribute is a pair of device arrays: one for vertex ids and one for
/// the corresponding values. Both arrays must have the same length.
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

impl<V: CugraphElement, W: CugraphElement> Default for ForceAtlas2VertexAttrs<'_, V, W> {
    fn default() -> Self {
        Self { radius: None, mobility: None, mass: None }
    }
}

impl<V: CugraphElement, W: CugraphElement> std::fmt::Debug for InitialPositions<'_, V, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InitialPositions").finish_non_exhaustive()
    }
}

/// Optional initial positions for Force Atlas 2.
///
/// Used for refinement passes where a previous layout result is fed back as
/// starting positions.
pub struct InitialPositions<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Vertex ids for the initial positions.
    pub vertices: &'a DeviceArrayView<'a, V>,
    /// X-axis initial positions.
    pub x: &'a DeviceArrayView<'a, W>,
    /// Y-axis initial positions.
    pub y: &'a DeviceArrayView<'a, W>,
}

/// Result of a Force Atlas 2 layout computation.
///
/// Provides access to vertex ids and their (x, y) coordinates. The result
/// owns the GPU memory and frees it on drop.
pub struct LayoutResult {
    raw: *mut cugraph_layout_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
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
    unsafe fn from_raw(
        raw: *mut cugraph_layout_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "LayoutResult::from_raw", "layout_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        // SAFETY: `self.raw` is a live layout result owned by this wrapper for
        // the duration of the call.
        let view_ptr = require_non_null(
            unsafe { cugraph_layout_result_get_vertices(self.raw) },
            "LayoutResult::vertex_ids",
            "vertices",
        )?;
        // SAFETY: cuGraph allocates view metadata; this wrapper frees it while the result retains the backing arrays.
        let raw = unsafe {
            RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), "LayoutResult::vertex_ids", "vertices")
        }?;
        VertexIds::from_raw(raw)
    }

    /// Get the x-axis coordinates as a runtime-typed device array view.
    ///
    /// Returns layout x-axis coordinates.
    ///
    /// The current cuGraph ForceAtlas2 C API returns Float32 coordinates.
    pub fn x(&self) -> Result<ScoreValues<'_>> {
        // SAFETY: `self.raw` is a live layout result owned by this wrapper for
        // the duration of the call.
        let view_ptr =
            require_non_null(unsafe { cugraph_layout_result_get_x_axis(self.raw) }, "LayoutResult::x", "x_axis")?;
        // SAFETY: cuGraph allocates view metadata; this wrapper frees it while the result retains the backing arrays.
        let raw =
            unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), "LayoutResult::x", "x_axis") }?;
        ScoreValues::from_raw(raw)
    }

    /// Get the y-axis coordinates as a runtime-typed device array view.
    ///
    /// Returns layout y-axis coordinates.
    ///
    /// The current cuGraph ForceAtlas2 C API returns Float32 coordinates.
    pub fn y(&self) -> Result<ScoreValues<'_>> {
        // SAFETY: `self.raw` is a live layout result owned by this wrapper for
        // the duration of the call.
        let view_ptr =
            require_non_null(unsafe { cugraph_layout_result_get_y_axis(self.raw) }, "LayoutResult::y", "y_axis")?;
        // SAFETY: cuGraph allocates view metadata; this wrapper frees it while the result retains the backing arrays.
        let raw =
            unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), "LayoutResult::y", "y_axis") }?;
        ScoreValues::from_raw(raw)
    }

    /// Convert this layout result to a cudf DataFrame.
    ///
    /// Returns a 3-column DataFrame:
    /// - `"vertex"`: vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"x"`: x-axis coordinates as `Float32`
    /// - `"y"`: y-axis coordinates as `Float32`
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        layout_result_to_dataframe(self, runtime).map_err(|e: Error| e.with_operation("LayoutResult::to_dataframe"))
    }
}

impl Drop for LayoutResult {
    fn drop(&mut self) {
        drop_device_resource(&mut self.raw, self.device, "LayoutResult::drop", DeviceResourceKind::LayoutResult);
    }
}

fn validate_force_atlas2_vertex_helper_dtype<T, V: CugraphElement>(
    graph: &Graph<T>,
    input_name: &'static str,
) -> Result<()> {
    let dtype = V::data_type();
    if !dtype.is_supported_index_type() {
        return Err(Error::new(ErrorKind::UnsupportedType, "force_atlas2")
            .with_message(format!("{input_name} require Int32 or Int64 vertex ids, got {dtype}")));
    }

    if dtype != graph.vertex_dtype() {
        return Err(Error::new(ErrorKind::UnsupportedType, "force_atlas2").with_message(format!(
            "{input_name} vertex dtype {dtype} does not match graph vertex dtype {}",
            graph.vertex_dtype()
        )));
    }

    Ok(())
}

fn validate_force_atlas2_value_helper_dtype<T, W: CugraphElement>(
    _graph: &Graph<T>,
    input_name: &'static str,
) -> Result<()> {
    let dtype = W::data_type();
    if dtype != DataType::Float32 {
        return Err(Error::new(ErrorKind::UnsupportedType, "force_atlas2")
            .with_message(format!("{input_name} require Float32 values, got {dtype}")));
    }

    Ok(())
}

fn graph_vertex_count<T: Transposition>(handle: &EnteredContext<'_>, graph: &mut Graph<T>) -> Result<usize> {
    let degrees = graph.out_degrees(handle, VertexSelection::<i32>::All)?;
    let vertex_ids = degrees.vertex_ids()?;
    Ok(vertex_ids.len())
}

/// Compute a Force Atlas 2 layout for a graph.
///
/// Prefer using [`Graph::force_atlas2`](crate::graph::Graph::force_atlas2),
/// which calls this function.
pub(crate) fn force_atlas2<T: Transposition, V: CugraphElement, W: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<T>,
    params: &ForceAtlas2Params,
    vertex_attrs: Option<&ForceAtlas2VertexAttrs<'_, V, W>>,
    initial_pos: Option<&InitialPositions<'_, V, W>>,
) -> Result<LayoutResult> {
    let graph_vertex_count =
        if initial_pos.is_some() || vertex_attrs.is_some() { Some(graph_vertex_count(handle, graph)?) } else { None };

    if let Some(pos) = initial_pos {
        let graph_vertex_count = graph_vertex_count.expect("graph vertex count should exist for initial positions");
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

        if n != graph_vertex_count {
            return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                .with_message("initial positions must provide one coordinate pair for each graph vertex")
                .with_context("graph_vertex_count", graph_vertex_count.to_string())
                .with_context("vertices_len", n.to_string()));
        }
    }

    // Validate vertex attribute array pairs have consistent lengths.
    if let Some(attrs) = vertex_attrs {
        let graph_vertex_count = graph_vertex_count.expect("graph vertex count should exist for vertex attrs");
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
            if vals.len() != graph_vertex_count {
                return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                    .with_message("vertex radius must provide one value for each graph vertex")
                    .with_context("graph_vertex_count", graph_vertex_count.to_string())
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
            if vals.len() != graph_vertex_count {
                return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                    .with_message("vertex mobility must provide one value for each graph vertex")
                    .with_context("graph_vertex_count", graph_vertex_count.to_string())
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
            if vals.len() != graph_vertex_count {
                return Err(Error::new(ErrorKind::InvalidInput, "force_atlas2")
                    .with_message("vertex mass must provide one value for each graph vertex")
                    .with_context("graph_vertex_count", graph_vertex_count.to_string())
                    .with_context("values_len", vals.len().to_string()));
            }
        }
    }

    let (start_verts_ptr, x_start_ptr, y_start_ptr) = match initial_pos {
        Some(pos) => (pos.vertices.as_ptr(), pos.x.as_ptr(), pos.y.as_ptr()),
        None => (ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
    };

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
    let graph_ptr = graph.as_ptr(handle, "force_atlas2")?;

    // SAFETY: every optional pointer is either null or borrowed from a live
    // device array view whose dtype/length was validated above, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_force_atlas2(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
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
        check_error(code, error, handle.device(), "force_atlas2")?;
    }

    let result = require_non_null(result, "force_atlas2", "layout_result")?;

    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    unsafe { LayoutResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

#[cfg(test)]
mod layout_result_constructor_tests {
    use std::ptr;

    use super::*;

    #[test]
    fn layout_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = unsafe { LayoutResult::from_raw(ptr::null_mut(), 0, None) }.expect_err("null must be rejected");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "LayoutResult::from_raw");
    }
}
