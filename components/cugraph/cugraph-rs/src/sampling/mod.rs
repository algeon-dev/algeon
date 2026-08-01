//! Random-walk and sampling algorithms.

use std::{ffi::CString, num::NonZeroUsize, ptr};

use cugraph_sys::{
    cugraph_biased_random_walks, cugraph_compression_type_t, cugraph_compression_type_t_COO,
    cugraph_compression_type_t_CSC, cugraph_compression_type_t_CSR, cugraph_compression_type_t_DCSC,
    cugraph_compression_type_t_DCSR, cugraph_error_t, cugraph_homogeneous_biased_neighbor_sample,
    cugraph_homogeneous_biased_temporal_neighbor_sample, cugraph_homogeneous_uniform_neighbor_sample,
    cugraph_homogeneous_uniform_temporal_neighbor_sample, cugraph_prior_sources_behavior_t,
    cugraph_prior_sources_behavior_t_CARRY_OVER, cugraph_prior_sources_behavior_t_DEFAULT,
    cugraph_prior_sources_behavior_t_EXCLUDE, cugraph_random_walk_result_free,
    cugraph_random_walk_result_get_max_path_length, cugraph_random_walk_result_get_paths,
    cugraph_random_walk_result_get_weights, cugraph_random_walk_result_t, cugraph_sample_result_free,
    cugraph_sample_result_get_edge_end_time, cugraph_sample_result_get_edge_id,
    cugraph_sample_result_get_edge_renumber_map, cugraph_sample_result_get_edge_renumber_map_offsets,
    cugraph_sample_result_get_edge_start_time, cugraph_sample_result_get_edge_type,
    cugraph_sample_result_get_edge_weight, cugraph_sample_result_get_hop, cugraph_sample_result_get_label_hop_offsets,
    cugraph_sample_result_get_label_type_hop_offsets, cugraph_sample_result_get_major_offsets,
    cugraph_sample_result_get_majors, cugraph_sample_result_get_minors, cugraph_sample_result_get_renumber_map,
    cugraph_sample_result_get_renumber_map_offsets, cugraph_sample_result_get_start_labels, cugraph_sample_result_t,
    cugraph_sampling_options_create, cugraph_sampling_options_free, cugraph_sampling_options_t,
    cugraph_sampling_set_compress_per_hop, cugraph_sampling_set_compression_type, cugraph_sampling_set_dedupe_sources,
    cugraph_sampling_set_disjoint_sampling, cugraph_sampling_set_prior_sources_behavior,
    cugraph_sampling_set_renumber_results, cugraph_sampling_set_retain_seeds, cugraph_sampling_set_return_hops,
    cugraph_sampling_set_temporal_sampling_comparison, cugraph_sampling_set_with_replacement,
    cugraph_select_random_vertices, cugraph_temporal_sampling_comparison_t,
    cugraph_temporal_sampling_comparison_t_LAST, cugraph_temporal_sampling_comparison_t_MONOTONICALLY_DECREASING,
    cugraph_temporal_sampling_comparison_t_MONOTONICALLY_INCREASING,
    cugraph_temporal_sampling_comparison_t_STRICTLY_DECREASING,
    cugraph_temporal_sampling_comparison_t_STRICTLY_INCREASING, cugraph_type_erased_device_array_t,
    cugraph_uniform_random_walks,
};

use crate::{
    array::{DeviceArray, DeviceArrayView, HostArrayView, RawDeviceArrayView},
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::{EnteredContext, ResourceHandleLease},
    params::PositiveF64,
    random::RngState,
    structure::{IntegerValues, ScoreValues, SizeValues, VertexIds},
    types::{CugraphElement, DataType, to_bool_t},
};

/// Parameters shared by uniform and biased random walks.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RandomWalkParams {
    /// Maximum number of edges traversed per walk.
    pub max_length: NonZeroUsize,
}

impl Default for RandomWalkParams {
    fn default() -> Self {
        Self { max_length: NonZeroUsize::new(1).expect("literal is non-zero") }
    }
}

/// Parameters for node2vec random walks.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Node2VecRandomWalkParams {
    /// Maximum number of edges traversed per walk.
    pub max_length: NonZeroUsize,
    /// Return parameter.
    pub p: PositiveF64,
    /// In/out parameter.
    pub q: PositiveF64,
}

impl Default for Node2VecRandomWalkParams {
    fn default() -> Self {
        Self {
            max_length: NonZeroUsize::new(1).expect("literal is non-zero"),
            p: PositiveF64::new(1.0).expect("literal is positive"),
            q: PositiveF64::new(1.0).expect("literal is positive"),
        }
    }
}

/// Sampling output compression mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum SamplingCompression {
    #[default]
    Coo,
    Csr,
    Csc,
    Dcsr,
    Dcsc,
}

impl SamplingCompression {
    fn to_c(self) -> cugraph_compression_type_t {
        match self {
            Self::Coo => cugraph_compression_type_t_COO,
            Self::Csr => cugraph_compression_type_t_CSR,
            Self::Csc => cugraph_compression_type_t_CSC,
            Self::Dcsr => cugraph_compression_type_t_DCSR,
            Self::Dcsc => cugraph_compression_type_t_DCSC,
        }
    }
}

/// How prior-hop sources contribute to the next sampling frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum PriorSourcesBehavior {
    #[default]
    Default,
    CarryOver,
    Exclude,
}

impl PriorSourcesBehavior {
    fn to_c(self) -> cugraph_prior_sources_behavior_t {
        match self {
            Self::Default => cugraph_prior_sources_behavior_t_DEFAULT,
            Self::CarryOver => cugraph_prior_sources_behavior_t_CARRY_OVER,
            Self::Exclude => cugraph_prior_sources_behavior_t_EXCLUDE,
        }
    }
}

/// Temporal edge-time comparison used by temporal neighbor sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum TemporalSamplingComparison {
    #[default]
    StrictlyIncreasing,
    MonotonicallyIncreasing,
    StrictlyDecreasing,
    MonotonicallyDecreasing,
    Last,
}

impl TemporalSamplingComparison {
    fn to_c(self) -> cugraph_temporal_sampling_comparison_t {
        match self {
            Self::StrictlyIncreasing => cugraph_temporal_sampling_comparison_t_STRICTLY_INCREASING,
            Self::MonotonicallyIncreasing => cugraph_temporal_sampling_comparison_t_MONOTONICALLY_INCREASING,
            Self::StrictlyDecreasing => cugraph_temporal_sampling_comparison_t_STRICTLY_DECREASING,
            Self::MonotonicallyDecreasing => cugraph_temporal_sampling_comparison_t_MONOTONICALLY_DECREASING,
            Self::Last => cugraph_temporal_sampling_comparison_t_LAST,
        }
    }
}

/// Options shared by homogeneous neighbor-sampling APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub struct SamplingOptions {
    pub retain_seeds: bool,
    pub renumber_results: bool,
    pub compress_per_hop: bool,
    pub with_replacement: bool,
    pub return_hops: bool,
    pub compression: SamplingCompression,
    pub prior_sources_behavior: PriorSourcesBehavior,
    pub dedupe_sources: bool,
    pub temporal_sampling_comparison: TemporalSamplingComparison,
    pub disjoint_sampling: bool,
}

/// Parameters for homogeneous neighbor sampling.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct NeighborSamplingParams {
    /// Per-hop fanout values. cuGraph requires `i32` fanout entries.
    pub fan_out: Vec<i32>,
    pub options: SamplingOptions,
    pub do_expensive_check: bool,
}

impl NeighborSamplingParams {
    pub fn new(fan_out: Vec<i32>) -> Self {
        Self { fan_out, options: SamplingOptions::default(), do_expensive_check: false }
    }
}

impl Default for NeighborSamplingParams {
    fn default() -> Self {
        Self::new(vec![1])
    }
}

struct NativeSamplingOptions {
    raw: *mut cugraph_sampling_options_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl NativeSamplingOptions {
    fn new(options: &SamplingOptions) -> Result<Self> {
        let mut raw: *mut cugraph_sampling_options_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        // SAFETY: `raw` and `error` are valid out-pointers.
        unsafe {
            let code = cugraph_sampling_options_create(&mut raw, &mut error);
            check_error(code, error, "NativeSamplingOptions::new")?;
        }
        let raw = require_non_null(raw, "NativeSamplingOptions::new", "sampling_options")?;

        // SAFETY: `raw` is a live sampling-options object owned by this
        // wrapper, and setters only mutate that object.
        unsafe {
            cugraph_sampling_set_retain_seeds(raw, to_bool_t(options.retain_seeds));
            cugraph_sampling_set_renumber_results(raw, to_bool_t(options.renumber_results));
            cugraph_sampling_set_compress_per_hop(raw, to_bool_t(options.compress_per_hop));
            cugraph_sampling_set_with_replacement(raw, to_bool_t(options.with_replacement));
            cugraph_sampling_set_return_hops(raw, to_bool_t(options.return_hops));
            cugraph_sampling_set_compression_type(raw, options.compression.to_c());
            cugraph_sampling_set_prior_sources_behavior(raw, options.prior_sources_behavior.to_c());
            cugraph_sampling_set_dedupe_sources(raw, to_bool_t(options.dedupe_sources));
            cugraph_sampling_set_temporal_sampling_comparison(raw, options.temporal_sampling_comparison.to_c());
            cugraph_sampling_set_disjoint_sampling(raw, to_bool_t(options.disjoint_sampling));
        }

        Ok(Self { raw, _nosend_nosync: std::marker::PhantomData })
    }

    fn as_ptr(&self) -> *const cugraph_sampling_options_t {
        self.raw
    }
}

impl Drop for NativeSamplingOptions {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `self.raw` is owned by this wrapper and freed exactly
            // once here.
            unsafe { cugraph_sampling_options_free(self.raw) };
        }
    }
}

/// Result of a random-walk computation.
pub struct RandomWalkResult {
    raw: *mut cugraph_random_walk_result_t,
    device: i32,
    has_weights: bool,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for RandomWalkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomWalkResult")
            .field("max_path_length", &self.max_path_length())
            .field("has_weights", &self.has_weights)
            .finish_non_exhaustive()
    }
}

impl RandomWalkResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph
    /// random-walk C API. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_random_walk_result_t,
        device: i32,
        has_weights: bool,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "RandomWalkResult::from_raw", "random_walk_result")?;
        Ok(Self {
            raw,
            device,
            has_weights,
            _resource_handle_lease: resource_handle_lease,
            _nosend_nosync: std::marker::PhantomData,
        })
    }

    /// Maximum number of edges requested per walk.
    pub fn max_path_length(&self) -> usize {
        // SAFETY: `self.raw` is a live random-walk result handle.
        unsafe { cugraph_random_walk_result_get_max_path_length(self.raw) }
    }

    /// Dense row-major vertex path matrix.
    ///
    /// The row count is the number of start vertices and each row contains
    /// `max_path_length() + 1` vertex ids. Native cuGraph uses a negative
    /// sentinel in later columns when a walk terminates early.
    pub fn paths(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_random_walk_result_get_paths, "RandomWalkResult::paths", "paths")?)
    }

    /// Dense row-major edge-weight matrix, if the input graph had weights.
    ///
    /// Each row contains `max_path_length()` edge weights. For unweighted
    /// graphs the native result has no weight column, so this returns `None`.
    pub fn weights(&self) -> Result<Option<ScoreValues<'_>>> {
        if !self.has_weights {
            return Ok(None);
        }

        let raw = self.raw_view(cugraph_random_walk_result_get_weights, "RandomWalkResult::weights", "weights")?;
        ScoreValues::from_raw(raw).map(Some)
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_random_walk_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is owned by this wrapper and remains live for
            // the duration of this call.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: random-walk getters allocate view metadata for the caller.
        // The backing device arrays are owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }
}

impl Drop for RandomWalkResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "RandomWalkResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and freed at most
            // once here.
            unsafe { cugraph_random_walk_result_free(self.raw) };
        }
    }
}

/// Result of a homogeneous neighbor-sampling computation.
pub struct SampleResult {
    raw: *mut cugraph_sample_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for SampleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampleResult").finish_non_exhaustive()
    }
}

impl SampleResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph sampling
    /// C API. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_sample_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "SampleResult::from_raw", "sample_result")?;
        Ok(Self {
            raw,
            device,
            _resource_handle_lease: resource_handle_lease,
            _nosend_nosync: std::marker::PhantomData,
        })
    }

    /// Sampled major vertices, or `None` when the output is compressed.
    pub fn majors(&self) -> Result<Option<VertexIds<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_majors, "SampleResult::majors")
            .and_then(|raw| raw.map(VertexIds::from_raw).transpose())
    }

    /// Sampled minor vertices.
    pub fn minors(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.required_raw_view(
            cugraph_sample_result_get_minors,
            "SampleResult::minors",
            "minors",
        )?)
    }

    /// Major offsets for compressed outputs.
    pub fn major_offsets(&self) -> Result<Option<SizeValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_major_offsets, "SampleResult::major_offsets")
            .and_then(|raw| raw.map(SizeValues::from_raw).transpose())
    }

    /// Optional start labels.
    pub fn start_labels(&self) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_start_labels, "SampleResult::start_labels")
            .and_then(|raw| raw.map(IntegerValues::from_raw).transpose())
    }

    /// Optional sampled edge ids.
    pub fn edge_ids(&self) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_edge_id, "SampleResult::edge_ids")
            .and_then(|raw| raw.map(IntegerValues::from_raw).transpose())
    }

    /// Optional sampled edge type ids.
    pub fn edge_types(&self) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_edge_type, "SampleResult::edge_types")
            .and_then(|raw| raw.map(IntegerValues::from_raw).transpose())
    }

    /// Optional sampled edge weights.
    pub fn edge_weights(&self) -> Result<Option<ScoreValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_edge_weight, "SampleResult::edge_weights")
            .and_then(|raw| raw.map(ScoreValues::from_raw).transpose())
    }

    /// Optional sampled edge start times.
    pub fn edge_start_times(&self) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_edge_start_time, "SampleResult::edge_start_times")
            .and_then(|raw| raw.map(IntegerValues::from_raw).transpose())
    }

    /// Optional sampled edge end times.
    pub fn edge_end_times(&self) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_edge_end_time, "SampleResult::edge_end_times")
            .and_then(|raw| raw.map(IntegerValues::from_raw).transpose())
    }

    /// Optional hop ids.
    pub fn hops(&self) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_hop, "SampleResult::hops")
            .and_then(|raw| raw.map(IntegerValues::from_raw).transpose())
    }

    /// Optional label-hop offsets.
    pub fn label_hop_offsets(&self) -> Result<Option<SizeValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_label_hop_offsets, "SampleResult::label_hop_offsets")
            .and_then(|raw| raw.map(SizeValues::from_raw).transpose())
    }

    /// Optional label-type-hop offsets.
    pub fn label_type_hop_offsets(&self) -> Result<Option<SizeValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_label_type_hop_offsets, "SampleResult::label_type_hop_offsets")
            .and_then(|raw| raw.map(SizeValues::from_raw).transpose())
    }

    /// Optional renumber map.
    pub fn renumber_map(&self) -> Result<Option<VertexIds<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_renumber_map, "SampleResult::renumber_map")
            .and_then(|raw| raw.map(VertexIds::from_raw).transpose())
    }

    /// Optional renumber-map offsets.
    pub fn renumber_map_offsets(&self) -> Result<Option<SizeValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_renumber_map_offsets, "SampleResult::renumber_map_offsets")
            .and_then(|raw| raw.map(SizeValues::from_raw).transpose())
    }

    /// Optional edge-renumber map.
    pub fn edge_renumber_map(&self) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_sample_result_get_edge_renumber_map, "SampleResult::edge_renumber_map")
            .and_then(|raw| raw.map(IntegerValues::from_raw).transpose())
    }

    /// Optional edge-renumber-map offsets.
    pub fn edge_renumber_map_offsets(&self) -> Result<Option<SizeValues<'_>>> {
        self.optional_raw_view(
            cugraph_sample_result_get_edge_renumber_map_offsets,
            "SampleResult::edge_renumber_map_offsets",
        )
        .and_then(|raw| raw.map(SizeValues::from_raw).transpose())
    }

    fn optional_raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *const cugraph_sample_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
    ) -> Result<Option<RawDeviceArrayView<'_>>> {
        // SAFETY: `self.raw` is owned by this wrapper and remains live for
        // the duration of this call.
        let view_ptr = unsafe { getter(self.raw) };
        if view_ptr.is_null() {
            return Ok(None);
        }

        // SAFETY: sample-result getters allocate view metadata for the caller.
        // The backing device arrays are owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, "view").map(Some) }
    }

    fn required_raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *const cugraph_sample_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        self.optional_raw_view(getter, operation)?.ok_or_else(|| {
            Error::new(ErrorKind::InvalidState, operation)
                .with_message(format!("native sample result returned null {pointer_name}"))
        })
    }
}

impl Drop for SampleResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "SampleResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and freed at most
            // once here.
            unsafe { cugraph_sample_result_free(self.raw) };
        }
    }
}

type RandomWalkFn = unsafe extern "C" fn(
    *const cugraph_sys::cugraph_resource_handle_t,
    *mut cugraph_sys::cugraph_rng_state_t,
    *mut cugraph_sys::cugraph_graph_t,
    *const cugraph_sys::cugraph_type_erased_device_array_view_t,
    usize,
    *mut *mut cugraph_random_walk_result_t,
    *mut *mut cugraph_error_t,
) -> cugraph_sys::cugraph_error_code_t;

pub(crate) fn uniform_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    params: &RandomWalkParams,
) -> Result<RandomWalkResult> {
    run_random_walks(
        handle,
        rng_state,
        graph,
        start_vertices,
        params.max_length.get(),
        cugraph_uniform_random_walks,
        "uniform_random_walks",
    )
}

pub(crate) fn biased_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    params: &RandomWalkParams,
) -> Result<RandomWalkResult> {
    run_random_walks(
        handle,
        rng_state,
        graph,
        start_vertices,
        params.max_length.get(),
        cugraph_biased_random_walks,
        "biased_random_walks",
    )
}

fn run_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    max_length: usize,
    function: RandomWalkFn,
    operation: &'static str,
) -> Result<RandomWalkResult> {
    graph.validate_vertex_input_dtype(V::data_type(), operation, "start_vertices")?;
    let mut result: *mut cugraph_random_walk_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, operation)?;

    // SAFETY: `graph`, `rng_state`, and `start_vertices` are live cuGraph
    // objects, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = function(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            start_vertices.as_ptr(),
            max_length,
            &mut result,
            &mut error,
        );
        check_error(code, error, operation)?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { RandomWalkResult::from_raw(result, handle.device(), graph.has_weights(), handle.runtime_bound_lease()) }
}

pub(crate) fn node2vec_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    params: &Node2VecRandomWalkParams,
) -> Result<RandomWalkResult> {
    graph.validate_vertex_input_dtype(V::data_type(), "node2vec_random_walks", "start_vertices")?;
    let mut result: *mut cugraph_random_walk_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "node2vec_random_walks")?;

    // SAFETY: `graph`, `rng_state`, and `start_vertices` are live cuGraph
    // objects, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_sys::cugraph_node2vec_random_walks(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            start_vertices.as_ptr(),
            params.max_length.get(),
            params.p.get(),
            params.q.get(),
            &mut result,
            &mut error,
        );
        check_error(code, error, "node2vec_random_walks")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { RandomWalkResult::from_raw(result, handle.device(), graph.has_weights(), handle.runtime_bound_lease()) }
}

pub(crate) fn select_random_vertices<T, V>(
    handle: &EnteredContext<'_>,
    graph: &Graph<T>,
    rng_state: &mut RngState,
    num_vertices: usize,
) -> Result<DeviceArray<V>>
where
    V: CugraphElement,
{
    graph.validate_vertex_input_dtype(V::data_type(), "select_random_vertices", "vertices")?;
    let mut vertices: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "select_random_vertices")?;

    // SAFETY: `graph` and `rng_state` are live cuGraph objects, and
    // `vertices`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_select_random_vertices(
            handle.handle_ptr(),
            graph_ptr,
            rng_state.as_ptr(),
            num_vertices,
            &mut vertices,
            &mut error,
        );
        check_error(code, error, "select_random_vertices")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the device array.
    unsafe {
        DeviceArray::from_raw(
            vertices,
            handle.device(),
            "select_random_vertices",
            "vertices",
            handle.runtime_bound_lease(),
        )
    }
}

fn fan_out_view(params: &NeighborSamplingParams) -> Result<HostArrayView<'_>> {
    if params.fan_out.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "NeighborSamplingParams::fan_out")
            .with_message("fan_out must contain at least one hop"));
    }

    let byte_len = params.fan_out.len() * std::mem::size_of::<i32>();
    // SAFETY: `fan_out` is a contiguous `Vec<i32>` and the byte slice is
    // borrowed for no longer than `params`.
    let bytes = unsafe { std::slice::from_raw_parts(params.fan_out.as_ptr().cast::<u8>(), byte_len) };
    HostArrayView::new(bytes, params.fan_out.len(), DataType::Int32)
}

pub(crate) fn homogeneous_uniform_neighbor_sample<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
    params: &NeighborSamplingParams,
) -> Result<SampleResult> {
    graph.validate_vertex_input_dtype(V::data_type(), "homogeneous_uniform_neighbor_sample", "start_vertices")?;
    let fan_out = fan_out_view(params)?;
    let options = NativeSamplingOptions::new(&params.options)?;
    let mut result: *mut cugraph_sample_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let label_offsets = starting_vertex_label_offsets.map(|v| v.as_ptr()).unwrap_or(ptr::null_mut());
    let graph_ptr = graph.as_ptr(handle, "homogeneous_uniform_neighbor_sample")?;

    // SAFETY: all input pointers are either null or live cuGraph view handles,
    // and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_homogeneous_uniform_neighbor_sample(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            start_vertices.as_ptr(),
            label_offsets,
            fan_out.as_ptr(),
            options.as_ptr(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "homogeneous_uniform_neighbor_sample")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { SampleResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

pub(crate) fn homogeneous_biased_neighbor_sample<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
    params: &NeighborSamplingParams,
) -> Result<SampleResult> {
    graph.validate_vertex_input_dtype(V::data_type(), "homogeneous_biased_neighbor_sample", "start_vertices")?;
    let fan_out = fan_out_view(params)?;
    let options = NativeSamplingOptions::new(&params.options)?;
    let mut result: *mut cugraph_sample_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let label_offsets = starting_vertex_label_offsets.map(|v| v.as_ptr()).unwrap_or(ptr::null_mut());
    let graph_ptr = graph.as_ptr(handle, "homogeneous_biased_neighbor_sample")?;

    // SAFETY: all input pointers are either null or live cuGraph view handles,
    // and `result`/`error` are valid out-pointers. Edge biases are null because
    // the upstream C constructors for independent edge-property views are not
    // exposed in this header; native cuGraph then uses graph weights as biases
    // when present.
    unsafe {
        let code = cugraph_homogeneous_biased_neighbor_sample(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            ptr::null(),
            start_vertices.as_ptr(),
            label_offsets,
            fan_out.as_ptr(),
            options.as_ptr(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "homogeneous_biased_neighbor_sample")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { SampleResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

pub(crate) fn homogeneous_uniform_temporal_neighbor_sample<V, Tm>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    temporal_property_name: &str,
    start_vertices: &DeviceArrayView<'_, V>,
    starting_vertex_times: Option<&DeviceArrayView<'_, Tm>>,
    starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
    params: &NeighborSamplingParams,
) -> Result<SampleResult>
where
    V: CugraphElement,
    Tm: CugraphElement,
{
    graph.validate_vertex_input_dtype(
        V::data_type(),
        "homogeneous_uniform_temporal_neighbor_sample",
        "start_vertices",
    )?;
    if let Some(times) = starting_vertex_times {
        if times.len() != start_vertices.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "homogeneous_uniform_temporal_neighbor_sample")
                .with_message("starting_vertex_times length must match start_vertices length")
                .with_context("start_vertices_len", start_vertices.len().to_string())
                .with_context("starting_vertex_times_len", times.len().to_string()));
        }
    }

    let temporal_property_name = CString::new(temporal_property_name).map_err(|_| {
        Error::new(ErrorKind::InvalidInput, "homogeneous_uniform_temporal_neighbor_sample")
            .with_message("temporal_property_name must not contain a NUL byte")
    })?;
    let fan_out = fan_out_view(params)?;
    let options = NativeSamplingOptions::new(&params.options)?;
    let mut result: *mut cugraph_sample_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let times = starting_vertex_times.map(|v| v.as_ptr()).unwrap_or(ptr::null_mut());
    let label_offsets = starting_vertex_label_offsets.map(|v| v.as_ptr()).unwrap_or(ptr::null_mut());
    let graph_ptr = graph.as_ptr(handle, "homogeneous_uniform_temporal_neighbor_sample")?;

    // SAFETY: all input pointers are either null or live cuGraph view handles,
    // `temporal_property_name` is a live NUL-terminated C string, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_homogeneous_uniform_temporal_neighbor_sample(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            temporal_property_name.as_ptr(),
            start_vertices.as_ptr(),
            times,
            label_offsets,
            fan_out.as_ptr(),
            options.as_ptr(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "homogeneous_uniform_temporal_neighbor_sample")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { SampleResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

pub(crate) fn homogeneous_biased_temporal_neighbor_sample<V, Tm>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    temporal_property_name: &str,
    start_vertices: &DeviceArrayView<'_, V>,
    starting_vertex_times: Option<&DeviceArrayView<'_, Tm>>,
    starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
    params: &NeighborSamplingParams,
) -> Result<SampleResult>
where
    V: CugraphElement,
    Tm: CugraphElement,
{
    graph.validate_vertex_input_dtype(
        V::data_type(),
        "homogeneous_biased_temporal_neighbor_sample",
        "start_vertices",
    )?;
    if let Some(times) = starting_vertex_times {
        if times.len() != start_vertices.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "homogeneous_biased_temporal_neighbor_sample")
                .with_message("starting_vertex_times length must match start_vertices length")
                .with_context("start_vertices_len", start_vertices.len().to_string())
                .with_context("starting_vertex_times_len", times.len().to_string()));
        }
    }

    let temporal_property_name = CString::new(temporal_property_name).map_err(|_| {
        Error::new(ErrorKind::InvalidInput, "homogeneous_biased_temporal_neighbor_sample")
            .with_message("temporal_property_name must not contain a NUL byte")
    })?;
    let fan_out = fan_out_view(params)?;
    let options = NativeSamplingOptions::new(&params.options)?;
    let mut result: *mut cugraph_sample_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let times = starting_vertex_times.map(|v| v.as_ptr()).unwrap_or(ptr::null_mut());
    let label_offsets = starting_vertex_label_offsets.map(|v| v.as_ptr()).unwrap_or(ptr::null_mut());
    let graph_ptr = graph.as_ptr(handle, "homogeneous_biased_temporal_neighbor_sample")?;

    // SAFETY: all input pointers are either null or live cuGraph view handles,
    // `temporal_property_name` is a live NUL-terminated C string, and
    // `result`/`error` are valid out-pointers. Edge biases are null for the
    // same reason as the non-temporal biased wrapper.
    unsafe {
        let code = cugraph_homogeneous_biased_temporal_neighbor_sample(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            temporal_property_name.as_ptr(),
            ptr::null(),
            start_vertices.as_ptr(),
            times,
            label_offsets,
            fan_out.as_ptr(),
            options.as_ptr(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "homogeneous_biased_temporal_neighbor_sample")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { SampleResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn random_walk_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { RandomWalkResult::from_raw(ptr::null_mut(), 0, false, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "RandomWalkResult::from_raw");
    }

    #[test]
    fn sample_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { SampleResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "SampleResult::from_raw");
    }

    #[test]
    fn neighbor_sampling_fan_out_rejects_empty() {
        let params = NeighborSamplingParams::new(Vec::new());

        let err = fan_out_view(&params).expect_err("empty fan_out must be rejected");

        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "NeighborSamplingParams::fan_out");
    }
}
