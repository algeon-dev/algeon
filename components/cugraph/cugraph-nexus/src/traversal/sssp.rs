//! Single-Source Shortest Path (SSSP) traversal algorithm.

use std::{collections::HashMap, fmt::Debug, ptr};

use cugraph_sys::{cugraph_error_t, cugraph_paths_result_t, cugraph_sssp, cugraph_sssp_workspace_preflight};

use crate::{
    array::DeviceArray,
    error::{Error, ErrorKind, Result, check_error},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::NonNegativeF64,
    structure::{
        VertexSelection,
        typed_results::{Distances, VertexIds},
    },
    traversal::paths::{ExtractPathsResult, PathsResult},
    types::{CugraphElement, DataType, to_bool_t},
};

const SSSP_PREFLIGHT_OPERATION: &str = "sssp_workspace_preflight";

/// Bounded SSSP shape used by the native allocation preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct SsspWorkspacePreflightInput {
    edge_count: usize,
    weight_type: DataType,
    has_edge_weights: bool,
}

impl SsspWorkspacePreflightInput {
    /// Create a preflight input for the graph shape passed to SSSP.
    pub fn new(edge_count: usize, weight_type: DataType, has_edge_weights: bool) -> Result<Self> {
        if !weight_type.is_float() {
            return Err(Error::new(ErrorKind::UnsupportedType, "SsspWorkspacePreflightInput::new")
                .with_message(format!("SSSP weight_type must be Float32 or Float64, got {weight_type}")));
        }
        Ok(Self { edge_count, weight_type, has_edge_weights })
    }

    /// Physical edge count or admission-time hard upper bound.
    pub const fn edge_count(self) -> usize {
        self.edge_count
    }

    /// Graph weight dtype, including the default Float32 dtype of an unweighted graph.
    pub const fn weight_type(self) -> DataType {
        self.weight_type
    }

    /// Whether the graph already owns the edge-weight property consumed by SSSP.
    pub const fn has_edge_weights(self) -> bool {
        self.has_edge_weights
    }
}

/// Owner-reported additional unit-weight allocation for one SSSP call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct SsspWorkspacePreflight {
    unit_weight_bytes: u64,
}

impl SsspWorkspacePreflight {
    /// Bytes allocated for call-local unit weights, or zero for a weighted graph.
    pub const fn unit_weight_bytes(self) -> u64 {
        self.unit_weight_bytes
    }
}

/// Query the native SSSP unit-weight allocation without accessing a CUDA device.
pub fn sssp_workspace_preflight(input: &SsspWorkspacePreflightInput) -> Result<SsspWorkspacePreflight> {
    let mut unit_weight_bytes = 0usize;
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    // SAFETY: the safe input validates the dtype and both output pointers are
    // writable for the duration of this allocation-free native query.
    let code = unsafe {
        cugraph_sssp_workspace_preflight(
            input.edge_count,
            input.weight_type.into(),
            to_bool_t(input.has_edge_weights),
            &mut unit_weight_bytes,
            &mut error,
        )
    };
    // SAFETY: consume the native error object exactly once on failure.
    unsafe { crate::error::check_host_error(code, error, SSSP_PREFLIGHT_OPERATION) }?;
    Ok(SsspWorkspacePreflight {
        unit_weight_bytes: u64::try_from(unit_weight_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, SSSP_PREFLIGHT_OPERATION)
                .with_message("native size_t SSSP unit-weight bound does not fit u64")
        })?,
    })
}

/// Parameters for the SSSP traversal algorithm.
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct SsspParams {
    /// Maximum edge-weight sum to consider. `None` means no cutoff.
    pub cutoff: Option<NonNegativeF64>,
    /// Whether predecessor vertices should be included in the result.
    pub compute_predecessors: bool,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}

impl SsspParams {
    /// Enable computation of predecessor vertices in the SSSP result.
    pub fn with_predecessors(mut self) -> Self {
        self.compute_predecessors = true;
        self
    }

    /// Set the maximum edge-weight sum to consider.
    pub fn with_cutoff(mut self, cutoff: NonNegativeF64) -> Self {
        self.cutoff = Some(cutoff);
        self
    }

    /// Enable expensive input validation checks in the C library.
    pub fn with_expensive_check(mut self) -> Self {
        self.do_expensive_check = true;
        self
    }

    /// The maximum edge-weight sum, or `None` for unlimited traversal.
    pub fn cutoff(&self) -> Option<NonNegativeF64> {
        self.cutoff
    }

    /// Whether predecessor vertices will be computed.
    pub fn compute_predecessors(&self) -> bool {
        self.compute_predecessors
    }

    /// Whether expensive input validation checks are enabled.
    pub fn do_expensive_check(&self) -> bool {
        self.do_expensive_check
    }
}

pub(crate) fn sssp<V>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    source: V,
    params: &SsspParams,
) -> Result<SsspResult>
where
    V: CugraphElement + TryInto<usize> + Copy + Debug,
{
    let source = source.try_into().map_err(|_| {
        Error::new(ErrorKind::InvalidInput, "sssp")
            .with_message("source vertex must be representable as a non-negative size_t")
            .with_context("source", format!("{source:?}"))
    })?;
    let cutoff = params.cutoff.map(f64::from).unwrap_or(f64::MAX);

    let mut result: *mut cugraph_paths_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "sssp")?;

    // SAFETY: `graph` stays valid for the duration of the call, `source` was
    // validated by `Graph::sssp`, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_sssp(
            handle.handle_ptr(),
            graph_ptr,
            source,
            cutoff,
            to_bool_t(params.compute_predecessors),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "sssp")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the paths result to
    // the caller.
    let paths = unsafe { PathsResult::from_raw(result, handle.device(), "sssp", handle.runtime_bound_lease()) }?;
    Ok(SsspResult { paths })
}

/// Result of an SSSP computation.
pub struct SsspResult {
    pub(crate) paths: PathsResult,
}

impl std::fmt::Debug for SsspResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsspResult").finish_non_exhaustive()
    }
}

impl SsspResult {
    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.paths.vertex_ids("SsspResult::vertex_ids")
    }

    /// Get the weighted SSSP distances as a runtime-typed device array view.
    ///
    /// Returns [`Distances::F32`] or [`Distances::F64`] depending on the
    /// graph's edge-weight type.
    pub fn distances(&self) -> Result<Distances<'_>> {
        self.paths.distances("SsspResult::distances")
    }

    /// Get the predecessor vertices as a runtime-typed device array view.
    ///
    /// Returns `None` if `compute_predecessors` was `false` in the SSSP call.
    pub fn predecessors(&self) -> Result<Option<VertexIds<'_>>> {
        self.paths.predecessors("SsspResult::predecessors")
    }

    /// Extract concrete row-major paths to the requested destination vertices.
    ///
    /// `sources` must contain the single source vertex originally used to
    /// compute this SSSP result. cuGraph requires this input again when
    /// extracting paths from the predecessor tree.
    pub fn extract_paths<'a, V>(
        &self,
        handle: &EnteredContext<'_>,
        graph: &mut Graph<Forward>,
        sources: impl Into<VertexSelection<'a, V>>,
        destinations: impl Into<VertexSelection<'a, V>>,
    ) -> Result<ExtractPathsResult>
    where
        V: CugraphElement + TryInto<i64> + Copy + Debug + 'a,
        <V as TryInto<i64>>::Error: Debug,
    {
        // The native C extract_paths implementation currently feeds the
        // paths_result distance column into extract_bfs_paths as `vertex_t`.
        // That is correct for BFS hop counts but not for SSSP float weights,
        // so SSSP reconstructs from predecessors instead.
        extract_sssp_paths_from_predecessors(handle, graph, &self.paths, sources, destinations)
    }

    /// Convert this SSSP result to a cudf DataFrame.
    ///
    /// Returns `vertex` and `distance` columns, plus a `predecessor` column
    /// when predecessors were computed.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::sssp_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("SsspResult::to_dataframe"))
    }
}

fn extract_sssp_paths_from_predecessors<'a, V>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    paths: &PathsResult,
    sources: impl Into<VertexSelection<'a, V>>,
    destinations: impl Into<VertexSelection<'a, V>>,
) -> Result<ExtractPathsResult>
where
    V: CugraphElement + TryInto<i64> + Copy + Debug + 'a,
    <V as TryInto<i64>>::Error: Debug,
{
    let operation = "SsspResult::extract_paths";
    let predecessors = paths.predecessors(operation)?.ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation).with_message(
            "paths result has no predecessor column; enable predecessor computation before extracting paths",
        )
    })?;

    let sources = sources.into();
    graph.validate_vertex_selection(&sources, operation, "sources", false)?;
    let sources = sources.resolve(handle)?;
    let sources_view = graph.resolved_vertex_view(&sources, operation, "sources")?;
    let source_values = source_values_to_i64(sources_view.to_vec(handle)?, operation)?;
    if source_values.len() != 1 {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("SSSP path extraction requires exactly one source vertex")
            .with_context("source_count", source_values.len().to_string()));
    }
    let source = source_values[0];

    let destinations = destinations.into();
    graph.validate_vertex_selection(&destinations, operation, "destinations", false)?;
    let destinations = destinations.resolve(handle)?;
    let destinations_view = graph.resolved_vertex_view(&destinations, operation, "destinations")?;
    let destination_values = source_values_to_i64(destinations_view.to_vec(handle)?, operation)?;

    let vertices = paths.vertex_ids(operation)?.to_vec_i64(handle)?;
    let predecessors = predecessors.to_vec_i64(handle)?;
    if vertices.len() != predecessors.len() {
        return Err(Error::new(ErrorKind::InvalidState, operation)
            .with_message("SSSP result returned mismatched vertex and predecessor lengths")
            .with_context("vertex_count", vertices.len().to_string())
            .with_context("predecessor_count", predecessors.len().to_string()));
    }

    let predecessor_by_vertex: HashMap<i64, i64> = vertices.into_iter().zip(predecessors).collect();
    let mut rows = Vec::with_capacity(destination_values.len());
    let mut max_path_length = 0usize;

    for destination in destination_values {
        let path = reconstruct_path(source, destination, &predecessor_by_vertex, operation)?;
        max_path_length = max_path_length.max(path.len());
        rows.push(path);
    }

    let max_path_length = max_path_length.max(1);
    match graph.vertex_dtype() {
        DataType::Int32 => {
            let flattened = flatten_paths_i32(&rows, max_path_length, operation)?;
            let paths = DeviceArray::from_slice(handle, &flattened)?;
            Ok(ExtractPathsResult::from_i32_paths(paths, max_path_length))
        },
        DataType::Int64 => {
            let flattened = flatten_paths_i64(&rows, max_path_length);
            let paths = DeviceArray::from_slice(handle, &flattened)?;
            Ok(ExtractPathsResult::from_i64_paths(paths, max_path_length))
        },
        other => Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("graph vertex dtype {other} is not supported for SSSP path extraction"))),
    }
}

fn source_values_to_i64<V>(values: Vec<V>, operation: &'static str) -> Result<Vec<i64>>
where
    V: TryInto<i64> + Copy + Debug,
    <V as TryInto<i64>>::Error: Debug,
{
    values
        .into_iter()
        .map(|value| {
            value.try_into().map_err(|error| {
                Error::new(ErrorKind::InvalidInput, operation)
                    .with_message("vertex value must be representable as i64")
                    .with_context("value", format!("{value:?}"))
                    .with_context("error", format!("{error:?}"))
            })
        })
        .collect()
}

fn reconstruct_path(
    source: i64,
    destination: i64,
    predecessor_by_vertex: &HashMap<i64, i64>,
    operation: &'static str,
) -> Result<Vec<i64>> {
    let mut reversed = vec![destination];
    let mut current = destination;

    for _ in 0..=predecessor_by_vertex.len() {
        if current == source {
            reversed.reverse();
            return Ok(reversed);
        }

        let predecessor = predecessor_by_vertex.get(&current).copied().ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, operation)
                .with_message("destination vertex is not present in the SSSP predecessor result")
                .with_context("destination", destination.to_string())
                .with_context("current", current.to_string())
        })?;

        if predecessor < 0 {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("destination vertex is unreachable from the SSSP source")
                .with_context("source", source.to_string())
                .with_context("destination", destination.to_string()));
        }

        current = predecessor;
        reversed.push(current);
    }

    Err(Error::new(ErrorKind::InvalidState, operation)
        .with_message("cycle detected while reconstructing SSSP predecessor path")
        .with_context("source", source.to_string())
        .with_context("destination", destination.to_string()))
}

fn flatten_paths_i64(rows: &[Vec<i64>], max_path_length: usize) -> Vec<i64> {
    let mut flattened = vec![-1_i64; rows.len() * max_path_length];
    for (row_index, path) in rows.iter().enumerate() {
        let start = row_index * max_path_length + (max_path_length - path.len());
        flattened[start..start + path.len()].copy_from_slice(path);
    }
    flattened
}

fn flatten_paths_i32(rows: &[Vec<i64>], max_path_length: usize, operation: &'static str) -> Result<Vec<i32>> {
    flatten_paths_i64(rows, max_path_length)
        .into_iter()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                Error::new(ErrorKind::InvalidState, operation)
                    .with_message("SSSP path vertex exceeded i32 range for an i32 graph")
                    .with_context("value", value.to_string())
            })
        })
        .collect()
}

#[cfg(test)]
mod preflight_tests {
    use super::*;

    #[test]
    fn input_preserves_weight_mode_and_dtype() {
        let input = SsspWorkspacePreflightInput::new(17, DataType::Float64, false).expect("valid SSSP shape");
        assert_eq!(input.edge_count(), 17);
        assert_eq!(input.weight_type(), DataType::Float64);
        assert!(!input.has_edge_weights());
    }

    #[test]
    fn input_rejects_non_float_weight_dtype() {
        let error = SsspWorkspacePreflightInput::new(1, DataType::Int32, false)
            .expect_err("SSSP preflight must reject an unsupported weight dtype");
        assert_eq!(error.kind(), ErrorKind::UnsupportedType);
        assert_eq!(error.operation(), "SsspWorkspacePreflightInput::new");
    }

    #[test]
    fn native_preflight_prices_only_unit_weights() {
        let float32 =
            sssp_workspace_preflight(&SsspWorkspacePreflightInput::new(3, DataType::Float32, false).unwrap()).unwrap();
        let float64 =
            sssp_workspace_preflight(&SsspWorkspacePreflightInput::new(3, DataType::Float64, false).unwrap()).unwrap();
        let weighted =
            sssp_workspace_preflight(&SsspWorkspacePreflightInput::new(3, DataType::Float64, true).unwrap()).unwrap();
        assert_eq!(float32.unit_weight_bytes(), 12);
        assert_eq!(float64.unit_weight_bytes(), 24);
        assert_eq!(weighted.unit_weight_bytes(), 0);
    }

    #[test]
    fn native_preflight_rejects_overflow() {
        let error =
            sssp_workspace_preflight(&SsspWorkspacePreflightInput::new(usize::MAX, DataType::Float64, false).unwrap())
                .expect_err("overflow must fail closed");
        assert_eq!(error.operation(), SSSP_PREFLIGHT_OPERATION);
    }
}
