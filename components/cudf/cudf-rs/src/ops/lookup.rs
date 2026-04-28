//! Dedicated lookup operators and reusable lookup-table mappings.
//!
//! Lookup is a row-aligned probe operation:
//! - input row order is preserved;
//! - missing keys become nulls;
//! - duplicate lookup keys are rejected;
//! - the output dtype comes from the lookup values.
//!
//! The current backend uses a distinct-hash build/probe/gather pipeline
//! internally, but that is an implementation detail rather than the public
//! contract.

use std::{
    fmt,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    Series,
    core::{
        column::{Column, ColumnView},
        error::{
            self, Error, ErrorCode, FACT_DISTINCT_KEYS_COUNT, FACT_KEYS_COUNT, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID,
            FACT_TYPE_ID, Result,
        },
        table::TableView,
        types::{DataType, TypeId},
    },
    ops::{copying, stream_compaction},
    runtime::{
        BoundaryKind, BoundaryObservation, ExecutionPlacement, Runtime, execution_context::ExecutionOverrides,
        execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
        record_boundary_observation_in_domain,
    },
};

struct ProbeMetadata {
    input_type_id: TypeId,
    keys_type_id: TypeId,
    value_type_id: TypeId,
    values_dtype: DataType,
    input_count: usize,
    keys_count: usize,
    values_count: usize,
}

impl ProbeMetadata {
    fn collect(
        input: &ColumnView<'_>,
        keys: &ColumnView<'_>,
        values: &ColumnView<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let input_dtype = input.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
        let keys_dtype = keys.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
        let values_dtype = values.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
        let input_count = input.len().map_err(|e| e.reanchor_if_needed(operation))?;
        let keys_count = keys.len().map_err(|e| e.reanchor_if_needed(operation))?;
        let values_count = values.len().map_err(|e| e.reanchor_if_needed(operation))?;

        Ok(Self {
            input_type_id: input_dtype.id(),
            keys_type_id: keys_dtype.id(),
            value_type_id: values_dtype.id(),
            values_dtype,
            input_count,
            keys_count,
            values_count,
        })
    }

    fn decorate(&self, err: Error, operation: &'static str) -> Error {
        let err = err.reanchor_if_needed(operation);
        let err = err.with_fact_if_absent(FACT_LEFT_TYPE_ID, self.input_type_id);
        let err = err.with_fact_if_absent(FACT_RIGHT_TYPE_ID, self.keys_type_id);
        let err = err.with_fact_if_absent(FACT_TYPE_ID, self.value_type_id);
        let err = err.with_fact_if_absent(FACT_KEYS_COUNT, self.keys_count);
        let err = err.with_detail_if_absent("input_count", self.input_count.to_string());
        err.with_detail_if_absent("values_count", self.values_count.to_string())
    }

    fn length_mismatch(&self, operation: &'static str, message: impl Into<String>) -> Error {
        self.decorate(Error::new_with(ErrorCode::ColumnLengthMismatch, operation, message), operation)
    }

    fn type_mismatch(&self, operation: &'static str, message: impl Into<String>) -> Error {
        self.decorate(Error::new_with(ErrorCode::TypeMismatch, operation, message), operation)
    }
}

fn validate_probe_inputs(
    input: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    values: &ColumnView<'_>,
    operation: &'static str,
) -> Result<ProbeMetadata> {
    let meta = ProbeMetadata::collect(input, keys, values, operation)?;

    if meta.input_type_id != meta.keys_type_id {
        return Err(meta.type_mismatch(operation, "input and lookup keys must have the same type"));
    }

    if meta.keys_count != meta.values_count {
        return Err(meta.length_mismatch(operation, "lookup keys and values must have the same length"));
    }

    Ok(meta)
}

fn ensure_unique_keys(keys: &ColumnView<'_>, key_count: usize, operation: &'static str) -> Result<()> {
    let key_table = TableView::from_column_views(&[keys]).map_err(|e| e.reanchor_if_needed(operation))?;
    let distinct = stream_compaction::distinct(
        &key_table,
        &[0],
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .map_err(|e| e.reanchor_if_needed(operation))?;
    let distinct_count = distinct
        .view()
        .map_err(|e| e.reanchor_if_needed(operation))?
        .num_rows()
        .map_err(|e| e.reanchor_if_needed(operation))?;

    if distinct_count != key_count {
        return Err(Error::new_with(ErrorCode::DuplicateKey, operation, "lookup keys must be unique")
            .with_fact(FACT_KEYS_COUNT, key_count)
            .with_fact(FACT_DISTINCT_KEYS_COUNT, distinct_count));
    }

    Ok(())
}

#[derive(Debug)]
struct DistinctHashJoinHandle {
    handle: NonNull<c_void>,
}

// SAFETY: libcudf distinct_hash_join is documented as probe-safe for repeated
// calls and possible parallel use. The handle is process-agnostic and not tied
// to the constructing thread.
unsafe impl Send for DistinctHashJoinHandle {}
unsafe impl Sync for DistinctHashJoinHandle {}

impl DistinctHashJoinHandle {
    fn new(build: &TableView<'_>, domain: &Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let exec_options = domain.explicit_exec_options_raw();
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_distinct_hash_join_create_ex(
                    build.raw_handle(),
                    cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
                    &mut handle,
                    exec_options.as_ref().map_or(ptr::null(), |options| options as *const _),
                )
            },
            operation,
            [("execution_context_provided", exec_options.is_some().to_string())],
        )?;
        Ok(Self { handle: error::require_non_null(handle, operation, "distinct hash join")? })
    }

    fn left_join_build_indices(
        &self,
        probe: &TableView<'_>,
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Column> {
        let exec_options = output_domain.explicit_exec_options_raw();
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_distinct_hash_join_left_join_ex(
                    self.handle.as_ptr(),
                    probe.raw_handle(),
                    &mut handle,
                    exec_options.as_ref().map_or(ptr::null(), |options| options as *const _),
                )
            },
            operation,
            [("execution_context_provided", exec_options.is_some().to_string())],
        )?;
        unsafe { Column::from_raw_with_domain(handle, Arc::clone(output_domain)) }
    }
}

impl Drop for DistinctHashJoinHandle {
    fn drop(&mut self) {
        unsafe { cudf_sys::cudf_distinct_hash_join_destroy(self.handle.as_ptr()) };
    }
}

fn execute_probe(
    input: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    values: &ColumnView<'_>,
    validate_unique_keys: bool,
    operation: &'static str,
) -> Result<Column> {
    let meta = validate_probe_inputs(input, keys, values, operation)?;
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::many(
        input.execution_domain(),
        [keys.execution_domain(), values.execution_domain()],
        &ctx,
        operation,
    )
    .map_err(|e| meta.decorate(e, operation))?;

    if validate_unique_keys {
        ensure_unique_keys(keys, meta.keys_count, operation).map_err(|e| meta.decorate(e, operation))?;
    }

    if meta.input_count == 0 {
        return Column::empty_with_dtype_in_domain(meta.values_dtype, Arc::clone(execution.output_domain()), operation)
            .map_err(|e| meta.decorate(e, operation));
    }

    let keys_table = TableView::from_column_views(&[keys]).map_err(|e| meta.decorate(e, operation))?;
    let backend = DistinctHashJoinHandle::new(&keys_table, execution.output_domain(), operation)
        .map_err(|e| meta.decorate(e, operation))?;
    execute_reusable_probe(
        input,
        values,
        &backend,
        &meta,
        execution.output_domain(),
        Some(execution.exec_options()),
        operation,
    )
}

fn execute_reusable_probe(
    input: &ColumnView<'_>,
    values: &ColumnView<'_>,
    backend: &DistinctHashJoinHandle,
    meta: &ProbeMetadata,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    if meta.input_count == 0 {
        return Column::empty_with_dtype_in_domain(meta.values_dtype, Arc::clone(output_domain), operation)
            .map_err(|e| meta.decorate(e, operation));
    }

    let input_table = TableView::from_column_views(&[input]).map_err(|e| meta.decorate(e, operation))?;
    let build_indices = backend
        .left_join_build_indices(&input_table, output_domain, operation)
        .map_err(|e| meta.decorate(e, operation))?;
    let build_indices_view = build_indices.view().map_err(|e| meta.decorate(e, operation))?;
    let values_table = TableView::from_column_views(&[values]).map_err(|e| meta.decorate(e, operation))?;
    let gathered = copying::gather_impl(
        &values_table,
        &build_indices_view,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
        "nullify",
        output_domain,
        exec_options,
        operation,
    )
    .map_err(|e| meta.decorate(e, operation))?;
    let mut columns = gathered.into_columns().map_err(|e| meta.decorate(e, operation))?;
    Ok(columns.remove(0))
}

fn mapping_length_error(keys: &Series, values: &Series, operation: &'static str) -> Error {
    let mut err =
        Error::new_with(ErrorCode::ColumnLengthMismatch, operation, "lookup keys and values must have the same length")
            .with_fact(FACT_KEYS_COUNT, keys.len())
            .with_detail("values_count", values.len().to_string());

    if let Ok(dtype) = keys.dtype() {
        err = err.with_fact(FACT_RIGHT_TYPE_ID, dtype.id());
    } else {
        err = err.with_detail("keys_type_id", "?");
    }

    if let Ok(dtype) = values.dtype() {
        err = err.with_fact(FACT_TYPE_ID, dtype.id());
    } else {
        err = err.with_detail("values_type_id", "?");
    }

    err
}

fn decorate_mapping_domain_error(err: Error, keys: &Series, values: &Series, operation: &'static str) -> Error {
    err.reanchor_if_needed(operation)
        .with_fact_if_absent(FACT_KEYS_COUNT, keys.len())
        .with_detail_if_absent("values_count", values.len().to_string())
}

fn require_mapping_same_provenance(keys: &Series, values: &Series, operation: &'static str) -> Result<()> {
    ExecutionDomain::require_same_device_for_mixed_inputs(
        keys.as_column().execution_domain(),
        values.as_column().execution_domain(),
        operation,
    )
    .map_err(|e| decorate_mapping_domain_error(e, keys, values, operation))?;

    ExecutionDomain::require_same_provenance(
        keys.as_column().execution_domain(),
        values.as_column().execution_domain(),
        operation,
        "lookup keys and values must share one retained provenance",
        "use LookupTable::new_in_runtime(...) to normalize reusable lookup state into one Runtime before building it",
    )
    .map_err(|e| decorate_mapping_domain_error(e, keys, values, operation))
}

fn clone_or_copy_series_into_domain(
    series: &Series,
    output_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Series> {
    if ExecutionDomain::same_provenance(series.as_column().execution_domain(), output_domain) {
        return Ok(series.clone());
    }

    ExecutionDomain::insert_stream_dependency(series.as_column().execution_domain(), output_domain, operation)
        .map_err(|e| e.with_detail("target_domain", output_domain.debug_label()))?;
    let column = series
        .as_column()
        .copy_into_domain(Arc::clone(output_domain), operation)
        .map_err(|e| e.with_detail("target_domain", output_domain.debug_label()))?;
    Series::from_column(column).map_err(|e| e.reanchor_if_needed(operation))
}

/// Probe a key/value mapping with a borrowed input column.
///
/// The result is row-aligned with `input`, preserves input order, and returns
/// nulls for missing keys. Duplicate lookup keys are rejected.
pub fn probe(input: &ColumnView<'_>, keys: &ColumnView<'_>, values: &ColumnView<'_>) -> Result<Column> {
    execute_probe(input, keys, values, true, "lookup::probe")
}

pub(crate) fn probe_prevalidated(
    input: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    values: &ColumnView<'_>,
    operation: &'static str,
) -> Result<Column> {
    execute_probe(input, keys, values, false, operation)
}

/// An owned reusable lookup mapping for repeated probes.
#[derive(Clone)]
pub struct LookupTable {
    backend: Arc<DistinctHashJoinHandle>,
    keys: Series,
    values: Series,
    domain: Arc<ExecutionDomain>,
}

impl fmt::Debug for LookupTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookupTable")
            .field("keys", &self.keys)
            .field("values", &self.values)
            .field("placement", &self.placement())
            .finish()
    }
}

impl LookupTable {
    /// Build a reusable lookup table from owned key/value series.
    ///
    /// Construction validates equal lengths, rejects duplicate keys, and
    /// rejects key/value pairs that do not already share one retained
    /// provenance. Use [`LookupTable::new_in_runtime`] when the mapping parts
    /// should be normalized into a specific runtime before building reusable
    /// GPU state.
    pub fn new(keys: Series, values: Series) -> Result<Self> {
        let operation = "lookup_table::new";
        Self::build_from_placed(keys, values, operation)
    }

    /// Build a reusable lookup table inside `runtime`.
    ///
    /// Keys and values already placed in `runtime` are shared. Other same-device
    /// inputs are fenced and deep-copied into `runtime` before the reusable
    /// backend is built. Cross-device inputs fail; relocate them explicitly
    /// before construction.
    pub fn new_in_runtime(runtime: &Runtime, keys: Series, values: Series) -> Result<Self> {
        let operation = "lookup_table::new_in_runtime";
        if keys.len() != values.len() {
            return Err(mapping_length_error(&keys, &values, operation));
        }

        let keys = clone_or_copy_series_into_domain(&keys, runtime.execution_domain(), operation)?;
        let values = clone_or_copy_series_into_domain(&values, runtime.execution_domain(), operation)?;
        Self::build_from_placed(keys, values, operation)
    }

    fn build_from_placed(keys: Series, values: Series, operation: &'static str) -> Result<Self> {
        if keys.len() != values.len() {
            return Err(mapping_length_error(&keys, &values, operation));
        }

        require_mapping_same_provenance(&keys, &values, operation)?;
        let domain = Arc::clone(keys.as_column().execution_domain());

        let backend = {
            let keys_view = keys.view().map_err(|e| e.reanchor_if_needed(operation))?;
            ensure_unique_keys(&keys_view, keys.len(), operation)?;
            let build_table =
                TableView::from_column_views(&[&keys_view]).map_err(|e| e.reanchor_if_needed(operation))?;
            DistinctHashJoinHandle::new(&build_table, keys.as_column().execution_domain(), operation)
                .map_err(|e| e.reanchor_if_needed(operation))?
        };

        Ok(Self { backend: Arc::new(backend), keys, values, domain })
    }

    /// Copy this reusable lookup state into `runtime`.
    ///
    /// If the table already has the target runtime's retained provenance, this
    /// returns a cheap clone. Otherwise keys and values are deep-copied into the
    /// target runtime and the reusable backend is rebuilt there.
    pub fn copy_into_runtime(&self, runtime: &Runtime) -> Result<Self> {
        let operation = "lookup_table::copy_into_runtime";
        if ExecutionDomain::same_provenance(&self.domain, runtime.execution_domain()) {
            return Ok(self.clone());
        }

        let keys = clone_or_copy_series_into_domain(&self.keys, runtime.execution_domain(), operation)?;
        let values = clone_or_copy_series_into_domain(&self.values, runtime.execution_domain(), operation)?;
        let table = Self::build_from_placed(keys, values, operation)?;
        record_boundary_observation_in_domain(
            runtime.execution_domain(),
            BoundaryObservation {
                kind: BoundaryKind::DeviceToDeviceCopy,
                placement: runtime.placement(),
                rows: self.len(),
                columns: 2,
                synchronized_for_host: false,
            },
        );
        Ok(table)
    }

    pub fn keys(&self) -> &Series {
        &self.keys
    }

    pub fn values(&self) -> &Series {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Return a diagnostic summary of where this reusable GPU lookup state is retained.
    pub fn placement(&self) -> ExecutionPlacement {
        self.domain.placement()
    }

    pub fn probe(&self, input: &Series) -> Result<Series> {
        let operation = "lookup_table::probe";
        let ctx = ExecutionOverrides::new();
        self.probe_with_context(input, &ctx, operation)
    }

    /// Probe this reusable lookup table with a borrowed execution override.
    ///
    /// The output keeps the input workflow identity and uses the concrete
    /// stream / memory-resource bundle resolved from `ctx`. The reusable table
    /// state and the input are ordered into that output stream before libcudf
    /// work is submitted.
    pub fn probe_in(&self, input: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.probe_with_context(input, ctx, "lookup_table::probe_in")
    }

    fn probe_with_context(
        &self,
        input: &Series,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let input_view = input.view().map_err(|e| e.reanchor_if_needed(operation))?;
        let keys_view = self.keys.view().map_err(|e| e.reanchor_if_needed(operation))?;
        let values_view = self.values.view().map_err(|e| e.reanchor_if_needed(operation))?;
        let meta = validate_probe_inputs(&input_view, &keys_view, &values_view, operation)?;
        let execution = ResolvedExecution::many(
            input_view.execution_domain(),
            [keys_view.execution_domain(), values_view.execution_domain()],
            ctx,
            operation,
        )
        .map_err(|e| meta.decorate(e, operation))?;
        let result = execute_reusable_probe(
            &input_view,
            &values_view,
            self.backend.as_ref(),
            &meta,
            execution.output_domain(),
            Some(execution.exec_options()),
            operation,
        )
        .map_err(|e| e.reanchor_if_needed(operation))?;
        Series::from_column(result).map_err(|e| e.reanchor_if_needed(operation))
    }
}
