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
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    Series,
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorKind, Result},
        table::TableView,
        types::{DataType, TypeId},
    },
    ops::{copying, stream_compaction},
    runtime::execution_domain::ExecutionDomain,
};

fn type_label(type_id: TypeId) -> String {
    format!("{:?}", type_id).to_lowercase()
}

struct ProbeMetadata {
    input_type_id: TypeId,
    keys_type_id: TypeId,
    values_type_id: TypeId,
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
        let input_dtype = input.data_type().map_err(|e| e.with_operation(operation))?;
        let keys_dtype = keys.data_type().map_err(|e| e.with_operation(operation))?;
        let values_dtype = values.data_type().map_err(|e| e.with_operation(operation))?;
        let input_count = input.len().map_err(|e| e.with_operation(operation))?;
        let keys_count = keys.len().map_err(|e| e.with_operation(operation))?;
        let values_count = values.len().map_err(|e| e.with_operation(operation))?;

        Ok(Self {
            input_type_id: input_dtype.id(),
            keys_type_id: keys_dtype.id(),
            values_type_id: values_dtype.id(),
            values_dtype,
            input_count,
            keys_count,
            values_count,
        })
    }

    fn decorate(&self, err: Error, operation: &'static str) -> Error {
        err.with_operation(operation)
            .with_context("input_type_id", type_label(self.input_type_id))
            .with_context("keys_type_id", type_label(self.keys_type_id))
            .with_context("values_type_id", type_label(self.values_type_id))
            .with_context("input_count", self.input_count.to_string())
            .with_context("keys_count", self.keys_count.to_string())
            .with_context("values_count", self.values_count.to_string())
    }

    fn invalid(&self, operation: &'static str, message: impl Into<String>) -> Error {
        self.decorate(Error::new(ErrorKind::InvalidOperation, message.into()), operation)
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
        return Err(meta.invalid(operation, "input and lookup keys must have the same type"));
    }

    if meta.keys_count != meta.values_count {
        return Err(meta.invalid(operation, "lookup keys and values must have the same length"));
    }

    ExecutionDomain::require_same_device_for_mixed_inputs(input.execution_domain(), keys.execution_domain(), operation)
        .map_err(|e| meta.decorate(e, operation))?;
    ExecutionDomain::require_same_device_for_mixed_inputs(
        input.execution_domain(),
        values.execution_domain(),
        operation,
    )
    .map_err(|e| meta.decorate(e, operation))?;

    if !ExecutionDomain::same_provenance(keys.execution_domain(), input.execution_domain()) {
        keys.execution_domain().fence_stream(operation).map_err(|e| meta.decorate(e, operation))?;
    }
    if !ExecutionDomain::same_provenance(values.execution_domain(), input.execution_domain()) {
        values.execution_domain().fence_stream(operation).map_err(|e| meta.decorate(e, operation))?;
    }

    Ok(meta)
}

fn ensure_unique_keys(keys: &ColumnView<'_>, key_count: usize, operation: &'static str) -> Result<()> {
    let key_table = TableView::from_column_views(&[keys]).map_err(|e| e.with_operation(operation))?;
    let distinct = stream_compaction::distinct(
        &key_table,
        &[0],
        stream_compaction::DuplicateKeepOption::KeepAny,
        stream_compaction::NullEquality::Equal,
        stream_compaction::NanEquality::AllEqual,
    )
    .map_err(|e| e.with_operation(operation))?;
    let distinct_count = distinct
        .view()
        .map_err(|e| e.with_operation(operation))?
        .num_rows()
        .map_err(|e| e.with_operation(operation))?;

    if distinct_count != key_count {
        return Err(Error::new(ErrorKind::InvalidOperation, "lookup keys must be unique")
            .with_operation(operation)
            .with_context("keys_count", key_count.to_string())
            .with_context("distinct_keys_count", distinct_count.to_string()));
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
        error::check_status_in_with(
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
        Ok(Self { handle: NonNull::new(handle).expect("null distinct-hash-join handle on success") })
    }

    fn left_join_build_indices(
        &self,
        probe: &TableView<'_>,
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Column> {
        let exec_options = output_domain.explicit_exec_options_raw();
        let mut handle = ptr::null_mut();
        error::check_status_in_with(
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
        Ok(unsafe { Column::from_raw_with_domain(handle, Arc::clone(output_domain)) })
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

    if validate_unique_keys {
        ensure_unique_keys(keys, meta.keys_count, operation).map_err(|e| meta.decorate(e, operation))?;
    }

    if meta.input_count == 0 {
        return Column::empty_with_dtype_in_domain(meta.values_dtype, Arc::clone(input.execution_domain()), operation)
            .map_err(|e| meta.decorate(e, operation));
    }

    let keys_table = TableView::from_column_views(&[keys]).map_err(|e| meta.decorate(e, operation))?;
    let backend = DistinctHashJoinHandle::new(&keys_table, input.execution_domain(), operation)
        .map_err(|e| meta.decorate(e, operation))?;
    execute_reusable_probe(input, values, &backend, &meta, operation)
}

fn execute_reusable_probe(
    input: &ColumnView<'_>,
    values: &ColumnView<'_>,
    backend: &DistinctHashJoinHandle,
    meta: &ProbeMetadata,
    operation: &'static str,
) -> Result<Column> {
    if meta.input_count == 0 {
        return Column::empty_with_dtype_in_domain(meta.values_dtype, Arc::clone(input.execution_domain()), operation)
            .map_err(|e| meta.decorate(e, operation));
    }

    let input_table = TableView::from_column_views(&[input]).map_err(|e| meta.decorate(e, operation))?;
    let build_indices = backend
        .left_join_build_indices(&input_table, input.execution_domain(), operation)
        .map_err(|e| meta.decorate(e, operation))?;
    let build_indices_view = build_indices.view().map_err(|e| meta.decorate(e, operation))?;
    let values_table = TableView::from_column_views(&[values]).map_err(|e| meta.decorate(e, operation))?;
    let exec_options = input.execution_domain().explicit_exec_options_raw();
    let gathered = copying::gather_impl(
        &values_table,
        &build_indices_view,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
        "nullify",
        input.execution_domain(),
        exec_options.as_ref(),
        operation,
    )
    .map_err(|e| meta.decorate(e, operation))?;
    let mut columns = gathered.into_columns().map_err(|e| meta.decorate(e, operation))?;
    Ok(columns.remove(0))
}

fn mapping_length_error(keys: &Series, values: &Series, operation: &'static str) -> Error {
    let keys_type_id = keys.dtype().map(|dtype| type_label(dtype.id())).unwrap_or_else(|_| "?".to_string());
    let values_type_id = values.dtype().map(|dtype| type_label(dtype.id())).unwrap_or_else(|_| "?".to_string());

    Error::new(ErrorKind::InvalidOperation, "lookup keys and values must have the same length")
        .with_operation(operation)
        .with_context("keys_type_id", keys_type_id)
        .with_context("values_type_id", values_type_id)
        .with_context("keys_count", keys.len().to_string())
        .with_context("values_count", values.len().to_string())
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
#[derive(Debug, Clone)]
pub struct LookupTable {
    backend: Arc<DistinctHashJoinHandle>,
    keys: Series,
    values: Series,
}

impl LookupTable {
    /// Build a reusable lookup table from owned key/value series.
    ///
    /// Construction validates equal lengths, rejects duplicate keys, and
    /// rejects cross-device key/value pairs.
    pub fn new(keys: Series, values: Series) -> Result<Self> {
        let operation = "lookup_table::new";

        if keys.len() != values.len() {
            return Err(mapping_length_error(&keys, &values, operation));
        }

        ExecutionDomain::require_same_device_for_mixed_inputs(
            keys.as_column().execution_domain(),
            values.as_column().execution_domain(),
            operation,
        )
        .map_err(|e| {
            e.with_context("keys_count", keys.len().to_string()).with_context("values_count", values.len().to_string())
        })?;

        let backend = {
            let keys_view = keys.view().map_err(|e| e.with_operation(operation))?;
            ensure_unique_keys(&keys_view, keys.len(), operation)?;
            let build_table = TableView::from_column_views(&[&keys_view]).map_err(|e| e.with_operation(operation))?;
            DistinctHashJoinHandle::new(&build_table, keys.as_column().execution_domain(), operation)
                .map_err(|e| e.with_operation(operation))?
        };

        Ok(Self { backend: Arc::new(backend), keys, values })
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

    pub fn probe(&self, input: &Series) -> Result<Series> {
        let operation = "lookup_table::probe";
        let input_view = input.view().map_err(|e| e.with_operation(operation))?;
        let keys_view = self.keys.view().map_err(|e| e.with_operation(operation))?;
        let values_view = self.values.view().map_err(|e| e.with_operation(operation))?;
        let meta = validate_probe_inputs(&input_view, &keys_view, &values_view, operation)?;
        let result = execute_reusable_probe(&input_view, &values_view, self.backend.as_ref(), &meta, operation)
            .map_err(|e| e.with_operation(operation))?;
        let result = Series::from_column(result).map_err(|e| e.with_operation(operation))?;

        Ok(match self.values.name() {
            Some(name) => result.rename(name),
            None => result,
        })
    }

    pub fn probe_as(&self, input: &Series, output_name: &str) -> Result<Series> {
        self.probe(input).map(|result| result.rename(output_name))
    }
}
