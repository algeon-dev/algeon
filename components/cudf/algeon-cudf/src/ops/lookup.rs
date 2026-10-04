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
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, FACT_KEYS_COUNT, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_TYPE_ID, Result},
        native_teardown::drop_device_owned_handle,
        table::TableView,
        types::{DataType, TypeId},
    },
    ops::copying,
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
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

struct DistinctHashJoinHandle {
    handle: NonNull<cudf_sys::cudf_distinct_hash_join_handle_state>,
    _domain: Arc<ExecutionDomain>,
}

// SAFETY: libcudf distinct_hash_join is documented as probe-safe for repeated
// calls and possible parallel use. The handle is process-agnostic and not tied
// to the constructing thread.
unsafe impl Send for DistinctHashJoinHandle {}
unsafe impl Sync for DistinctHashJoinHandle {}

impl DistinctHashJoinHandle {
    fn new(build: &TableView<'_>, domain: &Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let exec_options = domain.enter_execution()?;
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_distinct_hash_join_create_ex(
                    build.raw_handle(),
                    cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                    &mut handle,
                    exec_options.options(),
                )
            },
            operation,
            [("execution_authority", "retained_domain".to_string())],
        )?;
        Ok(Self {
            handle: error::require_non_null(handle, operation, "distinct hash join")?,
            _domain: Arc::clone(domain),
        })
    }

    fn left_join_build_indices(
        &self,
        probe: &TableView<'_>,
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Column> {
        let exec_options = output_domain.enter_execution()?;
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_distinct_hash_join_left_join_ex(
                    self.handle.as_ptr(),
                    probe.raw_handle(),
                    &mut handle,
                    exec_options.options(),
                )
            },
            operation,
            [("execution_authority", "retained_domain".to_string())],
        )?;
        unsafe { Column::from_raw_with_domain(handle, Arc::clone(output_domain)) }
    }
}

impl Drop for DistinctHashJoinHandle {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self._domain.device_id(),
            "DistinctHashJoinHandle::drop",
            |owned| unsafe { cudf_sys::cudf_distinct_hash_join_destroy(owned) },
        );
    }
}

fn execute_probe_with_context(
    input: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    values: &ColumnView<'_>,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<Column> {
    let meta = validate_probe_inputs(input, keys, values, operation)?;
    let execution = ResolvedExecution::many(
        input.execution_domain(),
        [keys.execution_domain(), values.execution_domain()],
        ctx,
        operation,
    )
    .map_err(|e| meta.decorate(e, operation))?;

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
        execution.exec_options(),
        operation,
    )
}

fn execute_reusable_probe(
    input: &ColumnView<'_>,
    values: &ColumnView<'_>,
    backend: &DistinctHashJoinHandle,
    meta: &ProbeMetadata,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
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
        cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
        "nullify",
        output_domain,
        exec_options,
        operation,
    )
    .map_err(|e| meta.decorate(e, operation))?;
    let mut columns = gathered.into_columns().map_err(|e| meta.decorate(e, operation))?;
    Ok(columns.remove(0))
}

pub(crate) fn probe_prevalidated(
    input: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    values: &ColumnView<'_>,
    operation: &'static str,
) -> Result<Column> {
    let ctx = Exec::RETAINED;
    execute_probe_with_context(input, keys, values, ctx, operation)
}

pub(crate) fn probe_prevalidated_in(
    input: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    values: &ColumnView<'_>,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<Column> {
    execute_probe_with_context(input, keys, values, ctx, operation)
}
