//! List schema and offset validators for numeric export proofs.
use std::{cell::Cell, sync::Arc};

use arrow::datatypes::FieldRef;
use rapids_interop::{
    BufferId, ColumnId, ColumnLayout, PhysicalValidity, PreparedAccess, PreparedColumnExport, ProducerReadiness,
};

use super::{
    super::DataFrame,
    list_export::{FixedSizeListStructuralProof, ListExportContract, ListExportParts, UniformListProof},
};
use crate::{
    core::{
        Column,
        column::ColumnView,
        error::{
            self, Error, ErrorCode, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_VALIDATION_REASON,
            Result,
        },
        table::TableView,
        types::{DataType, TypeId},
    },
    interop::arrow_host::{cudf_dtype_can_export_to_arrow_type, cudf_dtype_from_arrow_type},
    ops::{
        binary::{self, BinaryOperator},
        filter,
        reduction::{self, ReduceAgg},
        unary,
    },
    runtime::{
        BoundaryKind, BoundaryObservation, QueryMemoryObservation, matrix_element_bytes,
        record_boundary_observation_in_domain,
    },
};

thread_local! {
    static FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn validate_list_schema_contract(
    view: &ColumnView<'_>,
    child_field: &FieldRef,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<()> {
    let child_view = view.list_sliced_child().map_err(|error| error.reanchor_if_needed(operation))?;
    let actual_child_dtype = child_view.data_type().map_err(|error| error.reanchor_if_needed(operation))?;
    let expected_child_dtype =
        cudf_dtype_from_arrow_type(child_field.data_type()).map_err(|error| error.reanchor_if_needed(operation))?;
    if !cudf_dtype_can_export_to_arrow_type(actual_child_dtype, child_field.data_type()) {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            format!(
                "Arrow schema field '{field_name}' requires list child {:?} but the DataFrame list child is {:?}",
                child_field.data_type(),
                actual_child_dtype.id()
            ),
        )
        .with_fact(FACT_LEFT_TYPE_ID, actual_child_dtype.id())
        .with_fact(FACT_RIGHT_TYPE_ID, expected_child_dtype.id())
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("actual_child_cudf_type", actual_child_dtype.id().name())
        .with_detail("expected_child_arrow_type", format!("{:?}", child_field.data_type()))
        .with_detail("reason", "schema_list_child_dtype_mismatch"));
    }
    let child_null_count = child_view.null_count().map_err(|error| error.reanchor_if_needed(operation))?;
    if !child_field.is_nullable() && child_null_count > 0 {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            operation,
            format!(
                "Arrow schema field '{field_name}' declares a non-nullable list child but the DataFrame child contains nulls"
            ),
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("child_null_count", child_null_count.to_string())
        .with_detail("reason", "schema_list_child_nullability_mismatch"));
    }

    Ok(())
}

pub(super) fn validate_fixed_size_list_offsets(
    view: &ColumnView<'_>,
    producer: &Arc<Column>,
    dimension: i32,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<FixedSizeListStructuralProof> {
    let expected_dimension = fixed_size_list_expected_dimension(dimension, field_name, field_index, operation)?;
    let element = view
        .list_sliced_child()
        .map_err(|error| error.reanchor(operation))?
        .data_type()
        .map_err(|error| error.reanchor(operation))?
        .id();

    let (rows, child_offset, child_len, expected_child_len) =
        fixed_size_list_active_child_span(view, expected_dimension, field_name, field_index, operation)?;
    if rows == 0 && child_len != expected_child_len {
        return Err(error::foreign_invariant_violation(
            operation,
            "empty FixedSizeList active child range must also be empty",
            "list_child",
            "fixed_size_list_active_child_len_mismatch",
            "ensure the cudf list active child range matches the fixed-size schema",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("row_count", rows.to_string())
        .with_detail("dimension", expected_dimension.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("actual_child_len", child_len.to_string()));
    }
    if rows == 0 {
        return Ok(record_fixed_size_list_structural_proof(
            field_index,
            dimension,
            element,
            rows,
            child_offset,
            child_len,
            producer,
            None,
        ));
    }
    let widths = fixed_size_list_row_widths(view, rows, field_name, field_index, operation)?;
    let (minimum, maximum, observation) =
        fixed_size_list_width_bounds(&widths, view, producer, rows, dimension, field_name, field_index, operation)?;
    if minimum != dimension || maximum != dimension {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            "cudf list row widths do not match the Arrow FixedSizeList dimension",
        )
        .with_fact(FACT_EXPECTED_ROWS, expected_dimension)
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("expected_dimension", expected_dimension.to_string())
        .with_detail("minimum_actual_dimension", minimum.to_string())
        .with_detail("maximum_actual_dimension", maximum.to_string())
        .with_detail("evidence", "device_reduced_list_offset_width_range")
        .with_detail("reason", "schema_fixed_size_list_dimension_mismatch"));
    }

    Ok(record_fixed_size_list_structural_proof(
        field_index,
        dimension,
        element,
        rows,
        child_offset,
        child_len,
        producer,
        Some(observation),
    ))
}

fn fixed_size_list_expected_dimension(
    dimension: i32,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<usize> {
    let expected_dimension = usize::try_from(dimension).map_err(|_| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "Arrow FixedSizeList dimension must be positive")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("dimension", dimension.to_string())
            .with_detail("reason", "non_positive_fixed_size_list_dimension")
    })?;
    if expected_dimension == 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "Arrow FixedSizeList dimension must be positive",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("dimension", "0")
        .with_detail("reason", "non_positive_fixed_size_list_dimension"));
    }
    Ok(expected_dimension)
}

fn fixed_size_list_active_child_span(
    view: &ColumnView<'_>,
    expected_dimension: usize,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<(usize, usize, usize, usize)> {
    let rows = view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_view = view.list_sliced_child().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset_raw = child_view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset = usize::try_from(child_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "FixedSizeList child offset must be non-negative",
            "list_child",
            "negative_fixed_size_list_child_offset",
            "ensure the cudf list column exposes a valid active child range",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("child_offset", child_offset_raw.to_string())
    })?;
    let child_len = child_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    let expected_child_len = rows.checked_mul(expected_dimension).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList active child length overflowed usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("dimension", expected_dimension.to_string())
            .with_detail("reason", "fixed_size_list_child_len_overflow")
    })?;
    Ok((rows, child_offset, child_len, expected_child_len))
}

fn fixed_size_list_row_widths(
    view: &ColumnView<'_>,
    rows: usize,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<Column> {
    let root_offset_raw = view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let root_offset = usize::try_from(root_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "cudf list root offset must be non-negative",
            "list_offsets",
            "negative_list_root_offset",
            "ensure the cudf list column exposes a valid active range",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("root_offset", root_offset_raw.to_string())
    })?;
    let root_end = root_offset.checked_add(rows).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList active offset range overflowed usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("reason", "fixed_size_list_offset_range_overflow")
    })?;
    let offsets_view = view.child(0).map_err(|error| error.reanchor_if_needed(operation))?;
    let offsets_len = offsets_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    if offsets_len <= root_end {
        return Err(error::foreign_invariant_violation(
            operation,
            "cudf list offsets are too short for the FixedSizeList active range",
            "list_offsets",
            "fixed_size_list_offsets_too_short",
            "ensure the cudf list column exposes one more offset than active rows",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("required_offset_index", root_end.to_string())
        .with_detail("offset_count", offsets_len.to_string()));
    }

    let rows_i32 = i32::try_from(rows).map_err(|_| {
        error::range_limit_error(operation, "row_count", "i32::MAX", rows)
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
    })?;
    let end_offset = root_offset_raw.checked_add(1).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList end-offset slice start overflowed i32")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset_raw.to_string())
            .with_detail("reason", "fixed_size_list_end_offset_start_overflow")
    })?;
    let starts = offsets_view
        .slice_with_operation(root_offset_raw, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let ends = offsets_view
        .slice_with_operation(end_offset, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let int32_type = DataType::new(TypeId::Int32);
    let widths = binary::binary_operation_vv(&ends, &starts, BinaryOperator::Sub, &int32_type)
        .map_err(|error| error.reanchor(operation))?;
    Ok(widths)
}

fn fixed_size_list_width_bounds(
    widths: &Column,
    view: &ColumnView<'_>,
    producer: &Arc<Column>,
    rows: usize,
    dimension: i32,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<(i32, i32, BoundaryObservation)> {
    let widths_view = widths.view().map_err(|error| error.reanchor_if_needed(operation))?;
    let root_null_count = view.null_count().map_err(|error| error.reanchor_if_needed(operation))?;
    let filtered_widths = if root_null_count == 0 || root_null_count == rows {
        None
    } else {
        let valid_rows = unary::is_valid(view).map_err(|error| error.reanchor(operation))?;
        let valid_rows_view = valid_rows.view().map_err(|error| error.reanchor_if_needed(operation))?;
        let widths_table =
            TableView::from_column_views(&[&widths_view]).map_err(|error| error.reanchor_if_needed(operation))?;
        let mut columns = filter::apply_boolean_mask(&widths_table, &valid_rows_view)
            .map_err(|error| error.reanchor(operation))?
            .into_columns()
            .map_err(|error| error.reanchor_if_needed(operation))?;
        Some(columns.pop().ok_or_else(|| {
            error::foreign_invariant_violation(
                operation,
                "filtering FixedSizeList widths returned no column",
                "list_offsets",
                "missing_filtered_fixed_size_list_widths",
                "ensure cuDF preserves the single input column when filtering valid list rows",
            )
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
        })?)
    };
    let int32_type = DataType::new(TypeId::Int32);
    let (minimum, maximum, downloaded_scalars) = if root_null_count == rows {
        (dimension, dimension, 0usize)
    } else {
        let reduction_view = match filtered_widths.as_ref() {
            Some(filtered) => filtered.view().map_err(|error| error.reanchor_if_needed(operation))?,
            None => widths_view,
        };
        let minimum = reduction::reduce(&reduction_view, &ReduceAgg::Min, &int32_type)
            .map_err(|error| error.reanchor(operation))?;
        let maximum = reduction::reduce(&reduction_view, &ReduceAgg::Max, &int32_type)
            .map_err(|error| error.reanchor(operation))?;
        (
            minimum.value::<i32>().map_err(|error| error.reanchor(operation))?,
            maximum.value::<i32>().map_err(|error| error.reanchor(operation))?,
            2usize,
        )
    };
    let observation = BoundaryObservation::new(
        BoundaryKind::BoundedControlScalar,
        producer.execution_domain().placement(),
        rows,
        downloaded_scalars,
        true,
    )
    .with_memory(
        QueryMemoryObservation::empty().with_downloaded_bytes(downloaded_scalars * std::mem::size_of::<i32>()),
    );
    record_boundary_observation_in_domain(producer.execution_domain(), observation);
    Ok((minimum, maximum, observation))
}

pub(super) fn validate_uniform_list_offsets(
    view: &ColumnView<'_>,
    producer: &Arc<Column>,
    field_index: usize,
    operation: &'static str,
) -> Result<UniformListProof> {
    let rows = view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    if rows == 0 {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            operation,
            "variable-width List<Float32> dimension is unavailable for an empty active range",
        )
        .with_fact(FACT_VALIDATION_REASON, "uniform_list_dimension_unavailable")
        .with_detail("field_index", field_index.to_string())
        .with_detail("row_count", "0")
        .with_detail("remedy", "provide at least one vector row or use a trusted FixedSizeList schema dimension"));
    }

    let (widths, root_offset) = uniform_list_offset_widths(view, rows, field_index, operation)?;
    let (minimum, maximum, offsets_observation) = uniform_list_reduced_width_range(&widths, producer, rows, operation)?;

    if minimum < 0 {
        return Err(error::foreign_invariant_violation(
            operation,
            "List<Float32> offsets are not monotonic over the active range",
            "list_offsets",
            "list_offsets_not_monotonic",
            "repair the list offsets before requesting a dense-vector export",
        )
        .with_fact(FACT_VALIDATION_REASON, "list_offsets_not_monotonic")
        .with_detail("field_index", field_index.to_string())
        .with_detail("minimum_row_width", minimum.to_string())
        .with_detail("maximum_row_width", maximum.to_string())
        .with_detail("validation_evidence", "device_reduced_list_offset_width_range"));
    }
    if minimum == 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "uniform List<Float32> dimension must be positive",
        )
        .with_fact(FACT_VALIDATION_REASON, "non_positive_vector_dimension")
        .with_detail("field_index", field_index.to_string())
        .with_detail("minimum_row_width", minimum.to_string())
        .with_detail("maximum_row_width", maximum.to_string())
        .with_detail("validation_evidence", "device_reduced_list_offset_width_range"));
    }
    if minimum != maximum {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            "List<Float32> rows do not have one uniform dimension",
        )
        .with_fact(FACT_VALIDATION_REASON, "non_uniform_vector_dimension")
        .with_detail("field_index", field_index.to_string())
        .with_detail("minimum_row_width", minimum.to_string())
        .with_detail("maximum_row_width", maximum.to_string())
        .with_detail("validation_evidence", "device_reduced_list_offset_width_range"));
    }
    let dimension = usize::try_from(minimum).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "positive uniform list dimension must fit usize",
            "list_offsets",
            "uniform_list_dimension_conversion_failed",
            "retain cudf-compatible list offsets",
        )
        .with_detail("dimension", minimum.to_string())
    })?;

    let (child_offset, child_len) = uniform_list_active_child_span(view, rows, dimension, field_index, operation)?;

    Ok(UniformListProof {
        field_index,
        dimension,
        element: view
            .list_sliced_child()
            .map_err(|error| error.reanchor(operation))?
            .data_type()
            .map_err(|error| error.reanchor(operation))?
            .id(),
        rows,
        root_offset,
        child_offset,
        child_len,
        producer_column_identity: Arc::as_ptr(producer) as usize,
        binding_id: producer.execution_domain().binding_id(),
        placement: producer.execution_domain().placement(),
        offsets_observation,
    })
}

fn uniform_list_offset_widths(
    view: &ColumnView<'_>,
    rows: usize,
    field_index: usize,
    operation: &'static str,
) -> Result<(Column, usize)> {
    let root_offset_raw = view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let root_offset = usize::try_from(root_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "cudf list root offset must be non-negative",
            "list_offsets",
            "negative_list_root_offset",
            "retain a valid cudf list active range",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("root_offset", root_offset_raw.to_string())
    })?;
    let root_end = root_offset.checked_add(rows).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "uniform list active offset range overflowed usize")
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("reason", "uniform_list_offset_range_overflow")
    })?;
    let offsets_view = view.child(0).map_err(|error| error.reanchor_if_needed(operation))?;
    let offsets_len = offsets_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    if offsets_len <= root_end {
        return Err(error::foreign_invariant_violation(
            operation,
            "cudf list offsets are too short for the active uniform-list range",
            "list_offsets",
            "uniform_list_offsets_too_short",
            "retain one more list offset than active rows",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("required_offset_index", root_end.to_string())
        .with_detail("offset_count", offsets_len.to_string()));
    }

    let rows_i32 =
        i32::try_from(rows).map_err(|_| error::range_limit_error(operation, "row_count", "i32::MAX", rows))?;
    let end_offset = root_offset_raw.checked_add(1).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "uniform-list end-offset start overflowed i32")
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset_raw.to_string())
            .with_detail("reason", "uniform_list_end_offset_start_overflow")
    })?;
    let starts = offsets_view
        .slice_with_operation(root_offset_raw, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let ends = offsets_view
        .slice_with_operation(end_offset, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let int32_type = DataType::new(TypeId::Int32);
    let widths = binary::binary_operation_vv(&ends, &starts, BinaryOperator::Sub, &int32_type)
        .map_err(|error| error.reanchor(operation))?;
    Ok((widths, root_offset))
}

fn uniform_list_reduced_width_range(
    widths: &Column,
    producer: &Arc<Column>,
    rows: usize,
    operation: &'static str,
) -> Result<(i32, i32, BoundaryObservation)> {
    let widths_view = widths.view().map_err(|error| error.reanchor_if_needed(operation))?;
    let int32_type = DataType::new(TypeId::Int32);
    let minimum =
        reduction::reduce(&widths_view, &ReduceAgg::Min, &int32_type).map_err(|error| error.reanchor(operation))?;
    let maximum =
        reduction::reduce(&widths_view, &ReduceAgg::Max, &int32_type).map_err(|error| error.reanchor(operation))?;
    let minimum = minimum.value::<i32>().map_err(|error| error.reanchor(operation))?;
    let maximum = maximum.value::<i32>().map_err(|error| error.reanchor(operation))?;
    let offsets_observation = BoundaryObservation::new(
        BoundaryKind::BoundedControlScalar,
        producer.execution_domain().placement(),
        rows,
        2,
        true,
    )
    .with_memory(
        QueryMemoryObservation::empty().with_downloaded_bytes(2usize.saturating_mul(std::mem::size_of::<i32>())),
    );
    record_boundary_observation_in_domain(producer.execution_domain(), offsets_observation);
    Ok((minimum, maximum, offsets_observation))
}

fn uniform_list_active_child_span(
    view: &ColumnView<'_>,
    rows: usize,
    dimension: usize,
    field_index: usize,
    operation: &'static str,
) -> Result<(usize, usize)> {
    let child_view = view.list_sliced_child().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset_raw = child_view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset = usize::try_from(child_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "uniform List<Float32> child offset must be non-negative",
            "list_child",
            "negative_uniform_list_child_offset",
            "retain a valid cudf list active child range",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("child_offset", child_offset_raw.to_string())
    })?;
    let child_len = child_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    let expected_child_len = rows.checked_mul(dimension).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "uniform List<Float32> child length overflowed usize")
            .with_detail("field_index", field_index.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("dimension", dimension.to_string())
            .with_detail("reason", "uniform_list_child_len_overflow")
    })?;
    if child_len != expected_child_len {
        return Err(error::foreign_invariant_violation(
            operation,
            "offset-proven uniform list span differs from the active child range",
            "list_child",
            "uniform_list_active_child_len_mismatch",
            "retain a cudf list child range derived from the active offsets",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("dimension", dimension.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("actual_child_len", child_len.to_string()));
    }
    Ok((child_offset, child_len))
}

fn record_fixed_size_list_structural_proof(
    field_index: usize,
    dimension: i32,
    element: TypeId,
    rows: usize,
    child_offset: usize,
    child_len: usize,
    producer: &Arc<Column>,
    observation: Option<BoundaryObservation>,
) -> FixedSizeListStructuralProof {
    FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT.with(|count| count.set(count.get() + 1));
    FixedSizeListStructuralProof {
        field_index,
        dimension,
        element,
        rows,
        child_offset,
        child_len,
        producer_column_identity: Arc::as_ptr(producer) as usize,
        binding_id: producer.execution_domain().binding_id(),
        placement: producer.execution_domain().placement(),
        observation,
    }
}

pub(super) fn rebind_fixed_size_list_structural_proof(
    proof: FixedSizeListStructuralProof,
    frame: &DataFrame,
) -> Result<FixedSizeListStructuralProof> {
    const OPERATION: &str = "dataframe::GpuRecordBatch::with_validated_slice";
    let (_, producer) = frame.columns_map().get_index(proof.field_index).ok_or_else(|| {
        Error::new_with(
            ErrorCode::InvariantViolation,
            OPERATION,
            "validated FixedSizeList field disappeared while slicing a GPU record batch",
        )
        .with_detail("field_index", proof.field_index.to_string())
    })?;
    let view = producer.view().map_err(|error| error.reanchor(OPERATION))?;
    let rows = view.len().map_err(|error| error.reanchor(OPERATION))?;
    let child = view.list_sliced_child().map_err(|error| error.reanchor(OPERATION))?;
    let child_offset_raw = child.offset().map_err(|error| error.reanchor(OPERATION))?;
    let child_offset = usize::try_from(child_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            OPERATION,
            "sliced FixedSizeList child offset must be non-negative",
            "list_child",
            "negative_fixed_size_list_child_offset",
            "retain a valid cudf list slice",
        )
        .with_detail("field_index", proof.field_index.to_string())
        .with_detail("child_offset", child_offset_raw.to_string())
    })?;
    let child_len = child.len().map_err(|error| error.reanchor(OPERATION))?;
    let expected_child_len = rows.checked_mul(proof.dimension as usize).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, OPERATION, "sliced FixedSizeList child length overflowed usize")
            .with_detail("field_index", proof.field_index.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("dimension", proof.dimension.to_string())
    })?;
    let root_null_count = view.null_count().map_err(|error| error.reanchor(OPERATION))?;
    if root_null_count == 0 && child_len != expected_child_len {
        return Err(error::invariant_violation(
            OPERATION,
            "slicing a structurally validated FixedSizeList changed its fixed-width child span",
        )
        .with_detail("field_index", proof.field_index.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("actual_child_len", child_len.to_string()));
    }

    Ok(FixedSizeListStructuralProof {
        field_index: proof.field_index,
        dimension: proof.dimension,
        element: proof.element,
        rows,
        child_offset,
        child_len,
        producer_column_identity: Arc::as_ptr(producer) as usize,
        binding_id: producer.execution_domain().binding_id(),
        placement: producer.execution_domain().placement(),
        observation: proof.observation,
    })
}

pub(super) fn validate_list_export<'a>(
    export: PreparedColumnExport<'a>,
    contract: ListExportContract,
    operation: &'static str,
) -> Result<ListExportParts<'a>> {
    let (offsets_buffer_id, child_column_id) = check_list_export_root(&export, &contract, operation)?;
    let child_values_buffer_id = check_list_export_child(&export, &contract, child_column_id, operation)?;
    check_list_export_child_values(&export, &contract, child_values_buffer_id, operation)?;

    Ok(ListExportParts {
        root_column_id: export.column().id(),
        offsets_buffer_id,
        child_column_id,
        child_values_buffer_id,
        export,
    })
}

fn list_export_contract_error(
    contract: &ListExportContract,
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
) -> Error {
    error::foreign_invariant_violation(
        operation,
        message,
        "fixed_size_list_export",
        reason,
        "rebuild the export from the validated GpuRecordBatch producer",
    )
    .with_detail("field_index", contract.field_index.to_string())
    .with_detail("dimension", contract.dimension.to_string())
}

fn check_list_export_root(
    export: &PreparedColumnExport<'_>,
    contract: &ListExportContract,
    operation: &'static str,
) -> Result<(BufferId, ColumnId)> {
    let root = export.column();
    if root.data_type().id() != rapids_interop::TypeId::List || root.len() != contract.rows {
        return Err(list_export_contract_error(
            contract,
            operation,
            "prepared export root does not match the validated List<Float32> row contract",
            "list_export_root_mismatch",
        )
        .with_detail("root_type", format!("{:?}", root.data_type().id()))
        .with_detail("root_rows", root.len().to_string())
        .with_detail("proof_rows", contract.rows.to_string()));
    }
    if let Some(root_offset) = contract.root_offset
        && root.offset() != root_offset
    {
        return Err(list_export_contract_error(
            contract,
            operation,
            "prepared list export root offset differs from the proven active range",
            "list_export_root_offset_mismatch",
        )
        .with_detail("proof_root_offset", root_offset.to_string())
        .with_detail("export_root_offset", root.offset().to_string()));
    }
    if !validity_proves_no_nulls(root.validity()) {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            operation,
            "validated List<Float32> vector rows must not contain nulls",
        )
        .with_detail("field_index", contract.field_index.to_string())
        .with_detail("reason", "null_vector_row"));
    }

    let (offsets_buffer_id, child_column_id) = match root.layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        _ => {
            return Err(list_export_contract_error(
                contract,
                operation,
                "prepared export root does not retain a list layout",
                "fixed_size_list_export_layout_mismatch",
            ));
        },
    };
    if !export.bindings().iter().any(|binding| binding.id() == offsets_buffer_id) {
        return Err(list_export_contract_error(
            contract,
            operation,
            "prepared List<Float32> export lost its offsets binding",
            "list_float32_offsets_binding_missing",
        ));
    }
    Ok((offsets_buffer_id, child_column_id))
}

fn check_list_export_child(
    export: &PreparedColumnExport<'_>,
    contract: &ListExportContract,
    child_column_id: ColumnId,
    operation: &'static str,
) -> Result<BufferId> {
    let child = export.semantic_columns().iter().find(|column| column.id() == child_column_id).ok_or_else(|| {
        list_export_contract_error(
            contract,
            operation,
            "prepared List<Float32> export lost its semantic child column",
            "list_float32_child_column_missing",
        )
    })?;
    let expected = match contract.element {
        TypeId::Float32 => rapids_interop::TypeId::Float32,
        TypeId::Float64 => rapids_interop::TypeId::Float64,
        TypeId::Int8 => rapids_interop::TypeId::Int8,
        TypeId::UInt8 => rapids_interop::TypeId::UInt8,
        _ => {
            return Err(list_export_contract_error(
                contract,
                operation,
                "unsupported list child type",
                "unsupported_list_child_type",
            ));
        },
    };
    if child.data_type().id() != expected {
        return Err(list_export_contract_error(
            contract,
            operation,
            "prepared list child differs from proven element",
            "list_float32_child_type_mismatch",
        )
        .with_detail("child_type", format!("{:?}", child.data_type().id())));
    }
    if child.len() != contract.child_len || child.offset() != contract.child_offset {
        return Err(list_export_contract_error(
            contract,
            operation,
            "prepared List<Float32> child active range differs from the structural proof",
            "list_float32_child_range_mismatch",
        )
        .with_detail("proof_child_offset", contract.child_offset.to_string())
        .with_detail("export_child_offset", child.offset().to_string())
        .with_detail("proof_child_len", contract.child_len.to_string())
        .with_detail("export_child_len", child.len().to_string()));
    }
    if !validity_proves_no_nulls(child.validity()) {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            operation,
            "validated List<Float32> child must not contain nulls",
        )
        .with_detail("field_index", contract.field_index.to_string())
        .with_detail("reason", "null_vector_value"));
    }
    let child_values_buffer_id = match child.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        _ => {
            return Err(list_export_contract_error(
                contract,
                operation,
                "prepared List<Float32> child is not fixed-width",
                "list_float32_child_layout_mismatch",
            ));
        },
    };
    Ok(child_values_buffer_id)
}

fn check_list_export_child_values(
    export: &PreparedColumnExport<'_>,
    contract: &ListExportContract,
    child_values_buffer_id: BufferId,
    operation: &'static str,
) -> Result<()> {
    let child_values =
        export.bindings().iter().find(|binding| binding.id() == child_values_buffer_id).ok_or_else(|| {
            list_export_contract_error(
                contract,
                operation,
                "prepared List<Float32> export lost its child values binding",
                "list_float32_child_values_binding_missing",
            )
        })?;
    let required_child_bytes = contract
        .child_offset
        .checked_add(contract.child_len)
        .and_then(|elements| elements.checked_mul(matrix_element_bytes(contract.element, operation).ok()?))
        .ok_or_else(|| {
            list_export_contract_error(
                contract,
                operation,
                "validated List<Float32> child byte range overflowed usize",
                "list_float32_child_byte_range_overflow",
            )
        })?;
    if child_values.bytes().byte_len() < required_child_bytes {
        return Err(list_export_contract_error(
            contract,
            operation,
            "prepared List<Float32> child values binding is smaller than the proven active range",
            "list_float32_child_values_binding_too_small",
        )
        .with_detail("required_child_bytes", required_child_bytes.to_string())
        .with_detail("binding_bytes", child_values.bytes().byte_len().to_string()));
    }

    let expected_device = contract.placement.device.ordinal() as i32;
    if !matches!(
        child_values.placement(),
        rapids_interop::Placement::CudaDevice { device } if *device == expected_device
    ) {
        return Err(list_export_contract_error(
            contract,
            operation,
            "prepared List<Float32> child values binding is not on the proven CUDA device",
            "list_float32_child_device_mismatch",
        ));
    }
    match export.access() {
        PreparedAccess::Device { readiness: ProducerReadiness::DeviceStreamOrdered { stream }, .. }
            if stream.device() == expected_device && stream.as_raw() as usize == contract.placement.stream_debug => {},
        _ => {
            return Err(list_export_contract_error(
                contract,
                operation,
                "prepared List<Float32> export readiness differs from the proven execution stream",
                "list_export_readiness_mismatch",
            ));
        },
    }
    Ok(())
}

fn validity_proves_no_nulls(validity: &PhysicalValidity) -> bool {
    matches!(validity, PhysicalValidity::NoValidityBuffer | PhysicalValidity::ValidityMask { null_count: Some(0), .. })
}

pub(crate) fn reset_fixed_size_list_structural_proof_build_count_for_test() {
    FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT.with(|count| count.set(0));
}

pub(crate) fn fixed_size_list_structural_proof_build_count_for_test() -> usize {
    FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT.with(Cell::get)
}
