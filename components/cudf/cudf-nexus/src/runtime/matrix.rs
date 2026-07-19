//! Runtime-owned dense matrix packing for fixed-width Float32 columns.

use std::{ffi::c_void, fmt, sync::Arc};

use super::{
    BoundaryObservation, ExecutionBindingId, ExecutionPlacement, Runtime, execution_domain::ExecutionDomain,
    execution_resolver::ResolvedExecution,
};
use crate::{
    core::{
        Column,
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_ROWS, FACT_COLUMN_INDEX, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID,
            FACT_RIGHT_TYPE_ID, Result,
        },
        types::TypeId,
    },
    ops::{
        concatenate::concatenate_columns_with_exec, reshape::interleave_columns_with_exec,
        validation::FiniteFloat32ValidationProof,
    },
};

const PACK_OPERATION: &str = "runtime::pack_float32_columns";
const PACK_FLAT_OPERATION: &str = "runtime::pack_flat_float32_column";

/// Physical layout of a packed dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseMatrixLayout {
    /// Adjacent values belong to adjacent columns of one logical row.
    RowMajor,
    /// Adjacent values belong to adjacent rows of one logical column.
    ColumnMajor,
}

/// A contiguous Float32 matrix owned by a cudf runtime execution domain.
///
/// The underlying [`Column`] retains the exact CUDA device, stream, and memory
/// resource that produced the packed allocation. The matrix exposes only a
/// const device pointer; algorithms that need writable output should allocate
/// a [`crate::core::DeviceBuffer`] from the caller's [`Runtime`].
pub struct PackedFloat32Matrix {
    values: Column,
    rows: usize,
    columns: usize,
    layout: DenseMatrixLayout,
}

impl PackedFloat32Matrix {
    /// Logical matrix shape as `[rows, columns]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    /// Number of logical rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of logical columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Number of packed Float32 elements.
    pub fn len(&self) -> usize {
        self.rows * self.columns
    }

    /// Whether the packed allocation contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Physical matrix layout.
    pub fn layout(&self) -> DenseMatrixLayout {
        self.layout
    }

    /// Stride, in Float32 elements, between adjacent logical rows.
    pub fn row_stride_elements(&self) -> usize {
        match self.layout {
            DenseMatrixLayout::RowMajor => self.columns,
            DenseMatrixLayout::ColumnMajor => 1,
        }
    }

    /// Stride, in Float32 elements, between adjacent logical columns.
    pub fn column_stride_elements(&self) -> usize {
        match self.layout {
            DenseMatrixLayout::RowMajor => 1,
            DenseMatrixLayout::ColumnMajor => self.rows,
        }
    }

    /// Borrow the owned flat Float32 column.
    pub fn as_column(&self) -> &Column {
        &self.values
    }

    /// Consume the matrix and return its owned flat Float32 column.
    pub fn into_column(self) -> Column {
        self.values
    }

    /// Diagnostic summary of the retained execution placement.
    pub fn placement(&self) -> ExecutionPlacement {
        self.values.execution_domain().placement()
    }

    /// Stable identity for the exact retained stream and memory-resource
    /// binding.
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.values.execution_domain().binding_id()
    }

    /// Validate that every packed matrix value is finite before handing the
    /// device pointer to a downstream GPU consumer.
    pub fn validate_finite(&self) -> Result<ValidatedPackedFloat32Matrix<'_>> {
        let proof = self
            .values
            .validate_finite_float32()
            .map_err(|error| error.reanchor("packed_float32_matrix::validate_finite"))?;
        Ok(ValidatedPackedFloat32Matrix { matrix: self, proof })
    }
}

/// A packed Float32 matrix whose complete payload was proven finite on its
/// retained execution stream.
///
/// This borrowing gate is the only matrix surface that exposes the device
/// pointer used by downstream GPU consumers. The borrow keeps the validated
/// allocation alive and prevents the proof from being detached from a
/// different packed matrix.
pub struct ValidatedPackedFloat32Matrix<'a> {
    matrix: &'a PackedFloat32Matrix,
    proof: FiniteFloat32ValidationProof,
}

impl ValidatedPackedFloat32Matrix<'_> {
    /// Logical matrix shape as `[rows, columns]`.
    pub fn shape(&self) -> [usize; 2] {
        self.matrix.shape()
    }

    /// Number of logical rows.
    pub fn rows(&self) -> usize {
        self.matrix.rows()
    }

    /// Number of logical columns.
    pub fn columns(&self) -> usize {
        self.matrix.columns()
    }

    /// Number of validated Float32 elements.
    pub fn len(&self) -> usize {
        self.proof.validated_values()
    }

    /// Whether the validated allocation contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Physical matrix layout.
    pub fn layout(&self) -> DenseMatrixLayout {
        self.matrix.layout()
    }

    /// Stride, in Float32 elements, between adjacent logical rows.
    pub fn row_stride_elements(&self) -> usize {
        self.matrix.row_stride_elements()
    }

    /// Stride, in Float32 elements, between adjacent logical columns.
    pub fn column_stride_elements(&self) -> usize {
        self.matrix.column_stride_elements()
    }

    /// Read-only pointer to the finite packed device allocation.
    pub fn as_ptr(&self) -> Result<*const c_void> {
        self.matrix.values.device_data_ptr().map_err(|error| error.reanchor("validated_packed_float32_matrix::as_ptr"))
    }

    /// Exact stream/memory-resource binding used for finite validation.
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.proof.binding_id()
    }

    /// Execution placement associated with the validated payload.
    pub fn placement(&self) -> ExecutionPlacement {
        self.proof.placement()
    }

    /// Bounded control-scalar transfer observation from finite validation.
    pub fn observation(&self) -> BoundaryObservation {
        self.proof.observation()
    }
}

impl fmt::Debug for ValidatedPackedFloat32Matrix<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedPackedFloat32Matrix")
            .field("rows", &self.rows())
            .field("columns", &self.columns())
            .field("layout", &self.layout())
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for PackedFloat32Matrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PackedFloat32Matrix")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("layout", &self.layout)
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}

fn checked_matrix_element_count(rows: usize, columns: usize) -> Result<usize> {
    checked_matrix_element_count_in(rows, columns, PACK_OPERATION)
}

fn checked_matrix_element_count_in(rows: usize, columns: usize, operation: &'static str) -> Result<usize> {
    let elements = rows.checked_mul(columns).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "packed matrix element count overflowed usize")
            .with_detail("rows", rows.to_string())
            .with_detail("columns", columns.to_string())
            .with_detail("resource", "matrix_elements")
            .with_detail("reason", "shape_product_overflow")
            .with_detail("remedy", "reduce the matrix row or column count")
    })?;
    if elements > i32::MAX as usize {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "packed matrix element count exceeds cudf size_type",
        )
        .with_detail("rows", rows.to_string())
        .with_detail("columns", columns.to_string())
        .with_detail("matrix_elements", elements.to_string())
        .with_detail("max_matrix_elements", i32::MAX.to_string())
        .with_detail("resource", "matrix_elements")
        .with_detail("reason", "matrix_element_count_exceeds_cudf_size_type")
        .with_detail("remedy", "reduce the matrix row or column count so their product fits within i32::MAX"));
    }
    Ok(elements)
}

fn validate_float32_columns(columns: &[&Column]) -> Result<(usize, usize)> {
    let Some(first) = columns.first() else {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            PACK_OPERATION,
            "dense matrix packing requires at least one input column",
        )
        .with_detail("resource", "input_columns")
        .with_detail("reason", "empty_input")
        .with_detail("remedy", "provide at least one non-null Float32 column"));
    };

    let expected_rows = first.len().map_err(|error| error.reanchor(PACK_OPERATION))?;
    for (column_index, column) in columns.iter().enumerate() {
        let actual_type = column.data_type().map_err(|error| error.reanchor(PACK_OPERATION))?.id();
        if actual_type != TypeId::Float32 {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                PACK_OPERATION,
                "dense matrix packing requires Float32 input columns",
            )
            .with_fact(FACT_COLUMN_INDEX, column_index)
            .with_fact(FACT_LEFT_TYPE_ID, actual_type)
            .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Float32));
        }

        let actual_rows = column.len().map_err(|error| error.reanchor(PACK_OPERATION))?;
        if actual_rows != expected_rows {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                PACK_OPERATION,
                "dense matrix input columns must have the same row count",
            )
            .with_fact(FACT_COLUMN_INDEX, column_index)
            .with_fact(FACT_EXPECTED_ROWS, expected_rows)
            .with_fact(FACT_ACTUAL_ROWS, actual_rows));
        }

        let null_count = column.null_count().map_err(|error| error.reanchor(PACK_OPERATION))?;
        if null_count != 0 {
            return Err(Error::new_with(
                ErrorCode::NullUnexpected,
                PACK_OPERATION,
                "dense matrix input columns must not contain null values",
            )
            .with_fact(FACT_COLUMN_INDEX, column_index)
            .with_detail("null_count", null_count.to_string())
            .with_detail("reason", "null_feature_value")
            .with_detail("remedy", "replace or reject null feature values before packing"));
        }
    }

    checked_matrix_element_count(expected_rows, columns.len())?;
    Ok((expected_rows, columns.len()))
}

impl Runtime {
    /// Copy one borrowed device-resident row-major Float32 span into a packed
    /// matrix on this runtime's exact execution domain.
    ///
    /// # Safety
    ///
    /// `values` must address at least `rows * columns` readable Float32 values
    /// on this runtime's CUDA device. Prior writes must be ordered before this
    /// runtime's stream, and the allocation must remain alive until the stream
    /// completes the enqueued copy.
    pub(crate) unsafe fn pack_borrowed_flat_float32_device_values(
        &self,
        values: *const c_void,
        input_domain: &Arc<ExecutionDomain>,
        rows: usize,
        columns: usize,
        layout: DenseMatrixLayout,
        operation: &'static str,
    ) -> Result<PackedFloat32Matrix> {
        if rows == 0 || columns == 0 {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                operation,
                "flat Float32 matrix packing requires positive rows and columns",
            )
            .with_detail("rows", rows.to_string())
            .with_detail("columns", columns.to_string())
            .with_detail("reason", "non_positive_matrix_shape"));
        }
        checked_matrix_element_count_in(rows, columns, operation)?;
        if values.is_null() {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                operation,
                "borrowed Float32 matrix values pointer must not be null",
            )
            .with_detail("reason", "null_borrowed_matrix_values"));
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| Error::new_with(ErrorCode::ValueOutOfRange, operation, "matrix rows exceed cudf size_type"))?;
        let columns_i32 = i32::try_from(columns).map_err(|_| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "matrix columns exceed cudf size_type")
        })?;
        let execution =
            ResolvedExecution::with_output_domain(Arc::clone(self.execution_domain()), [input_domain], operation)?;
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            // SAFETY: the caller supplies the documented device pointer,
            // ordering, and lifetime guarantees; shape validation above
            // bounds the exact Float32 span consumed by the native call.
            unsafe {
                cudf_sys::cudf_pack_borrowed_float32_matrix_ex(
                    values,
                    rows_i32,
                    columns_i32,
                    i32::from(matches!(layout, DenseMatrixLayout::ColumnMajor)),
                    execution.exec_options(),
                    &mut out,
                )
            },
            operation,
        )?;
        // SAFETY: success transfers one native owning column allocated on the
        // exact output runtime execution domain entered above.
        let values = unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }
            .map_err(|error| error.reanchor(operation))?;
        // `execution` drops after the consuming call and records the reverse
        // completion fence. Cross-stream input deallocation is therefore
        // ordered after the copy without synchronizing the host.
        Ok(PackedFloat32Matrix { values, rows, columns, layout })
    }

    /// Pack equal-length, non-null Float32 columns into a contiguous matrix on
    /// this runtime's exact device, stream, and memory-resource binding.
    ///
    /// Inputs may originate from other streams on the same device; the
    /// runtime inserts device-side ordering in both directions. Cross-device
    /// inputs are rejected before launching the pack operation.
    pub fn pack_float32_columns(&self, columns: &[&Column], layout: DenseMatrixLayout) -> Result<PackedFloat32Matrix> {
        let (rows, column_count) = validate_float32_columns(columns)?;
        let views = columns
            .iter()
            .map(|column| column.view().map_err(|error| error.reanchor(PACK_OPERATION)))
            .collect::<Result<Vec<_>>>()?;
        let view_refs = views.iter().collect::<Vec<_>>();
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(self.execution_domain()),
            columns.iter().map(|column| column.execution_domain()),
            PACK_OPERATION,
        )?;
        let values = match layout {
            DenseMatrixLayout::RowMajor => {
                interleave_columns_with_exec(&view_refs, execution.output_domain(), execution.exec_options())
            },
            DenseMatrixLayout::ColumnMajor => {
                concatenate_columns_with_exec(&view_refs, execution.output_domain(), execution.exec_options())
            },
        }
        .map_err(|error| error.reanchor(PACK_OPERATION))?;

        Ok(PackedFloat32Matrix { values, rows, columns: column_count, layout })
    }

    /// Pack one flat, non-null Float32 column whose values describe a
    /// row-major `[rows, columns]` matrix into the requested dense layout.
    ///
    /// This is the nested-vector counterpart to [`pack_float32_columns`](Self::pack_float32_columns): it retains
    /// the runtime's exact device, stream, and memory-resource provenance and
    /// performs any row-major to column-major rearrangement entirely on that
    /// stream. It never exposes or copies vector payloads through host memory.
    pub fn pack_flat_float32_column(
        &self,
        values: &Column,
        rows: usize,
        columns: usize,
        layout: DenseMatrixLayout,
    ) -> Result<PackedFloat32Matrix> {
        if rows == 0 || columns == 0 {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                PACK_FLAT_OPERATION,
                "flat Float32 matrix packing requires positive rows and columns",
            )
            .with_detail("rows", rows.to_string())
            .with_detail("columns", columns.to_string())
            .with_detail("reason", "non_positive_matrix_shape"));
        }
        let expected_len = checked_matrix_element_count_in(rows, columns, PACK_FLAT_OPERATION)?;
        let data_type = values.data_type().map_err(|error| error.reanchor(PACK_FLAT_OPERATION))?;
        if data_type.id() != TypeId::Float32 {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                PACK_FLAT_OPERATION,
                "flat matrix packing requires a Float32 value column",
            )
            .with_fact(FACT_LEFT_TYPE_ID, data_type.id())
            .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Float32));
        }
        let null_count = values.null_count().map_err(|error| error.reanchor(PACK_FLAT_OPERATION))?;
        if null_count != 0 {
            return Err(Error::new_with(
                ErrorCode::NullUnexpected,
                PACK_FLAT_OPERATION,
                "flat matrix packing requires non-null Float32 values",
            )
            .with_detail("null_count", null_count.to_string())
            .with_detail("reason", "nullable_flat_matrix_values"));
        }
        let actual_len = values.len().map_err(|error| error.reanchor(PACK_FLAT_OPERATION))?;
        if actual_len != expected_len {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                PACK_FLAT_OPERATION,
                "flat matrix value length does not match the requested shape",
            )
            .with_fact(FACT_EXPECTED_ROWS, expected_len)
            .with_fact(FACT_ACTUAL_ROWS, actual_len)
            .with_detail("rows", rows.to_string())
            .with_detail("columns", columns.to_string())
            .with_detail("reason", "flat_matrix_shape_mismatch"));
        }

        let view = values.view().map_err(|error| error.reanchor(PACK_FLAT_OPERATION))?;
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(self.execution_domain()),
            [values.execution_domain()],
            PACK_FLAT_OPERATION,
        )?;
        let rows_i32 = i32::try_from(rows).map_err(|_| {
            Error::new_with(ErrorCode::ValueOutOfRange, PACK_FLAT_OPERATION, "matrix rows exceed cudf size_type")
        })?;
        let columns_i32 = i32::try_from(columns).map_err(|_| {
            Error::new_with(ErrorCode::ValueOutOfRange, PACK_FLAT_OPERATION, "matrix columns exceed cudf size_type")
        })?;
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_pack_flat_float32_matrix_ex(
                    view.raw_handle(),
                    rows_i32,
                    columns_i32,
                    i32::from(matches!(layout, DenseMatrixLayout::ColumnMajor)),
                    execution.exec_options(),
                    &mut out,
                )
            },
            PACK_FLAT_OPERATION,
        )?;
        // SAFETY: a successful native entry transfers one owned column handle
        // produced on the exact retained output execution domain.
        let values = unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }
            .map_err(|error| error.reanchor(PACK_FLAT_OPERATION))?;
        Ok(PackedFloat32Matrix { values, rows, columns, layout })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_element_count_overflow_is_structured() {
        let error = checked_matrix_element_count(usize::MAX, 2).expect_err("shape multiplication must overflow");
        assert_eq!(error.code(), ErrorCode::SizeOverflow);
        assert_eq!(error.operation(), PACK_OPERATION);
        assert_eq!(error.detail_value("reason"), Some("shape_product_overflow"));
    }

    #[test]
    fn matrix_element_count_accepts_cudf_size_type_maximum() {
        assert_eq!(checked_matrix_element_count(i32::MAX as usize, 1).unwrap(), i32::MAX as usize);
    }

    #[test]
    fn matrix_element_count_rejects_one_past_cudf_size_type_maximum() {
        let error = checked_matrix_element_count(i32::MAX as usize + 1, 1)
            .expect_err("matrix output length must fit cudf size_type");
        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(error.operation(), PACK_OPERATION);
        assert_eq!(error.detail_value("matrix_elements"), Some((i32::MAX as usize + 1).to_string().as_str()));
        assert_eq!(error.detail_value("max_matrix_elements"), Some(i32::MAX.to_string().as_str()));
        assert_eq!(error.detail_value("reason"), Some("matrix_element_count_exceeds_cudf_size_type"));
    }
}
