//! Runtime-owned dense matrix packing for supported fixed-width numeric columns.

use std::{ffi::c_void, fmt, sync::Arc};

use super::{
    DeviceId, ExecutionBindingId, ExecutionPlacement, Runtime, execution_domain::ExecutionDomain,
    execution_resolver::ResolvedExecution,
};
use crate::{
    Exec,
    core::{
        Column,
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_ROWS, FACT_COLUMN_INDEX, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID,
            FACT_RIGHT_TYPE_ID, Result,
        },
        types::{DataType, TypeId},
    },
    frame::Series,
    ops::{
        concatenate::concatenate_columns_with_exec, reshape::interleave_columns_with_exec, unary::cast_impl,
        validation::validate_finite_float64,
    },
};

const PACK_OPERATION: &str = "runtime::pack_columns";
const PACK_FLAT_OPERATION: &str = "runtime::pack_flat_column";
const WIDEN_OPERATION: &str = "runtime::widen_packed_matrix_to_float64";
const UNPACK_OPERATION: &str = "runtime::unpack_dense_matrix";

pub(crate) fn matrix_element_bytes(element: TypeId, operation: &'static str) -> Result<usize> {
    match element {
        TypeId::Float32 => Ok(4),
        TypeId::Float64 => Ok(8),
        TypeId::Int8 | TypeId::UInt8 => Ok(1),
        _ => Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "matrix element must be Float32, Float64, Int8, or UInt8",
        )
        .with_fact(FACT_LEFT_TYPE_ID, element)
        .with_detail("supported_elements", "Float32, Float64, Int8, UInt8")),
    }
}

/// Physical layout of a packed dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseMatrixLayout {
    /// Adjacent values belong to adjacent columns of one logical row.
    RowMajor,
    /// Adjacent values belong to adjacent rows of one logical column.
    ColumnMajor,
}

/// A contiguous supported numeric matrix owned by a cudf runtime execution domain.
///
/// The underlying [`Column`] retains the exact CUDA device, stream, and memory
/// resource that produced the packed allocation. The matrix exposes only a
/// const device pointer; algorithms that need writable output should allocate
/// a [`crate::core::DeviceBuffer`] from the caller's [`Runtime`].
pub struct PackedMatrix {
    values: Column,
    element: TypeId,
    rows: usize,
    columns: usize,
    layout: DenseMatrixLayout,
}

impl PackedMatrix {
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

    /// Number of packed elements.
    pub fn len(&self) -> usize {
        self.rows * self.columns
    }

    /// Whether the packed allocation contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element type retained by this matrix.
    pub fn element(&self) -> TypeId {
        self.element
    }

    /// Physical matrix layout.
    pub fn layout(&self) -> DenseMatrixLayout {
        self.layout
    }

    /// Stride, in elements, between adjacent logical rows.
    pub fn row_stride_elements(&self) -> usize {
        match self.layout {
            DenseMatrixLayout::RowMajor => self.columns,
            DenseMatrixLayout::ColumnMajor => 1,
        }
    }

    /// Stride, in elements, between adjacent logical columns.
    pub fn column_stride_elements(&self) -> usize {
        match self.layout {
            DenseMatrixLayout::RowMajor => 1,
            DenseMatrixLayout::ColumnMajor => self.rows,
        }
    }

    /// Borrow the owned flat numeric column.
    pub fn as_column(&self) -> &Column {
        &self.values
    }

    /// Consume the matrix and return its owned flat numeric column.
    pub fn into_column(self) -> Column {
        self.values
    }

    /// Diagnostic summary of the retained execution placement.
    pub fn placement(&self) -> ExecutionPlacement {
        self.values.execution_domain().placement()
    }

    /// The CUDA device the packed values live on.
    pub fn device(&self) -> DeviceId {
        self.values.execution_domain().device_id()
    }

    /// Stable identity for the exact retained stream and memory-resource
    /// binding.
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.values.execution_domain().binding_id()
    }

    /// Validate floating values for finiteness, or retain integer shape and binding facts.
    pub fn validate(&self) -> Result<ValidatedPackedMatrix<'_>> {
        match self.element {
            TypeId::Float32 => {
                self.values.validate_finite_float32().map_err(|error| error.reanchor("packed_matrix::validate"))?;
            },
            TypeId::Float64 => {
                let view = self.values.view().map_err(|error| error.reanchor("packed_matrix::validate"))?;
                validate_finite_float64(&view).map_err(|error| error.reanchor("packed_matrix::validate"))?;
            },
            TypeId::Int8 | TypeId::UInt8 => {},
            _ => unreachable!("packed matrix element is checked at construction"),
        }
        Ok(ValidatedPackedMatrix { matrix: self })
    }
}

/// A packed matrix whose floating payload was proven finite on its
/// retained execution stream. Integer payloads have no finite scan.
///
/// This borrowing gate is the only matrix surface that exposes the device
/// pointer used by downstream GPU consumers. The borrow keeps the validated
/// allocation alive and prevents the proof from being detached from a
/// different packed matrix.
pub struct ValidatedPackedMatrix<'a> {
    matrix: &'a PackedMatrix,
}

impl ValidatedPackedMatrix<'_> {
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

    /// Number of validated elements.
    pub fn len(&self) -> usize {
        self.matrix.len()
    }

    /// Whether the validated allocation contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validated matrix element type.
    pub fn element(&self) -> TypeId {
        self.matrix.element()
    }

    /// Physical matrix layout.
    pub fn layout(&self) -> DenseMatrixLayout {
        self.matrix.layout()
    }

    /// Stride, in elements, between adjacent logical rows.
    pub fn row_stride_elements(&self) -> usize {
        self.matrix.row_stride_elements()
    }

    /// Stride, in elements, between adjacent logical columns.
    pub fn column_stride_elements(&self) -> usize {
        self.matrix.column_stride_elements()
    }

    /// Read-only pointer to the validated packed device allocation.
    pub fn as_ptr(&self) -> Result<*const c_void> {
        self.matrix.values.device_data_ptr().map_err(|error| error.reanchor("validated_packed_matrix::as_ptr"))
    }

    /// Exact retained stream/memory-resource binding.
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.matrix.binding_id()
    }

    /// Execution placement associated with the validated payload.
    pub fn placement(&self) -> ExecutionPlacement {
        self.matrix.placement()
    }

    /// The CUDA device the validated payload lives on.
    pub fn device(&self) -> DeviceId {
        self.matrix.device()
    }
}

impl fmt::Debug for ValidatedPackedMatrix<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedPackedMatrix")
            .field("rows", &self.rows())
            .field("columns", &self.columns())
            .field("layout", &self.layout())
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for PackedMatrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PackedMatrix")
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

fn validate_columns(columns: &[&Column]) -> Result<(usize, usize, TypeId)> {
    let Some(first) = columns.first() else {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            PACK_OPERATION,
            "dense matrix packing requires at least one input column",
        )
        .with_detail("resource", "input_columns")
        .with_detail("reason", "empty_input")
        .with_detail("remedy", "provide at least one non-null supported numeric column"));
    };

    let expected_rows = first.len().map_err(|error| error.reanchor(PACK_OPERATION))?;
    let element = first.data_type().map_err(|error| error.reanchor(PACK_OPERATION))?.id();
    matrix_element_bytes(element, PACK_OPERATION).map_err(|error| error.with_fact(FACT_COLUMN_INDEX, 0usize))?;
    for (column_index, column) in columns.iter().enumerate() {
        let actual_type = column.data_type().map_err(|error| error.reanchor(PACK_OPERATION))?.id();
        if actual_type != element {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                PACK_OPERATION,
                "dense matrix input columns must have the same supported element type",
            )
            .with_fact(FACT_COLUMN_INDEX, column_index)
            .with_fact(FACT_LEFT_TYPE_ID, actual_type)
            .with_fact(FACT_RIGHT_TYPE_ID, element)
            .with_detail("supported_elements", "Float32, Float64, Int8, UInt8"));
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
    Ok((expected_rows, columns.len(), element))
}

impl Runtime {
    /// Split a flat dense numeric column into device-resident feature series.
    /// Row-major values are gathered by stride; column-major values use contiguous slices.
    pub fn unpack_dense_matrix(
        &self,
        values: Column,
        rows: usize,
        columns: usize,
        layout: DenseMatrixLayout,
    ) -> Result<Vec<Series>> {
        if columns == 0 || rows == 0 {
            return Err(Error::new_with(ErrorCode::EmptyInput, UNPACK_OPERATION, "matrix dimensions must be positive"));
        }
        let expected = checked_matrix_element_count_in(rows, columns, UNPACK_OPERATION)?;
        let element = values.data_type().map_err(|e| e.reanchor(UNPACK_OPERATION))?.id();
        matrix_element_bytes(element, UNPACK_OPERATION)?;
        let actual = values.len().map_err(|e| e.reanchor(UNPACK_OPERATION))?;
        if actual != expected {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                UNPACK_OPERATION,
                "flat matrix length differs from its shape",
            )
            .with_fact(FACT_EXPECTED_ROWS, expected)
            .with_fact(FACT_ACTUAL_ROWS, actual));
        }
        if values.null_count().map_err(|e| e.reanchor(UNPACK_OPERATION))? != 0 {
            return Err(Error::new_with(ErrorCode::NullUnexpected, UNPACK_OPERATION, "matrix values must be non-null"));
        }
        let flat = Series::from_column(values).map_err(|e| e.reanchor(UNPACK_OPERATION))?;
        (0..columns)
            .map(|index| {
                let series = match layout {
                    DenseMatrixLayout::ColumnMajor => flat.on(Exec::RETAINED).slice(index * rows, rows),
                    DenseMatrixLayout::RowMajor => {
                        let indices = self.sequence(index as i32, columns as i32, rows)?;
                        flat.on(Exec::RETAINED).gather(&indices)
                    },
                }
                .map_err(|e| e.reanchor(UNPACK_OPERATION))?;
                series.copy_into_runtime(self).map_err(|e| e.reanchor(UNPACK_OPERATION))
            })
            .collect()
    }

    /// Copy one borrowed device-resident row-major numeric span into a packed
    /// matrix on this runtime's exact execution domain.
    ///
    /// # Safety
    ///
    /// `values` must address at least `rows * columns` readable values of `element`
    /// on this runtime's CUDA device. Prior writes must be ordered before this
    /// runtime's stream, and the allocation must remain alive until the stream
    /// completes the enqueued copy.
    pub(crate) unsafe fn pack_borrowed_flat_device_values(
        &self,
        values: *const c_void,
        input_domain: &Arc<ExecutionDomain>,
        element: TypeId,
        rows: usize,
        columns: usize,
        layout: DenseMatrixLayout,
        operation: &'static str,
    ) -> Result<PackedMatrix> {
        if rows == 0 || columns == 0 {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                operation,
                "flat matrix packing requires positive rows and columns",
            )
            .with_detail("rows", rows.to_string())
            .with_detail("columns", columns.to_string())
            .with_detail("reason", "non_positive_matrix_shape"));
        }
        checked_matrix_element_count_in(rows, columns, operation)?;
        matrix_element_bytes(element, operation)?;
        if values.is_null() {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                operation,
                "borrowed matrix values pointer must not be null",
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
            // bounds the exact element span consumed by the native call.
            unsafe {
                cudf_sys::cudf_pack_borrowed_matrix_ex(
                    values,
                    element.to_raw() as i32,
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
        Ok(PackedMatrix { values, element, rows, columns, layout })
    }

    /// Pack equal-length, non-null columns of one supported element type.
    /// The output retains this runtime's exact device, stream, and memory-resource binding.
    ///
    /// Inputs may originate from other streams on the same device; the
    /// runtime inserts device-side ordering in both directions. Cross-device
    /// inputs are rejected before launching the pack operation.
    pub fn pack_columns(&self, columns: &[&Column], layout: DenseMatrixLayout) -> Result<PackedMatrix> {
        let (rows, column_count, element) = validate_columns(columns)?;
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

        Ok(PackedMatrix { values, element, rows, columns: column_count, layout })
    }

    /// Widen a Float32 packed matrix to Float64 on this runtime's exact
    /// execution domain, preserving shape and layout. Every Float32 value is
    /// exactly representable in Float64. Any other source element type is a
    /// `TypeMismatch`. The source matrix stays valid.
    pub fn widen_packed_matrix_to_float64(&self, matrix: &PackedMatrix) -> Result<PackedMatrix> {
        if matrix.element != TypeId::Float32 {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                WIDEN_OPERATION,
                "only a Float32 matrix can be widened to Float64",
            )
            .with_fact(FACT_LEFT_TYPE_ID, matrix.element)
            .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Float64));
        }
        let view = matrix.values.view().map_err(|error| error.reanchor(WIDEN_OPERATION))?;
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(self.execution_domain()),
            [matrix.values.execution_domain()],
            WIDEN_OPERATION,
        )?;
        let values = cast_impl(
            &view,
            &DataType::new(TypeId::Float64),
            execution.output_domain(),
            execution.exec_options(),
            WIDEN_OPERATION,
        )?;
        Ok(PackedMatrix {
            values,
            element: TypeId::Float64,
            rows: matrix.rows,
            columns: matrix.columns,
            layout: matrix.layout,
        })
    }

    /// Pack one flat, non-null numeric column whose values describe a
    /// row-major `[rows, columns]` matrix into the requested dense layout.
    ///
    /// This is the nested-vector counterpart to [`pack_columns`](Self::pack_columns): it retains
    /// the runtime's exact device, stream, and memory-resource provenance and
    /// performs any row-major to column-major rearrangement entirely on that
    /// stream. It never exposes or copies vector payloads through host memory.
    pub fn pack_flat_column(
        &self,
        values: &Column,
        rows: usize,
        columns: usize,
        layout: DenseMatrixLayout,
    ) -> Result<PackedMatrix> {
        if rows == 0 || columns == 0 {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                PACK_FLAT_OPERATION,
                "flat matrix packing requires positive rows and columns",
            )
            .with_detail("rows", rows.to_string())
            .with_detail("columns", columns.to_string())
            .with_detail("reason", "non_positive_matrix_shape"));
        }
        let expected_len = checked_matrix_element_count_in(rows, columns, PACK_FLAT_OPERATION)?;
        let data_type = values.data_type().map_err(|error| error.reanchor(PACK_FLAT_OPERATION))?;
        let element = data_type.id();
        matrix_element_bytes(element, PACK_FLAT_OPERATION)?;
        let null_count = values.null_count().map_err(|error| error.reanchor(PACK_FLAT_OPERATION))?;
        if null_count != 0 {
            return Err(Error::new_with(
                ErrorCode::NullUnexpected,
                PACK_FLAT_OPERATION,
                "flat matrix packing requires non-null values",
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
                cudf_sys::cudf_pack_flat_matrix_ex(
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
        Ok(PackedMatrix { values, element, rows, columns, layout })
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
