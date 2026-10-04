use std::{marker::PhantomData, mem, sync::Arc};

use cudf::{
    DenseMatrixLayout, PackedMatrix, Runtime, Series, TypeId, ValidatedPackedMatrix,
    core::{Column, DeviceBuffer as CudfDeviceBuffer},
    runtime::ExecutionBindingId,
};

use crate::{
    error::{ErrorCode, FailurePhase, Result, error, from_cudf, from_cudf_output, invalid_input},
    resources::CumlResources,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

impl From<DenseMatrixLayout> for MatrixLayout {
    fn from(v: DenseMatrixLayout) -> Self {
        match v {
            DenseMatrixLayout::RowMajor => Self::RowMajor,
            DenseMatrixLayout::ColumnMajor => Self::ColumnMajor,
        }
    }
}

impl From<MatrixLayout> for DenseMatrixLayout {
    fn from(v: MatrixLayout) -> Self {
        match v {
            MatrixLayout::RowMajor => Self::RowMajor,
            MatrixLayout::ColumnMajor => Self::ColumnMajor,
        }
    }
}

pub trait DeviceElement: cudf::CudfType + Copy + 'static {}
impl DeviceElement for f32 {}
impl DeviceElement for f64 {}
impl DeviceElement for i32 {}
impl DeviceElement for i64 {}

pub struct DeviceBuffer<T: DeviceElement> {
    inner: CudfDeviceBuffer,
    runtime: Arc<Runtime>,
    len: usize,
    validated: bool,
    _type: PhantomData<T>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<T: DeviceElement> DeviceBuffer<T> {
    pub fn new(resources: &CumlResources, len: usize) -> Result<Self> {
        let op = "DeviceBuffer::new";
        if len == 0 {
            return Err(invalid_input(op, "len must be greater than zero"));
        }
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| error(ErrorCode::SizeOverflow, FailurePhase::Preflight, op, "allocation size overflow"))?;
        let runtime = Arc::clone(resources.runtime());
        let inner = runtime.device_buffer(bytes).map_err(|e| from_cudf(op, e))?;
        Ok(Self { inner, runtime, len, validated: false, _type: PhantomData, _not_send_sync: PhantomData })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.inner.binding_id()
    }
    pub fn as_ptr(&self) -> Result<*const std::ffi::c_void> {
        self.inner.as_ptr().map_err(|e| from_cudf("DeviceBuffer::as_ptr", e))
    }
    pub(crate) fn as_mut_ptr(&mut self) -> Result<*mut std::ffi::c_void> {
        // A native writer invalidates any earlier proof about this buffer's contents.
        self.validated = false;
        self.inner.as_mut_ptr().map_err(|e| from_cudf("DeviceBuffer::as_mut_ptr", e))
    }
    /// Transfers a native output only after the corresponding content check succeeded.
    pub fn into_column(self) -> Result<Column> {
        let op = "DeviceBuffer::into_column";
        if !self.validated {
            return Err(error(
                ErrorCode::InvariantViolation,
                FailurePhase::OutputValidation,
                op,
                "native output was not validated",
            ));
        }
        self.runtime.column_from_device_buffer(self.inner, self.len, T::TYPE_ID).map_err(|e| from_cudf(op, e))
    }
    pub fn into_series(self) -> Result<Series> {
        Series::from_column(self.into_column()?).map_err(|e| from_cudf("DeviceBuffer::into_series", e))
    }
    /// Transfers a native output that leaves some elements unspecified. The
    /// caller must validate every element it consumes from the column.
    pub(crate) fn into_partially_written_column(self) -> Result<Column> {
        let op = "DeviceBuffer::into_partially_written_column";
        self.runtime.column_from_device_buffer(self.inner, self.len, T::TYPE_ID).map_err(|e| from_cudf(op, e))
    }
}

impl DeviceBuffer<f32> {
    pub(crate) fn validate_finite(&mut self, op: &'static str) -> Result<()> {
        let proof =
            self.inner.validate_finite_float32().map_err(|e| from_cudf_output(op, e, ErrorCode::OutputNotFinite))?;
        if proof.len() != self.len {
            return Err(error(
                ErrorCode::InvariantViolation,
                FailurePhase::OutputValidation,
                op,
                "finite validation covered an unexpected length",
            ));
        }
        self.validated = true;
        Ok(())
    }
}

impl DeviceBuffer<f64> {
    pub(crate) fn validate_finite_float64(&mut self, op: &'static str) -> Result<()> {
        self.inner.validate_finite_float64().map_err(|e| from_cudf_output(op, e, ErrorCode::OutputNotFinite))?;
        self.validated = true;
        Ok(())
    }
}

impl DeviceBuffer<i64> {
    pub(crate) fn validate_range(&mut self, min: i64, max: i64, op: &'static str) -> Result<()> {
        let proof = self
            .inner
            .validate_int64_range(min, max)
            .map_err(|e| from_cudf_output(op, e, ErrorCode::OutputOutOfRange))?;
        if proof.len() != self.len {
            return Err(error(
                ErrorCode::InvariantViolation,
                FailurePhase::OutputValidation,
                op,
                "range validation covered an unexpected length",
            ));
        }
        self.validated = true;
        Ok(())
    }
}

pub struct DeviceMatrix<'a> {
    ptr: *const std::ffi::c_void,
    rows: usize,
    columns: usize,
    layout: MatrixLayout,
    element: TypeId,
    binding_id: ExecutionBindingId,
    _owner: PhantomData<&'a ()>,
}

impl<'a> DeviceMatrix<'a> {
    pub fn from_validated(matrix: &'a ValidatedPackedMatrix<'_>) -> Result<Self> {
        Self::with_element(matrix, TypeId::Float32, "DeviceMatrix::from_validated")
    }
    /// Views Float64 values for consumers with a double-precision native
    /// path, such as forest inference on a double-threshold model.
    pub fn from_validated_float64(matrix: &'a ValidatedPackedMatrix<'_>) -> Result<Self> {
        Self::with_element(matrix, TypeId::Float64, "DeviceMatrix::from_validated_float64")
    }
    fn with_element(matrix: &'a ValidatedPackedMatrix<'_>, element: TypeId, op: &'static str) -> Result<Self> {
        if matrix.element() != element {
            return Err(invalid_input(op, format!("matrix values must be {element:?}")));
        }
        Ok(Self {
            ptr: matrix.as_ptr().map_err(|e| from_cudf(op, e))?,
            rows: matrix.rows(),
            columns: matrix.columns(),
            layout: matrix.layout().into(),
            element,
            binding_id: matrix.binding_id(),
            _owner: PhantomData,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }
    pub fn element(&self) -> TypeId {
        self.element
    }
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.binding_id
    }
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }
}

enum MatrixStorage {
    Buffer(DeviceBuffer<f32>),
    Packed(PackedMatrix),
}

/// An owned Float32 matrix bound to one exact cuDF execution binding.
/// The packed allocation is exclusively owned, so native writers cannot mutate source columns.
pub struct OwnedFloat32Matrix {
    storage: MatrixStorage,
    rows: usize,
    columns: usize,
    layout: MatrixLayout,
}

impl OwnedFloat32Matrix {
    pub fn new(resources: &CumlResources, rows: usize, columns: usize, layout: MatrixLayout) -> Result<Self> {
        let len = rows.checked_mul(columns).ok_or_else(|| {
            error(ErrorCode::SizeOverflow, FailurePhase::Preflight, "OwnedFloat32Matrix::new", "matrix shape overflow")
        })?;
        if rows == 0 || columns == 0 {
            return Err(invalid_input("OwnedFloat32Matrix::new", "dimensions must be nonzero"));
        }
        Ok(Self { storage: MatrixStorage::Buffer(DeviceBuffer::new(resources, len)?), rows, columns, layout })
    }
    pub fn from_packed(resources: &CumlResources, matrix: PackedMatrix) -> Result<Self> {
        const OP: &str = "OwnedFloat32Matrix::from_packed";
        resources.validate_exact_binding(OP)?;
        if matrix.element() != TypeId::Float32 {
            return Err(error(
                ErrorCode::UnsupportedDtype,
                FailurePhase::Preflight,
                OP,
                "cuML requires Float32 matrix values",
            ));
        }
        let validated = matrix.validate().map_err(|e| from_cudf(OP, e))?;
        if validated.rows() == 0 || validated.columns() == 0 {
            return Err(invalid_input(OP, "dimensions must be nonzero"));
        }
        if validated.binding_id() != resources.binding_id() || validated.device() != resources.device() {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                OP,
                "packed matrix and cuML resources must share an exact binding",
            )
            .with_detail("matrix_binding_id", validated.binding_id())
            .with_detail("resource_binding_id", resources.binding_id())
            .with_detail("matrix_device", validated.device().ordinal())
            .with_detail("resource_device", resources.device().ordinal()));
        }
        let (rows, columns, layout) = (validated.rows(), validated.columns(), validated.layout().into());
        Ok(Self { storage: MatrixStorage::Packed(matrix), rows, columns, layout })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }
    /// Transfer a validated native output to cuDF without copying its device allocation.
    pub fn into_column(self) -> Result<Column> {
        match self.storage {
            MatrixStorage::Buffer(buffer) => buffer.into_column(),
            MatrixStorage::Packed(_) => {
                Err(invalid_input("OwnedFloat32Matrix::into_column", "packed input is not a native output"))
            },
        }
    }
    pub fn binding_id(&self) -> ExecutionBindingId {
        match &self.storage {
            MatrixStorage::Buffer(buffer) => buffer.binding_id(),
            MatrixStorage::Packed(matrix) => matrix.binding_id(),
        }
    }
    pub fn as_mut_ptr(&mut self) -> Result<*mut std::ffi::c_void> {
        match &mut self.storage {
            MatrixStorage::Buffer(buffer) => buffer.as_mut_ptr(),
            // The packed allocation is exclusively owned by self; no safe alias survives the move.
            MatrixStorage::Packed(matrix) => matrix
                .as_column()
                .device_data_ptr()
                .map(|ptr| ptr as *mut std::ffi::c_void)
                .map_err(|e| from_cudf("OwnedFloat32Matrix::as_mut_ptr", e)),
        }
    }
    pub fn as_ptr(&self) -> Result<*const std::ffi::c_void> {
        match &self.storage {
            MatrixStorage::Buffer(buffer) => buffer.as_ptr(),
            MatrixStorage::Packed(matrix) => {
                matrix.as_column().device_data_ptr().map_err(|e| from_cudf("OwnedFloat32Matrix::as_ptr", e))
            },
        }
    }
    pub fn validate_finite(&mut self, op: &'static str) -> Result<()> {
        match &mut self.storage {
            MatrixStorage::Buffer(buffer) => buffer.validate_finite(op),
            MatrixStorage::Packed(matrix) => {
                let validated = matrix.validate().map_err(|e| from_cudf_output(op, e, ErrorCode::OutputNotFinite))?;
                if validated.len() != self.rows * self.columns {
                    return Err(error(
                        ErrorCode::InvariantViolation,
                        FailurePhase::OutputValidation,
                        op,
                        "finite validation covered an unexpected length",
                    ));
                }
                Ok(())
            },
        }
    }
}
