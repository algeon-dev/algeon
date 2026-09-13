use std::{marker::PhantomData, mem, sync::Arc};

use cudf::{
    DenseMatrixLayout, Runtime, Series, ValidatedPackedFloat32Matrix,
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
    binding_id: ExecutionBindingId,
    _owner: PhantomData<&'a ()>,
}

impl<'a> DeviceMatrix<'a> {
    pub fn from_validated(matrix: &'a ValidatedPackedFloat32Matrix<'_>) -> Result<Self> {
        Ok(Self {
            ptr: matrix.as_ptr().map_err(|e| from_cudf("DeviceMatrix::from_validated", e))?,
            rows: matrix.rows(),
            columns: matrix.columns(),
            layout: matrix.layout().into(),
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
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.binding_id
    }
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }
}

pub struct OwnedFloat32Matrix {
    inner: DeviceBuffer<f32>,
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
        Ok(Self { inner: DeviceBuffer::new(resources, len)?, rows, columns, layout })
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
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.inner.binding_id()
    }
    pub fn as_mut_ptr(&mut self) -> Result<*mut std::ffi::c_void> {
        self.inner.as_mut_ptr()
    }
    pub fn as_ptr(&self) -> Result<*const std::ffi::c_void> {
        self.inner.as_ptr()
    }
    pub fn validate_finite(&mut self, op: &'static str) -> Result<()> {
        self.inner.validate_finite(op)
    }
}
