use std::{ffi::c_void, fmt, marker::PhantomData, mem, ptr::NonNull, sync::Arc};

use cudf::{
    CudfType, DenseMatrixLayout, PackedMatrix, Runtime, ValidatedPackedMatrix,
    core::{Column, DeviceBuffer as CudfDeviceBuffer},
    interop::contract::TypeId as InteropTypeId,
    runtime::ExecutionBindingId,
};

use crate::{
    error::{Result, from_cudf, invalid_input, require_nonzero},
    resources::CuvsResources,
};

/// Element type of a cuVS device buffer or tensor.
///
/// Implemented for `f32`, `f64`, `i8`, `u8`, `i32`, and `i64`. The set is closed: the supertrait
/// [`CudfType`] is sealed, so downstream crates cannot implement it for their
/// own types, and the orphan rule prevents them from implementing this trait
/// for other foreign types.
pub trait DeviceElement: CudfType + Copy + 'static {
    /// DLPack type code used, with [`DLPACK_BITS`](Self::DLPACK_BITS) and one
    /// lane, to describe tensors of this element to cuVS.
    const DLPACK_CODE: u8;
    /// DLPack element width in bits.
    const DLPACK_BITS: u8;
    /// RAPIDS interop type a prepared cuDF column must carry to be read as a
    /// tensor of this element.
    const INTEROP_TYPE: InteropTypeId;

    /// Returns whether host values are valid native cuVS input.
    ///
    /// Floating elements require finite values; integer elements accept every value.
    fn valid_native_input(_values: &[Self]) -> bool {
        true
    }

    fn validate_native_buffer(_buffer: &CudfDeviceBuffer, _operation: &'static str) -> Result<()> {
        Ok(())
    }
}

impl DeviceElement for f32 {
    const DLPACK_CODE: u8 = cuvs_sys::DLDataTypeCode::kDLFloat as u8;
    const DLPACK_BITS: u8 = 32;
    const INTEROP_TYPE: InteropTypeId = InteropTypeId::Float32;

    fn valid_native_input(values: &[Self]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn validate_native_buffer(buffer: &CudfDeviceBuffer, operation: &'static str) -> Result<()> {
        buffer.validate_finite_float32().map_err(|error| from_cudf(operation, error))?;
        Ok(())
    }
}

impl DeviceElement for f64 {
    const DLPACK_CODE: u8 = cuvs_sys::DLDataTypeCode::kDLFloat as u8;
    const DLPACK_BITS: u8 = 64;
    const INTEROP_TYPE: InteropTypeId = InteropTypeId::Float64;

    fn valid_native_input(values: &[Self]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn validate_native_buffer(buffer: &CudfDeviceBuffer, operation: &'static str) -> Result<()> {
        buffer.validate_finite_float64().map_err(|error| from_cudf(operation, error))?;
        Ok(())
    }
}

impl DeviceElement for i8 {
    const DLPACK_CODE: u8 = cuvs_sys::DLDataTypeCode::kDLInt as u8;
    const DLPACK_BITS: u8 = 8;
    const INTEROP_TYPE: InteropTypeId = InteropTypeId::Int8;
}

impl DeviceElement for u8 {
    const DLPACK_CODE: u8 = cuvs_sys::DLDataTypeCode::kDLUInt as u8;
    const DLPACK_BITS: u8 = 8;
    const INTEROP_TYPE: InteropTypeId = InteropTypeId::UInt8;
}

impl DeviceElement for i32 {
    const DLPACK_CODE: u8 = cuvs_sys::DLDataTypeCode::kDLInt as u8;
    const DLPACK_BITS: u8 = 32;
    const INTEROP_TYPE: InteropTypeId = InteropTypeId::Int32;
}

impl DeviceElement for i64 {
    const DLPACK_CODE: u8 = cuvs_sys::DLDataTypeCode::kDLInt as u8;
    const DLPACK_BITS: u8 = 64;
    const INTEROP_TYPE: InteropTypeId = InteropTypeId::Int64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

impl From<DenseMatrixLayout> for MatrixLayout {
    fn from(layout: DenseMatrixLayout) -> Self {
        match layout {
            DenseMatrixLayout::RowMajor => Self::RowMajor,
            DenseMatrixLayout::ColumnMajor => Self::ColumnMajor,
        }
    }
}

impl From<MatrixLayout> for DenseMatrixLayout {
    fn from(layout: MatrixLayout) -> Self {
        match layout {
            MatrixLayout::RowMajor => Self::RowMajor,
            MatrixLayout::ColumnMajor => Self::ColumnMajor,
        }
    }
}

/// A typed owner for a cudf runtime-allocated device buffer.
pub struct DeviceBuffer<T: DeviceElement> {
    inner: CudfDeviceBuffer,
    runtime: Arc<Runtime>,
    len: usize,
    native_input_validated: bool,
    _element: PhantomData<T>,
}

impl<T: DeviceElement> DeviceBuffer<T> {
    pub fn new(resources: &CuvsResources, len: usize) -> Result<Self> {
        let operation = "DeviceBuffer::new";
        require_nonzero(len, operation, "len")?;
        let size_bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| invalid_input(operation, format!("allocation size overflow for {len} elements")))?;
        let runtime = Arc::clone(resources.runtime());
        let inner = runtime.device_buffer(size_bytes).map_err(|error| from_cudf(operation, error))?;
        Ok(Self { inner, runtime, len, native_input_validated: false, _element: PhantomData })
    }

    pub fn from_slice(resources: &CuvsResources, values: &[T]) -> Result<Self> {
        let operation = "DeviceBuffer::from_slice";
        require_nonzero(values.len(), operation, "values.len")?;
        if !T::valid_native_input(values) {
            return Err(invalid_input(operation, "floating native inputs must contain only finite values"));
        }
        let runtime = Arc::clone(resources.runtime());
        let inner = runtime.device_buffer_from_host_data(values).map_err(|error| from_cudf(operation, error))?;
        Ok(Self { inner, runtime, len: values.len(), native_input_validated: true, _element: PhantomData })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len * mem::size_of::<T>()
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.inner.binding_id()
    }

    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    pub(crate) fn native_input_validated(&self) -> bool {
        self.native_input_validated
    }

    pub(crate) fn validate_native_input(&mut self, operation: &'static str) -> Result<()> {
        T::validate_native_buffer(&self.inner, operation)?;
        self.native_input_validated = true;
        Ok(())
    }

    pub fn as_validated_tensor<'a>(
        &'a self,
        resources: &CuvsResources,
        shape: &[usize],
        strides: Option<&[usize]>,
        byte_offset: usize,
    ) -> Result<ValidatedDeviceTensor<'a, T>> {
        let operation = "DeviceBuffer::as_validated_tensor";
        if !self.native_input_validated {
            return Err(invalid_input(operation, "device buffer has not passed input validation"));
        }
        self.as_tensor(resources, shape, strides, byte_offset).map(ValidatedDeviceTensor::new)
    }

    pub fn as_validated_matrix(
        &self,
        resources: &CuvsResources,
        rows: usize,
        columns: usize,
    ) -> Result<ValidatedDeviceTensor<'_, T>> {
        self.as_validated_tensor(resources, &[rows, columns], None, 0)
    }

    pub fn as_tensor<'a>(
        &'a self,
        resources: &CuvsResources,
        shape: &[usize],
        strides: Option<&[usize]>,
        byte_offset: usize,
    ) -> Result<DeviceTensor<'a, T>> {
        let operation = "DeviceBuffer::as_tensor";
        ensure_buffer_binding(resources, &self.inner, operation)?;
        let data = self.inner.as_ptr().map_err(|error| from_cudf(operation, error))? as *mut c_void;
        DeviceTensor::from_parts(
            data,
            resources.device().ordinal() as i32,
            shape,
            strides,
            self.size_bytes(),
            byte_offset,
            self.binding_id(),
            operation,
        )
    }

    pub fn as_tensor_mut<'a>(
        &'a mut self,
        resources: &CuvsResources,
        shape: &[usize],
        strides: Option<&[usize]>,
        byte_offset: usize,
    ) -> Result<DeviceTensorMut<'a, T>> {
        let operation = "DeviceBuffer::as_tensor_mut";
        ensure_buffer_binding(resources, &self.inner, operation)?;
        self.native_input_validated = false;
        let size_bytes = self.size_bytes();
        let data = self.inner.as_mut_ptr().map_err(|error| from_cudf(operation, error))?;
        DeviceTensorMut::from_parts(
            data,
            resources.device().ordinal() as i32,
            shape,
            strides,
            size_bytes,
            byte_offset,
            self.binding_id(),
            operation,
        )
    }

    pub fn as_matrix(&self, resources: &CuvsResources, rows: usize, columns: usize) -> Result<DeviceTensor<'_, T>> {
        self.as_tensor(resources, &[rows, columns], None, 0)
    }

    pub fn as_matrix_mut(
        &mut self,
        resources: &CuvsResources,
        rows: usize,
        columns: usize,
    ) -> Result<DeviceTensorMut<'_, T>> {
        self.as_tensor_mut(resources, &[rows, columns], None, 0)
    }

    pub fn inner(&self) -> &CudfDeviceBuffer {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut CudfDeviceBuffer {
        self.native_input_validated = false;
        &mut self.inner
    }

    pub fn into_inner(self) -> CudfDeviceBuffer {
        self.inner
    }

    /// Adopt this allocation as one flat cudf column without copying it.
    pub fn into_column(self) -> Result<Column> {
        let operation = "DeviceBuffer::into_column";
        let Self { inner, runtime, len, .. } = self;
        runtime.column_from_device_buffer(inner, len, T::TYPE_ID).map_err(|error| from_cudf(operation, error))
    }
}

impl DeviceBuffer<f32> {
    pub(crate) fn validate_native_distances_for_valid_neighbors(
        &self,
        neighbors: &DeviceBuffer<i64>,
        sentinel: i64,
        operation: &'static str,
    ) -> Result<()> {
        self.inner
            .validate_finite_float32_unless_int64_equals(neighbors.inner(), sentinel)
            .map_err(|error| from_cudf(operation, error))
    }
}

impl DeviceBuffer<i64> {
    pub(crate) fn validate_native_index_range_or_sentinel(
        &self,
        min_inclusive: i64,
        max_exclusive: i64,
        sentinel: i64,
        operation: &'static str,
    ) -> Result<()> {
        self.inner
            .validate_int64_range_or_sentinel(min_inclusive, max_exclusive, sentinel)
            .map_err(|error| from_cudf(operation, error))
    }

    pub(crate) fn validate_native_index_range(
        &self,
        min_inclusive: i64,
        max_exclusive: i64,
        operation: &'static str,
    ) -> Result<()> {
        let proof = self
            .inner
            .validate_int64_range(min_inclusive, max_exclusive)
            .map_err(|error| from_cudf(operation, error))?;
        if proof.len() != self.len {
            return Err(invalid_input(
                operation,
                format!("index range validation covered {} values, expected {}", proof.len(), self.len),
            ));
        }
        if proof.binding_id() != self.binding_id() {
            return Err(invalid_input(operation, "index range validation used a different execution binding")
                .with_context("validated_binding_id", proof.binding_id())
                .with_context("buffer_binding_id", self.binding_id()));
        }
        if proof.placement().device != self.inner.device() {
            return Err(invalid_input(operation, "index range validation used a different CUDA device")
                .with_context("validated_device", proof.placement().device.ordinal())
                .with_context("buffer_device", self.inner.device().ordinal()));
        }
        if proof.min_inclusive() != min_inclusive || proof.max_exclusive() != max_exclusive {
            return Err(invalid_input(operation, "index validation proof covered a different value range"));
        }
        Ok(())
    }
}

impl<T: DeviceElement> fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceBuffer")
            .field("element", &std::any::type_name::<T>())
            .field("len", &self.len)
            .field("binding_id", &self.binding_id())
            .finish_non_exhaustive()
    }
}

enum MatrixStorage<T: DeviceElement> {
    Buffer(DeviceBuffer<T>),
    Packed(PackedMatrix),
}

/// An owned contiguous numeric matrix on one exact cudf execution binding.
///
/// The packed variant consumes a cudf-owned packed matrix. Because the owner
/// is exclusive, `as_tensor_mut` can safely expose the
/// allocation to native algorithms that update their input without mutating
/// any source cudf columns used to create the pack.
pub struct OwnedMatrix<T: DeviceElement> {
    storage: MatrixStorage<T>,
    rows: usize,
    columns: usize,
    layout: MatrixLayout,
    input_validated: bool,
}

impl<T: DeviceElement> OwnedMatrix<T> {
    pub fn new(resources: &CuvsResources, rows: usize, columns: usize, layout: MatrixLayout) -> Result<Self> {
        let operation = "OwnedMatrix::new";
        require_nonzero(rows, operation, "rows")?;
        require_nonzero(columns, operation, "columns")?;
        let len = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid_input(operation, "matrix element count overflowed usize"))?;
        Ok(Self {
            storage: MatrixStorage::Buffer(DeviceBuffer::new(resources, len)?),
            rows,
            columns,
            layout,
            input_validated: false,
        })
    }

    pub fn from_packed(resources: &CuvsResources, matrix: PackedMatrix) -> Result<Self> {
        let operation = "OwnedMatrix::from_packed";
        resources.validate_exact_binding(operation)?;
        if matrix.element() != T::TYPE_ID {
            return Err(invalid_input(operation, "packed matrix element does not match requested cuVS element")
                .with_context("packed_element", format!("{:?}", matrix.element()))
                .with_context("requested_element", format!("{:?}", T::TYPE_ID)));
        }
        let (rows, columns, layout) = {
            let validated = matrix.validate().map_err(|error| from_cudf(operation, error))?;
            require_nonzero(validated.rows(), operation, "rows")?;
            require_nonzero(validated.columns(), operation, "columns")?;
            if validated.binding_id() != resources.binding_id() {
                return Err(invalid_input(
                    operation,
                    "packed matrix and cuVS resources must use the same exact runtime binding",
                )
                .with_context("matrix_binding_id", validated.binding_id())
                .with_context("resource_binding_id", resources.binding_id()));
            }
            if validated.device() != resources.device() {
                return Err(invalid_input(operation, "packed matrix and cuVS resources must use the same CUDA device")
                    .with_context("matrix_device", validated.device().ordinal())
                    .with_context("resource_device", resources.device().ordinal()));
            }
            (validated.rows(), validated.columns(), validated.layout().into())
        };
        Ok(Self { storage: MatrixStorage::Packed(matrix), rows, columns, layout, input_validated: true })
    }

    pub(crate) fn from_buffer(
        resources: &CuvsResources,
        buffer: DeviceBuffer<T>,
        rows: usize,
        columns: usize,
        layout: MatrixLayout,
        operation: &'static str,
    ) -> Result<Self> {
        resources.validate_exact_binding(operation)?;
        let expected = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid_input(operation, "matrix element count overflowed usize"))?;
        if buffer.len() != expected {
            return Err(invalid_input(
                operation,
                format!("matrix buffer contains {} elements, expected {expected}", buffer.len()),
            ));
        }
        ensure_buffer_binding(resources, buffer.inner(), operation)?;
        let input_validated = buffer.native_input_validated();
        Ok(Self { storage: MatrixStorage::Buffer(buffer), rows, columns, layout, input_validated })
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
        match &self.storage {
            MatrixStorage::Buffer(buffer) => buffer.binding_id(),
            MatrixStorage::Packed(matrix) => matrix.binding_id(),
        }
    }

    pub fn is_input_validated(&self) -> bool {
        self.input_validated
    }

    pub fn as_validated_tensor<'a>(&'a self, resources: &CuvsResources) -> Result<ValidatedDeviceTensor<'a, T>> {
        let operation = "OwnedMatrix::as_validated_tensor";
        if !self.input_validated {
            return Err(invalid_input(operation, "owned matrix has not passed input validation"));
        }
        self.as_tensor(resources).map(ValidatedDeviceTensor::new)
    }

    pub(crate) fn as_tensor<'a>(&'a self, resources: &CuvsResources) -> Result<DeviceTensor<'a, T>> {
        let operation = "OwnedMatrix::as_tensor";
        self.validate_binding(resources, operation)?;
        let data = match &self.storage {
            MatrixStorage::Buffer(buffer) => {
                buffer.inner().as_ptr().map_err(|error| from_cudf(operation, error))? as *mut c_void
            },
            MatrixStorage::Packed(matrix) => {
                matrix.as_column().device_data_ptr().map_err(|error| from_cudf(operation, error))? as *mut c_void
            },
        };
        self.tensor_from_ptr(data, operation)
    }

    pub(crate) fn as_tensor_mut<'a>(&'a mut self, resources: &CuvsResources) -> Result<DeviceTensorMut<'a, T>> {
        let operation = "OwnedMatrix::as_tensor_mut";
        self.validate_binding(resources, operation)?;
        self.input_validated = false;
        let data = match &mut self.storage {
            MatrixStorage::Buffer(buffer) => {
                buffer.inner_mut().as_mut_ptr().map_err(|error| from_cudf(operation, error))?
            },
            // The packed matrix is exclusively owned by `self`; no safe API
            // can retain an alias after it was moved into this owner.
            MatrixStorage::Packed(matrix) => {
                matrix.as_column().device_data_ptr().map_err(|error| from_cudf(operation, error))? as *mut c_void
            },
        };
        let strides = self.strides();
        DeviceTensorMut::from_parts(
            data,
            resources.device().ordinal() as i32,
            &[self.rows, self.columns],
            strides.as_ref().map(<[usize; 2]>::as_slice),
            self.byte_len(),
            0,
            self.binding_id(),
            operation,
        )
    }

    pub(crate) fn validate_native_output(&mut self, resources: &CuvsResources, operation: &'static str) -> Result<()> {
        self.validate_binding(resources, operation)?;
        match &mut self.storage {
            MatrixStorage::Buffer(buffer) => buffer.validate_native_input(operation)?,
            MatrixStorage::Packed(matrix) => {
                let proof = matrix.validate().map_err(|error| from_cudf(operation, error))?;
                if proof.len() != self.rows * self.columns {
                    return Err(invalid_input(
                        operation,
                        format!(
                            "output validation covered {} values, expected {}",
                            proof.len(),
                            self.rows * self.columns
                        ),
                    ));
                }
                if proof.binding_id() != resources.binding_id() {
                    return Err(invalid_input(operation, "output validation used a different execution binding")
                        .with_context("validated_binding_id", proof.binding_id())
                        .with_context("resource_binding_id", resources.binding_id()));
                }
            },
        }
        self.input_validated = true;
        Ok(())
    }

    fn validate_binding(&self, resources: &CuvsResources, operation: &'static str) -> Result<()> {
        resources.validate_exact_binding(operation)?;
        if self.binding_id() != resources.binding_id() {
            return Err(invalid_input(operation, "owned matrix and cuVS resources must use the same exact binding")
                .with_context("matrix_binding_id", self.binding_id())
                .with_context("resource_binding_id", resources.binding_id()));
        }
        Ok(())
    }

    fn byte_len(&self) -> usize {
        self.rows * self.columns * mem::size_of::<T>()
    }

    fn strides(&self) -> Option<[usize; 2]> {
        match self.layout {
            MatrixLayout::RowMajor => None,
            MatrixLayout::ColumnMajor => Some([1, self.rows]),
        }
    }

    fn tensor_from_ptr<'a>(&'a self, data: *mut c_void, operation: &'static str) -> Result<DeviceTensor<'a, T>> {
        let strides = self.strides();
        DeviceTensor::from_parts(
            data,
            self.device_ordinal(),
            &[self.rows, self.columns],
            strides.as_ref().map(<[usize; 2]>::as_slice),
            self.byte_len(),
            0,
            self.binding_id(),
            operation,
        )
    }

    fn device_ordinal(&self) -> i32 {
        match &self.storage {
            MatrixStorage::Buffer(buffer) => buffer.inner().device().ordinal() as i32,
            MatrixStorage::Packed(matrix) => matrix.device().ordinal() as i32,
        }
    }
}

impl<T: DeviceElement> fmt::Debug for OwnedMatrix<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedMatrix")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("layout", &self.layout)
            .field("binding_id", &self.binding_id())
            .finish_non_exhaustive()
    }
}

fn ensure_buffer_binding(resources: &CuvsResources, buffer: &CudfDeviceBuffer, operation: &'static str) -> Result<()> {
    resources.validate_exact_binding(operation)?;
    if buffer.binding_id() != resources.binding_id() {
        return Err(invalid_input(
            operation,
            "device buffer and cuVS resources must use the same exact runtime binding",
        )
        .with_context("buffer_binding", format!("{:?}", buffer.binding_id()))
        .with_context("resource_binding", format!("{:?}", resources.binding_id())));
    }
    if buffer.device() != resources.device() {
        return Err(invalid_input(operation, "device buffer and cuVS resources must use the same CUDA device")
            .with_context("buffer_device", buffer.device().ordinal())
            .with_context("resource_device", resources.device().ordinal()));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct TensorParts<'a, T: DeviceElement> {
    pub(crate) data: NonNull<c_void>,
    pub(crate) device: i32,
    pub(crate) shape: &'a [i64],
    pub(crate) strides: Option<&'a [i64]>,
    pub(crate) byte_len: usize,
    pub(crate) byte_offset: usize,
    _element: PhantomData<T>,
}

/// A lifetime-bound, read-only CUDA tensor descriptor.
pub struct DeviceTensor<'a, T: DeviceElement> {
    data: NonNull<c_void>,
    device: i32,
    shape: Box<[i64]>,
    strides: Option<Box<[i64]>>,
    byte_len: usize,
    byte_offset: usize,
    binding_id: ExecutionBindingId,
    _borrow: PhantomData<&'a T>,
}

impl<T: DeviceElement> DeviceTensor<'_, T> {
    #[expect(clippy::too_many_arguments, reason = "all tensor descriptor parts are required")]
    pub(crate) fn from_parts(
        data: *mut c_void,
        device: i32,
        shape: &[usize],
        strides: Option<&[usize]>,
        byte_len: usize,
        byte_offset: usize,
        binding_id: ExecutionBindingId,
        operation: &'static str,
    ) -> Result<Self> {
        let (data, shape, strides) = validate_tensor::<T>(data, shape, strides, byte_len, byte_offset, operation)?;
        Ok(Self { data, device, shape, strides, byte_len, byte_offset, binding_id, _borrow: PhantomData })
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn strides(&self) -> Option<&[i64]> {
        self.strides.as_deref()
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.binding_id
    }

    pub fn len(&self) -> usize {
        self.shape.iter().map(|value| *value as usize).product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn rows(&self) -> Result<usize> {
        require_matrix(self.shape(), "DeviceTensor::rows").map(|shape| shape[0] as usize)
    }

    pub fn columns(&self) -> Result<usize> {
        require_matrix(self.shape(), "DeviceTensor::columns").map(|shape| shape[1] as usize)
    }

    pub(crate) fn parts(&self) -> TensorParts<'_, T> {
        TensorParts {
            data: self.data,
            device: self.device,
            shape: &self.shape,
            strides: self.strides.as_deref(),
            byte_len: self.byte_len,
            byte_offset: self.byte_offset,
            _element: PhantomData,
        }
    }
}

/// A read-only tensor with finite-scanned floating values or validated integer shape.
pub struct ValidatedDeviceTensor<'a, T: DeviceElement> {
    inner: DeviceTensor<'a, T>,
}

impl<'a, T: DeviceElement> ValidatedDeviceTensor<'a, T> {
    pub fn from_packed(resources: &CuvsResources, matrix: &'a ValidatedPackedMatrix<'_>) -> Result<Self> {
        let operation = "ValidatedDeviceTensor::from_packed";
        resources.validate_exact_binding(operation)?;
        if matrix.element() != T::TYPE_ID {
            return Err(invalid_input(operation, "packed matrix element does not match requested cuVS element")
                .with_context("packed_element", format!("{:?}", matrix.element()))
                .with_context("requested_element", format!("{:?}", T::TYPE_ID)));
        }
        if matrix.binding_id() != resources.binding_id() {
            return Err(invalid_input(
                operation,
                "packed matrix and cuVS resources must use the same exact runtime binding",
            )
            .with_context("tensor_binding_id", matrix.binding_id())
            .with_context("resource_binding_id", resources.binding_id()));
        }
        if matrix.device() != resources.device() {
            return Err(invalid_input(operation, "packed matrix and cuVS resources must use the same CUDA device")
                .with_context("tensor_device", matrix.device().ordinal())
                .with_context("resource_device", resources.device().ordinal()));
        }
        let data = matrix.as_ptr().map_err(|error| from_cudf(operation, error))? as *mut c_void;
        let strides = match MatrixLayout::from(matrix.layout()) {
            MatrixLayout::RowMajor => None,
            MatrixLayout::ColumnMajor => Some([1, matrix.rows()]),
        };
        DeviceTensor::from_parts(
            data,
            resources.device().ordinal() as i32,
            &[matrix.rows(), matrix.columns()],
            strides.as_ref().map(<[usize; 2]>::as_slice),
            matrix.len() * mem::size_of::<T>(),
            0,
            matrix.binding_id(),
            operation,
        )
        .map(Self::new)
    }

    pub(crate) fn new(inner: DeviceTensor<'a, T>) -> Self {
        Self { inner }
    }

    pub(crate) fn as_tensor(&self) -> &DeviceTensor<'a, T> {
        &self.inner
    }

    pub fn device(&self) -> i32 {
        self.inner.device()
    }

    pub fn shape(&self) -> &[i64] {
        self.inner.shape()
    }

    pub fn strides(&self) -> Option<&[i64]> {
        self.inner.strides()
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.inner.binding_id()
    }

    pub fn rows(&self) -> Result<usize> {
        self.inner.rows()
    }

    pub fn columns(&self) -> Result<usize> {
        self.inner.columns()
    }
}

impl<T: DeviceElement> fmt::Debug for ValidatedDeviceTensor<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("ValidatedDeviceTensor").field(&self.inner).finish()
    }
}

impl<T: DeviceElement> fmt::Debug for DeviceTensor<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceTensor")
            .field("element", &std::any::type_name::<T>())
            .field("device", &self.device)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("byte_len", &self.byte_len)
            .field("byte_offset", &self.byte_offset)
            .field("binding_id", &self.binding_id)
            .finish_non_exhaustive()
    }
}

/// A lifetime-bound, exclusive CUDA tensor descriptor.
pub struct DeviceTensorMut<'a, T: DeviceElement> {
    inner: DeviceTensor<'a, T>,
    _exclusive: PhantomData<&'a mut T>,
}

impl<T: DeviceElement> DeviceTensorMut<'_, T> {
    #[expect(clippy::too_many_arguments, reason = "all tensor descriptor parts are required")]
    fn from_parts(
        data: *mut c_void,
        device: i32,
        shape: &[usize],
        strides: Option<&[usize]>,
        byte_len: usize,
        byte_offset: usize,
        binding_id: ExecutionBindingId,
        operation: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            inner: DeviceTensor::from_parts(
                data,
                device,
                shape,
                strides,
                byte_len,
                byte_offset,
                binding_id,
                operation,
            )?,
            _exclusive: PhantomData,
        })
    }

    pub fn as_const(&self) -> DeviceTensor<'_, T> {
        DeviceTensor {
            data: self.inner.data,
            device: self.inner.device,
            shape: self.inner.shape.clone(),
            strides: self.inner.strides.clone(),
            byte_len: self.inner.byte_len,
            byte_offset: self.inner.byte_offset,
            binding_id: self.inner.binding_id,
            _borrow: PhantomData,
        }
    }

    pub fn device(&self) -> i32 {
        self.inner.device()
    }

    pub fn shape(&self) -> &[i64] {
        self.inner.shape()
    }

    pub fn strides(&self) -> Option<&[i64]> {
        self.inner.strides()
    }

    pub fn byte_len(&self) -> usize {
        self.inner.byte_len()
    }

    pub fn byte_offset(&self) -> usize {
        self.inner.byte_offset()
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.inner.binding_id()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn rows(&self) -> Result<usize> {
        self.inner.rows()
    }

    pub fn columns(&self) -> Result<usize> {
        self.inner.columns()
    }

    pub(crate) fn parts(&self) -> TensorParts<'_, T> {
        self.inner.parts()
    }
}

impl<T: DeviceElement> fmt::Debug for DeviceTensorMut<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

type ValidatedTensorParts = (NonNull<c_void>, Box<[i64]>, Option<Box<[i64]>>);

fn validate_tensor<T: DeviceElement>(
    data: *mut c_void,
    shape: &[usize],
    strides: Option<&[usize]>,
    byte_len: usize,
    byte_offset: usize,
    operation: &'static str,
) -> Result<ValidatedTensorParts> {
    if shape.is_empty() {
        return Err(invalid_input(operation, "tensor rank must be greater than zero"));
    }
    if strides.is_some_and(|strides| strides.len() != shape.len()) {
        return Err(invalid_input(operation, "tensor strides must have the same rank as shape"));
    }
    if !byte_offset.is_multiple_of(mem::align_of::<T>()) {
        return Err(invalid_input(operation, "tensor byte offset is not aligned for its element type"));
    }

    let mut logical_elements = 1usize;
    let mut shape_i64 = Vec::with_capacity(shape.len());
    for &dimension in shape {
        require_nonzero(dimension, operation, "shape dimension")?;
        logical_elements = logical_elements
            .checked_mul(dimension)
            .ok_or_else(|| invalid_input(operation, "tensor logical element count overflowed usize"))?;
        shape_i64.push(i64::try_from(dimension).map_err(|_| invalid_input(operation, "tensor dimension exceeds i64"))?);
    }

    let strides_i64 = strides
        .map(|strides| {
            strides
                .iter()
                .map(|&stride| {
                    require_nonzero(stride, operation, "stride")?;
                    i64::try_from(stride).map_err(|_| invalid_input(operation, "tensor stride exceeds i64"))
                })
                .collect::<Result<Vec<_>>>()
                .map(Vec::into_boxed_slice)
        })
        .transpose()?;
    let last_element = match strides {
        Some(strides) => shape.iter().zip(strides).try_fold(0usize, |offset, (&dimension, &stride)| {
            offset
                .checked_add(
                    (dimension - 1)
                        .checked_mul(stride)
                        .ok_or_else(|| invalid_input(operation, "tensor stride footprint overflowed usize"))?,
                )
                .ok_or_else(|| invalid_input(operation, "tensor stride footprint overflowed usize"))
        })?,
        None => logical_elements - 1,
    };
    let footprint = last_element
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(mem::size_of::<T>()))
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or_else(|| invalid_input(operation, "tensor byte footprint overflowed usize"))?;
    if footprint > byte_len {
        return Err(invalid_input(
            operation,
            format!("tensor footprint {footprint} bytes exceeds backing buffer length {byte_len}"),
        ));
    }
    let data =
        NonNull::new(data).ok_or_else(|| invalid_input(operation, "non-empty CUDA tensor data must not be null"))?;
    Ok((data, shape_i64.into_boxed_slice(), strides_i64))
}

pub(crate) fn require_matrix<'a>(shape: &'a [i64], operation: &'static str) -> Result<&'a [i64]> {
    if shape.len() != 2 {
        return Err(invalid_input(operation, format!("expected a rank-2 matrix, found rank {}", shape.len())));
    }
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    fn resources() -> (Arc<Runtime>, CuvsResources) {
        let runtime = Arc::new(Runtime::new().expect("runtime"));
        let resources = CuvsResources::for_runtime(Arc::clone(&runtime)).expect("cuVS resources");
        (runtime, resources)
    }

    #[test]
    fn native_output_validation_rejects_non_finite_buffer_before_restoring_proof() {
        let (runtime, resources) = resources();
        let inner = runtime.device_buffer_from_host_data(&[f32::INFINITY]).expect("raw native output");
        let buffer =
            DeviceBuffer::<f32> { inner, runtime, len: 1, native_input_validated: false, _element: PhantomData };
        let mut matrix =
            OwnedMatrix::from_buffer(&resources, buffer, 1, 1, MatrixLayout::RowMajor, "test").expect("matrix");

        let error = matrix
            .validate_native_output(&resources, "test")
            .expect_err("non-finite native output must not restore the finite-input proof");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(!matrix.is_input_validated());
    }

    #[test]
    fn float64_native_output_rejects_non_finite_values() {
        let (runtime, _resources) = resources();
        let inner = runtime.device_buffer_from_host_data(&[f64::NAN]).expect("raw native output");
        let mut buffer =
            DeviceBuffer::<f64> { inner, runtime, len: 1, native_input_validated: false, _element: PhantomData };
        let error = buffer.validate_native_input("test").expect_err("non-finite Float64 output must be rejected");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(!buffer.native_input_validated());
    }

    #[test]
    fn tensor_stride_footprint_overflow_returns_an_error() {
        let half_max = usize::MAX / 2;
        let error = validate_tensor::<f32>(
            NonNull::<f32>::dangling().as_ptr().cast(),
            &[2, 2, 2],
            Some(&[half_max, half_max, 1]),
            usize::MAX,
            0,
            "test",
        )
        .expect_err("an unrepresentable strided footprint must not panic or wrap");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(error.message().contains("footprint overflowed"));
    }
}
