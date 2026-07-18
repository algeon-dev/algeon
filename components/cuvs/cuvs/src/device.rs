use std::{ffi::c_void, marker::PhantomData, mem, ptr::NonNull};

use crate::{
    cuda::synchronize_device,
    dlpack::DlpackTensor,
    error::{Result, check_cuda, invalid_input, invalid_state, require_nonzero},
};

pub trait DeviceCopy: Copy + Default + 'static {}

impl DeviceCopy for f32 {}
impl DeviceCopy for i32 {}
impl DeviceCopy for i64 {}

#[derive(Debug)]
pub struct DeviceBuffer<T: DeviceCopy> {
    ptr: NonNull<T>,
    len: usize,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    pub fn new(len: usize) -> Result<Self> {
        require_nonzero(len, "DeviceBuffer::new", "len")?;
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| invalid_input("DeviceBuffer::new", format!("allocation size overflow for len {len}")))?;
        let mut raw = std::ptr::null_mut::<c_void>();
        unsafe { check_cuda(cuvs_sys::cudaMalloc(&mut raw, bytes), "DeviceBuffer::new")? };
        let ptr = NonNull::new(raw.cast::<T>())
            .ok_or_else(|| invalid_state("DeviceBuffer::new", "cudaMalloc returned a null pointer"))?;
        Ok(Self { ptr, len, _not_send_sync: PhantomData })
    }

    pub fn from_slice(values: &[T]) -> Result<Self> {
        require_nonzero(values.len(), "DeviceBuffer::from_slice", "values.len")?;
        let buffer = Self::new(values.len())?;
        buffer.copy_from(values)?;
        Ok(buffer)
    }

    pub fn copy_from(&self, values: &[T]) -> Result<()> {
        if values.len() != self.len {
            return Err(invalid_input(
                "DeviceBuffer::copy_from",
                format!("source length {} does not match device length {}", values.len(), self.len),
            ));
        }
        let bytes = self.len * mem::size_of::<T>();
        unsafe {
            check_cuda(
                cuvs_sys::cudaMemcpy(
                    self.ptr.as_ptr().cast::<c_void>(),
                    values.as_ptr().cast::<c_void>(),
                    bytes,
                    cuvs_sys::cudaMemcpyHostToDevice,
                ),
                "DeviceBuffer::copy_from",
            )
        }
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        synchronize_device()?;
        let mut values = vec![T::default(); self.len];
        let bytes = self.len * mem::size_of::<T>();
        unsafe {
            check_cuda(
                cuvs_sys::cudaMemcpy(
                    values.as_mut_ptr().cast::<c_void>(),
                    self.ptr.as_ptr().cast::<c_void>(),
                    bytes,
                    cuvs_sys::cudaMemcpyDeviceToHost,
                ),
                "DeviceBuffer::to_vec",
            )?;
        }
        Ok(values)
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = cuvs_sys::cudaFree(self.ptr.as_ptr().cast::<c_void>());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug)]
pub struct DeviceMatrixF32 {
    data: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    layout: MatrixLayout,
}

impl DeviceMatrixF32 {
    pub fn from_host(rows: usize, cols: usize, layout: MatrixLayout, values: &[f32]) -> Result<Self> {
        validate_matrix_len(rows, cols, values.len(), "DeviceMatrixF32::from_host")?;
        Ok(Self { data: DeviceBuffer::from_slice(values)?, rows, cols, layout })
    }

    pub fn new_uninit(rows: usize, cols: usize, layout: MatrixLayout) -> Result<Self> {
        validate_matrix_len(rows, cols, rows.saturating_mul(cols), "DeviceMatrixF32::new_uninit")?;
        Ok(Self { data: DeviceBuffer::new(rows * cols)?, rows, cols, layout })
    }

    pub fn as_view(&self) -> DeviceMatrixViewF32<'_> {
        DeviceMatrixViewF32 {
            ptr: NonNull::new(self.data.as_mut_ptr()).expect("device buffers are non-null"),
            rows: self.rows,
            cols: self.cols,
            layout: self.layout,
            _marker: PhantomData,
        }
    }

    pub fn to_host(&self) -> Result<Vec<f32>> {
        self.data.to_vec()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceMatrixViewF32<'a> {
    ptr: NonNull<f32>,
    rows: usize,
    cols: usize,
    layout: MatrixLayout,
    _marker: PhantomData<&'a f32>,
}

impl<'a> DeviceMatrixViewF32<'a> {
    /// Constructs a view from a raw device pointer.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a contiguous, properly aligned region of GPU
    ///   memory holding at least `rows * cols` valid `f32` elements.
    /// - The memory behind `ptr` must remain valid for the lifetime of this
    ///   view (`'a`).
    /// - The caller must ensure no aliasing or data races occur over the
    ///   referenced memory for the lifetime of this view.
    pub unsafe fn from_raw_parts(ptr: *mut f32, rows: usize, cols: usize, layout: MatrixLayout) -> Result<Self> {
        validate_matrix_len(rows, cols, rows.saturating_mul(cols), "DeviceMatrixViewF32::from_raw_parts")?;
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| invalid_input("DeviceMatrixViewF32::from_raw_parts", "ptr must not be null"))?;
        Ok(Self { ptr, rows, cols, layout, _marker: PhantomData })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.ptr.as_ptr()
    }

    pub fn to_dlpack(&self) -> Result<DlpackTensor> {
        DlpackTensor::device_matrix_f32(self.ptr.as_ptr(), self.rows, self.cols, self.layout)
    }
}

pub(crate) fn validate_matrix_len(rows: usize, cols: usize, len: usize, operation: &'static str) -> Result<()> {
    require_nonzero(rows, operation, "rows")?;
    require_nonzero(cols, operation, "cols")?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid_input(operation, format!("matrix size overflow for shape [{rows}, {cols}]")))?;
    if len != expected {
        return Err(invalid_input(
            operation,
            format!("matrix data length {len} does not match shape [{rows}, {cols}] ({expected})"),
        ));
    }
    Ok(())
}
