//! Type-erased device and host array wrappers.
//!
//! These types wrap the cuGraph C API's type-erased arrays and provide a safe
//! Rust interface with proper ownership semantics:
//!
//! - [`DeviceArray`] — owning device array, freed on drop
//! - [`DeviceArrayView`] — non-owning view into device memory, lifetime-tied
//! - [`HostArrayView`] — non-owning view into host memory, lifetime-tied

use std::marker::PhantomData;
use std::ptr;

use cugraph_sys::{
    cugraph_error_t, cugraph_type_erased_device_array_create,
    cugraph_type_erased_device_array_free, cugraph_type_erased_device_array_t,
    cugraph_type_erased_device_array_view, cugraph_type_erased_device_array_view_copy_from_host,
    cugraph_type_erased_device_array_view_copy_to_host,
    cugraph_type_erased_device_array_view_create, cugraph_type_erased_device_array_view_free,
    cugraph_type_erased_device_array_view_pointer, cugraph_type_erased_device_array_view_size,
    cugraph_type_erased_device_array_view_t, cugraph_type_erased_device_array_view_type,
    cugraph_type_erased_host_array_view_create, cugraph_type_erased_host_array_view_free,
    cugraph_type_erased_host_array_view_t,
};

use crate::error::{check_error, Error, ErrorKind, Result};
use crate::handle::ResourceHandle;
use crate::types::DataType;

/// An owning device (GPU) array.
///
/// Allocates GPU memory on creation and frees it on drop. Create a
/// [`DeviceArrayView`] via [`view()`](DeviceArray::view) for zero-copy access.
pub struct DeviceArray {
    raw: *mut cugraph_type_erased_device_array_t,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl DeviceArray {
    /// Allocate a new device array with `len` elements of type `dtype`.
    pub fn new(handle: &ResourceHandle, len: usize, dtype: DataType) -> Result<Self> {
        let mut raw: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        unsafe {
            let code = cugraph_type_erased_device_array_create(
                handle.as_ptr(),
                len,
                dtype.into(),
                &mut raw,
                &mut error,
            );
            check_error(code, error, "DeviceArray::new")?;
        }

        Ok(Self {
            raw,
            _nosend_nosync: PhantomData,
        })
    }

    /// Create a non-owning view of this array.
    pub fn view(&self) -> DeviceArrayView<'_> {
        let raw = unsafe { cugraph_type_erased_device_array_view(self.raw) };
        DeviceArrayView {
            raw,
            owned: true,
            _lifetime: PhantomData,
            _nosend_nosync: PhantomData,
        }
    }

    /// Copy data from a host byte slice into this device array.
    ///
    /// The slice length must match `len * dtype.size_of()`.
    pub fn copy_from_host(&mut self, handle: &ResourceHandle, data: &[u8]) -> Result<()> {
        let view = self.view();
        let expected = view.len() * view.dtype()?.size_of();
        if data.len() != expected {
            return Err(
                Error::new(ErrorKind::InvalidInput, "DeviceArray::copy_from_host")
                    .with_message("host input length must exactly match device array byte length")
                    .with_context("expected_len", expected.to_string())
                    .with_context("actual_len", data.len().to_string()),
            );
        }

        let mut error: *mut cugraph_error_t = ptr::null_mut();

        unsafe {
            let code = cugraph_type_erased_device_array_view_copy_from_host(
                handle.as_ptr(),
                view.raw,
                data.as_ptr() as *const i8,
                &mut error,
            );
            check_error(code, error, "DeviceArray::copy_from_host")?;
        }

        Ok(())
    }

    /// Copy data from this device array to a host byte slice.
    ///
    /// The slice length must be exactly `len * dtype.size_of()`.
    pub fn copy_to_host(&self, handle: &ResourceHandle, dest: &mut [u8]) -> Result<()> {
        let view = self.view();
        let expected = view.len() * view.dtype()?.size_of();
        if dest.len() != expected {
            return Err(
                Error::new(ErrorKind::InvalidInput, "DeviceArray::copy_to_host")
                    .with_message("destination length must exactly match device array byte length")
                    .with_context("expected_len", expected.to_string())
                    .with_context("actual_len", dest.len().to_string()),
            );
        }

        let mut error: *mut cugraph_error_t = ptr::null_mut();

        unsafe {
            let code = cugraph_type_erased_device_array_view_copy_to_host(
                handle.as_ptr(),
                dest.as_mut_ptr() as *mut i8,
                view.raw,
                &mut error,
            );
            check_error(code, error, "DeviceArray::copy_to_host")?;
        }

        Ok(())
    }
}

impl Drop for DeviceArray {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_type_erased_device_array_free(self.raw) };
        }
    }
}

/// Contract for a type that safely exposes a contiguous device-memory region.
///
/// This trait is intended for interop crates (for example GPU DataFrame wrappers)
/// to provide a safe bridge into [`DeviceArrayView::from_device_buffer`].
///
/// # Safety
///
/// Implementors must guarantee that:
/// - `device_ptr()` returns a pointer to valid device memory
/// - the allocation contains at least `len()` elements of type `dtype()`
/// - the returned pointer remains valid for as long as `self` is alive
pub unsafe trait DeviceBuffer {
    /// Raw device pointer to the first element.
    fn device_ptr(&self) -> *mut std::ffi::c_void;

    /// Number of logical elements.
    fn len(&self) -> usize;

    /// Element type of the allocation.
    ///
    /// Returns an error if the underlying type has no cuGraph equivalent (e.g.
    /// a timestamp or string type from an external library).
    fn dtype(&self) -> Result<DataType>;
}

/// A non-owning view into device (GPU) memory.
///
/// The lifetime parameter `'a` ties this view to whatever owns the underlying
/// memory (a [`DeviceArray`], a result object, etc.), preventing
/// use-after-free.
pub struct DeviceArrayView<'a> {
    pub(crate) raw: *mut cugraph_type_erased_device_array_view_t,
    /// Whether this view struct itself was allocated (and should be freed on
    /// drop). Views obtained from result accessors are NOT owned — the result
    /// object owns the view memory.
    owned: bool,
    _lifetime: PhantomData<&'a ()>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl<'a> DeviceArrayView<'a> {
    /// Wrap a raw C view pointer that is NOT owned by us (e.g., returned from a
    /// result accessor). The caller must ensure the pointer stays valid for `'a`.
    pub(crate) unsafe fn from_borrowed(raw: *mut cugraph_type_erased_device_array_view_t) -> Self {
        DeviceArrayView {
            raw,
            owned: false,
            _lifetime: PhantomData,
            _nosend_nosync: PhantomData,
        }
    }

    /// Create a view from an existing raw device pointer.
    ///
    /// Useful for zero-copy interop with external GPU libraries that expose raw
    /// device pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following:
    /// - `pointer` points to valid device memory
    /// - the allocation contains at least `len` elements of type `dtype`
    /// - the underlying memory outlives the returned view's lifetime `'a`
    pub unsafe fn from_raw(pointer: *mut std::ffi::c_void, len: usize, dtype: DataType) -> Self {
        let raw = cugraph_type_erased_device_array_view_create(pointer, len, dtype.into());
        DeviceArrayView {
            raw,
            owned: true,
            _lifetime: PhantomData,
            _nosend_nosync: PhantomData,
        }
    }

    /// Create a view from a safe [`DeviceBuffer`] implementor.
    ///
    /// This is the ergonomic constructor for zero-copy interop. The lifetime
    /// of the returned view is tied to `buffer`, preventing accidental use
    /// after the external allocation is dropped.
    pub fn from_device_buffer<B>(buffer: &'a B) -> Result<Self>
    where
        B: DeviceBuffer + ?Sized,
    {
        let dtype = buffer.dtype()?;
        let raw = unsafe {
            cugraph_type_erased_device_array_view_create(
                buffer.device_ptr(),
                buffer.len(),
                dtype.into(),
            )
        };
        Ok(DeviceArrayView {
            raw,
            owned: true,
            _lifetime: PhantomData,
            _nosend_nosync: PhantomData,
        })
    }

    /// Device pointer to the first element of this view.
    pub fn device_ptr(&self) -> *mut std::ffi::c_void {
        unsafe { cugraph_type_erased_device_array_view_pointer(self.raw) as *mut std::ffi::c_void }
    }

    /// Number of elements in this view.
    pub fn len(&self) -> usize {
        unsafe { cugraph_type_erased_device_array_view_size(self.raw) }
    }

    /// Whether this view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The element data type.
    pub fn dtype(&self) -> Result<DataType> {
        let c_type = unsafe { cugraph_type_erased_device_array_view_type(self.raw) };
        DataType::try_from(c_type).map_err(|_| {
            Error::new(ErrorKind::UnsupportedType, "DeviceArrayView::dtype").with_message(
                format!("unknown cugraph_data_type_id_t value: {}", c_type.0),
            )
        })
    }

    /// Get the raw C view pointer. For internal use.
    pub(crate) fn as_ptr(&self) -> *mut cugraph_type_erased_device_array_view_t {
        self.raw
    }

    /// Copy the device data to a host Vec.
    pub fn to_host_vec(&self, handle: &ResourceHandle) -> Result<Vec<u8>> {
        let byte_len = self.len() * self.dtype()?.size_of();
        let mut buf = vec![0u8; byte_len];
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        unsafe {
            let code = cugraph_type_erased_device_array_view_copy_to_host(
                handle.as_ptr(),
                buf.as_mut_ptr() as *mut i8,
                self.raw,
                &mut error,
            );
            check_error(code, error, "DeviceArrayView::to_host_vec")?;
        }

        Ok(buf)
    }
}

impl<'a> Drop for DeviceArrayView<'a> {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            unsafe { cugraph_type_erased_device_array_view_free(self.raw) };
        }
    }
}

/// A non-owning view into host (CPU) memory, usable by the C API.
///
/// Created from a Rust slice and freed on drop. The lifetime `'a` ties it to
/// the original slice.
pub struct HostArrayView<'a> {
    raw: *mut cugraph_type_erased_host_array_view_t,
    _lifetime: PhantomData<&'a ()>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl<'a> HostArrayView<'a> {
    /// Create a host array view from a byte slice.
    ///
    /// `n_elems` is the number of logical elements (not bytes). The slice must
    /// contain at least `n_elems * dtype.size_of()` bytes.
    pub fn new(data: &'a [u8], n_elems: usize, dtype: DataType) -> Self {
        let raw = unsafe {
            cugraph_type_erased_host_array_view_create(
                data.as_ptr() as *mut std::ffi::c_void,
                n_elems,
                dtype.into(),
            )
        };
        HostArrayView {
            raw,
            _lifetime: PhantomData,
            _nosend_nosync: PhantomData,
        }
    }
}

impl<'a> Drop for HostArrayView<'a> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_type_erased_host_array_view_free(self.raw) };
        }
    }
}

/// Helper: create a [`DeviceArray`] from a typed host slice, copying the data
/// to the GPU.
pub fn device_array_from_host<T: Copy>(
    handle: &ResourceHandle,
    data: &[T],
    dtype: DataType,
) -> Result<DeviceArray> {
    let type_size = std::mem::size_of::<T>();
    let dtype_size = dtype.size_of();
    if type_size != dtype_size {
        return Err(
            Error::new(ErrorKind::InvalidInput, "device_array_from_host")
                .with_message("Rust type size does not match the specified DataType size")
                .with_context("rust_type_size", type_size.to_string())
                .with_context("dtype_size", dtype_size.to_string()),
        );
    }

    let byte_len = std::mem::size_of_val(data);
    let mut arr = DeviceArray::new(handle, data.len(), dtype)?;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    arr.copy_from_host(handle, bytes)?;
    Ok(arr)
}

unsafe impl DeviceBuffer for DeviceArray {
    fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.view().device_ptr()
    }

    fn len(&self) -> usize {
        self.view().len()
    }

    fn dtype(&self) -> Result<DataType> {
        self.view().dtype()
    }
}

unsafe impl<'a> DeviceBuffer for DeviceArrayView<'a> {
    fn device_ptr(&self) -> *mut std::ffi::c_void {
        DeviceArrayView::device_ptr(self)
    }

    fn len(&self) -> usize {
        DeviceArrayView::len(self)
    }

    fn dtype(&self) -> Result<DataType> {
        DeviceArrayView::dtype(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_not_impl_any;

    #[test]
    fn test_device_array_roundtrip() {
        let handle = ResourceHandle::new().expect("handle");
        let src: Vec<i32> = vec![10, 20, 30, 40, 50];
        let arr = device_array_from_host(&handle, &src, DataType::Int32).expect("create");

        let view = arr.view();
        assert_eq!(view.len(), 5);
        assert_eq!(view.dtype().expect("dtype"), DataType::Int32);

        let mut dest = vec![0i32; 5];
        let dest_bytes =
            unsafe { std::slice::from_raw_parts_mut(dest.as_mut_ptr() as *mut u8, 5 * 4) };
        arr.copy_to_host(&handle, dest_bytes).expect("copy_to_host");
        assert_eq!(dest, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_device_array_view_to_host_vec() {
        let handle = ResourceHandle::new().expect("handle");
        let src: Vec<f32> = vec![1.0, 2.0, 3.0];
        let arr = device_array_from_host(&handle, &src, DataType::Float32).expect("create");

        let view = arr.view();
        let bytes = view.to_host_vec(&handle).expect("to_host_vec");
        assert_eq!(bytes.len(), 3 * 4);

        let result: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_device_array_empty() {
        let handle = ResourceHandle::new().expect("handle");
        let arr = DeviceArray::new(&handle, 0, DataType::Int32).expect("create empty");
        let view = arr.view();
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
    }

    #[test]
    fn test_copy_from_host_rejects_wrong_length() {
        let handle = ResourceHandle::new().expect("handle");
        let mut arr = DeviceArray::new(&handle, 4, DataType::Int32).expect("create");

        // Expected 16 bytes, provide only 15.
        let bad = vec![0u8; 15];
        let err = arr
            .copy_from_host(&handle, &bad)
            .expect_err("must reject incorrect host input length");
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "DeviceArray::copy_from_host");
    }

    #[test]
    fn test_copy_to_host_rejects_wrong_length() {
        let handle = ResourceHandle::new().expect("handle");
        let arr = DeviceArray::new(&handle, 4, DataType::Int32).expect("create");

        // Expected 16 bytes, provide 20.
        let mut bad = vec![0u8; 20];
        let err = arr
            .copy_to_host(&handle, &mut bad)
            .expect_err("must reject incorrect destination length");
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "DeviceArray::copy_to_host");
    }

    #[test]
    fn test_array_types_not_send_sync() {
        assert_not_impl_any!(DeviceArray: Send, Sync);
        assert_not_impl_any!(DeviceArrayView<'static>: Send, Sync);
        assert_not_impl_any!(HostArrayView<'static>: Send, Sync);
    }

    #[test]
    fn test_from_raw_roundtrip() {
        // Create a device array, get its raw pointer, create a view from it,
        // and verify data survives the round trip.
        let handle = ResourceHandle::new().expect("handle");
        let src: Vec<i32> = vec![100, 200, 300];
        let arr = device_array_from_host(&handle, &src, DataType::Int32).expect("create");

        // Get the raw device pointer from the owned array's view.
        let owned_view = arr.view();
        let raw_ptr = owned_view.device_ptr();

        // SAFETY: `raw_ptr` comes from a live device allocation with exactly 3 i32 values,
        // and `arr` is kept alive for the duration of `raw_view` usage.
        let raw_view = unsafe { DeviceArrayView::from_raw(raw_ptr, 3, DataType::Int32) };
        assert_eq!(raw_view.len(), 3);
        assert_eq!(raw_view.dtype().expect("dtype"), DataType::Int32);

        // Read back through the raw view.
        let bytes = raw_view.to_host_vec(&handle).expect("to_host_vec");
        let result: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(result, vec![100, 200, 300]);
    }

    #[test]
    fn test_from_device_buffer_roundtrip() {
        struct ExternalBuffer<'a> {
            arr: &'a DeviceArray,
            dtype: DataType,
        }

        unsafe impl DeviceBuffer for ExternalBuffer<'_> {
            fn device_ptr(&self) -> *mut std::ffi::c_void {
                self.arr.view().device_ptr()
            }

            fn len(&self) -> usize {
                self.arr.view().len()
            }

            fn dtype(&self) -> crate::error::Result<DataType> {
                Ok(self.dtype)
            }
        }

        let handle = ResourceHandle::new().expect("handle");
        let src: Vec<i32> = vec![7, 8, 9, 10];
        let arr = device_array_from_host(&handle, &src, DataType::Int32).expect("create");

        let ext = ExternalBuffer {
            arr: &arr,
            dtype: DataType::Int32,
        };

        let view = DeviceArrayView::from_device_buffer(&ext).expect("from_device_buffer");
        assert_eq!(view.len(), 4);
        assert_eq!(view.dtype().expect("dtype"), DataType::Int32);

        let bytes = view.to_host_vec(&handle).expect("to_host_vec");
        let result: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(result, src);
    }

    #[test]
    fn test_device_array_from_host_rejects_type_size_mismatch() {
        let handle = ResourceHandle::new().expect("handle");
        // i64 (8 bytes) does not match Int32 (4 bytes)
        let data: Vec<i64> = vec![1, 2, 3];
        let err = match device_array_from_host(&handle, &data, DataType::Int32) {
            Ok(_) => panic!("must reject type size mismatch"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "device_array_from_host");
        assert!(err.message().contains("size"));
    }
}
