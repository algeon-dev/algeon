//! Typed device and host array wrappers.
//!
//! These types wrap the cuGraph C API's type-erased arrays and provide a safe
//! Rust interface with proper ownership semantics:
//!
//! - [`DeviceArray<T>`] — owning device array, freed on drop
//! - [`DeviceArrayView<'a, T>`] — non-owning view into device memory, lifetime-tied
//! - internal `HostArrayView` — non-owning view into host memory, lifetime-tied
//!
//! The type parameter `T: CugraphElement` carries the element type at compile
//! time, eliminating the need for runtime type checks.

use std::{fmt, marker::PhantomData, ptr};

use cugraph_sys::*;

use crate::{
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    handle::{EnteredContext, ResourceHandleLease},
    types::{CugraphElement, DataType},
};

// ═══════════════════════════════════════════════════════════════════════════
// Raw (type-erased) internal types — used by FFI and algorithm result types
// ═══════════════════════════════════════════════════════════════════════════

/// Internal type-erased owning device array. Used for C API interop.
pub(crate) struct RawDeviceArray {
    raw: *mut cugraph_type_erased_device_array_t,
    raw_view: *mut cugraph_type_erased_device_array_view_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

fn checked_byte_len(len: usize, dtype: DataType, operation: &'static str) -> Result<usize> {
    len.checked_mul(dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message("array byte size overflow")
            .with_context("len", len.to_string())
            .with_context("dtype", format!("{:?}", dtype))
            .with_context("element_size", dtype.size_of().to_string())
    })
}

impl RawDeviceArray {
    /// Wrap a native-owned cuGraph device-array handle returned by an
    /// algorithm. Ownership of `raw` transfers to this wrapper.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by the cuGraph C API,
    /// and no other owner may free it after this call succeeds.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_type_erased_device_array_t,
        device: i32,
        operation: &'static str,
        pointer_name: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, pointer_name)?;
        // SAFETY: `raw` is a live cuGraph device-array handle transferred to
        // this wrapper. The returned view metadata is separately owned.
        let raw_view = unsafe { cugraph_type_erased_device_array_view(raw) };
        let raw_view = match require_non_null(raw_view, operation, "device_array_view") {
            Ok(raw_view) => raw_view,
            Err(error) => {
                let _guard = DeviceGuard::switch_to_for_drop(device, operation).ok();
                // SAFETY: construction has not succeeded, so `raw` is still
                // uniquely owned here and must be released on error.
                unsafe { cugraph_type_erased_device_array_free(raw) };
                return Err(error);
            },
        };

        Ok(Self { raw, raw_view, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    pub(crate) fn new(handle: &EnteredContext<'_>, len: usize, dtype: DataType) -> Result<Self> {
        let mut raw: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        // SAFETY: `handle` outlives the call and `raw`/`error` are valid
        // out-pointers for the C API to initialize.
        unsafe {
            let code =
                cugraph_type_erased_device_array_create(handle.handle_ptr(), len, dtype.into(), &mut raw, &mut error);
            check_error(code, error, "RawDeviceArray::new")?;
        }

        let raw = require_non_null(raw, "RawDeviceArray::new", "device_array")?;
        // SAFETY: `raw` is a freshly created cuGraph device-array handle that
        // remains owned by this wrapper after construction.
        let raw_view = unsafe { cugraph_type_erased_device_array_view(raw) };
        let raw_view = match require_non_null(raw_view, "RawDeviceArray::new", "device_array_view") {
            Ok(raw_view) => raw_view,
            Err(error) => {
                let _guard = DeviceGuard::switch_to(handle.device(), "RawDeviceArray::new").ok();
                // SAFETY: `raw` is still uniquely owned here because
                // construction has not succeeded yet.
                unsafe { cugraph_type_erased_device_array_free(raw) };
                return Err(error);
            },
        };

        Ok(Self {
            raw,
            raw_view,
            device: handle.device(),
            _resource_handle_lease: handle.runtime_bound_lease(),
            _nosend_nosync: PhantomData,
        })
    }

    pub(crate) fn raw_view(&self) -> Result<RawDeviceArrayView<'_>> {
        let raw = require_non_null(self.raw_view, "RawDeviceArray::raw_view", "device_array_view")?;
        Ok(RawDeviceArrayView {
            raw,
            owned: false,
            device: Some(self.device),
            _lifetime: PhantomData,
            _nosend_nosync: PhantomData,
        })
    }

    fn copy_from_host(&mut self, handle: &EnteredContext<'_>, data: &[u8]) -> Result<()> {
        let view = self.raw_view()?;
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        // SAFETY: `view.raw` is a live device-array view owned by `self`,
        // `data` points to initialized host bytes, and `error` is a valid
        // out-pointer.
        unsafe {
            let code = cugraph_type_erased_device_array_view_copy_from_host(
                handle.handle_ptr(),
                view.raw,
                data.as_ptr() as *const i8,
                &mut error,
            );
            check_error(code, error, "RawDeviceArray::copy_from_host")?;
        }

        Ok(())
    }
}

impl Drop for RawDeviceArray {
    fn drop(&mut self) {
        if !self.raw_view.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "RawDeviceArray::drop").ok();
            // SAFETY: `self.raw_view` is owned by this wrapper and is freed at
            // most once here.
            unsafe { cugraph_type_erased_device_array_view_free(self.raw_view) };
        }
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "RawDeviceArray::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and has not been
            // freed yet.
            unsafe { cugraph_type_erased_device_array_free(self.raw) };
        }
    }
}

/// Type-erased non-owning view into device memory.
///
/// Used internally for values whose concrete type is only known at runtime
/// (e.g., degree arrays that may be `i32` or `i64` depending on the graph's
/// edge type). For compile-time-typed access, use [`DeviceArrayView<T>`].
pub(crate) struct RawDeviceArrayView<'a> {
    pub(crate) raw: *mut cugraph_type_erased_device_array_view_t,
    owned: bool,
    device: Option<i32>,
    _lifetime: PhantomData<&'a ()>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl fmt::Debug for RawDeviceArrayView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawDeviceArrayView")
            .field("len", &self.len())
            .field("dtype", &self.dtype().ok())
            .finish_non_exhaustive()
    }
}

impl RawDeviceArrayView<'_> {
    /// Wrap a raw C view pointer that is NOT owned by us.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `raw` points to a live `cugraph_type_erased_device_array_view_t`
    /// - the underlying device allocation remains valid for the returned
    ///   lifetime `'a`
    /// - this wrapper does not outlive the owner responsible for freeing the
    ///   C view and its backing storage
    pub(crate) unsafe fn from_borrowed(
        raw: *mut cugraph_type_erased_device_array_view_t,
        device: Option<i32>,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, pointer_name)?;
        Ok(Self::from_borrowed_unchecked(raw, device))
    }

    pub(crate) unsafe fn from_borrowed_unchecked(
        raw: *mut cugraph_type_erased_device_array_view_t,
        device: Option<i32>,
    ) -> Self {
        RawDeviceArrayView { raw, owned: false, device, _lifetime: PhantomData, _nosend_nosync: PhantomData }
    }

    /// Wrap a raw C view pointer whose metadata ownership transfers to this
    /// Rust wrapper.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `raw` points to a live `cugraph_type_erased_device_array_view_t`
    /// - only the view metadata is owned by this wrapper
    /// - the backing device allocation remains valid for the returned lifetime
    pub(crate) unsafe fn from_owned_view(
        raw: *mut cugraph_type_erased_device_array_view_t,
        device: Option<i32>,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, pointer_name)?;
        Ok(RawDeviceArrayView { raw, owned: true, device, _lifetime: PhantomData, _nosend_nosync: PhantomData })
    }

    /// Create a view from an existing raw device pointer.
    ///
    /// # Safety
    ///
    /// Same safety requirements as `DeviceArrayView::from_raw`.
    pub(crate) unsafe fn try_from_raw(
        pointer: *mut std::ffi::c_void,
        len: usize,
        dtype: DataType,
        operation: &'static str,
    ) -> Result<Self> {
        checked_byte_len(len, dtype, operation)?;
        // SAFETY: the caller upholds the raw pointer, element count, and dtype
        // contract documented on this constructor.
        let raw = unsafe { cugraph_type_erased_device_array_view_create(pointer, len, dtype.into()) };
        if raw.is_null() {
            return Err(Error::new(ErrorKind::OutOfMemory, operation)
                .with_message("failed to allocate device array view metadata")
                .set_temporary());
        }
        Ok(RawDeviceArrayView { raw, owned: true, device: None, _lifetime: PhantomData, _nosend_nosync: PhantomData })
    }

    /// Device pointer to the first element of this view.
    pub(crate) fn device_ptr(&self) -> *mut std::ffi::c_void {
        // SAFETY: `self.raw` is a live device-array-view handle.
        unsafe { cugraph_type_erased_device_array_view_pointer(self.raw) as *mut std::ffi::c_void }
    }

    /// Number of elements in this view.
    pub(crate) fn len(&self) -> usize {
        // SAFETY: `self.raw` is a live device-array-view handle.
        unsafe { cugraph_type_erased_device_array_view_size(self.raw) }
    }

    /// Whether this view is empty.
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Runtime element type of this view.
    pub(crate) fn dtype(&self) -> Result<DataType> {
        // SAFETY: `self.raw` is a live device-array-view handle.
        let c_type = unsafe { cugraph_type_erased_device_array_view_type(self.raw) };
        DataType::try_from(c_type).map_err(|err| err.with_operation("RawDeviceArrayView::dtype"))
    }

    pub(crate) fn as_ptr(&self) -> *mut cugraph_type_erased_device_array_view_t {
        self.raw
    }

    pub(crate) fn device(&self) -> Option<i32> {
        self.device
    }

    /// Borrow this raw view without transferring ownership.
    pub(crate) fn raw_view(&self) -> RawDeviceArrayView<'_> {
        // SAFETY: the returned view borrows the same underlying C view pointer
        // for no longer than `&self`, and does not take ownership.
        unsafe { RawDeviceArrayView::from_borrowed_unchecked(self.raw, self.device) }
    }

    /// Copy device array data to a raw host byte vector.
    ///
    /// The returned bytes can be reinterpreted as the element type indicated
    /// by [`dtype()`](Self::dtype).
    pub(crate) fn to_host_vec(&self, handle: &EnteredContext<'_>) -> Result<Vec<u8>> {
        let dtype = self.dtype()?;
        let byte_len = checked_byte_len(self.len(), dtype, "RawDeviceArrayView::to_host_vec")?;
        let mut buf = vec![0u8; byte_len];
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        // SAFETY: `self.raw` is a live device-array-view handle, `buf` points
        // to enough writable host memory for the copy, and `error` is a valid
        // out-pointer.
        unsafe {
            let code = cugraph_type_erased_device_array_view_copy_to_host(
                handle.handle_ptr(),
                buf.as_mut_ptr() as *mut i8,
                self.raw,
                &mut error,
            );
            check_error(code, error, "RawDeviceArrayView::to_host_vec")?;
        }

        Ok(buf)
    }
}

impl Drop for RawDeviceArrayView<'_> {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            // SAFETY: owned views created by this module are freed exactly once
            // here when their wrapper is dropped.
            unsafe { cugraph_type_erased_device_array_view_free(self.raw) };
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Typed public API
// ═══════════════════════════════════════════════════════════════════════════

/// An owning device (GPU) array with compile-time element type `T`.
///
/// Allocates GPU memory on creation and frees it on drop. Create a
/// [`DeviceArrayView`] via [`view()`](DeviceArray::view) for zero-copy access.
///
/// # Example
///
/// ```no_run
/// # use cudf::Runtime;
/// # use cugraph::{DeviceArray, ResourceHandle};
/// let runtime = std::sync::Arc::new(Runtime::new().unwrap());
/// let handle =
///     ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
/// let entered = handle.enter("DeviceArray docs").unwrap();
/// let arr = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).unwrap();
/// let ids: Vec<i32> = arr.view().unwrap().to_vec(&entered).unwrap();
/// assert_eq!(ids, vec![0, 1, 2]);
/// ```
pub struct DeviceArray<T: CugraphElement> {
    inner: RawDeviceArray,
    _element: PhantomData<T>,
}

impl<T: CugraphElement> fmt::Debug for DeviceArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("DeviceArray");
        d.field("dtype", &T::data_type());
        match self.inner.raw_view() {
            Ok(view) => {
                d.field("len", &view.len());
            },
            Err(error) => {
                d.field("len_error", &error.to_string());
            },
        }
        d.finish_non_exhaustive()
    }
}

impl<T: CugraphElement> DeviceArray<T> {
    /// Wrap a native-owned cuGraph device-array handle returned by an
    /// algorithm. Ownership of `raw` transfers to the returned [`DeviceArray`].
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by the cuGraph C API.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_type_erased_device_array_t,
        device: i32,
        operation: &'static str,
        pointer_name: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: the caller transfers ownership of the native array handle.
        let inner = unsafe { RawDeviceArray::from_raw(raw, device, operation, pointer_name, resource_handle_lease) }?;
        let dtype = inner.raw_view()?.dtype()?;
        if dtype != T::data_type() {
            return Err(Error::new(ErrorKind::InvalidState, operation).with_message(format!(
                "{pointer_name} dtype {dtype} does not match requested dtype {}",
                T::data_type()
            )));
        }
        Ok(Self { inner, _element: PhantomData })
    }

    /// Allocate a new device array with `len` elements of type `T`.
    pub fn new(handle: &EnteredContext<'_>, len: usize) -> Result<Self> {
        let inner = RawDeviceArray::new(handle, len, T::data_type())?;
        Ok(Self { inner, _element: PhantomData })
    }

    /// Create a device array from a host slice, copying the data to the GPU.
    ///
    /// This is the primary constructor for typed device arrays.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cudf::Runtime;
    /// # use cugraph::{DeviceArray, ResourceHandle};
    /// let runtime = std::sync::Arc::new(Runtime::new().unwrap());
    /// let handle =
    ///     ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
    /// let entered = handle.enter("DeviceArray::from_slice docs").unwrap();
    /// let arr = DeviceArray::from_slice(&entered, &[1i32, 2, 3]).unwrap();
    /// assert_eq!(arr.view().unwrap().len(), 3);
    /// ```
    pub fn from_slice(handle: &EnteredContext<'_>, data: &[T]) -> Result<Self> {
        let mut arr = Self::new(handle, data.len())?;
        let byte_len = std::mem::size_of_val(data);
        // SAFETY: `data` is a contiguous slice of `T`, so reinterpreting its
        // initialized bytes as `u8` for a host-to-device copy is sound.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        arr.inner.copy_from_host(handle, bytes)?;
        Ok(arr)
    }

    /// Create a non-owning typed view of this array.
    pub fn view(&self) -> Result<DeviceArrayView<'_, T>> {
        Ok(DeviceArrayView { inner: self.inner.raw_view()?, _element: PhantomData })
    }

    /// Get the raw C pointer to the underlying array. For internal use.
    #[allow(dead_code)]
    pub(crate) fn raw_view(&self) -> Result<RawDeviceArrayView<'_>> {
        self.inner.raw_view()
    }
}

/// Contract for a type that safely exposes a contiguous device-memory region
/// of elements of type `T`.
///
/// This trait is intended for interop crates (for example GPU DataFrame wrappers)
/// to provide a safe bridge into [`DeviceArrayView::from_device_buffer`].
///
/// # Safety
///
/// Implementors must guarantee that:
/// - `device_ptr()` returns a pointer to valid device memory
/// - the allocation contains at least `len()` elements of type `T`
/// - the returned pointer remains valid for as long as `self` is alive
pub unsafe trait DeviceBuffer<T: CugraphElement> {
    /// Raw device pointer to the first element.
    fn device_ptr(&self) -> *mut std::ffi::c_void;

    /// Number of logical elements.
    fn len(&self) -> usize;

    /// Returns `true` if the buffer contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A non-owning view into device (GPU) memory with compile-time element type `T`.
///
/// The lifetime parameter `'a` ties this view to whatever owns the underlying
/// memory (a [`DeviceArray`], a result object, etc.), preventing
/// use-after-free.
pub struct DeviceArrayView<'a, T: CugraphElement> {
    inner: RawDeviceArrayView<'a>,
    _element: PhantomData<T>,
}

impl<T: CugraphElement> fmt::Debug for DeviceArrayView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceArrayView")
            .field("len", &self.len())
            .field("dtype", &T::data_type())
            .finish_non_exhaustive()
    }
}

impl<'a, T: CugraphElement> DeviceArrayView<'a, T> {
    /// Wrap a raw C view pointer that is NOT owned by us (e.g., returned from a
    /// result accessor). The caller must ensure the pointer stays valid for `'a`
    /// and that the underlying data is of type `T`.
    ///
    /// # Safety
    ///
    /// - `raw` must point to valid device memory containing elements of type `T`
    /// - the memory must outlive `'a`
    #[allow(dead_code)] // kept for potential direct use in typed algorithm paths
    pub(crate) unsafe fn from_borrowed(raw: *mut cugraph_type_erased_device_array_view_t) -> Self {
        DeviceArrayView { inner: RawDeviceArrayView::from_borrowed_unchecked(raw, None), _element: PhantomData }
    }

    /// Create a view from an existing raw device pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following:
    /// - `pointer` points to valid device memory
    /// - the allocation contains at least `len` elements of type `T`
    /// - the underlying memory outlives the returned view's lifetime `'a`
    #[cfg(test)] // used in array round-trip tests
    pub(crate) unsafe fn from_raw(pointer: *mut std::ffi::c_void, len: usize) -> Result<Self> {
        // SAFETY: the caller upholds the raw device pointer contract for `T`.
        let inner =
            unsafe { RawDeviceArrayView::try_from_raw(pointer, len, T::data_type(), "DeviceArrayView::from_raw") }?;
        Ok(DeviceArrayView { inner, _element: PhantomData })
    }

    /// Wrap a [`RawDeviceArrayView`] without checking the element type.
    ///
    /// The caller must have already verified that the raw view's dtype matches `T`.
    /// This is used by [`DegreeValues::from_raw`](crate::structure::degrees::DegreeValues::from_raw)
    /// after a runtime dtype check.
    pub(crate) fn from_raw_unchecked(raw: RawDeviceArrayView<'a>) -> Self {
        DeviceArrayView { inner: raw, _element: PhantomData }
    }

    /// Create a view from a safe [`DeviceBuffer`] implementor.
    ///
    /// This is the ergonomic constructor for zero-copy interop. The lifetime
    /// of the returned view is tied to `buffer`, preventing accidental use
    /// after the external allocation is dropped.
    pub fn from_device_buffer<B>(buffer: &'a B) -> Result<Self>
    where B: DeviceBuffer<T> + ?Sized {
        Self::try_from_device_buffer(buffer)
    }

    /// Fallible variant of [`from_device_buffer`](Self::from_device_buffer).
    pub fn try_from_device_buffer<B>(buffer: &'a B) -> Result<Self>
    where B: DeviceBuffer<T> + ?Sized {
        // SAFETY: `DeviceBuffer` implementors guarantee that the returned
        // device pointer and length describe a valid contiguous buffer of `T`
        // that lives for `'a`.
        let inner = unsafe {
            RawDeviceArrayView::try_from_raw(
                buffer.device_ptr(),
                buffer.len(),
                T::data_type(),
                "DeviceArrayView::try_from_device_buffer",
            )?
        };
        Ok(DeviceArrayView { inner, _element: PhantomData })
    }

    /// Device pointer to the first element of this view.
    pub fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.inner.device_ptr()
    }

    /// Number of elements in this view.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether this view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the raw C view pointer. For internal use.
    pub(crate) fn as_ptr(&self) -> *mut cugraph_type_erased_device_array_view_t {
        self.inner.as_ptr()
    }

    /// Clone this typed view as a crate-private raw view.
    pub(crate) fn raw_view(&self) -> RawDeviceArrayView<'_> {
        self.inner.raw_view()
    }

    /// Copy device array data to a typed host vector.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cudf::Runtime;
    /// # use cugraph::{DeviceArray, ResourceHandle};
    /// # let runtime = std::sync::Arc::new(Runtime::new().unwrap());
    /// # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
    /// # let entered = handle.enter("DeviceArrayView::to_vec docs").unwrap();
    /// let arr = DeviceArray::from_slice(&entered, &[10i32, 20, 30]).unwrap();
    /// let values: Vec<i32> = arr.view().unwrap().to_vec(&entered).unwrap();
    /// assert_eq!(values, vec![10, 20, 30]);
    /// ```
    pub fn to_vec(&self, handle: &EnteredContext<'_>) -> Result<Vec<T>> {
        let type_size = std::mem::size_of::<T>();
        let bytes = self.inner.to_host_vec(handle)?;
        let len = bytes.len() / type_size;

        let mut result = Vec::with_capacity(len);
        for chunk in bytes.chunks_exact(type_size) {
            // SAFETY: chunk has exactly `size_of::<T>()` bytes and `T: Copy` has no
            // drop glue. `read_unaligned` avoids any alignment requirement on the
            // source pointer, which is safe since `Vec<u8>` may not be aligned to `T`.
            let value = unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const T) };
            result.push(value);
        }

        Ok(result)
    }
}

// SAFETY: `DeviceArray<T>` owns a contiguous GPU allocation of `T` elements
// for its full lifetime and exposes the same pointer/length pair that backs
// its typed view APIs.
unsafe impl<T: CugraphElement> DeviceBuffer<T> for DeviceArray<T> {
    fn device_ptr(&self) -> *mut std::ffi::c_void {
        if self.inner.raw_view.is_null() {
            debug_assert!(!self.inner.raw_view.is_null(), "DeviceArray invariant violated: missing raw view handle");
            return ptr::null_mut();
        }
        // SAFETY: `self.inner.raw_view` is created during `RawDeviceArray::new`
        // and remains owned by the array until drop.
        unsafe { cugraph_type_erased_device_array_view_pointer(self.inner.raw_view) as *mut std::ffi::c_void }
    }

    fn len(&self) -> usize {
        if self.inner.raw_view.is_null() {
            debug_assert!(!self.inner.raw_view.is_null(), "DeviceArray invariant violated: missing raw view handle");
            return 0;
        }
        // SAFETY: `self.inner.raw_view` is created during `RawDeviceArray::new`
        // and remains owned by the array until drop.
        unsafe { cugraph_type_erased_device_array_view_size(self.inner.raw_view) }
    }
}

// SAFETY: `DeviceArrayView<'a, T>` is a borrowed view over a contiguous GPU
// allocation of `T` elements, and its lifetime ensures the underlying memory
// outlives the trait accessors.
unsafe impl<T: CugraphElement> DeviceBuffer<T> for DeviceArrayView<'_, T> {
    fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.inner.device_ptr()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HostArrayView (stays type-erased, pub(crate))
// ═══════════════════════════════════════════════════════════════════════════

/// A non-owning view into host (CPU) memory, usable by the C API.
///
/// Created from a Rust slice and freed on drop. The lifetime `'a` ties it to
/// the original slice.
#[allow(dead_code)]
pub(crate) struct HostArrayView<'a> {
    pub(crate) raw: *mut cugraph_type_erased_host_array_view_t,
    _lifetime: PhantomData<&'a ()>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl fmt::Debug for HostArrayView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostArrayView").finish_non_exhaustive()
    }
}

impl<'a> HostArrayView<'a> {
    /// Create a host array view from a byte slice.
    ///
    /// `n_elems` is the number of logical elements (not bytes). The slice must
    /// contain at least `n_elems * dtype.size_of()` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if the slice is too small to hold
    /// `n_elems` elements of the given `dtype`.
    #[allow(dead_code)]
    pub(crate) fn new(data: &'a [u8], n_elems: usize, dtype: DataType) -> Result<Self> {
        let required = checked_byte_len(n_elems, dtype, "HostArrayView::new")?;
        if data.len() < required {
            return Err(Error::new(ErrorKind::InvalidInput, "HostArrayView::new")
                .with_message("slice is too small for the requested number of elements")
                .with_context("required_bytes", required.to_string())
                .with_context("actual_bytes", data.len().to_string())
                .with_context("n_elems", n_elems.to_string())
                .with_context("dtype", format!("{:?}", dtype)));
        }

        // The cuGraph C API takes `void*` here even though the host buffer is
        // treated as read-only input. Keep the const-to-mut conversion local to
        // this FFI boundary so the rest of the Rust API remains immutable.
        let data_ptr = data.as_ptr().cast::<std::ffi::c_void>().cast_mut();

        // SAFETY: `data` is borrowed for `'a`, points to at least `required`
        // bytes, and the view metadata does not outlive the slice.
        let raw = unsafe { cugraph_type_erased_host_array_view_create(data_ptr, n_elems, dtype.into()) };
        if raw.is_null() {
            return Err(Error::new(ErrorKind::OutOfMemory, "HostArrayView::new")
                .with_message("failed to allocate host array view metadata")
                .set_temporary());
        }
        Ok(HostArrayView { raw, _lifetime: PhantomData, _nosend_nosync: PhantomData })
    }

    pub(crate) fn as_ptr(&self) -> *mut cugraph_type_erased_host_array_view_t {
        self.raw
    }
}

impl Drop for HostArrayView<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `self.raw` is owned by this wrapper and is freed exactly
            // once here.
            unsafe { cugraph_type_erased_host_array_view_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_not_impl_any;

    use super::*;
    use crate::ResourceHandle;

    #[test]
    fn test_device_array_roundtrip() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_device_array_roundtrip").expect("entered");
        let arr = DeviceArray::from_slice(&entered, &[10i32, 20, 30, 40, 50]).expect("create");

        let view = arr.view().expect("view");
        assert_eq!(view.len(), 5);

        let result: Vec<i32> = view.to_vec(&entered).expect("to_vec");
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_device_array_f32_roundtrip() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_device_array_f32_roundtrip").expect("entered");
        let arr = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0]).expect("create");

        let view = arr.view().expect("view");
        let result: Vec<f32> = view.to_vec(&entered).expect("to_vec");
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_device_array_empty() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_device_array_empty").expect("entered");
        let arr = DeviceArray::<i32>::new(&entered, 0).expect("create empty");
        let view = arr.view().expect("view");
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
    }

    #[test]
    fn test_array_types_not_send_sync() {
        assert_not_impl_any!(DeviceArray<i32>: Send, Sync);
        assert_not_impl_any!(DeviceArrayView<'static, i32>: Send, Sync);
        assert_not_impl_any!(HostArrayView<'static>: Send, Sync);
    }

    #[test]
    fn test_from_raw_roundtrip() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_from_raw_roundtrip").expect("entered");
        let arr = DeviceArray::from_slice(&entered, &[100i32, 200, 300]).expect("create");

        let owned_view = arr.view().expect("owned_view");
        let raw_ptr = owned_view.device_ptr();

        // SAFETY: `raw_ptr` comes from a live device allocation with exactly 3 i32 values,
        // and `arr` is kept alive for the duration of `raw_view` usage.
        let raw_view = unsafe { DeviceArrayView::<i32>::from_raw(raw_ptr, 3) }.expect("raw_view");
        assert_eq!(raw_view.len(), 3);

        let result: Vec<i32> = raw_view.to_vec(&entered).expect("to_vec");
        assert_eq!(result, vec![100, 200, 300]);
    }

    #[test]
    fn test_from_device_buffer_roundtrip() {
        struct ExternalBuffer<'a, T: CugraphElement> {
            arr: &'a DeviceArray<T>,
        }

        // SAFETY: `ExternalBuffer` forwards the pointer and length from a live `DeviceArray`
        // borrow, so the reported device region remains valid for the lifetime of `self`.
        unsafe impl<T: CugraphElement> DeviceBuffer<T> for ExternalBuffer<'_, T> {
            fn device_ptr(&self) -> *mut std::ffi::c_void {
                self.arr.view().expect("view").device_ptr()
            }

            fn len(&self) -> usize {
                self.arr.view().expect("view").len()
            }
        }

        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_from_device_buffer_roundtrip").expect("entered");
        let arr = DeviceArray::from_slice(&entered, &[7i32, 8, 9, 10]).expect("create");

        let ext = ExternalBuffer { arr: &arr };

        let view = DeviceArrayView::<i32>::from_device_buffer(&ext).expect("device view");
        assert_eq!(view.len(), 4);

        let result: Vec<i32> = view.to_vec(&entered).expect("to_vec");
        assert_eq!(result, vec![7, 8, 9, 10]);
    }

    #[test]
    fn test_try_from_device_buffer_roundtrip() {
        struct ExternalBuffer<'a, T: CugraphElement> {
            arr: &'a DeviceArray<T>,
        }

        // SAFETY: `ExternalBuffer` forwards the pointer and length from a live `DeviceArray`
        // borrow, so the reported device region remains valid for the lifetime of `self`.
        unsafe impl<T: CugraphElement> DeviceBuffer<T> for ExternalBuffer<'_, T> {
            fn device_ptr(&self) -> *mut std::ffi::c_void {
                self.arr.view().expect("view").device_ptr()
            }

            fn len(&self) -> usize {
                self.arr.view().expect("view").len()
            }
        }

        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_try_from_device_buffer_roundtrip").expect("entered");
        let arr = DeviceArray::from_slice(&entered, &[11i32, 12, 13]).expect("create");

        let ext = ExternalBuffer { arr: &arr };

        let view = DeviceArrayView::<i32>::try_from_device_buffer(&ext).expect("device view");
        assert_eq!(view.len(), 3);

        let result: Vec<i32> = view.to_vec(&entered).expect("to_vec");
        assert_eq!(result, vec![11, 12, 13]);
    }

    #[test]
    fn test_to_vec_roundtrip_i32() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_to_vec_roundtrip_i32").expect("entered");
        let src: Vec<i32> = vec![10, 20, 30];
        let arr = DeviceArray::from_slice(&entered, &src).expect("create");
        let result: Vec<i32> = arr.view().expect("view").to_vec(&entered).expect("roundtrip");
        assert_eq!(result, src);
    }

    #[test]
    fn test_to_vec_roundtrip_f64() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_to_vec_roundtrip_f64").expect("entered");
        let src: Vec<f64> = vec![1.5, 2.5, 3.5];
        let arr = DeviceArray::from_slice(&entered, &src).expect("create");
        let result: Vec<f64> = arr.view().expect("view").to_vec(&entered).expect("roundtrip");
        assert_eq!(result, src);
    }

    #[test]
    fn test_host_array_view_rejects_undersized_slice() {
        let err = HostArrayView::new(&[0u8; 3], 2, DataType::Int32).expect_err("must reject undersized slice");
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "HostArrayView::new");
    }

    #[test]
    fn test_host_array_view_rejects_overflowing_size() {
        let err = HostArrayView::new(&[], usize::MAX, DataType::Int64).expect_err("must reject overflowing size math");
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "HostArrayView::new");
    }

    #[test]
    fn test_raw_device_array_view_try_from_raw_rejects_overflowing_size() {
        // SAFETY: the constructor must reject the overflow before touching the
        // null pointer, so this test does not rely on a valid device buffer.
        let err = unsafe {
            RawDeviceArrayView::try_from_raw(
                std::ptr::null_mut(),
                usize::MAX,
                DataType::Int64,
                "RawDeviceArrayView::try_from_raw",
            )
        }
        .expect_err("must reject overflowing size math");
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "RawDeviceArrayView::try_from_raw");
    }

    #[test]
    fn test_raw_device_array_view_from_borrowed_rejects_null() {
        // SAFETY: this validates the constructor's null guard and does not
        // dereference the pointer.
        let err = unsafe { RawDeviceArrayView::from_borrowed(std::ptr::null_mut(), None, "test_op", "view") }
            .expect_err("null borrowed view should be rejected");
        assert_eq!(err.kind(), crate::error::ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn test_host_array_view_accepts_exact_size() {
        let view = HostArrayView::new(&[0u8; 8], 2, DataType::Int32).expect("exact size should succeed");
        drop(view);
    }

    #[test]
    fn test_host_array_view_accepts_oversized_slice() {
        let view = HostArrayView::new(&[0u8; 16], 2, DataType::Int32).expect("oversized slice should succeed");
        drop(view);
    }

    #[test]
    fn test_device_array_distinct_types() {
        // Compile-time test: DeviceArray<i32> and DeviceArray<f32> are distinct types.
        fn takes_i32(_arr: &DeviceArray<i32>) {}
        fn takes_f32(_arr: &DeviceArray<f32>) {}

        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_device_array_distinct_types").expect("entered");
        let arr_i32 = DeviceArray::from_slice(&entered, &[1i32, 2, 3]).expect("i32");
        let arr_f32 = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0]).expect("f32");
        takes_i32(&arr_i32);
        takes_f32(&arr_f32);
    }
}
