use std::{marker::PhantomData, os::raw::c_void};

use crate::core::types::TypeId;

/// An opaque descriptor of a contiguous region of GPU device memory.
///
/// `DeviceSlice` is a safe, typed handle that carries a device pointer, element
/// count, and type information. It does **not** copy or dereference the pointer —
/// device pointers are not accessible from host code, so holding one cannot cause
/// undefined behavior.
///
/// This is a low-level raw-pointer helper, not the preferred cross-crate GPU
/// interop contract. For external borrowed interop, prefer
/// [`crate::interop::BorrowedColumnView`] and
/// [`crate::interop::TryBorrowColumnView`], which also carry null-mask
/// semantics, base-pointer-plus-offset behavior, and execution provenance.
///
/// Create a `DeviceSlice` from a [`ColumnView`] via
/// [`device_slice()`](ColumnView::device_slice). Use [`split_at()`](Self::split_at)
/// to partition it into sub-regions (e.g., source and destination vertex IDs).
///
/// `DeviceSlice` is `Send + Sync` because the underlying data lives on the GPU
/// and is not host-dereferenceable.
///
/// The lifetime `'a` ties this slice to the [`ColumnView`] (or [`Column`]) that
/// produced it, preventing the slice from outliving the GPU memory it points to.
#[derive(Debug)]
pub struct DeviceSlice<'a> {
    pub(crate) ptr: *mut c_void,
    pub(crate) len: usize,
    pub(crate) type_id: TypeId,
    pub(crate) elem_size: usize,
    pub(crate) _lifetime: PhantomData<&'a ()>,
}

// SAFETY: Device pointers are not dereferenceable from host code, so sharing
// across threads cannot create a host-side race on device memory. The host-side
// metadata (ptr, len, type_id, elem_size) is immutable after construction, and
// the lifetime parameter ensures the underlying GPU memory remains valid.
//
// NOTE: Unlike `ColumnView` and `TableView`, which are `!Send + !Sync`,
// `DeviceSlice` opts _in_ to Send + Sync. This is an intentional design choice:
// `DeviceSlice` is a thin descriptor (pointer + length + type tag) with no FFI
// methods that perform device operations, so there is no risk of CUDA-context or
// stream misuse from another thread. It exists solely to pass device pointers to
// external GPU libraries (e.g., cugraph).
//
// If `DeviceSlice` ever gains methods that launch kernels, enqueue copies, or
// interact with a CUDA stream, the Send + Sync impls must be re-evaluated —
// those operations may require stream synchronization or CUDA-context affinity
// that a cross-thread send would violate.
unsafe impl Send for DeviceSlice<'_> {}
unsafe impl Sync for DeviceSlice<'_> {}

impl DeviceSlice<'_> {
    /// Number of elements in this device slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this slice contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The cudf type of the elements.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Returns the raw device pointer.
    ///
    /// This pointer refers to GPU memory and cannot be dereferenced from host
    /// code. It is intended for passing to GPU libraries (e.g., cugraph) that
    /// accept device pointers.
    ///
    /// This method exposes only a pointer. It does not communicate null-mask
    /// semantics, original column offset, or execution provenance.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Returns the size in bytes of one element.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// Split this slice into two at the given element index.
    ///
    /// Returns `(left, right)` where `left` has `mid` elements and `right`
    /// has `len - mid` elements.
    ///
    /// # Panics
    ///
    /// Panics if `mid > self.len`.
    pub fn split_at(&self, mid: usize) -> (DeviceSlice<'_>, DeviceSlice<'_>) {
        assert!(mid <= self.len, "split_at: mid ({mid}) > len ({})", self.len);

        // SAFETY: Pointer arithmetic within the device buffer range. mid <= len
        // is asserted above. The result is never dereferenced on the host.
        let right_ptr = if self.ptr.is_null() || self.elem_size == 0 {
            self.ptr
        } else {
            unsafe { (self.ptr as *const u8).add(mid * self.elem_size) as *mut c_void }
        };

        (
            DeviceSlice {
                ptr: self.ptr,
                len: mid,
                type_id: self.type_id,
                elem_size: self.elem_size,
                _lifetime: PhantomData,
            },
            DeviceSlice {
                ptr: right_ptr,
                len: self.len - mid,
                type_id: self.type_id,
                elem_size: self.elem_size,
                _lifetime: PhantomData,
            },
        )
    }
}
