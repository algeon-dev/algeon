use std::{marker::PhantomData, os::raw::c_void};

use crate::core::types::TypeId;

/// A typed descriptor of a contiguous region of GPU device memory.
///
/// `DeviceSlice` bundles a device pointer, element count, and type descriptor.
/// It does not copy or dereference the pointer. Descriptors borrow their source
/// column immutably; obtaining `&mut` access to the slice does not guarantee
/// exclusive access to the underlying device allocation.
///
/// The lifetime `'a` ties this slice to the source [`Column`](crate::core::Column),
/// preventing access after the allocation is freed.
///
/// For interop across crates, [`rapids_interop::PreparedColumnExport`] via
/// [`Column::try_prepare_device_column_export`](crate::core::Column::try_prepare_device_column_export)
/// provides null-mask, offset, and execution provenance semantics.
#[derive(Debug)]
pub struct DeviceSlice<'a> {
    pub(crate) ptr: *mut c_void,
    pub(crate) len: usize,
    pub(crate) type_id: TypeId,
    pub(crate) elem_size: usize,
    pub(crate) _lifetime: PhantomData<&'a ()>,
}

// SAFETY: Every safe operation on this type reads the host-side metadata
// (ptr, len, type_id, elem_size), which is immutable after construction, and
// returns the address without touching device memory or a CUDA stream, so
// sharing or sending the descriptor cannot race on the host. This reasoning
// covers the descriptor alone and says nothing about what a caller does with
// the returned pointer. A method that launches a kernel, enqueues a copy, or
// binds a stream would require re-evaluating these impls for thread and
// CUDA-context affinity.
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

    /// Returns the device address as a read-only pointer.
    ///
    /// The pointer refers to GPU memory and is intended for passing to GPU
    /// libraries (e.g., cugraph) that accept device pointers. It carries no
    /// null-mask semantics, original column offset, or execution provenance.
    /// A shared reference yields `*const` because `DeviceSlice` is `Sync`;
    /// see [`as_mut_ptr()`](Self::as_mut_ptr) for the mutable address.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }

    /// Returns the device address without granting exclusive access to the
    /// allocation.
    ///
    /// `&mut self` prevents obtaining the mutable address through a shared
    /// reference to this descriptor and nothing more; other descriptors of the
    /// same column may exist at the same time. Native writes through this
    /// pointer require the caller to preserve
    /// storage lifetime and coordinate access with other descriptors and with
    /// GPU work on the same allocation.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
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
