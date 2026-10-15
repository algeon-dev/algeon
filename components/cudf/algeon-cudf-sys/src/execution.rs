use std::marker::PhantomData;

use crate::{
    CUDF_EXEC_FLAG_DEVICE_EXPLICIT, cudf_event_handle, cudf_exec_options_t, cudf_memory_resource_handle,
    cudf_stream_handle,
};

/// Borrowed CUDA stream owner-handle token.
///
/// Wraps a raw `cudf_stream_handle` bound to lifetime `'a` without taking
/// resource ownership or destroying the underlying stream on drop.
///
/// Implements `Send` but not `Sync`. CUDA stream handles may transfer between
/// threads, while concurrent shared-reference access requires external
/// synchronization in the owning layer.
#[derive(Debug)]
pub struct BorrowedStream<'a> {
    raw: cudf_stream_handle,
    _owner: PhantomData<&'a ()>,
}

// SAFETY: A borrowed CUDA stream handle may be moved to another thread for
// submission/query, but concurrent shared-reference use still requires
// external synchronization by the owning layer.
unsafe impl Send for BorrowedStream<'_> {}

impl BorrowedStream<'_> {
    /// Bind a raw stream owner-handle token to an external lifetime.
    ///
    /// # Safety
    /// - `raw` must refer to a valid, non-destroyed CUDA stream owner handle.
    /// - The underlying stream resource must remain valid for the entire
    ///   lifetime `'a`.
    /// - The caller must not destroy the resource while any
    ///   `BorrowedStream<'a>` derived from it is still live.
    pub unsafe fn from_raw(raw: cudf_stream_handle) -> Self {
        Self { raw, _owner: PhantomData }
    }

    pub fn as_raw(&self) -> cudf_stream_handle {
        self.raw
    }
}

/// Borrowed CUDA event owner-handle token.
///
/// Wraps a raw `cudf_event_handle` bound to lifetime `'a` without taking
/// resource ownership or destroying the underlying event on drop.
#[derive(Debug)]
pub struct BorrowedEvent<'a> {
    raw: cudf_event_handle,
    _owner: PhantomData<&'a ()>,
}

// SAFETY: Borrowed CUDA event handles may be moved across threads and queried
// concurrently; the wrapper does not expose ownership or mutable Rust state.
unsafe impl Send for BorrowedEvent<'_> {}
unsafe impl Sync for BorrowedEvent<'_> {}

impl BorrowedEvent<'_> {
    /// Bind a raw event owner-handle token to an external lifetime.
    ///
    /// # Safety
    /// - `raw` must refer to a valid, non-destroyed CUDA event owner handle.
    /// - The underlying event resource must remain valid for the entire
    ///   lifetime `'a`.
    /// - The caller must not destroy the resource while any
    ///   `BorrowedEvent<'a>` derived from it is still live.
    pub unsafe fn from_raw(raw: cudf_event_handle) -> Self {
        Self { raw, _owner: PhantomData }
    }

    pub fn as_raw(&self) -> cudf_event_handle {
        self.raw
    }
}

/// Borrowed device-memory-resource owner-handle token.
///
/// Wraps a raw `cudf_memory_resource_handle` bound to lifetime `'a` without
/// taking resource ownership or destroying the underlying allocator on drop.
#[derive(Debug)]
pub struct BorrowedMemoryResource<'a> {
    raw: cudf_memory_resource_handle,
    _owner: PhantomData<&'a ()>,
}

// SAFETY: Borrowed memory-resource handles may be moved and shared across
// threads; the wrapper does not transfer ownership and RMM allocators are
// intended for concurrent use.
unsafe impl Send for BorrowedMemoryResource<'_> {}
unsafe impl Sync for BorrowedMemoryResource<'_> {}

impl BorrowedMemoryResource<'_> {
    /// Bind a raw memory-resource owner-handle token to an external lifetime.
    ///
    /// # Safety
    /// - `raw` must refer to a valid, non-destroyed memory-resource owner
    ///   handle.
    /// - The underlying memory resource must remain valid for the entire
    ///   lifetime `'a`.
    /// - The caller must not destroy the resource while any
    ///   `BorrowedMemoryResource<'a>` derived from it is still live.
    pub unsafe fn from_raw(raw: cudf_memory_resource_handle) -> Self {
        Self { raw, _owner: PhantomData }
    }

    pub fn as_raw(&self) -> cudf_memory_resource_handle {
        self.raw
    }
}

/// Typed builder for `cudf_exec_options_t`.
///
/// Null stream / memory-resource fields preserve cuDF defaults for permissive
/// `_ex` calls. Strict `_ex` calls require an explicit device, stream, and
/// memory resource. If an explicit memory resource is attached, it must outlive
/// any outputs allocated through the `_ex` call that consumes this context.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionContext<'a> {
    raw: cudf_exec_options_t,
    _stream: PhantomData<BorrowedStream<'a>>,
    _mr: PhantomData<BorrowedMemoryResource<'a>>,
}

impl<'a> ExecutionContext<'a> {
    pub fn new() -> Self {
        Self { raw: cudf_exec_options_t::defaults(), _stream: PhantomData, _mr: PhantomData }
    }

    pub fn with_device(mut self, device: i32) -> Self {
        self.raw.device = device;
        self.raw.flags |= CUDF_EXEC_FLAG_DEVICE_EXPLICIT;
        self
    }

    pub fn with_stream(mut self, stream: &'a BorrowedStream<'a>) -> Self {
        self.raw.stream = stream.as_raw();
        self
    }

    pub fn with_memory_resource(mut self, mr: &'a BorrowedMemoryResource<'a>) -> Self {
        self.raw.mr = mr.as_raw();
        self
    }

    pub fn raw(&self) -> &cudf_exec_options_t {
        &self.raw
    }

    pub fn as_raw(&self) -> *const cudf_exec_options_t {
        self.raw()
    }
}

impl Default for ExecutionContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}
