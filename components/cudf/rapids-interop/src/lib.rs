//! Borrowed GPU-view interop ABI for RAPIDS Rust crates.
//!
//! This crate defines the v1 shared Rust-native contract for lending
//! GPU-resident column and table views across crate boundaries.
//!
//! # Contract Summary
//!
//! - v1 is a borrowed GPU-view ABI.
//! - Borrowing does not transfer ownership.
//! - Borrowing avoids host copies on the input path and is zero-copy where
//!   possible because it describes existing device buffers.
//! - Consumer algorithms may still allocate or copy for outputs; v1 does not
//!   guarantee zero-copy output.
//! - v1 supports only fixed-width types.
//! - `String`, `List`, `Struct`, and other nested or child-bearing layouts are
//!   unsupported in v1.
//! - Execution binding semantics are part of the contract and must be honored
//!   by consumers before they read borrowed GPU data.
//! - `cudf-rs` producer adapters require explicit execution provenance and
//!   reject ambient/default-only objects at the interop boundary.
//! - User-facing types such as `cudf::Series` and `cudf::DataFrame` may
//!   provide convenience adapters, but they are not the ABI itself.
//!
//! [`BorrowedColumnView`] and [`BorrowedTableView`] are low-level fact records
//! about producer-owned device memory. Consumers must not treat them as
//! ownership-bearing handles or cache them beyond the lifetime of the producer
//! borrow that created them.

use core::{ffi::c_void, fmt, marker::PhantomData, ptr::NonNull};

pub type Result<T> = core::result::Result<T, InteropError>;

/// Logical type identifiers shared by the borrowed interop ABI.
///
/// Borrowed interop v1 validates only the fixed-width subset of these values.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeId {
    Empty = 0,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool8,
    TimestampDays,
    TimestampSeconds,
    TimestampMilliseconds,
    TimestampMicroseconds,
    TimestampNanoseconds,
    DurationDays,
    DurationSeconds,
    DurationMilliseconds,
    DurationMicroseconds,
    DurationNanoseconds,
    Dictionary32,
    String,
    List,
    Decimal32,
    Decimal64,
    Decimal128,
    Struct,
}

impl TypeId {
    pub fn fixed_width_size_bytes(self) -> Option<usize> {
        match self {
            Self::Int8 | Self::UInt8 | Self::Bool8 => Some(1),
            Self::Int16 | Self::UInt16 => Some(2),
            Self::Int32
            | Self::UInt32
            | Self::Float32
            | Self::TimestampDays
            | Self::TimestampSeconds
            | Self::TimestampMilliseconds
            | Self::DurationDays
            | Self::DurationSeconds
            | Self::DurationMilliseconds
            | Self::Decimal32
            | Self::Dictionary32 => Some(4),
            Self::Int64
            | Self::UInt64
            | Self::Float64
            | Self::TimestampMicroseconds
            | Self::TimestampNanoseconds
            | Self::DurationMicroseconds
            | Self::DurationNanoseconds
            | Self::Decimal64 => Some(8),
            Self::Decimal128 => Some(16),
            Self::Empty | Self::String | Self::List | Self::Struct => None,
        }
    }
}

/// A logical data type for borrowed interop descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataType {
    id: TypeId,
    scale: i32,
}

impl DataType {
    pub fn new(id: TypeId) -> Self {
        Self { id, scale: 0 }
    }

    pub fn fixed_point(id: TypeId, scale: i32) -> Self {
        Self { id, scale }
    }

    pub fn id(&self) -> TypeId {
        self.id
    }

    pub fn scale(&self) -> i32 {
        self.scale
    }
}

/// High-level reasons why a borrowed interop projection or validation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteropErrorKind {
    UnsupportedType,
    UnsupportedLayout,
    ExecutionBindingUnavailable,
    AmbientInteropUnsupported,
    ExecutionBindingMismatch,
    NullMaskUnsupported,
    InvalidView,
}

/// Semantic error returned by borrowed interop validation and producer
/// adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteropError {
    kind: InteropErrorKind,
    message: String,
    context: Vec<(&'static str, String)>,
}

impl InteropError {
    pub fn new(kind: InteropErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), context: Vec::new() }
    }

    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    pub fn kind(&self) -> InteropErrorKind {
        self.kind
    }

    pub fn context(&self) -> &[(&'static str, String)] {
        &self.context
    }
}

impl fmt::Display for InteropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for InteropError {}

/// Borrowed opaque handle to a CUDA stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStreamHandle<'a> {
    raw: NonNull<c_void>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> CudaStreamHandle<'a> {
    /// Construct a borrowed stream handle from a raw non-null pointer.
    pub fn from_raw(raw: *mut c_void) -> Result<Self> {
        let raw = NonNull::new(raw).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "CUDA stream handle must be non-null")
                .with_context("field", "cuda_stream")
        })?;
        Ok(Self { raw, _lifetime: PhantomData })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }
}

/// Borrowed opaque handle to a device memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryResourceHandle<'a> {
    raw: NonNull<c_void>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> DeviceMemoryResourceHandle<'a> {
    /// Construct a borrowed memory-resource handle from a raw non-null
    /// pointer.
    pub fn from_raw(raw: *mut c_void) -> Result<Self> {
        let raw = NonNull::new(raw).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "device memory resource handle must be non-null")
                .with_context("field", "mr")
        })?;
        Ok(Self { raw, _lifetime: PhantomData })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }
}

/// Execution metadata that a consumer must honor before reading borrowed GPU
/// data.
///
/// Producers are tied to a specific device, CUDA stream, and memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionBinding<'a> {
    device: i32,
    cuda_stream: CudaStreamHandle<'a>,
    mr: DeviceMemoryResourceHandle<'a>,
}

impl<'a> ExecutionBinding<'a> {
    /// Construct explicit producer execution metadata.
    pub fn explicit(device: i32, cuda_stream: CudaStreamHandle<'a>, mr: DeviceMemoryResourceHandle<'a>) -> Self {
        Self { device, cuda_stream, mr }
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn cuda_stream(&self) -> CudaStreamHandle<'a> {
        self.cuda_stream
    }

    pub fn memory_resource(&self) -> DeviceMemoryResourceHandle<'a> {
        self.mr
    }
}

/// Distinguishes between the absence of a null mask and the presence of a null
/// mask whose null count may still be zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullMaskKind {
    Absent,
    Present { ptr: NonNull<c_void>, bit_len: usize, null_count: usize },
}

/// Lifetime-bound borrowed null-mask descriptor for a column view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NullMask<'a> {
    kind: NullMaskKind,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> NullMask<'a> {
    /// Construct the canonical "no null mask is present" descriptor.
    pub fn absent() -> Self {
        Self { kind: NullMaskKind::Absent, _lifetime: PhantomData }
    }

    /// Construct the canonical "a null mask is present" descriptor.
    ///
    /// `null_count == 0` remains a valid present-mask state.
    pub fn present(ptr: *const c_void, bit_len: usize, null_count: usize) -> Result<Self> {
        let ptr = NonNull::new(ptr as *mut c_void).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "present null mask must use a non-null pointer")
                .with_context("field", "null_mask")
        })?;
        Ok(Self { kind: NullMaskKind::Present { ptr, bit_len, null_count }, _lifetime: PhantomData })
    }

    pub fn kind(&self) -> &NullMaskKind {
        &self.kind
    }
}

/// Borrowed descriptor for a single GPU column view.
///
/// This is a low-level fact record about producer-owned memory. It does not
/// transfer ownership, does not allocate output, and must not outlive the
/// producer borrow that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BorrowedColumnView<'a> {
    data_type: DataType,
    len: usize,
    offset: usize,
    data_ptr: *const c_void,
    null_mask: NullMask<'a>,
    execution: ExecutionBinding<'a>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> BorrowedColumnView<'a> {
    /// Validate and construct a borrowed column descriptor.
    ///
    /// Borrowed interop v1 accepts only fixed-width types and rejects invalid
    /// physical view states such as null data pointers for non-empty views,
    /// overflowing active ranges, misaligned active addresses, and undersized
    /// null masks.
    pub fn try_new(
        data_type: DataType,
        len: usize,
        offset: usize,
        data_ptr: *const c_void,
        null_mask: NullMask<'a>,
        execution: ExecutionBinding<'a>,
    ) -> Result<Self> {
        let elem_size = data_type.id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::UnsupportedType, "borrowed interop v1 supports fixed-width types only")
                .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        })?;

        if len > 0 && data_ptr.is_null() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "non-empty view must use a non-null data pointer",
            )
            .with_context("len", len.to_string())
            .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
        }

        let active_start = offset.checked_mul(elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "active byte range overflowed while computing offset")
                .with_context("offset", offset.to_string())
                .with_context("elem_size", elem_size.to_string())
        })?;
        let active_len = len.checked_mul(elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "active byte range overflowed while computing length")
                .with_context("len", len.to_string())
                .with_context("elem_size", elem_size.to_string())
        })?;
        let _active_end = active_start.checked_add(active_len).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "active byte range overflowed while computing end offset")
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
                .with_context("elem_size", elem_size.to_string())
        })?;

        if !data_ptr.is_null() {
            let active_addr = (data_ptr as usize).checked_add(active_start).ok_or_else(|| {
                InteropError::new(InteropErrorKind::InvalidView, "active data address overflowed")
                    .with_context("offset", offset.to_string())
                    .with_context("elem_size", elem_size.to_string())
            })?;
            if active_addr % elem_size != 0 {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "active data pointer is misaligned for declared type",
                )
                .with_context("address", active_addr.to_string())
                .with_context("alignment", elem_size.to_string())
                .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
            }
        }

        match null_mask.kind() {
            NullMaskKind::Absent => {},
            NullMaskKind::Present { bit_len, null_count, .. } => {
                if *null_count > len {
                    return Err(InteropError::new(InteropErrorKind::InvalidView, "null count exceeds logical length")
                        .with_context("null_count", null_count.to_string())
                        .with_context("len", len.to_string()));
                }
                let required_bits = offset.checked_add(len).ok_or_else(|| {
                    InteropError::new(InteropErrorKind::InvalidView, "required null-mask range overflowed")
                        .with_context("offset", offset.to_string())
                        .with_context("len", len.to_string())
                })?;
                if *bit_len < required_bits {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "null mask is too small for the active logical range",
                    )
                    .with_context("bit_len", bit_len.to_string())
                    .with_context("required_bits", required_bits.to_string()));
                }
            },
        }

        Ok(Self { data_type, len, offset, data_ptr, null_mask, execution, _lifetime: PhantomData })
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn type_id(&self) -> TypeId {
        self.data_type.id()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn data_ptr(&self) -> *const c_void {
        self.data_ptr
    }

    pub fn null_mask(&self) -> &NullMask<'a> {
        &self.null_mask
    }

    pub fn execution(&self) -> ExecutionBinding<'a> {
        self.execution
    }

    pub fn element_size_bytes(&self) -> usize {
        self.type_id().fixed_width_size_bytes().expect("BorrowedColumnView always holds a fixed-width type")
    }
}

/// Borrowed descriptor for a GPU table view.
///
/// The table owns its descriptor collection so consumers do not need to keep
/// parallel temporary vectors alive just to preserve column descriptor
/// lifetimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowedTableView<'a> {
    num_rows: usize,
    columns: Vec<BorrowedColumnView<'a>>,
    execution: ExecutionBinding<'a>,
}

impl<'a> BorrowedTableView<'a> {
    /// Validate and construct a borrowed table descriptor whose columns all
    /// share the same row count and execution binding.
    pub fn try_new(
        num_rows: usize,
        columns: Vec<BorrowedColumnView<'a>>,
        execution: ExecutionBinding<'a>,
    ) -> Result<Self> {
        for (index, column) in columns.iter().enumerate() {
            if column.len() != num_rows {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "column length does not match borrowed table row count",
                )
                .with_context("column_index", index.to_string())
                .with_context("column_len", column.len().to_string())
                .with_context("num_rows", num_rows.to_string()));
            }
            if column.execution() != execution {
                return Err(InteropError::new(
                    InteropErrorKind::ExecutionBindingMismatch,
                    "all borrowed table columns must share the table execution binding",
                )
                .with_context("column_index", index.to_string()));
            }
        }

        Ok(Self { num_rows, columns, execution })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[BorrowedColumnView<'a>] {
        &self.columns
    }

    pub fn execution(&self) -> ExecutionBinding<'a> {
        self.execution
    }
}

/// Fallible producer adapter for a borrowed column descriptor.
///
/// The returned descriptor lifetime is intentionally tied to the specific
/// borrow of `self`.
pub trait TryBorrowColumnView {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>>;
}

/// Fallible producer adapter for a borrowed table descriptor.
///
/// The returned descriptor lifetime is intentionally tied to the specific
/// borrow of `self`.
pub trait TryBorrowTableView {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>>;
}

impl<'a> TryBorrowColumnView for BorrowedColumnView<'a> {
    fn try_borrow_column_view<'b>(&'b self) -> Result<BorrowedColumnView<'b>> {
        BorrowedColumnView::try_new(
            self.data_type(),
            self.len(),
            self.offset(),
            self.data_ptr(),
            *self.null_mask(),
            self.execution(),
        )
    }
}

impl<'a> TryBorrowTableView for BorrowedTableView<'a> {
    fn try_borrow_table_view<'b>(&'b self) -> Result<BorrowedTableView<'b>> {
        let columns =
            self.columns().iter().map(|column| column.try_borrow_column_view()).collect::<Result<Vec<_>>>()?;
        BorrowedTableView::try_new(self.num_rows(), columns, self.execution())
    }
}
