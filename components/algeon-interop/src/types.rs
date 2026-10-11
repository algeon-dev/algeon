use core::{ffi::c_void, fmt, marker::PhantomData, ptr::NonNull};
use std::sync::Arc;

pub type Result<T> = core::result::Result<T, InteropError>;

/// Logical type identifiers shared by interop exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeId {
    Empty,
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
            Self::Int32 | Self::UInt32 | Self::Float32 | Self::TimestampDays | Self::DurationDays | Self::Decimal32 => {
                Some(4)
            },
            Self::Int64
            | Self::UInt64
            | Self::Float64
            | Self::TimestampSeconds
            | Self::TimestampMilliseconds
            | Self::TimestampMicroseconds
            | Self::TimestampNanoseconds
            | Self::DurationSeconds
            | Self::DurationMilliseconds
            | Self::DurationMicroseconds
            | Self::DurationNanoseconds
            | Self::Decimal64 => Some(8),
            Self::Decimal128 => Some(16),
            Self::Empty | Self::Dictionary32 | Self::String | Self::List | Self::Struct => None,
        }
    }
}

/// A logical data type for interop column descriptors.
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

/// High-level reasons why shared interop projection, validation, or lease
/// activation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteropErrorKind {
    NotFound,
    UnsupportedType,
    UnsupportedLayout,
    ExecutionBindingUnavailable,
    ExecutionBindingMismatch,
    InvalidView,
    PlacementUnsupported,
    ReadinessUnavailable,
    ReadinessMismatch,
    MixedDeviceUnsupported,
    ProfileUnsupported,
}

/// Semantic error returned by interop validation and producer adapters.
#[derive(Debug, Clone)]
pub struct InteropError {
    kind: InteropErrorKind,
    message: String,
    context: Vec<(&'static str, String)>,
    source: Option<Arc<dyn std::error::Error + Send + Sync + 'static>>,
}

impl InteropError {
    pub fn new(kind: InteropErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), context: Vec::new(), source: None }
    }

    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    pub fn set_source<E>(mut self, source: E) -> Self
    where E: std::error::Error + Send + Sync + 'static {
        self.source = Some(Arc::new(source));
        self
    }

    pub fn kind(&self) -> InteropErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn context(&self) -> &[(&'static str, String)] {
        &self.context
    }
}

impl PartialEq for InteropError {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.message == other.message && self.context == other.context
    }
}

impl Eq for InteropError {}

impl fmt::Display for InteropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for InteropError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|source| source as &(dyn std::error::Error + 'static))
    }
}

fn validate_cuda_handle(
    raw: *mut c_void,
    device: i32,
    handle_name: &'static str,
    device_field: &'static str,
    handle_field: &'static str,
) -> Result<NonNull<c_void>> {
    if device < 0 {
        return Err(InteropError::new(InteropErrorKind::InvalidView, format!("{handle_name} device must be >= 0"))
            .with_context("field", device_field)
            .with_context("device", device.to_string()));
    }

    NonNull::new(raw).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, format!("{handle_name} handle must be non-null"))
            .with_context("field", handle_field)
    })
}

/// Borrowed opaque handle to a CUDA stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStreamHandle<'a> {
    raw: NonNull<c_void>,
    device: i32,
    _lifetime: PhantomData<&'a ()>,
}

impl CudaStreamHandle<'_> {
    /// Construct a borrowed stream handle from a raw non-null pointer plus its
    /// CUDA device identity.
    pub fn from_raw_on_device(raw: *mut c_void, device: i32) -> Result<Self> {
        let raw = validate_cuda_handle(raw, device, "CUDA stream", "cuda_stream.device", "cuda_stream")?;
        Ok(Self { raw, device, _lifetime: PhantomData })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    pub fn device(&self) -> i32 {
        self.device
    }
}

// SAFETY: This crate never dereferences the stream pointer or passes it to
// CUDA; it only copies and compares the address. Sending or sharing the handle
// therefore cannot race inside this crate. `from_raw_on_device` accepts any
// lifetime, so `'a` does not prove that the stream is alive: the producer
// guarantees that. Code that passes the pointer to CUDA relies on that
// guarantee and on CUDA accepting a stream from any host thread.
unsafe impl Send for CudaStreamHandle<'_> {}
unsafe impl Sync for CudaStreamHandle<'_> {}

/// Borrowed opaque handle to a CUDA event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaEventHandle<'a> {
    raw: NonNull<c_void>,
    device: i32,
    _lifetime: PhantomData<&'a ()>,
}

impl CudaEventHandle<'_> {
    /// Construct a borrowed event handle from a raw non-null pointer plus its
    /// CUDA device identity.
    pub fn from_raw_on_device(raw: *mut c_void, device: i32) -> Result<Self> {
        let raw = validate_cuda_handle(raw, device, "CUDA event", "cuda_event.device", "cuda_event")?;
        Ok(Self { raw, device, _lifetime: PhantomData })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    pub fn device(&self) -> i32 {
        self.device
    }
}

// SAFETY: As with `CudaStreamHandle`, this crate never dereferences the event
// pointer or passes it to CUDA. The producer, not `'a`, guarantees that the
// event outlives every use of the handle.
unsafe impl Send for CudaEventHandle<'_> {}
unsafe impl Sync for CudaEventHandle<'_> {}

/// Borrowed opaque handle to a device memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryResourceHandle<'a> {
    raw: NonNull<c_void>,
    _lifetime: PhantomData<&'a ()>,
}

impl DeviceMemoryResourceHandle<'_> {
    /// Construct a borrowed memory-resource handle from a raw non-null pointer.
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

// SAFETY: This crate never dereferences the memory-resource pointer. A caller
// that allocates through it relies on the producer keeping the resource alive
// and on the resource supporting concurrent use; `from_raw` accepts any
// lifetime, so `'a` cannot enforce either.
unsafe impl Send for DeviceMemoryResourceHandle<'_> {}
unsafe impl Sync for DeviceMemoryResourceHandle<'_> {}

/// Execution provenance of a producer runtime: its CUDA stream, and through
/// the stream its CUDA device, plus its memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionBinding<'a> {
    cuda_stream: CudaStreamHandle<'a>,
    mr: DeviceMemoryResourceHandle<'a>,
}

impl<'a> ExecutionBinding<'a> {
    pub fn new(cuda_stream: CudaStreamHandle<'a>, mr: DeviceMemoryResourceHandle<'a>) -> Self {
        Self { cuda_stream, mr }
    }

    pub fn device(&self) -> i32 {
        self.cuda_stream.device()
    }

    pub fn cuda_stream(&self) -> CudaStreamHandle<'a> {
        self.cuda_stream
    }

    pub fn memory_resource(&self) -> DeviceMemoryResourceHandle<'a> {
        self.mr
    }
}

/// Stable symbolic identifier for a bound byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u32);

impl BufferId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable symbolic identifier for a semantic column node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnId(u32);

impl ColumnId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Physical validity representation for a semantic column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalValidity {
    NoValidityBuffer,
    ValidityMask { mask: BufferId, null_count: Option<usize> },
}

/// Physical layout description for a semantic column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnLayout {
    FixedWidth { values: BufferId },
    BinaryLike { offsets: BufferId, data: BufferId },
    List { offsets: BufferId, child: ColumnId },
    Struct { children: Vec<ColumnId> },
    Dictionary { indices: BufferId, keys: ColumnId },
}

/// Semantic description of one column independent of transport handles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticColumn {
    id: ColumnId,
    data_type: DataType,
    len: usize,
    offset: usize,
    validity: PhysicalValidity,
    layout: ColumnLayout,
}

impl SemanticColumn {
    pub fn new(
        id: ColumnId,
        data_type: DataType,
        len: usize,
        offset: usize,
        validity: PhysicalValidity,
        layout: ColumnLayout,
    ) -> Self {
        Self { id, data_type, len, offset, validity, layout }
    }

    pub fn id(&self) -> ColumnId {
        self.id
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
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

    pub fn validity(&self) -> &PhysicalValidity {
        &self.validity
    }

    pub fn layout(&self) -> &ColumnLayout {
        &self.layout
    }
}

/// Semantic description of one table independent of resource bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticTable {
    num_rows: usize,
    columns: Vec<ColumnId>,
    names: Option<Vec<String>>,
}

impl SemanticTable {
    pub fn new(num_rows: usize, columns: Vec<ColumnId>, names: Option<Vec<String>>) -> Self {
        Self { num_rows, columns, names }
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn columns(&self) -> &[ColumnId] {
        &self.columns
    }

    pub fn names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }
}

/// Concrete placement of a bound byte span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Host,
    CudaDevice { device: i32 },
}

/// Borrowed byte span that backs an interop buffer binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundBytes<'a> {
    base_ptr: NonNull<c_void>,
    byte_len: usize,
    _lifetime: PhantomData<&'a ()>,
}

impl BoundBytes<'_> {
    pub fn from_raw_parts(base_ptr: *mut c_void, byte_len: usize) -> Result<Self> {
        let base_ptr = match NonNull::new(base_ptr) {
            Some(ptr) => ptr,
            None if byte_len == 0 => NonNull::dangling(),
            None => {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "bound bytes must use a non-null base pointer",
                )
                .with_context("field", "base_ptr")
                .with_context("byte_len", byte_len.to_string()));
            },
        };
        Ok(Self { base_ptr, byte_len, _lifetime: PhantomData })
    }

    /// Returns the base address. Holding a `BoundBytes` grants no access to
    /// the pointee.
    pub fn base_ptr(&self) -> *const c_void {
        self.base_ptr.as_ptr() as *const c_void
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

// SAFETY: `BoundBytes` is a `Copy` address-plus-length descriptor for host or
// device memory, and this crate never dereferences the address. Moving or
// sharing it across threads copies an integer-like value and cannot race here.
// `from_raw_parts` accepts any lifetime (cudf produces `'static` bindings), so
// the producer, not `'a`, guarantees that the storage outlives every read.
unsafe impl Send for BoundBytes<'_> {}
unsafe impl Sync for BoundBytes<'_> {}

/// Concrete binding from a symbolic buffer id to bytes plus placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferBinding<'a> {
    id: BufferId,
    bytes: BoundBytes<'a>,
    placement: Placement,
}

impl<'a> BufferBinding<'a> {
    pub fn new(id: BufferId, bytes: BoundBytes<'a>, placement: Placement) -> Self {
        Self { id, bytes, placement }
    }

    pub fn id(&self) -> BufferId {
        self.id
    }

    pub fn bytes(&self) -> &BoundBytes<'a> {
        &self.bytes
    }

    pub fn placement(&self) -> &Placement {
        &self.placement
    }
}

/// Device point after which a prepared device export's bytes may be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducerReadiness<'a> {
    DeviceStreamOrdered { stream: CudaStreamHandle<'a> },
    DeviceEventOrdered { event: CudaEventHandle<'a> },
}

impl ProducerReadiness<'_> {
    /// CUDA device of the readiness stream or event, which is the producer
    /// device of the export.
    pub fn device(&self) -> i32 {
        match self {
            Self::DeviceStreamOrdered { stream } => stream.device(),
            Self::DeviceEventOrdered { event } => event.device(),
        }
    }
}

/// Consumer-supplied read-start authority for an activated read lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadStartAuthority<'a> {
    SynchronousByReturn,
    ConsumerStream { stream: CudaStreamHandle<'a> },
}

/// Consumer-supplied completion authority for an activated read lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionAuthority<'a> {
    SynchronousByReturn,
    SameDeviceStream { stream: CudaStreamHandle<'a> },
}
