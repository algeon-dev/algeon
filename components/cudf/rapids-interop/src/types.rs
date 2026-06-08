use core::{ffi::c_void, fmt, marker::PhantomData, ptr::NonNull};

pub type Result<T> = core::result::Result<T, InteropError>;

/// Logical type identifiers shared by the borrowed interop ABI.
///
/// The legacy raw borrowed constructor validates only the fixed-width subset.
/// The current semantic borrowed compatibility slice additionally supports
/// `STRING`, `LIST`, `STRUCT`, and `DICTIONARY32`.
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

/// High-level reasons why shared interop projection, validation, or lease
/// activation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteropErrorKind {
    UnsupportedType,
    UnsupportedLayout,
    ExecutionBindingUnavailable,
    AmbientInteropUnsupported,
    ExecutionBindingMismatch,
    NullMaskUnsupported,
    InvalidView,
    PlacementUnsupported,
    ReadinessUnavailable,
    ReadinessMismatch,
    MixedDeviceUnsupported,
    ProfileUnsupported,
    LeaseAlreadyActive,
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
    device: i32,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> CudaStreamHandle<'a> {
    /// Reject constructing a borrowed stream handle without CUDA device
    /// identity.
    pub fn from_raw(raw: *mut c_void) -> Result<Self> {
        let _ = NonNull::new(raw).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "CUDA stream handle must be non-null")
                .with_context("field", "cuda_stream")
        })?;
        Err(InteropError::new(
            InteropErrorKind::ReadinessUnavailable,
            "CUDA stream handle must carry explicit device identity",
        )
        .with_context("field", "cuda_stream.device"))
    }

    /// Construct a borrowed stream handle from a raw non-null pointer plus its
    /// CUDA device identity.
    pub fn from_raw_on_device(raw: *mut c_void, device: i32) -> Result<Self> {
        if device < 0 {
            return Err(InteropError::new(InteropErrorKind::InvalidView, "CUDA stream device must be >= 0")
                .with_context("field", "cuda_stream.device")
                .with_context("device", device.to_string()));
        }
        Self::try_new(raw, device)
    }

    fn try_new(raw: *mut c_void, device: i32) -> Result<Self> {
        let raw = NonNull::new(raw).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "CUDA stream handle must be non-null")
                .with_context("field", "cuda_stream")
        })?;
        Ok(Self { raw, device, _lifetime: PhantomData })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    pub fn device(&self) -> i32 {
        self.device
    }
}

// SAFETY: CUDA streams are opaque runtime handles whose API calls are
// internally serialized by the CUDA driver, so the handle itself may be sent
// to or shared with another thread. Note that concurrent *submission* to the
// same stream from multiple threads is serialized by the driver but callers
// should still treat stream ordering as single-producer for correctness.
// This wrapper is borrowed-only: it never destroys the underlying stream and
// carries explicit device identity plus a lifetime tying validity to the
// producer-owned source handle.
unsafe impl Send for CudaStreamHandle<'_> {}
unsafe impl Sync for CudaStreamHandle<'_> {}

/// Borrowed opaque handle to a CUDA event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaEventHandle<'a> {
    raw: NonNull<c_void>,
    device: i32,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> CudaEventHandle<'a> {
    /// Reject constructing a borrowed event handle without CUDA device
    /// identity.
    pub fn from_raw(raw: *mut c_void) -> Result<Self> {
        let _ = NonNull::new(raw).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "CUDA event handle must be non-null")
                .with_context("field", "cuda_event")
        })?;
        Err(InteropError::new(
            InteropErrorKind::ReadinessUnavailable,
            "CUDA event handle must carry explicit device identity",
        )
        .with_context("field", "cuda_event.device"))
    }

    /// Construct a borrowed event handle from a raw non-null pointer plus its
    /// CUDA device identity.
    pub fn from_raw_on_device(raw: *mut c_void, device: i32) -> Result<Self> {
        if device < 0 {
            return Err(InteropError::new(InteropErrorKind::InvalidView, "CUDA event device must be >= 0")
                .with_context("field", "cuda_event.device")
                .with_context("device", device.to_string()));
        }
        Self::try_new(raw, device)
    }

    fn try_new(raw: *mut c_void, device: i32) -> Result<Self> {
        let raw = NonNull::new(raw).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "CUDA event handle must be non-null")
                .with_context("field", "cuda_event")
        })?;
        Ok(Self { raw, device, _lifetime: PhantomData })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    pub fn device(&self) -> i32 {
        self.device
    }
}

// SAFETY: CUDA events are opaque synchronization handles whose record, query,
// and wait operations are internally serialized by the CUDA driver, so the
// handle may safely be sent to or shared with another thread. This wrapper is
// borrowed-only, non-dropping, and lifetime-bound to the producer-owned source
// handle.
unsafe impl Send for CudaEventHandle<'_> {}
unsafe impl Sync for CudaEventHandle<'_> {}

/// Borrowed opaque handle to a device memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryResourceHandle<'a> {
    raw: NonNull<c_void>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> DeviceMemoryResourceHandle<'a> {
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

// SAFETY: RMM memory-resource handles denote allocator objects that are
// expected to support concurrent use. This wrapper is borrowed-only and does
// not own or destroy the underlying allocator; the lifetime ensures the source
// owner outlives the descriptor.
unsafe impl Send for DeviceMemoryResourceHandle<'_> {}
unsafe impl Sync for DeviceMemoryResourceHandle<'_> {}

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

/// Stable symbolic identifier for a semantic table node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableId(u32);

impl TableId {
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

/// Optional schema-level nullability fact distinct from the physical mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaNullability {
    Unknown,
    Nullable,
    NonNullable,
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
    schema_nullability: SchemaNullability,
    layout: ColumnLayout,
}

impl SemanticColumn {
    pub fn new(
        id: ColumnId,
        data_type: DataType,
        len: usize,
        offset: usize,
        validity: PhysicalValidity,
        schema_nullability: SchemaNullability,
        layout: ColumnLayout,
    ) -> Self {
        Self { id, data_type, len, offset, validity, schema_nullability, layout }
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

    pub fn schema_nullability(&self) -> SchemaNullability {
        self.schema_nullability
    }

    pub fn layout(&self) -> &ColumnLayout {
        &self.layout
    }
}

/// Semantic description of one table independent of resource bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticTable {
    id: TableId,
    num_rows: usize,
    columns: Vec<ColumnId>,
    names: Option<Vec<String>>,
}

impl SemanticTable {
    pub fn new(id: TableId, num_rows: usize, columns: Vec<ColumnId>, names: Option<Vec<String>>) -> Self {
        Self { id, num_rows, columns, names }
    }

    pub fn id(&self) -> TableId {
        self.id
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

impl<'a> BoundBytes<'a> {
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

    /// Returns the base address as a read-only raw pointer.
    ///
    /// This descriptor does not claim exclusive or mutable access to the
    /// pointee. Callers that need mutation-capable access must hold `&mut self`
    /// and use [`base_mut_ptr()`](Self::base_mut_ptr).
    pub fn base_ptr(&self) -> *const c_void {
        self.base_ptr.as_ptr() as *const c_void
    }

    /// Returns the base address as a mutable raw pointer.
    ///
    /// Requiring `&mut self` prevents safe code from obtaining multiple
    /// mutation-capable pointers from shared access when `BoundBytes` is shared
    /// across threads.
    pub fn base_mut_ptr(&mut self) -> *mut c_void {
        self.base_ptr.as_ptr()
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

// SAFETY: `BoundBytes` is a non-owning pointer-plus-length descriptor for
// CUDA device memory. CUDA device pointers are valid from any host thread —
// the CUDA runtime does not bind a device pointer to the thread that allocated
// it. The safe surface exposes only read-only raw pointers from `&self`;
// mutable raw access requires `&mut self`, preventing aliased mutable device
// pointers through shared references. The lifetime parameter ties the
// descriptor to the producer-owned allocation, ensuring the wrapper cannot
// outlive the storage it references.
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

/// Producer-side read-start fact for a prepared export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducerReadiness<'a> {
    HostVisible,
    DeviceStreamOrdered { stream: CudaStreamHandle<'a> },
    DeviceEventOrdered { event: CudaEventHandle<'a> },
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
    CrossDeviceEvent { event: CudaEventHandle<'a> },
}

/// Backend-facing borrowed producer-readiness facts captured on an active
/// lease.
///
/// # Safety
///
/// The `raw` pointers are lifetime-erased CUDA stream/event handles.  They
/// are valid only while the `'consumer` lifetime on the enclosing
/// `ActiveColumnReadLease` / `ActiveTableReadLease` is alive.  Do not store
/// these bindings beyond the lease's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseReadiness {
    HostVisible,
    DeviceStreamOrdered { raw: *mut c_void, device: i32 },
    DeviceEventOrdered { raw: *mut c_void, device: i32 },
}

/// Backend-facing read-start authority captured on an active lease.
///
/// # Safety
///
/// Same lifetime-erasure invariant as [`LeaseReadiness`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadStartBinding {
    SynchronousByReturn,
    ConsumerStream { raw: *mut c_void, device: i32 },
}
