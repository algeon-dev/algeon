//! Shared interop contract for RAPIDS Rust crates.
//!
//! This crate defines:
//!
//! - a borrowed GPU-view compatibility profile via [`BorrowedColumnView`] and
//!   [`BorrowedTableView`]
//! - a producer-rooted prepared export profile via [`PreparedColumnExport`] and
//!   [`PreparedTableExport`]
//! - activated read leases with explicit completion via
//!   [`ActiveColumnReadLease`] and [`ActiveTableReadLease`]
//! - an owned device transfer profile via [`OwnedColumnTransfer`] and
//!   [`OwnedTableTransfer`]
//! - explicit partitioned multi-device wrappers via
//!   [`PartitionedPreparedTableExport`] and
//!   [`PartitionedOwnedTableTransfer`]
//!
//! # Contract Summary
//!
//! - Borrowed compatibility and prepared export both describe existing buffers;
//!   neither transfers ownership.
//! - Borrowed compatibility covers low-level borrowed descriptors and their
//!   explicit execution/read-start facts.
//! - Prepared export plus activated lease covers read-start, completion, and
//!   backend-defined incomplete-drop behavior at the shared contract layer.
//! - Owned device transfer is a separate ownership-bearing contract with an
//!   explicit release contract.
//! - Partitioned multi-device export is explicit composition over multiple
//!   single-device table roots; it does not weaken the meaning of one
//!   [`PreparedTableExport`] or one [`OwnedTableTransfer`].
//! - Current device-export support covers fixed-width layouts, the binary-like
//!   string layout slice, `List`, `Struct`, and `Dictionary`.
//! - Nested-layout support is still intentionally narrow in this slice:
//!   current coverage is the recursive semantic-column graph that can be
//!   expressed by the existing fixed-width, binary-like string, list, struct,
//!   and dictionary slices on one CUDA device.
//! - Consumer algorithms may still allocate or copy for outputs; this crate
//!   does not promise zero-copy output.
//! - `cudf-rs` producer adapters require explicit execution provenance and
//!   reject ambient/default-only objects at the interop boundary.
//! - User-facing types such as `cudf::Series` and `cudf::DataFrame` may
//!   provide convenience adapters, but they are not the shared contract.
//!
//! [`BorrowedColumnView`] and [`BorrowedTableView`] are low-level fact records
//! about producer-owned device memory. Consumers must not treat them as
//! ownership-bearing handles or cache them beyond the lifetime of the producer
//! borrow that created them.
//!
//! [`PreparedColumnExport`] and [`PreparedTableExport`] are producer-rooted
//! export contracts. Consumers activate access explicitly by providing
//! [`CompletionAuthority`], then finish via [`ActiveColumnReadLease::complete`]
//! or [`ActiveTableReadLease::complete`], which returns a shared
//! [`LeaseCompletionResult`].

use core::{ffi::c_void, fmt, marker::PhantomData, ptr::NonNull};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub type Result<T> = core::result::Result<T, InteropError>;

/// Logical type identifiers shared by the borrowed interop ABI.
///
/// The current borrowed compatibility profile validates only the fixed-width
/// subset of these values.
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
            | Self::Decimal32 => Some(4),
            Self::Int64
            | Self::UInt64
            | Self::Float64
            | Self::TimestampMicroseconds
            | Self::TimestampNanoseconds
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

    pub fn base_ptr(&self) -> NonNull<c_void> {
        self.base_ptr
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

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

/// Consumer-supplied completion authority for an activated read lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionAuthority<'a> {
    SynchronousByReturn,
    SameDeviceStream { stream: CudaStreamHandle<'a> },
    CrossDeviceEvent { event: CudaEventHandle<'a> },
}

/// Backend-facing borrowed producer-readiness facts captured on an active
/// lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseReadiness {
    HostVisible,
    DeviceStreamOrdered { raw: *mut c_void, device: i32 },
    DeviceEventOrdered { raw: *mut c_void, device: i32 },
}

/// Backend-facing producer export facts retained for the lifetime of an active
/// lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseExportContext {
    producer_device: Option<i32>,
    readiness: LeaseReadiness,
}

impl LeaseExportContext {
    pub const fn new(producer_device: Option<i32>, readiness: LeaseReadiness) -> Self {
        Self { producer_device, readiness }
    }

    pub fn producer_device(&self) -> Option<i32> {
        self.producer_device
    }

    pub fn readiness(&self) -> LeaseReadiness {
        self.readiness
    }
}

/// Backend-facing completion authority captured on an active lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionBinding {
    SynchronousByReturn,
    SameDeviceStream { raw: *mut c_void, device: i32 },
    CrossDeviceEvent { raw: *mut c_void, device: i32 },
}

/// Classification of the completion witness produced by a finished lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionWitnessKind {
    ConsumerStream,
    ConsumerEvent,
}

/// Backend operations supported by a producer-owned completion witness.
pub trait CompletionWitnessBackend: Send + Sync {
    fn is_ready(&self) -> Result<bool>;
    fn synchronize(&self) -> Result<()>;
}

#[derive(Debug)]
struct ReadyCompletionWitnessBackend;

impl CompletionWitnessBackend for ReadyCompletionWitnessBackend {
    fn is_ready(&self) -> Result<bool> {
        Ok(true)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// Producer-owned completion witness returned by a completed lease.
#[derive(Clone)]
pub struct CompletionWitness {
    kind: CompletionWitnessKind,
    backend: Arc<dyn CompletionWitnessBackend>,
}

impl CompletionWitness {
    pub fn new(kind: CompletionWitnessKind) -> Self {
        Self { kind, backend: Arc::new(ReadyCompletionWitnessBackend) }
    }

    pub fn with_backend(kind: CompletionWitnessKind, backend: Arc<dyn CompletionWitnessBackend>) -> Self {
        Self { kind, backend }
    }

    pub const fn kind(&self) -> CompletionWitnessKind {
        self.kind
    }

    pub fn is_ready(&self) -> Result<bool> {
        self.backend.is_ready()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.backend.synchronize()
    }
}

impl fmt::Debug for CompletionWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletionWitness").field("kind", &self.kind).finish()
    }
}

impl PartialEq for CompletionWitness {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for CompletionWitness {}

/// Describes how a backend handles active leases that are dropped before
/// explicit completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompleteDropBehavior {
    GuaranteedFallback,
    ExplicitCompletionRequired,
}

/// Backend-owned activation state retained for the lifetime of one active
/// lease.
pub trait ActiveLeaseActivation: Send + Sync {}

/// Backend activation result retained by an active lease.
#[derive(Clone, Default)]
pub struct LeaseActivation {
    state: Option<Arc<dyn ActiveLeaseActivation>>,
}

impl LeaseActivation {
    pub fn noop() -> Self {
        Self { state: None }
    }

    pub fn with_state(state: Arc<dyn ActiveLeaseActivation>) -> Self {
        Self { state: Some(state) }
    }
}

impl fmt::Debug for LeaseActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeaseActivation").field("has_backend_state", &self.state.is_some()).finish()
    }
}

/// Backend implementation for shared active read leases.
pub trait ActiveLeaseBackend: Send + Sync {
    fn activate(&self, _export: &LeaseExportContext, _completion: CompletionBinding) -> Result<LeaseActivation> {
        Ok(LeaseActivation::noop())
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior;
    fn complete(&self, export: &LeaseExportContext, completion: CompletionBinding) -> Result<LeaseCompletionResult>;
    fn drop_incomplete(&self, export: &LeaseExportContext, completion: CompletionBinding) -> Result<()>;
}

#[derive(Debug, Clone, Copy)]
struct SemanticLeaseBackend {
    incomplete_drop_behavior: IncompleteDropBehavior,
}

impl ActiveLeaseBackend for SemanticLeaseBackend {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.incomplete_drop_behavior
    }

    fn complete(&self, _export: &LeaseExportContext, completion: CompletionBinding) -> Result<LeaseCompletionResult> {
        Ok(match completion {
            CompletionBinding::SynchronousByReturn => LeaseCompletionResult::FallbackSynchronized,
            CompletionBinding::SameDeviceStream { .. } => {
                LeaseCompletionResult::Completed(CompletionWitness::new(CompletionWitnessKind::ConsumerStream))
            },
            CompletionBinding::CrossDeviceEvent { .. } => {
                LeaseCompletionResult::Completed(CompletionWitness::new(CompletionWitnessKind::ConsumerEvent))
            },
        })
    }

    fn drop_incomplete(&self, _export: &LeaseExportContext, _completion: CompletionBinding) -> Result<()> {
        Ok(())
    }
}

/// Shared backend descriptor attached to prepared exports.
#[derive(Clone)]
pub struct LeaseBackend {
    inner: Arc<dyn ActiveLeaseBackend>,
}

impl LeaseBackend {
    pub fn new<B>(backend: B) -> Self
    where B: ActiveLeaseBackend + 'static {
        Self { inner: Arc::new(backend) }
    }

    /// Built-in shared backend that treats incomplete lease drop as safely
    /// fallback-completable.
    pub fn semantic_guaranteed_fallback() -> Self {
        Self::new(SemanticLeaseBackend { incomplete_drop_behavior: IncompleteDropBehavior::GuaranteedFallback })
    }

    /// Built-in shared backend that requires explicit lease completion and
    /// performs no drop-time fallback.
    pub fn semantic_explicit_completion_required() -> Self {
        Self::new(SemanticLeaseBackend { incomplete_drop_behavior: IncompleteDropBehavior::ExplicitCompletionRequired })
    }

    fn semantic_default() -> Self {
        Self::semantic_guaranteed_fallback()
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.inner.incomplete_drop_behavior()
    }

    fn activate(&self, export: &LeaseExportContext, completion: CompletionBinding) -> Result<LeaseActivation> {
        self.inner.activate(export, completion)
    }

    fn complete(&self, export: &LeaseExportContext, completion: CompletionBinding) -> Result<LeaseCompletionResult> {
        self.inner.complete(export, completion)
    }

    fn drop_incomplete(&self, export: &LeaseExportContext, completion: CompletionBinding) -> Result<()> {
        self.inner.drop_incomplete(export, completion)
    }
}

impl fmt::Debug for LeaseBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeaseBackend").field("incomplete_drop_behavior", &self.incomplete_drop_behavior()).finish()
    }
}

/// Shared completion outcome for an active read lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseCompletionResult {
    Completed(CompletionWitness),
    FallbackSynchronized,
}

/// Minimal diagnostics surface for the interop boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportProfile {
    BorrowedDeviceCompatibility,
    PreparedDeviceExport,
    PreparedHostExport,
    OwnedDeviceTransfer,
}

/// Minimal profile/debug metadata for an export root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportDiagnostics {
    profile: ExportProfile,
}

impl ExportDiagnostics {
    pub fn new(profile: ExportProfile) -> Self {
        Self { profile }
    }

    pub fn profile(&self) -> ExportProfile {
        self.profile
    }
}

#[derive(Clone, Default)]
struct RetainedHostBuffers {
    buffers: Vec<Arc<[u8]>>,
}

impl RetainedHostBuffers {
    fn new(buffers: Vec<Arc<[u8]>>) -> Self {
        Self { buffers }
    }
}

impl fmt::Debug for RetainedHostBuffers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedHostBuffers").field("count", &self.buffers.len()).finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferCompletionRule {
    CallerMustReleaseAfterConsumerWorkCompletes,
    ReleaseWaitsOnConsumerSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseSynchronization {
    Asynchronous,
    MaySynchronizeInternally,
    RequiresCallerSynchronization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseContract {
    release_device: i32,
    completion: TransferCompletionRule,
    synchronization: ReleaseSynchronization,
}

impl ReleaseContract {
    pub fn new(
        release_device: i32,
        completion: TransferCompletionRule,
        synchronization: ReleaseSynchronization,
    ) -> Self {
        Self { release_device, completion, synchronization }
    }

    pub fn release_device(&self) -> i32 {
        self.release_device
    }

    pub fn completion(&self) -> TransferCompletionRule {
        self.completion
    }

    pub fn synchronization(&self) -> ReleaseSynchronization {
        self.synchronization
    }
}

struct OwnedTransferOwner {
    state: Box<dyn Send + Sync>,
}

impl OwnedTransferOwner {
    fn new<Owner>(owner: Owner) -> Self
    where Owner: Send + Sync + 'static {
        Self { state: Box::new(owner) }
    }
}

impl fmt::Debug for OwnedTransferOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = &self.state;
        f.debug_struct("OwnedTransferOwner").finish()
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
    /// The current borrowed compatibility profile accepts only fixed-width
    /// types and rejects invalid physical view states such as null data
    /// pointers for non-empty views, overflowing active ranges, misaligned
    /// active addresses, and undersized null masks.
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

fn export_profile_name(profile: ExportProfile) -> &'static str {
    match profile {
        ExportProfile::BorrowedDeviceCompatibility => "borrowed_device_compatibility",
        ExportProfile::PreparedDeviceExport => "prepared_device_export",
        ExportProfile::PreparedHostExport => "prepared_host_export",
        ExportProfile::OwnedDeviceTransfer => "owned_device_transfer",
    }
}

fn profile_error(profile: ExportProfile, message: &'static str) -> InteropError {
    InteropError::new(InteropErrorKind::ProfileUnsupported, message)
        .with_context("profile", export_profile_name(profile))
}

fn prepared_profile_error(profile: ExportProfile, message: &'static str) -> InteropError {
    profile_error(profile, message)
}

fn owned_transfer_profile_error(profile: ExportProfile, message: &'static str) -> InteropError {
    profile_error(profile, message)
}

fn validate_prepared_profile(diagnostics: ExportDiagnostics) -> Result<ExportProfile> {
    match diagnostics.profile() {
        ExportProfile::PreparedDeviceExport | ExportProfile::PreparedHostExport => Ok(diagnostics.profile()),
        other => Err(prepared_profile_error(
            other,
            "prepared export values must declare a prepared export profile in diagnostics",
        )),
    }
}

fn validate_owned_transfer_profile(diagnostics: ExportDiagnostics) -> Result<ExportProfile> {
    match diagnostics.profile() {
        ExportProfile::OwnedDeviceTransfer => Ok(diagnostics.profile()),
        other => Err(owned_transfer_profile_error(
            other,
            "owned transfer values must declare an owned device transfer profile in diagnostics",
        )),
    }
}

fn validate_prepared_readiness<'a>(
    profile: ExportProfile,
    readiness: &ProducerReadiness<'a>,
    producer_device: Option<i32>,
) -> Result<()> {
    match profile {
        ExportProfile::PreparedDeviceExport => {
            let producer_device = producer_device.expect("prepared device export must resolve a producer device");
            match readiness {
                ProducerReadiness::DeviceStreamOrdered { stream } => {
                    let stream_device = stream.device();
                    if stream_device != producer_device {
                        return Err(InteropError::new(
                            InteropErrorKind::ReadinessMismatch,
                            "prepared export readiness stream must live on the producer CUDA device",
                        )
                        .with_context("readiness", "device_stream_ordered")
                        .with_context("producer_device", producer_device.to_string())
                        .with_context("stream_device", stream_device.to_string()));
                    }
                    Ok(())
                },
                ProducerReadiness::HostVisible => {
                    Err(prepared_profile_error(profile, "prepared device export requires device-ordered readiness")
                        .with_context("readiness", "host_visible"))
                },
                ProducerReadiness::DeviceEventOrdered { event } => {
                    let event_device = event.device();
                    if event_device != producer_device {
                        return Err(InteropError::new(
                            InteropErrorKind::ReadinessMismatch,
                            "prepared export readiness event must live on the producer CUDA device",
                        )
                        .with_context("readiness", "device_event_ordered")
                        .with_context("producer_device", producer_device.to_string())
                        .with_context("event_device", event_device.to_string()));
                    }
                    Ok(())
                },
            }
        },
        ExportProfile::PreparedHostExport => match readiness {
            ProducerReadiness::HostVisible => Ok(()),
            ProducerReadiness::DeviceStreamOrdered { .. } => {
                Err(prepared_profile_error(profile, "prepared host export requires host-visible readiness")
                    .with_context("readiness", "device_stream_ordered"))
            },
            ProducerReadiness::DeviceEventOrdered { .. } => {
                Err(prepared_profile_error(profile, "prepared host export requires host-visible readiness")
                    .with_context("readiness", "device_event_ordered"))
            },
        },
        ExportProfile::BorrowedDeviceCompatibility => unreachable!("borrowed compatibility is not a prepared export"),
        ExportProfile::OwnedDeviceTransfer => unreachable!("owned transfer is not a prepared export"),
    }
}

fn validate_owned_transfer_readiness<'a>(readiness: &ProducerReadiness<'a>, producer_device: i32) -> Result<()> {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            let stream_device = stream.device();
            if stream_device != producer_device {
                return Err(InteropError::new(
                    InteropErrorKind::ReadinessMismatch,
                    "owned device transfer readiness stream must live on the producer CUDA device",
                )
                .with_context("readiness", "device_stream_ordered")
                .with_context("producer_device", producer_device.to_string())
                .with_context("stream_device", stream_device.to_string()));
            }
            Ok(())
        },
        ProducerReadiness::DeviceEventOrdered { event } => {
            let event_device = event.device();
            if event_device != producer_device {
                return Err(InteropError::new(
                    InteropErrorKind::ReadinessMismatch,
                    "owned device transfer readiness event must live on the producer CUDA device",
                )
                .with_context("readiness", "device_event_ordered")
                .with_context("producer_device", producer_device.to_string())
                .with_context("event_device", event_device.to_string()));
            }
            Ok(())
        },
        ProducerReadiness::HostVisible => Err(owned_transfer_profile_error(
            ExportProfile::OwnedDeviceTransfer,
            "owned device transfer requires device-ordered readiness",
        )
        .with_context("readiness", "host_visible")),
    }
}

fn binding_map<'a>(bindings: &'a [BufferBinding<'a>]) -> Result<HashMap<BufferId, &'a BufferBinding<'a>>> {
    let mut map = HashMap::with_capacity(bindings.len());
    for binding in bindings {
        if map.insert(binding.id(), binding).is_some() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export bindings must not reuse one buffer id",
            )
            .with_context("buffer_id", binding.id().get().to_string()));
        }
    }
    Ok(map)
}

fn require_binding<'a>(
    map: &HashMap<BufferId, &'a BufferBinding<'a>>,
    buffer_id: BufferId,
) -> Result<&'a BufferBinding<'a>> {
    map.get(&buffer_id).copied().ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export referenced a missing buffer binding")
            .with_context("buffer_id", buffer_id.get().to_string())
    })
}

fn fixed_width_required_bytes(column: &SemanticColumn) -> Result<usize> {
    let elem_size = column.data_type().id().fixed_width_size_bytes().ok_or_else(|| {
        InteropError::new(
            InteropErrorKind::UnsupportedType,
            "prepared device export supports fixed-width logical types only for fixed-width layouts",
        )
        .with_context("column_id", column.id().get().to_string())
        .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase())
    })?;
    let active_elements = column.offset().checked_add(column.len()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export active element range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })?;
    active_elements.checked_mul(elem_size).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export active byte range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
            .with_context("elem_size", elem_size.to_string())
    })
}

fn binary_like_offsets_required_bytes(column: &SemanticColumn) -> Result<usize> {
    let required_offsets =
        column.offset().checked_add(column.len()).and_then(|v| v.checked_add(1)).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "prepared export binary-like offsets range overflowed")
                .with_context("column_id", column.id().get().to_string())
                .with_context("offset", column.offset().to_string())
                .with_context("len", column.len().to_string())
        })?;
    required_offsets.checked_mul(core::mem::size_of::<i32>()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export binary-like offsets byte range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })
}

fn dictionary_indices_required_bytes(column: &SemanticColumn) -> Result<usize> {
    if column.data_type().id() != TypeId::Dictionary32 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedType,
            "prepared device export dictionary layout requires the dictionary32 logical type in the current slice",
        )
        .with_context("column_id", column.id().get().to_string())
        .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
    }

    let active_elements = column.offset().checked_add(column.len()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export dictionary index range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })?;
    active_elements.checked_mul(core::mem::size_of::<i32>()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export dictionary index byte range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })
}

fn validate_semantic_layout_matches_data_type(column: &SemanticColumn) -> Result<()> {
    match column.layout() {
        ColumnLayout::FixedWidth { .. } => {
            if column.data_type().id().fixed_width_size_bytes().is_none() {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export fixed-width layout requires a fixed-width logical type",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
        ColumnLayout::BinaryLike { .. } => {
            if column.data_type().id() != TypeId::String {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export binary-like layout requires the string logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
        ColumnLayout::List { .. } => {
            if column.data_type().id() != TypeId::List {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export list layout requires the list logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
        ColumnLayout::Struct { .. } => {
            if column.data_type().id() != TypeId::Struct {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export struct layout requires the struct logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
        ColumnLayout::Dictionary { .. } => {
            if column.data_type().id() != TypeId::Dictionary32 {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export dictionary layout requires the dictionary32 logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
    }

    Ok(())
}

fn semantic_column_map<'a>(
    columns: &'a [SemanticColumn],
    context: &'static str,
) -> Result<HashMap<ColumnId, &'a SemanticColumn>> {
    let mut map = HashMap::with_capacity(columns.len());
    for column in columns {
        if map.insert(column.id(), column).is_some() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export semantic columns must not reuse one semantic column id",
            )
            .with_context(context, column.id().get().to_string()));
        }
    }
    Ok(map)
}

fn require_cuda_device<'a>(
    binding: &'a BufferBinding<'a>,
    kind: InteropErrorKind,
) -> Result<(i32, &'a BufferBinding<'a>)> {
    match binding.placement() {
        Placement::CudaDevice { device } => Ok((*device, binding)),
        Placement::Host => Err(InteropError::new(kind, "prepared device export requires CUDA-device buffer placement")
            .with_context("buffer_id", binding.id().get().to_string())
            .with_context("placement", "host")),
    }
}

fn require_host_binding<'a>(binding: &'a BufferBinding<'a>, kind: InteropErrorKind) -> Result<&'a BufferBinding<'a>> {
    match binding.placement() {
        Placement::Host => Ok(binding),
        Placement::CudaDevice { device } => {
            Err(InteropError::new(kind, "prepared host export requires host buffer placement")
                .with_context("buffer_id", binding.id().get().to_string())
                .with_context("placement", "cuda_device")
                .with_context("device", device.to_string()))
        },
    }
}

fn validate_prepared_host_column_against_bindings<'a>(
    column: &SemanticColumn,
    columns: &HashMap<ColumnId, &'a SemanticColumn>,
    bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    validated_columns: &mut HashSet<ColumnId>,
) -> Result<()> {
    validate_semantic_layout_matches_data_type(column)?;

    fn validate_host_validity_against_bindings<'a>(
        column: &SemanticColumn,
        bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    ) -> Result<()> {
        if let PhysicalValidity::ValidityMask { mask, null_count } = column.validity() {
            if let Some(null_count) = null_count {
                if *null_count > column.len() {
                    return Err(InteropError::new(InteropErrorKind::InvalidView, "null count exceeds logical length")
                        .with_context("column_id", column.id().get().to_string())
                        .with_context("null_count", null_count.to_string())
                        .with_context("len", column.len().to_string()));
                }
            }

            let mask_binding = require_binding(bindings, *mask)?;
            require_host_binding(mask_binding, InteropErrorKind::PlacementUnsupported)?;

            let required_bits = column.offset().checked_add(column.len()).ok_or_else(|| {
                InteropError::new(InteropErrorKind::InvalidView, "required null-mask range overflowed")
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("offset", column.offset().to_string())
                    .with_context("len", column.len().to_string())
            })?;
            let bit_len = mask_binding.bytes().byte_len().checked_mul(8).ok_or_else(|| {
                InteropError::new(InteropErrorKind::InvalidView, "null-mask bit range overflowed")
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", mask.get().to_string())
                    .with_context("byte_len", mask_binding.bytes().byte_len().to_string())
            })?;
            if bit_len < required_bits {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export validity mask is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", mask.get().to_string())
                .with_context("bit_len", bit_len.to_string())
                .with_context("required_bits", required_bits.to_string()));
            }
        }

        Ok(())
    }

    fn validate_host_leaf_or_string_column_against_bindings<'a>(
        column: &SemanticColumn,
        bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    ) -> Result<()> {
        match column.layout() {
            ColumnLayout::FixedWidth { values } => {
                let values_binding = require_binding(bindings, *values)?;
                require_host_binding(values_binding, InteropErrorKind::PlacementUnsupported)?;
                let required_bytes = fixed_width_required_bytes(column)?;
                if values_binding.bytes().byte_len() < required_bytes {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export value binding is too small for the active logical range",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", values.get().to_string())
                    .with_context("byte_len", values_binding.bytes().byte_len().to_string())
                    .with_context("required_bytes", required_bytes.to_string()));
                }
            },
            ColumnLayout::BinaryLike { offsets, data } => {
                let offsets_binding = require_binding(bindings, *offsets)?;
                require_host_binding(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
                let required_bytes = binary_like_offsets_required_bytes(column)?;
                if offsets_binding.bytes().byte_len() < required_bytes {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export binary-like offsets binding is too small for the active logical range",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", offsets.get().to_string())
                    .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                    .with_context("required_bytes", required_bytes.to_string()));
                }

                let data_binding = require_binding(bindings, *data)?;
                require_host_binding(data_binding, InteropErrorKind::PlacementUnsupported)?;
            },
            _ => {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedLayout,
                    "prepared host export currently supports fixed-width and binary-like string column layouts only on the non-list path",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            },
        }
        validate_host_validity_against_bindings(column, bindings)
    }

    fn validate_host_column_with_current_supported_child_projection<'a>(
        column: &SemanticColumn,
        columns: &HashMap<ColumnId, &'a SemanticColumn>,
        bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
        validated_columns: &mut HashSet<ColumnId>,
    ) -> Result<()> {
        validated_columns.insert(column.id());

        match column.layout() {
            ColumnLayout::List { offsets, child } => {
                let offsets_binding = require_binding(bindings, *offsets)?;
                require_host_binding(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
                let required_bytes = binary_like_offsets_required_bytes(column)?;
                if offsets_binding.bytes().byte_len() < required_bytes {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export list offsets binding is too small for the active logical range",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", offsets.get().to_string())
                    .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                    .with_context("required_bytes", required_bytes.to_string()));
                }

                let child_column = columns.get(child).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared host export list layout referenced a missing child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                })?;
                validate_prepared_host_column_against_bindings(child_column, columns, bindings, validated_columns)?;
                validate_host_validity_against_bindings(column, bindings)?;
            },
            _ => validate_host_leaf_or_string_column_against_bindings(column, bindings)?,
        }

        Ok(())
    }

    match column.layout() {
        ColumnLayout::Struct { children } => {
            validated_columns.insert(column.id());
            if children.is_empty() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared host export struct layout must reference at least one child semantic column",
                )
                .with_context("column_id", column.id().get().to_string()));
            }

            for child in children {
                let child_column = columns.get(child).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared host export struct layout referenced a missing child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                })?;
                validate_prepared_host_column_against_bindings(child_column, columns, bindings, validated_columns)?;
            }
            validate_host_validity_against_bindings(column, bindings)
        },
        ColumnLayout::Dictionary { indices, keys } => {
            validated_columns.insert(column.id());

            let indices_binding = require_binding(bindings, *indices)?;
            require_host_binding(indices_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = dictionary_indices_required_bytes(column)?;
            if indices_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export dictionary indices binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", indices.get().to_string())
                .with_context("byte_len", indices_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let keys_column = columns.get(keys).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared host export dictionary layout referenced a missing keys semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", keys.get().to_string())
            })?;
            validate_prepared_host_column_against_bindings(keys_column, columns, bindings, validated_columns)?;
            validate_host_validity_against_bindings(column, bindings)
        },
        _ => validate_host_column_with_current_supported_child_projection(column, columns, bindings, validated_columns),
    }
}

fn validate_prepared_column_against_bindings<'a>(
    column: &SemanticColumn,
    columns: &HashMap<ColumnId, &'a SemanticColumn>,
    bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    cross_binding_kind: InteropErrorKind,
    visiting: &mut HashSet<ColumnId>,
    validated_devices: &mut HashMap<ColumnId, i32>,
) -> Result<i32> {
    validate_semantic_layout_matches_data_type(column)?;

    if let Some(device) = validated_devices.get(&column.id()) {
        return Ok(*device);
    }
    if !visiting.insert(column.id()) {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            "prepared export semantic column graph must not contain cycles",
        )
        .with_context("column_id", column.id().get().to_string()));
    }

    let device = match column.layout() {
        ColumnLayout::FixedWidth { values } => {
            let values_binding = require_binding(bindings, *values)?;
            let (device, _) = require_cuda_device(values_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = fixed_width_required_bytes(column)?;
            if values_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export value binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", values.get().to_string())
                .with_context("byte_len", values_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }
            device
        },
        ColumnLayout::BinaryLike { offsets, data } => {
            let offsets_binding = require_binding(bindings, *offsets)?;
            let (device, _) = require_cuda_device(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = binary_like_offsets_required_bytes(column)?;
            if offsets_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export binary-like offsets binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", offsets.get().to_string())
                .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let data_binding = require_binding(bindings, *data)?;
            let (data_device, _) = require_cuda_device(data_binding, InteropErrorKind::PlacementUnsupported)?;
            if data_device != device {
                return Err(InteropError::new(
                    cross_binding_kind,
                    "prepared export bindings must live on one CUDA device",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", data.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", data_device.to_string()));
            }
            device
        },
        ColumnLayout::List { offsets, child } => {
            let offsets_binding = require_binding(bindings, *offsets)?;
            let (device, _) = require_cuda_device(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = binary_like_offsets_required_bytes(column)?;
            if offsets_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export list offsets binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", offsets.get().to_string())
                .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let child_column = columns.get(child).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export list layout referenced a missing child semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", child.get().to_string())
            })?;
            let child_device = validate_prepared_column_against_bindings(
                child_column,
                columns,
                bindings,
                cross_binding_kind,
                visiting,
                validated_devices,
            )?;
            if child_device != device {
                return Err(InteropError::new(
                    cross_binding_kind,
                    "prepared export bindings must live on one CUDA device",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", child.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", child_device.to_string()));
            }
            device
        },
        ColumnLayout::Struct { children } => {
            let mut child_iter = children.iter();
            let first_child = child_iter.next().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export struct layout must reference at least one child semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
            })?;
            let first_child_column = columns.get(first_child).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export struct layout referenced a missing child semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", first_child.get().to_string())
            })?;
            let device = validate_prepared_column_against_bindings(
                first_child_column,
                columns,
                bindings,
                cross_binding_kind,
                visiting,
                validated_devices,
            )?;
            for child in child_iter {
                let child_column = columns.get(child).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export struct layout referenced a missing child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                })?;
                let child_device = validate_prepared_column_against_bindings(
                    child_column,
                    columns,
                    bindings,
                    cross_binding_kind,
                    visiting,
                    validated_devices,
                )?;
                if child_device != device {
                    return Err(InteropError::new(
                        cross_binding_kind,
                        "prepared export bindings must live on one CUDA device",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                    .with_context("expected_device", device.to_string())
                    .with_context("actual_device", child_device.to_string()));
                }
            }
            device
        },
        ColumnLayout::Dictionary { indices, keys } => {
            let indices_binding = require_binding(bindings, *indices)?;
            let (device, _) = require_cuda_device(indices_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = dictionary_indices_required_bytes(column)?;
            if indices_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export dictionary indices binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", indices.get().to_string())
                .with_context("byte_len", indices_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let keys_column = columns.get(keys).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export dictionary layout referenced a missing keys semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", keys.get().to_string())
            })?;
            let keys_device = validate_prepared_column_against_bindings(
                keys_column,
                columns,
                bindings,
                cross_binding_kind,
                visiting,
                validated_devices,
            )?;
            if keys_device != device {
                return Err(InteropError::new(
                    cross_binding_kind,
                    "prepared export bindings must live on one CUDA device",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", keys.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", keys_device.to_string()));
            }
            device
        },
    };

    if let PhysicalValidity::ValidityMask { mask, null_count } = column.validity() {
        if let Some(null_count) = null_count {
            if *null_count > column.len() {
                return Err(InteropError::new(InteropErrorKind::InvalidView, "null count exceeds logical length")
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("null_count", null_count.to_string())
                    .with_context("len", column.len().to_string()));
            }
        }

        let mask_binding = require_binding(bindings, *mask)?;
        let (mask_device, _) = require_cuda_device(mask_binding, InteropErrorKind::PlacementUnsupported)?;
        if mask_device != device {
            return Err(InteropError::new(cross_binding_kind, "prepared export bindings must live on one CUDA device")
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", mask.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", mask_device.to_string()));
        }

        let required_bits = column.offset().checked_add(column.len()).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "required null-mask range overflowed")
                .with_context("column_id", column.id().get().to_string())
                .with_context("offset", column.offset().to_string())
                .with_context("len", column.len().to_string())
        })?;
        let bit_len = mask_binding.bytes().byte_len().checked_mul(8).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "null-mask bit range overflowed")
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", mask.get().to_string())
                .with_context("byte_len", mask_binding.bytes().byte_len().to_string())
        })?;
        if bit_len < required_bits {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export validity mask is too small for the active logical range",
            )
            .with_context("column_id", column.id().get().to_string())
            .with_context("buffer_id", mask.get().to_string())
            .with_context("bit_len", bit_len.to_string())
            .with_context("required_bits", required_bits.to_string()));
        }
    }

    visiting.remove(&column.id());
    validated_devices.insert(column.id(), device);
    Ok(device)
}

fn validate_completion_authority(
    profile: ExportProfile,
    completion: CompletionAuthority<'_>,
    producer_device: Option<i32>,
) -> Result<()> {
    match profile {
        ExportProfile::PreparedHostExport => match completion {
            CompletionAuthority::SynchronousByReturn => Ok(()),
            CompletionAuthority::SameDeviceStream { .. } => {
                Err(prepared_profile_error(profile, "prepared host export does not support CUDA completion authority")
                    .with_context("completion", "same_device_stream"))
            },
            CompletionAuthority::CrossDeviceEvent { .. } => {
                Err(prepared_profile_error(profile, "prepared host export does not support CUDA completion authority")
                    .with_context("completion", "cross_device_event"))
            },
        },
        ExportProfile::PreparedDeviceExport => {
            let producer_device = producer_device.expect("prepared device export must resolve a producer device");
            match completion {
                CompletionAuthority::SynchronousByReturn => Ok(()),
                CompletionAuthority::SameDeviceStream { stream } => {
                    let consumer_device = stream.device();
                    if consumer_device != producer_device {
                        return Err(InteropError::new(
                            InteropErrorKind::ReadinessMismatch,
                            "same-device stream completion must use a consumer stream on the producer CUDA device",
                        )
                        .with_context("completion", "same_device_stream")
                        .with_context("producer_device", producer_device.to_string())
                        .with_context("consumer_device", consumer_device.to_string()));
                    }
                    Ok(())
                },
                CompletionAuthority::CrossDeviceEvent { event } => {
                    let event_device = event.device();
                    if event_device == producer_device {
                        return Err(InteropError::new(
                            InteropErrorKind::InvalidView,
                            "cross-device event completion authority must use an event on a non-producer CUDA device; use SameDeviceStream for same-device completion",
                        )
                        .with_context("completion", "cross_device_event")
                        .with_context("event_device", event_device.to_string())
                        .with_context("producer_device", producer_device.to_string()));
                    }

                    Ok(())
                },
            }
        },
        ExportProfile::BorrowedDeviceCompatibility => unreachable!("borrowed compatibility is not a prepared export"),
        ExportProfile::OwnedDeviceTransfer => unreachable!("owned transfer is not a prepared export"),
    }
}

fn lease_readiness_binding(readiness: ProducerReadiness<'_>) -> LeaseReadiness {
    match readiness {
        ProducerReadiness::HostVisible => LeaseReadiness::HostVisible,
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            LeaseReadiness::DeviceStreamOrdered { raw: stream.as_raw(), device: stream.device() }
        },
        ProducerReadiness::DeviceEventOrdered { event } => {
            LeaseReadiness::DeviceEventOrdered { raw: event.as_raw(), device: event.device() }
        },
    }
}

fn completion_binding(completion: CompletionAuthority<'_>) -> CompletionBinding {
    match completion {
        CompletionAuthority::SynchronousByReturn => CompletionBinding::SynchronousByReturn,
        CompletionAuthority::SameDeviceStream { stream } => {
            CompletionBinding::SameDeviceStream { raw: stream.as_raw(), device: stream.device() }
        },
        CompletionAuthority::CrossDeviceEvent { event } => {
            CompletionBinding::CrossDeviceEvent { raw: event.as_raw(), device: event.device() }
        },
    }
}

#[derive(Debug, Clone, Default)]
struct LeaseRootState {
    active: Arc<AtomicBool>,
}

impl LeaseRootState {
    fn try_acquire(&self, export_kind: &'static str) -> Result<ActiveLeaseRegistration> {
        match self.active.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(ActiveLeaseRegistration { active: Arc::clone(&self.active) }),
            Err(_) => Err(InteropError::new(
                InteropErrorKind::LeaseAlreadyActive,
                "prepared export root does not support overlapping active read leases",
            )
            .with_context("export_kind", export_kind)),
        }
    }
}

#[derive(Debug)]
struct ActiveLeaseRegistration {
    active: Arc<AtomicBool>,
}

impl Drop for ActiveLeaseRegistration {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
    }
}

/// Consumer-activated read lease over a prepared column export.
#[derive(Debug)]
#[must_use = "lease should be completed explicitly via .complete(); dropping without completion follows the backend-defined incomplete-drop behavior"]
pub struct ActiveColumnReadLease<'export, 'producer, 'consumer> {
    export: &'export PreparedColumnExport<'producer>,
    completion: CompletionAuthority<'consumer>,
    completion_binding: CompletionBinding,
    export_context: LeaseExportContext,
    lease_backend: LeaseBackend,
    _activation: LeaseActivation,
    _registration: ActiveLeaseRegistration,
    completed: bool,
}

impl<'export, 'producer, 'consumer> ActiveColumnReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedColumnExport<'producer> {
        self.export
    }

    pub fn completion(&self) -> CompletionAuthority<'consumer> {
        self.completion
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    pub fn complete(mut self) -> Result<LeaseCompletionResult> {
        let result = self.lease_backend.complete(&self.export_context, self.completion_binding)?;
        self.completed = true;
        Ok(result)
    }

    #[doc(hidden)]
    pub fn release_after_external_completion(mut self) {
        self.completed = true;
    }
}

impl Drop for ActiveColumnReadLease<'_, '_, '_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.lease_backend.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback) {
            let _ = self.lease_backend.drop_incomplete(&self.export_context, self.completion_binding);
        }
    }
}

/// Consumer-activated read lease over a prepared table export.
#[derive(Debug)]
#[must_use = "lease should be completed explicitly via .complete(); dropping without completion follows the backend-defined incomplete-drop behavior"]
pub struct ActiveTableReadLease<'export, 'producer, 'consumer> {
    export: &'export PreparedTableExport<'producer>,
    completion: CompletionAuthority<'consumer>,
    completion_binding: CompletionBinding,
    export_context: LeaseExportContext,
    lease_backend: LeaseBackend,
    _activation: LeaseActivation,
    _registration: ActiveLeaseRegistration,
    completed: bool,
}

impl<'export, 'producer, 'consumer> ActiveTableReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedTableExport<'producer> {
        self.export
    }

    pub fn completion(&self) -> CompletionAuthority<'consumer> {
        self.completion
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    pub fn complete(mut self) -> Result<LeaseCompletionResult> {
        let result = self.lease_backend.complete(&self.export_context, self.completion_binding)?;
        self.completed = true;
        Ok(result)
    }

    #[doc(hidden)]
    pub fn release_after_external_completion(mut self) {
        self.completed = true;
    }
}

impl Drop for ActiveTableReadLease<'_, '_, '_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.lease_backend.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback) {
            let _ = self.lease_backend.drop_incomplete(&self.export_context, self.completion_binding);
        }
    }
}

/// Ownership-bearing device transfer root for one semantic column.
#[derive(Debug)]
pub struct OwnedColumnTransfer {
    column: SemanticColumn,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
    readiness: ProducerReadiness<'static>,
    diagnostics: ExportDiagnostics,
    release_contract: ReleaseContract,
    _owner: OwnedTransferOwner,
}

impl OwnedColumnTransfer {
    pub fn try_new<Owner>(
        column: SemanticColumn,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        Self::try_new_with_semantic_columns(
            column.clone(),
            vec![column],
            bindings,
            readiness,
            diagnostics,
            release_contract,
            owner,
        )
    }

    pub fn try_new_with_semantic_columns<Owner>(
        column: SemanticColumn,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        validate_owned_transfer_profile(diagnostics)?;
        if release_contract.release_device() < 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned device transfer release device must be >= 0",
            )
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let mapped_root = column_map.get(&column.id()).copied().ok_or_else(|| {
            InteropError::new(
                InteropErrorKind::InvalidView,
                "owned column transfer root must be present in the semantic column graph",
            )
            .with_context("column_id", column.id().get().to_string())
        })?;
        if mapped_root != &column {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned column transfer root must match the semantic column graph entry for its column id",
            )
            .with_context("column_id", column.id().get().to_string()));
        }

        let binding_map = binding_map(&bindings)?;
        let producer_device = validate_prepared_column_against_bindings(
            &column,
            &column_map,
            &binding_map,
            InteropErrorKind::PlacementUnsupported,
            &mut HashSet::new(),
            &mut HashMap::new(),
        )?;
        if column_map.len() != 1 {
            let mut visited = HashMap::new();
            validate_prepared_column_against_bindings(
                &column,
                &column_map,
                &binding_map,
                InteropErrorKind::PlacementUnsupported,
                &mut HashSet::new(),
                &mut visited,
            )?;
            if visited.len() != column_map.len() {
                let unreferenced = semantic_columns
                    .iter()
                    .find(|candidate| !visited.contains_key(&candidate.id()))
                    .expect("missing visited semantic column must exist");
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "owned column transfer included an unreferenced semantic column",
                )
                .with_context("column_id", unreferenced.id().get().to_string()));
            }
        }

        validate_owned_transfer_readiness(&readiness, producer_device)?;
        if release_contract.release_device() != producer_device {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned column transfer release device must match the producer CUDA device",
            )
            .with_context("producer_device", producer_device.to_string())
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        Ok(Self {
            column,
            semantic_columns,
            bindings,
            readiness,
            diagnostics,
            release_contract,
            _owner: OwnedTransferOwner::new(owner),
        })
    }

    pub fn column(&self) -> &SemanticColumn {
        &self.column
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'static>] {
        &self.bindings
    }

    pub fn readiness(&self) -> &ProducerReadiness<'static> {
        &self.readiness
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn release_contract(&self) -> &ReleaseContract {
        &self.release_contract
    }
}

/// Ownership-bearing device transfer root for one semantic table.
#[derive(Debug)]
pub struct OwnedTableTransfer {
    table: SemanticTable,
    columns: Vec<SemanticColumn>,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
    readiness: ProducerReadiness<'static>,
    diagnostics: ExportDiagnostics,
    release_contract: ReleaseContract,
    _owner: OwnedTransferOwner,
}

impl OwnedTableTransfer {
    pub fn try_new<Owner>(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        Self::try_new_with_semantic_columns(
            table,
            columns.clone(),
            columns,
            bindings,
            readiness,
            diagnostics,
            release_contract,
            owner,
        )
    }

    pub fn try_new_with_semantic_columns<Owner>(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        validate_owned_transfer_profile(diagnostics)?;
        if release_contract.release_device() < 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned device transfer release device must be >= 0",
            )
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        if let Some(names) = table.names() {
            if names.len() != table.columns().len() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "owned table transfer names must match the number of table schema columns",
                )
                .with_context("names_len", names.len().to_string())
                .with_context("num_columns", table.columns().len().to_string()));
            }
        }

        let top_level_map = semantic_column_map(&columns, "table_column_id")?;
        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let binding_map = binding_map(&bindings)?;
        let mut seen = HashSet::with_capacity(table.columns().len());
        let mut table_device = None;
        let mut validated_devices = HashMap::new();

        for column_id in table.columns() {
            let column = top_level_map.get(column_id).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "owned table transfer referenced a missing top-level semantic column",
                )
                .with_context("column_id", column_id.get().to_string())
            })?;
            seen.insert(*column_id);

            if column.len() != table.num_rows() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "owned table transfer row count must match every top-level semantic column length",
                )
                .with_context("column_id", column_id.get().to_string())
                .with_context("column_len", column.len().to_string())
                .with_context("num_rows", table.num_rows().to_string()));
            }

            let column_device = validate_prepared_column_against_bindings(
                column,
                &column_map,
                &binding_map,
                InteropErrorKind::PlacementUnsupported,
                &mut HashSet::new(),
                &mut validated_devices,
            )?;
            match table_device {
                None => table_device = Some(column_device),
                Some(expected) if expected != column_device => {
                    let mut err = InteropError::new(
                        InteropErrorKind::MixedDeviceUnsupported,
                        "owned table transfer requires all referenced bindings to live on one CUDA device",
                    )
                    .with_context("column_id", column_id.get().to_string())
                    .with_context("expected_device", expected.to_string())
                    .with_context("actual_device", column_device.to_string());
                    if let Some(buffer_id) = first_buffer_id(column) {
                        err = err.with_context("buffer_id", buffer_id.get().to_string());
                    }
                    return Err(err);
                },
                Some(_) => {},
            }
        }

        if seen.len() != top_level_map.len() {
            let unreferenced = top_level_map.keys().find(|column_id| !seen.contains(column_id)).copied().unwrap();
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned table transfer included an unreferenced top-level semantic column",
            )
            .with_context("column_id", unreferenced.get().to_string()));
        }

        if validated_devices.len() != column_map.len() {
            let unreferenced = semantic_columns
                .iter()
                .find(|candidate| !validated_devices.contains_key(&candidate.id()))
                .expect("missing validated semantic column must exist");
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned table transfer included an unreferenced semantic column",
            )
            .with_context("column_id", unreferenced.id().get().to_string()));
        }

        let producer_device = match (table_device, readiness) {
            (Some(device), _) => device,
            (None, ProducerReadiness::DeviceStreamOrdered { stream }) => stream.device(),
            (None, ProducerReadiness::DeviceEventOrdered { event }) => event.device(),
            (None, ProducerReadiness::HostVisible) => {
                return Err(owned_transfer_profile_error(
                    ExportProfile::OwnedDeviceTransfer,
                    "owned device transfer requires device-ordered readiness",
                )
                .with_context("readiness", "host_visible"));
            },
        };

        validate_owned_transfer_readiness(&readiness, producer_device)?;
        if release_contract.release_device() != producer_device {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned table transfer release device must match the producer CUDA device",
            )
            .with_context("producer_device", producer_device.to_string())
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        Ok(Self {
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            diagnostics,
            release_contract,
            _owner: OwnedTransferOwner::new(owner),
        })
    }

    pub fn table(&self) -> &SemanticTable {
        &self.table
    }

    pub fn columns(&self) -> &[SemanticColumn] {
        &self.columns
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'static>] {
        &self.bindings
    }

    pub fn readiness(&self) -> &ProducerReadiness<'static> {
        &self.readiness
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn release_contract(&self) -> &ReleaseContract {
        &self.release_contract
    }
}

/// One partition inside a partitioned prepared table export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedPreparedTablePartition<'a> {
    top_level_column_indices: Vec<usize>,
    export: PreparedTableExport<'a>,
}

impl<'a> PartitionedPreparedTablePartition<'a> {
    pub fn new(top_level_column_indices: Vec<usize>, export: PreparedTableExport<'a>) -> Self {
        Self { top_level_column_indices, export }
    }

    pub fn top_level_column_indices(&self) -> &[usize] {
        &self.top_level_column_indices
    }

    pub fn export(&self) -> &PreparedTableExport<'a> {
        &self.export
    }
}

/// Partitioned multi-device prepared device export for one logical table
/// schema split across multiple single-device prepared table roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedPreparedTableExport<'a> {
    schema_names: Option<Vec<String>>,
    num_rows: usize,
    partitions: Vec<PartitionedPreparedTablePartition<'a>>,
}

impl<'a> PartitionedPreparedTableExport<'a> {
    pub fn try_new(
        schema_names: Option<Vec<String>>,
        num_rows: usize,
        partitions: Vec<PartitionedPreparedTablePartition<'a>>,
    ) -> Result<Self> {
        validate_partitioned_table_partitions(
            "partitioned prepared table export",
            schema_names.as_deref(),
            num_rows,
            &partitions,
            |partition| partition.export().diagnostics().profile(),
            |partition| partition.export().table().num_rows(),
            |partition| partition.export().table().columns().len(),
        )?;

        Ok(Self { schema_names, num_rows, partitions })
    }

    pub fn schema_names(&self) -> Option<&[String]> {
        self.schema_names.as_deref()
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn partitions(&self) -> &[PartitionedPreparedTablePartition<'a>] {
        &self.partitions
    }
}

/// One partition inside a partitioned owned table transfer.
#[derive(Debug)]
pub struct PartitionedOwnedTableTransferPartition {
    top_level_column_indices: Vec<usize>,
    transfer: OwnedTableTransfer,
}

impl PartitionedOwnedTableTransferPartition {
    pub fn new(top_level_column_indices: Vec<usize>, transfer: OwnedTableTransfer) -> Self {
        Self { top_level_column_indices, transfer }
    }

    pub fn top_level_column_indices(&self) -> &[usize] {
        &self.top_level_column_indices
    }

    pub fn transfer(&self) -> &OwnedTableTransfer {
        &self.transfer
    }
}

/// Partitioned multi-device owned device transfer for one logical table schema
/// split across multiple single-device ownership-transfer roots.
#[derive(Debug)]
pub struct PartitionedOwnedTableTransfer {
    schema_names: Option<Vec<String>>,
    num_rows: usize,
    partitions: Vec<PartitionedOwnedTableTransferPartition>,
}

impl PartitionedOwnedTableTransfer {
    pub fn try_new(
        schema_names: Option<Vec<String>>,
        num_rows: usize,
        partitions: Vec<PartitionedOwnedTableTransferPartition>,
    ) -> Result<Self> {
        validate_partitioned_table_partitions(
            "partitioned owned table transfer",
            schema_names.as_deref(),
            num_rows,
            &partitions,
            |partition| partition.transfer().diagnostics().profile(),
            |partition| partition.transfer().table().num_rows(),
            |partition| partition.transfer().table().columns().len(),
        )?;

        Ok(Self { schema_names, num_rows, partitions })
    }

    pub fn schema_names(&self) -> Option<&[String]> {
        self.schema_names.as_deref()
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn partitions(&self) -> &[PartitionedOwnedTableTransferPartition] {
        &self.partitions
    }
}

fn validate_partitioned_table_partitions<P>(
    kind: &'static str,
    schema_names: Option<&[String]>,
    num_rows: usize,
    partitions: &[P],
    profile_of: impl Fn(&P) -> ExportProfile,
    num_rows_of: impl Fn(&P) -> usize,
    top_level_column_count_of: impl Fn(&P) -> usize,
) -> Result<()>
where
    P: PartitionTopLevelCoverage,
{
    if partitions.is_empty() {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            format!("{kind} must contain at least one partition"),
        ));
    }

    let partition_top_level_columns = partitions.iter().map(&top_level_column_count_of).sum::<usize>();
    let total_top_level_columns = schema_names.map_or(partition_top_level_columns, |names| names.len());
    if let Some(names) = schema_names {
        if names.len() < partition_top_level_columns {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                format!("{kind} names must cover every partitioned top-level column"),
            )
            .with_context("names_len", names.len().to_string())
            .with_context("partition_top_level_columns", partition_top_level_columns.to_string()));
        }
    }

    let expected_profile = match kind {
        "partitioned prepared table export" => ExportProfile::PreparedDeviceExport,
        "partitioned owned table transfer" => ExportProfile::OwnedDeviceTransfer,
        _ => unreachable!("unexpected partitioned table validation kind"),
    };

    let mut seen = vec![false; total_top_level_columns];
    for (partition_index, partition) in partitions.iter().enumerate() {
        let profile = profile_of(partition);
        if profile != expected_profile {
            return Err(InteropError::new(
                InteropErrorKind::ProfileUnsupported,
                format!("{kind} requires every partition to use one shared export profile"),
            )
            .with_context("partition_index", partition_index.to_string())
            .with_context("expected_profile", format!("{expected_profile:?}"))
            .with_context("actual_profile", format!("{profile:?}")));
        }

        let partition_num_rows = num_rows_of(partition);
        if partition_num_rows != num_rows {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                format!("{kind} requires every partition to have the same row count"),
            )
            .with_context("partition_index", partition_index.to_string())
            .with_context("partition_num_rows", partition_num_rows.to_string())
            .with_context("num_rows", num_rows.to_string()));
        }

        let top_level_column_indices = partition.top_level_column_indices();
        if top_level_column_indices.is_empty() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                format!("{kind} partitions must not be empty"),
            )
            .with_context("partition_index", partition_index.to_string()));
        }

        let expected_top_level_columns = top_level_column_count_of(partition);
        if top_level_column_indices.len() != expected_top_level_columns {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                format!("{kind} partition metadata must match the partition root column count"),
            )
            .with_context("partition_index", partition_index.to_string())
            .with_context("segment_columns", top_level_column_indices.len().to_string())
            .with_context("partition_columns", expected_top_level_columns.to_string()));
        }

        for &top_level_column_index in top_level_column_indices {
            if top_level_column_index >= total_top_level_columns {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    format!("{kind} top-level column index is out of range"),
                )
                .with_context("partition_index", partition_index.to_string())
                .with_context("top_level_column_index", top_level_column_index.to_string())
                .with_context("num_top_level_columns", total_top_level_columns.to_string()));
            }

            if seen[top_level_column_index] {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    format!("{kind} must not cover one top-level column more than once"),
                )
                .with_context("partition_index", partition_index.to_string())
                .with_context("top_level_column_index", top_level_column_index.to_string()));
            }

            seen[top_level_column_index] = true;
        }
    }

    if let Some((missing_index, _)) = seen.iter().enumerate().find(|(_, covered)| !**covered) {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            format!("{kind} must cover every original top-level column exactly once"),
        )
        .with_context("missing_top_level_column_index", missing_index.to_string()));
    }

    Ok(())
}

trait PartitionTopLevelCoverage {
    fn top_level_column_indices(&self) -> &[usize];
}

impl PartitionTopLevelCoverage for PartitionedPreparedTablePartition<'_> {
    fn top_level_column_indices(&self) -> &[usize] {
        self.top_level_column_indices()
    }
}

impl PartitionTopLevelCoverage for PartitionedOwnedTableTransferPartition {
    fn top_level_column_indices(&self) -> &[usize] {
        self.top_level_column_indices()
    }
}

/// Producer-rooted prepared export for one semantic column.
#[derive(Debug, Clone)]
pub struct PreparedColumnExport<'a> {
    column: SemanticColumn,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    readiness: ProducerReadiness<'a>,
    diagnostics: ExportDiagnostics,
    retained_host_buffers: RetainedHostBuffers,
    /// Cached producer device ordinal for device-backed exports.
    producer_device: Option<i32>,
    lease_state: LeaseRootState,
    lease_backend: LeaseBackend,
}

impl PartialEq for PreparedColumnExport<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.column == other.column
            && self.semantic_columns == other.semantic_columns
            && self.bindings == other.bindings
            && self.readiness == other.readiness
            && self.diagnostics == other.diagnostics
            && self.producer_device == other.producer_device
            && self.lease_backend.incomplete_drop_behavior() == other.lease_backend.incomplete_drop_behavior()
    }
}

impl Eq for PreparedColumnExport<'_> {}

impl<'a> PreparedColumnExport<'a> {
    pub fn try_new(
        column: SemanticColumn,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        Self::try_new_with_semantic_columns(column.clone(), vec![column], bindings, readiness, diagnostics)
    }

    pub fn try_new_with_semantic_columns(
        column: SemanticColumn,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        let profile = validate_prepared_profile(diagnostics)?;
        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let mapped_root = column_map.get(&column.id()).copied().ok_or_else(|| {
            InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared column export root must be present in the semantic column graph",
            )
            .with_context("column_id", column.id().get().to_string())
        })?;
        if mapped_root != &column {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared column export root must match the semantic column graph entry for its column id",
            )
            .with_context("column_id", column.id().get().to_string()));
        }
        let binding_map = binding_map(&bindings)?;
        let producer_device = match profile {
            ExportProfile::PreparedDeviceExport => {
                let producer_device = validate_prepared_column_against_bindings(
                    &column,
                    &column_map,
                    &binding_map,
                    InteropErrorKind::PlacementUnsupported,
                    &mut HashSet::new(),
                    &mut HashMap::new(),
                )?;
                if column_map.len() != 1 {
                    let mut visited = HashMap::new();
                    validate_prepared_column_against_bindings(
                        &column,
                        &column_map,
                        &binding_map,
                        InteropErrorKind::PlacementUnsupported,
                        &mut HashSet::new(),
                        &mut visited,
                    )?;
                    if visited.len() != column_map.len() {
                        let unreferenced = semantic_columns
                            .iter()
                            .find(|candidate| !visited.contains_key(&candidate.id()))
                            .expect("missing visited semantic column must exist");
                        return Err(InteropError::new(
                            InteropErrorKind::InvalidView,
                            "prepared column export included an unreferenced semantic column",
                        )
                        .with_context("column_id", unreferenced.id().get().to_string()));
                    }
                }
                Some(producer_device)
            },
            ExportProfile::PreparedHostExport => {
                let mut validated = HashSet::new();
                validate_prepared_host_column_against_bindings(&column, &column_map, &binding_map, &mut validated)?;
                if validated.len() != column_map.len() {
                    let unreferenced = semantic_columns
                        .iter()
                        .find(|candidate| !validated.contains(&candidate.id()))
                        .expect("missing validated semantic column must exist");
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared column export included an unreferenced semantic column",
                    )
                    .with_context("column_id", unreferenced.id().get().to_string()));
                }
                None
            },
            ExportProfile::BorrowedDeviceCompatibility => {
                unreachable!("borrowed compatibility is not a prepared export")
            },
            ExportProfile::OwnedDeviceTransfer => unreachable!("owned transfer is not a prepared export"),
        };
        validate_prepared_readiness(profile, &readiness, producer_device)?;
        Ok(Self {
            column,
            semantic_columns,
            bindings,
            readiness,
            diagnostics,
            retained_host_buffers: RetainedHostBuffers::default(),
            producer_device,
            lease_state: LeaseRootState::default(),
            lease_backend: LeaseBackend::semantic_default(),
        })
    }

    pub fn column(&self) -> &SemanticColumn {
        &self.column
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'a>] {
        &self.bindings
    }

    pub fn readiness(&self) -> &ProducerReadiness<'a> {
        &self.readiness
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn with_lease_backend(mut self, lease_backend: LeaseBackend) -> Self {
        self.lease_backend = lease_backend;
        self
    }

    #[doc(hidden)]
    pub fn with_retained_host_buffers(mut self, retained_host_buffers: Vec<Arc<[u8]>>) -> Self {
        self.retained_host_buffers = RetainedHostBuffers::new(retained_host_buffers);
        self
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    fn lease_export_context(&self) -> LeaseExportContext {
        LeaseExportContext::new(self.producer_device, lease_readiness_binding(self.readiness))
    }

    pub fn try_activate_read_lease<'export, 'consumer>(
        &'export self,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<ActiveColumnReadLease<'export, 'a, 'consumer>> {
        validate_completion_authority(self.diagnostics.profile(), completion, self.producer_device)?;
        let completion_binding = completion_binding(completion);
        let export_context = self.lease_export_context();
        let registration = self.lease_state.try_acquire("column")?;
        let activation = self.lease_backend.activate(&export_context, completion_binding)?;
        Ok(ActiveColumnReadLease {
            export: self,
            completion,
            completion_binding,
            export_context,
            lease_backend: self.lease_backend.clone(),
            _activation: activation,
            _registration: registration,
            completed: false,
        })
    }
}

/// Producer-rooted prepared export for one semantic table.
#[derive(Debug, Clone)]
pub struct PreparedTableExport<'a> {
    table: SemanticTable,
    columns: Vec<SemanticColumn>,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    readiness: ProducerReadiness<'a>,
    diagnostics: ExportDiagnostics,
    retained_host_buffers: RetainedHostBuffers,
    /// Cached producer device ordinal for device-backed exports.
    producer_device: Option<i32>,
    lease_state: LeaseRootState,
    lease_backend: LeaseBackend,
}

impl PartialEq for PreparedTableExport<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.table == other.table
            && self.columns == other.columns
            && self.semantic_columns == other.semantic_columns
            && self.bindings == other.bindings
            && self.readiness == other.readiness
            && self.diagnostics == other.diagnostics
            && self.producer_device == other.producer_device
            && self.lease_backend.incomplete_drop_behavior() == other.lease_backend.incomplete_drop_behavior()
    }
}

impl Eq for PreparedTableExport<'_> {}

impl<'a> PreparedTableExport<'a> {
    pub fn try_new(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        Self::try_new_with_semantic_columns(table, columns.clone(), columns, bindings, readiness, diagnostics)
    }

    pub fn try_new_with_semantic_columns(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        let profile = validate_prepared_profile(diagnostics)?;

        if let Some(names) = table.names() {
            if names.len() != table.columns().len() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared table names must match the number of table schema columns",
                )
                .with_context("names_len", names.len().to_string())
                .with_context("num_columns", table.columns().len().to_string()));
            }
        }

        let binding_map = binding_map(&bindings)?;
        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let mut top_level_map = HashMap::with_capacity(columns.len());
        for column in &columns {
            if top_level_map.insert(column.id(), column).is_some() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared table top-level columns must not reuse one semantic column id",
                )
                .with_context("column_id", column.id().get().to_string()));
            }
        }

        let mut seen = HashSet::with_capacity(table.columns().len());
        let mut table_device = None;
        let mut validated_devices = HashMap::new();
        let mut validated_columns = HashSet::new();
        for column_id in table.columns() {
            if !seen.insert(*column_id) {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared table schema must not repeat one semantic column id",
                )
                .with_context("column_id", column_id.get().to_string()));
            }
            let column = top_level_map.get(column_id).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared table schema referenced a missing top-level semantic column",
                )
                .with_context("column_id", column_id.get().to_string())
            })?;
            if column.len() != table.num_rows() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "column length does not match prepared table row count",
                )
                .with_context("column_id", column_id.get().to_string())
                .with_context("column_len", column.len().to_string())
                .with_context("num_rows", table.num_rows().to_string()));
            }

            match profile {
                ExportProfile::PreparedDeviceExport => {
                    let column_device = validate_prepared_column_against_bindings(
                        column,
                        &column_map,
                        &binding_map,
                        InteropErrorKind::PlacementUnsupported,
                        &mut HashSet::new(),
                        &mut validated_devices,
                    )?;
                    match table_device {
                        None => table_device = Some(column_device),
                        Some(expected) if expected != column_device => {
                            let mut err = InteropError::new(
                                InteropErrorKind::MixedDeviceUnsupported,
                                "prepared table export requires all referenced bindings to live on one CUDA device",
                            )
                            .with_context("column_id", column_id.get().to_string())
                            .with_context("expected_device", expected.to_string())
                            .with_context("actual_device", column_device.to_string());
                            if let Some(buffer_id) = first_buffer_id(column) {
                                err = err.with_context("buffer_id", buffer_id.get().to_string());
                            }
                            return Err(err);
                        },
                        Some(_) => {},
                    }
                },
                ExportProfile::PreparedHostExport => {
                    validate_prepared_host_column_against_bindings(
                        column,
                        &column_map,
                        &binding_map,
                        &mut validated_columns,
                    )?;
                },
                ExportProfile::BorrowedDeviceCompatibility => {
                    unreachable!("borrowed compatibility is not a prepared export")
                },
                ExportProfile::OwnedDeviceTransfer => unreachable!("owned transfer is not a prepared export"),
            }
        }

        if seen.len() != top_level_map.len() {
            let unreferenced = top_level_map.keys().find(|column_id| !seen.contains(column_id)).copied().unwrap();
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared table export included an unreferenced top-level semantic column",
            )
            .with_context("column_id", unreferenced.get().to_string()));
        }
        if validated_devices.len() != column_map.len() {
            if profile == ExportProfile::PreparedDeviceExport {
                let unreferenced = semantic_columns
                    .iter()
                    .find(|candidate| !validated_devices.contains_key(&candidate.id()))
                    .expect("missing validated semantic column must exist");
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared table export included an unreferenced semantic column",
                )
                .with_context("column_id", unreferenced.id().get().to_string()));
            }
        }
        if profile == ExportProfile::PreparedHostExport && validated_columns.len() != column_map.len() {
            let unreferenced = semantic_columns
                .iter()
                .find(|candidate| !validated_columns.contains(&candidate.id()))
                .expect("missing validated semantic column must exist");
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared table export included an unreferenced semantic column",
            )
            .with_context("column_id", unreferenced.id().get().to_string()));
        }

        let producer_device = match profile {
            ExportProfile::PreparedDeviceExport => Some(match (table_device, readiness) {
                (Some(device), _) => device,
                (None, ProducerReadiness::DeviceStreamOrdered { stream }) => stream.device(),
                (None, ProducerReadiness::DeviceEventOrdered { event }) => event.device(),
                (None, ProducerReadiness::HostVisible) => {
                    return Err(prepared_profile_error(
                        profile,
                        "prepared device export requires device-ordered readiness",
                    )
                    .with_context("readiness", "host_visible"));
                },
            }),
            ExportProfile::PreparedHostExport => None,
            ExportProfile::BorrowedDeviceCompatibility => {
                unreachable!("borrowed compatibility is not a prepared export")
            },
            ExportProfile::OwnedDeviceTransfer => unreachable!("owned transfer is not a prepared export"),
        };
        validate_prepared_readiness(profile, &readiness, producer_device)?;

        Ok(Self {
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            diagnostics,
            retained_host_buffers: RetainedHostBuffers::default(),
            producer_device,
            lease_state: LeaseRootState::default(),
            lease_backend: LeaseBackend::semantic_default(),
        })
    }

    pub fn table(&self) -> &SemanticTable {
        &self.table
    }

    pub fn columns(&self) -> &[SemanticColumn] {
        &self.columns
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'a>] {
        &self.bindings
    }

    pub fn readiness(&self) -> &ProducerReadiness<'a> {
        &self.readiness
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn with_lease_backend(mut self, lease_backend: LeaseBackend) -> Self {
        self.lease_backend = lease_backend;
        self
    }

    #[doc(hidden)]
    pub fn with_retained_host_buffers(mut self, retained_host_buffers: Vec<Arc<[u8]>>) -> Self {
        self.retained_host_buffers = RetainedHostBuffers::new(retained_host_buffers);
        self
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    fn lease_export_context(&self) -> LeaseExportContext {
        LeaseExportContext::new(self.producer_device, lease_readiness_binding(self.readiness))
    }

    pub fn try_activate_read_lease<'export, 'consumer>(
        &'export self,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<ActiveTableReadLease<'export, 'a, 'consumer>> {
        validate_completion_authority(self.diagnostics.profile(), completion, self.producer_device)?;
        let completion_binding = completion_binding(completion);
        let export_context = self.lease_export_context();
        let registration = self.lease_state.try_acquire("table")?;
        let activation = self.lease_backend.activate(&export_context, completion_binding)?;
        Ok(ActiveTableReadLease {
            export: self,
            completion,
            completion_binding,
            export_context,
            lease_backend: self.lease_backend.clone(),
            _activation: activation,
            _registration: registration,
            completed: false,
        })
    }
}

fn first_buffer_id(column: &SemanticColumn) -> Option<BufferId> {
    match column.layout() {
        ColumnLayout::FixedWidth { values } => Some(*values),
        ColumnLayout::BinaryLike { offsets, .. } => Some(*offsets),
        ColumnLayout::List { offsets, .. } => Some(*offsets),
        ColumnLayout::Struct { .. } => None,
        ColumnLayout::Dictionary { indices, .. } => Some(*indices),
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

/// Fallible producer adapter for a prepared device column export.
pub trait TryPrepareDeviceColumnExport {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>>;
}

/// Fallible producer adapter for a prepared device table export.
pub trait TryPrepareDeviceTableExport {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>>;
}

/// Fallible producer adapter for a partitioned prepared device table export.
pub trait TryPrepareDevicePartitionedTableExport {
    fn try_prepare_device_partitioned_table_export(&self) -> Result<PartitionedPreparedTableExport<'_>>;
}

/// Fallible producer adapter for a prepared host table export.
pub trait TryPrepareHostTableExport {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>>;
}

/// Fallible producer adapter for a prepared host column export.
pub trait TryPrepareHostColumnExport {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>>;
}

/// Fallible producer adapter for an owned device column transfer.
pub trait TryIntoOwnedDeviceColumnTransfer: Sized {
    fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer>;
}

/// Fallible producer adapter for an owned device table transfer.
pub trait TryIntoOwnedDeviceTableTransfer: Sized {
    fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer>;
}

/// Fallible producer adapter for a partitioned owned device table transfer.
pub trait TryIntoOwnedDevicePartitionedTableTransfer: Sized {
    fn try_into_owned_device_partitioned_table_transfer(self) -> Result<PartitionedOwnedTableTransfer>;
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

impl<'a> TryPrepareDeviceColumnExport for PreparedColumnExport<'a> {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>> {
        Ok(self.clone())
    }
}

impl<'a> TryPrepareDeviceTableExport for PreparedTableExport<'a> {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>> {
        Ok(self.clone())
    }
}

impl<'a> TryPrepareDevicePartitionedTableExport for PartitionedPreparedTableExport<'a> {
    fn try_prepare_device_partitioned_table_export(&self) -> Result<PartitionedPreparedTableExport<'_>> {
        Ok(self.clone())
    }
}

impl TryPrepareHostColumnExport for PreparedColumnExport<'static> {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>> {
        if self.diagnostics().profile() != ExportProfile::PreparedHostExport {
            return Err(prepared_profile_error(
                self.diagnostics().profile(),
                "prepared host export adapter requires prepared host export diagnostics",
            ));
        }
        Ok(self.clone())
    }
}

impl TryPrepareHostTableExport for PreparedTableExport<'static> {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>> {
        if self.diagnostics().profile() != ExportProfile::PreparedHostExport {
            return Err(prepared_profile_error(
                self.diagnostics().profile(),
                "prepared host export adapter requires prepared host export diagnostics",
            ));
        }
        Ok(self.clone())
    }
}

impl TryIntoOwnedDeviceColumnTransfer for OwnedColumnTransfer {
    fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer> {
        Ok(self)
    }
}

impl TryIntoOwnedDeviceTableTransfer for OwnedTableTransfer {
    fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer> {
        Ok(self)
    }
}

impl TryIntoOwnedDevicePartitionedTableTransfer for PartitionedOwnedTableTransfer {
    fn try_into_owned_device_partitioned_table_transfer(self) -> Result<PartitionedOwnedTableTransfer> {
        Ok(self)
    }
}
