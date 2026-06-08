use core::{any::Any, ffi::c_void, fmt};
use std::sync::Arc;

use crate::{LeaseReadiness, ReadStartBinding, types::Result};

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
///
/// # Safety
///
/// The `raw` pointers are lifetime-erased CUDA stream/event handles.  They
/// are valid only while the `'consumer` lifetime on the enclosing
/// `ActiveColumnReadLease` / `ActiveTableReadLease` is alive.  Do not store
/// these bindings beyond the lease's lifetime.
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
    #[doc(hidden)]
    fn as_any(&self) -> &dyn Any;
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

    fn as_any(&self) -> &dyn Any {
        self
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

    #[doc(hidden)]
    pub fn downcast_backend_ref<T: CompletionWitnessBackend + 'static>(&self) -> Option<&T> {
        self.backend.as_any().downcast_ref::<T>()
    }
}

impl fmt::Debug for CompletionWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletionWitness").field("kind", &self.kind).finish()
    }
}

/// Semantic equality: two witnesses are equal when they represent the same
/// completion outcome (`kind`).  The backend that produced the witness is
/// intentionally ignored because backends are trait objects without identity
/// equality — what matters to consumers is the observable completion state.
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
    fn activate(
        &self,
        _export: &LeaseExportContext,
        _read_start: ReadStartBinding,
        _completion: CompletionBinding,
    ) -> Result<LeaseActivation> {
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

    pub(crate) fn semantic_default() -> Self {
        Self::semantic_guaranteed_fallback()
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.inner.incomplete_drop_behavior()
    }

    pub(crate) fn activate(
        &self,
        export: &LeaseExportContext,
        read_start: ReadStartBinding,
        completion: CompletionBinding,
    ) -> Result<LeaseActivation> {
        self.inner.activate(export, read_start, completion)
    }

    pub(crate) fn complete(
        &self,
        export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> Result<LeaseCompletionResult> {
        self.inner.complete(export, completion)
    }

    pub(crate) fn drop_incomplete(&self, export: &LeaseExportContext, completion: CompletionBinding) -> Result<()> {
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
pub(crate) struct RetainedHostBuffers {
    buffers: Vec<Arc<[u8]>>,
}

impl RetainedHostBuffers {
    pub(crate) fn new(buffers: Vec<Arc<[u8]>>) -> Self {
        Self { buffers }
    }
}

impl fmt::Debug for RetainedHostBuffers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedHostBuffers").field("count", &self.buffers.len()).finish()
    }
}

/// Declares who is responsible for establishing that consumer work is complete
/// before an owned-transfer release may finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferCompletionRule {
    /// The caller must determine that consumer work has completed before it
    /// begins release.
    CallerMustReleaseAfterConsumerWorkCompletes,
    /// The release path itself waits on a consumer-provided completion signal.
    ReleaseWaitsOnConsumerSignal,
}

/// Declares whether owned-transfer release may block internally or requires
/// synchronization outside the release call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseSynchronization {
    /// Release proceeds asynchronously once started.
    Asynchronous,
    /// Release may block internally while carrying out the release contract.
    MaySynchronizeInternally,
    /// The caller must perform any required synchronization before release.
    RequiresCallerSynchronization,
}

/// Shared semantic release contract for an ownership-bearing device transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseContract {
    release_device: i32,
    completion: TransferCompletionRule,
    synchronization: ReleaseSynchronization,
}

impl ReleaseContract {
    /// Build a plain release-contract value. Semantic validation happens when
    /// the contract is used to construct an owned transfer root.
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

pub(crate) struct OwnedTransferOwner {
    state: Box<dyn Any + Send + Sync>,
}

impl OwnedTransferOwner {
    pub(crate) fn new<Owner>(owner: Owner) -> Self
    where Owner: Send + Sync + 'static {
        Self { state: Box::new(owner) }
    }

    pub(crate) fn try_into<Owner>(self) -> core::result::Result<Owner, Self>
    where Owner: Send + Sync + 'static {
        match self.state.downcast::<Owner>() {
            Ok(owner) => Ok(*owner),
            Err(state) => Err(Self { state }),
        }
    }
}

impl fmt::Debug for OwnedTransferOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = &self.state;
        f.debug_struct("OwnedTransferOwner").finish()
    }
}
