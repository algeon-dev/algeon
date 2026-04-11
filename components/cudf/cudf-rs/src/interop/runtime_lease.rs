mod active;
mod authorities;
mod backend;
mod sync;

pub use active::{
    RuntimeActiveColumnReadLease, RuntimeActiveTableReadLease, TryActivateRuntimeColumnReadLease,
    TryActivateRuntimeTableReadLease,
};
pub use authorities::{
    RuntimeCompletionAuthority, RuntimeCompletionWitness, RuntimeLeaseCompletionResult, RuntimeReadStartAuthority,
};
pub use backend::cuda_explicit_completion_required_lease_backend;
pub(crate) use backend::shared_cuda_lease_backend;
pub(crate) use sync::{synchronize_completion_fallback, synchronize_runtime_release_completion};

pub(super) const ACTIVATE_RUNTIME_COLUMN_LEASE_OP: &str =
    "interop::PreparedColumnExport::try_activate_runtime_read_lease";
pub(super) const ACTIVATE_RUNTIME_TABLE_LEASE_OP: &str =
    "interop::PreparedTableExport::try_activate_runtime_read_lease";
pub(super) const COMPLETE_RUNTIME_COLUMN_LEASE_OP: &str = "interop::RuntimeActiveColumnReadLease::complete";
pub(super) const COMPLETE_RUNTIME_TABLE_LEASE_OP: &str = "interop::RuntimeActiveTableReadLease::complete";
pub(super) const DROP_RUNTIME_COLUMN_LEASE_OP: &str = "interop::RuntimeActiveColumnReadLease::drop";
pub(super) const DROP_RUNTIME_TABLE_LEASE_OP: &str = "interop::RuntimeActiveTableReadLease::drop";
pub(super) const RUNTIME_COMPLETION_WITNESS_READY_OP: &str = "interop::RuntimeCompletionWitness::is_ready";
pub(super) const RUNTIME_COMPLETION_WITNESS_SYNC_OP: &str = "interop::RuntimeCompletionWitness::synchronize";
pub(super) const SHARED_ACTIVATE_LEASE_OP: &str = "interop::PreparedExport::try_activate_read_lease";
pub(super) const SHARED_COMPLETE_LEASE_OP: &str = "interop::ActiveReadLease::complete";
pub(super) const SHARED_DROP_LEASE_OP: &str = "interop::ActiveReadLease::drop";
pub(super) const SHARED_COMPLETION_WITNESS_READY_OP: &str = "interop::CompletionWitness::is_ready";
pub(super) const SHARED_COMPLETION_WITNESS_SYNC_OP: &str = "interop::CompletionWitness::synchronize";
