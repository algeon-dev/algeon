mod authorities;
mod backend;
mod sync;

pub use backend::cuda_explicit_completion_required_lease_backend;
pub(crate) use backend::shared_cuda_lease_backend_with_observation;
pub(super) use sync::{
    synchronize_completion_fallback, synchronize_cross_device_event_raw, synchronize_same_device_stream_raw,
};
pub(super) const SHARED_ACTIVATE_LEASE_OP: &str = "interop::PreparedExport::try_activate_read_lease";
pub(super) const SHARED_COMPLETE_LEASE_OP: &str = "interop::ActiveReadLease::complete";
pub(super) const SHARED_DROP_LEASE_OP: &str = "interop::ActiveReadLease::drop";
pub(super) const SHARED_COMPLETION_WITNESS_READY_OP: &str = "interop::CompletionWitness::is_ready";
pub(super) const SHARED_COMPLETION_WITNESS_SYNC_OP: &str = "interop::CompletionWitness::synchronize";
