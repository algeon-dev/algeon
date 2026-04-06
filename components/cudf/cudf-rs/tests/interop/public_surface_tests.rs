use std::sync::Arc;

use cudf::interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, CompletionBinding, IncompleteDropBehavior, InteropError,
    LeaseActivation, LeaseBackend, LeaseCompletionResult, LeaseExportContext, LeaseReadiness, ReadStartBinding,
    cuda_explicit_completion_required_lease_backend,
};

struct TestActivation;

impl ActiveLeaseActivation for TestActivation {}

struct TestBackend;

impl ActiveLeaseBackend for TestBackend {
    fn activate(
        &self,
        _export: &LeaseExportContext,
        _read_start: ReadStartBinding,
        _completion: CompletionBinding,
    ) -> std::result::Result<LeaseActivation, InteropError> {
        Ok(LeaseActivation::with_state(Arc::new(TestActivation)))
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        IncompleteDropBehavior::GuaranteedFallback
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> std::result::Result<LeaseCompletionResult, InteropError> {
        Ok(LeaseCompletionResult::FallbackSynchronized)
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> std::result::Result<(), InteropError> {
        Ok(())
    }
}

#[test]
fn cudf_interop_reexports_shared_activation_backend_surface() {
    let export = LeaseExportContext::new(None, LeaseReadiness::HostVisible);
    let wrapped = LeaseBackend::new(TestBackend);
    let backend = TestBackend;

    let _activation = backend
        .activate(&export, ReadStartBinding::SynchronousByReturn, CompletionBinding::SynchronousByReturn)
        .unwrap();
    assert_eq!(wrapped.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);
    assert_eq!(
        backend.complete(&export, CompletionBinding::SynchronousByReturn).unwrap(),
        LeaseCompletionResult::FallbackSynchronized
    );
}

#[test]
fn cudf_interop_exposes_builtin_explicit_completion_backend() {
    let wrapped = LeaseBackend::semantic_explicit_completion_required();

    assert_eq!(wrapped.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);
}

#[test]
fn cudf_interop_exposes_cuda_explicit_completion_backend() {
    let wrapped = cuda_explicit_completion_required_lease_backend();

    assert_eq!(wrapped.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);
}
