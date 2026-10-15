//! Internal operation-level execution placement resolver.
//!
//! This module centralizes the repeated rules used at libcudf operation
//! boundaries: resolve borrowed stream / memory-resource overrides, validate
//! same-device inputs, insert device-side stream dependencies, and prepare FFI
//! execution options for the selected output domain.

use std::sync::Arc;

use super::{
    execution_context::Exec,
    execution_domain::{ConsumerCompletionFence, EnteredExecution, ExecutionDomain},
};
use crate::core::error::Result;

/// Resolved placement for one consuming operation.
///
/// Besides the output domain and FFI exec options, this carries the
/// [`ConsumerCompletionFence`] for every ordered cross-stream input. Keep the
/// `ResolvedExecution` alive across the FFI call that enqueues the operation.
/// When it drops, the fence records a completion event on the output stream
/// and makes each cross-stream input's retained stream wait on it, so
/// stream-ordered deallocation of an input can never race the consumer.
pub(crate) struct ResolvedExecution {
    output_domain: Arc<ExecutionDomain>,
    exec_options: EnteredExecution,
    consumer_fence: ConsumerCompletionFence,
}

impl ResolvedExecution {
    pub(crate) fn unary(primary: &Arc<ExecutionDomain>, ctx: Exec<'_>, operation: &'static str) -> Result<Self> {
        Self::many(primary, [], ctx, operation)
    }

    pub(crate) fn binary(
        primary: &Arc<ExecutionDomain>,
        secondary: &Arc<ExecutionDomain>,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        Self::many(primary, [secondary], ctx, operation)
    }

    pub(crate) fn many<'a>(
        primary: &Arc<ExecutionDomain>,
        secondaries: impl IntoIterator<Item = &'a Arc<ExecutionDomain>>,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(primary)?;
        let mut resolved = Self::new(output_domain, operation)?;
        resolved.order_input(primary)?;
        for secondary in secondaries {
            resolved.order_input(secondary)?;
        }
        Ok(resolved)
    }

    pub(crate) fn with_output_domain<'a>(
        output_domain: Arc<ExecutionDomain>,
        secondaries: impl IntoIterator<Item = &'a Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        let mut resolved = Self::new(output_domain, operation)?;
        for secondary in secondaries {
            resolved.order_input(secondary)?;
        }
        Ok(resolved)
    }

    /// Order an additional input into the output domain (forward fence) and
    /// register it for the reverse completion fence fired on drop.
    pub(crate) fn order_input(&mut self, input: &Arc<ExecutionDomain>) -> Result<()> {
        self.consumer_fence.order_input(input)
    }

    pub(crate) fn output_domain(&self) -> &Arc<ExecutionDomain> {
        &self.output_domain
    }

    pub(crate) fn exec_options(&self) -> &cudf_sys::cudf_exec_options_t {
        self.exec_options.options()
    }

    fn new(output_domain: Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let exec_options = output_domain.enter_execution()?;
        let consumer_fence = ConsumerCompletionFence::for_output(&output_domain, operation);
        Ok(Self { output_domain, exec_options, consumer_fence })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    use cudf_sys::test_support;

    use super::ResolvedExecution;
    use crate::{
        core::{
            error::{ErrorCode, FactKeyName, FactValue},
            memory::DeviceMemoryResource,
        },
        runtime::{CudaStream, DeviceId, Exec, Runtime},
        testing::run_named_cases,
    };

    const TEST_POOL_BYTES: usize = 16 * 1024 * 1024;

    fn lock_event_observation() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        match LOCK.get_or_init(|| Mutex::new(())).lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn reset_dependency_observation() {
        unsafe { test_support::cudf_test_event_reset_observation() };
    }

    fn dependency_wait_call_count_for_stream(stream_raw: usize) -> i32 {
        let mut out = -1;
        let status =
            unsafe { test_support::cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        out
    }

    fn runtime() -> Runtime {
        Runtime::builder()
            .with_stream(CudaStream::new().expect("failed to create test stream"))
            .with_pool_memory_resource(TEST_POOL_BYTES, None)
            .build()
            .expect("failed to build test runtime")
    }

    fn unary_uses_primary_domain_and_exec_options_without_overrides() {
        let rt = runtime();
        let ctx = Exec::RETAINED;

        let execution =
            ResolvedExecution::unary(rt.execution_domain(), ctx, "execution_resolver::test::unary").unwrap();

        assert_eq!(execution.output_domain().debug_stream_value(), rt.stream().__debug_raw_handle_value());
        assert_eq!(execution.output_domain().debug_mr_value(), rt.memory_resource().__debug_raw_handle_value());
        assert_eq!(execution.exec_options().device, rt.device().ordinal_raw());
        assert_eq!(execution.exec_options().stream, rt.execution_domain().stream().as_raw());
        assert_eq!(execution.exec_options().mr, rt.execution_domain().memory_resource().as_raw());
    }

    fn unary_with_borrowed_overrides_orders_primary_into_output_domain() {
        let _event_observation = lock_event_observation();
        let rt = runtime();
        let override_stream = CudaStream::new().unwrap();
        let override_mr = DeviceMemoryResource::pool(TEST_POOL_BYTES, None).unwrap();
        let ctx = Exec::RETAINED.with_stream(&override_stream).with_memory_resource(&override_mr);
        let expected_stream = override_stream.__debug_raw_handle_value();
        let expected_mr = override_mr.__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::unary(
            rt.execution_domain(),
            ctx,
            "execution_resolver::test::unary_with_borrowed_overrides",
        )
        .unwrap();

        assert_eq!(execution.output_domain().debug_stream_value(), expected_stream);
        assert_eq!(execution.output_domain().debug_mr_value(), expected_mr);
        assert_eq!(dependency_wait_call_count_for_stream(expected_stream), 1);
    }

    fn binary_orders_secondary_into_primary_output_domain() {
        let _event_observation = lock_event_observation();
        let primary = runtime();
        let secondary = runtime();
        let ctx = Exec::RETAINED;
        let expected_stream = primary.stream().__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::binary(
            primary.execution_domain(),
            secondary.execution_domain(),
            ctx,
            "execution_resolver::test::binary",
        )
        .unwrap();

        assert_eq!(execution.output_domain().debug_stream_value(), expected_stream);
        assert_eq!(dependency_wait_call_count_for_stream(expected_stream), 1);
    }

    fn many_orders_each_secondary_into_primary_output_domain() {
        let _event_observation = lock_event_observation();
        let primary = runtime();
        let secondary_a = runtime();
        let secondary_b = runtime();
        let ctx = Exec::RETAINED;
        let expected_stream = primary.stream().__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::many(
            primary.execution_domain(),
            [secondary_a.execution_domain(), secondary_b.execution_domain()],
            ctx,
            "execution_resolver::test::many",
        )
        .unwrap();

        assert_eq!(execution.output_domain().debug_stream_value(), expected_stream);
        assert_eq!(dependency_wait_call_count_for_stream(expected_stream), 2);
    }

    fn with_output_domain_orders_sources_into_explicit_output_domain() {
        let _event_observation = lock_event_observation();
        let output = runtime();
        let source_a = runtime();
        let source_b = runtime();
        let expected_stream = output.stream().__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(output.execution_domain()),
            [source_a.execution_domain(), source_b.execution_domain()],
            "execution_resolver::test::with_output_domain",
        )
        .unwrap();

        assert_eq!(execution.output_domain().debug_stream_value(), expected_stream);
        assert_eq!(dependency_wait_call_count_for_stream(expected_stream), 2);
    }

    fn dropping_resolution_fences_cross_stream_inputs_after_consumer() {
        let _event_observation = lock_event_observation();
        let primary = runtime();
        let secondary = runtime();
        let ctx = Exec::RETAINED;
        let output_stream = primary.stream().__debug_raw_handle_value();
        let secondary_stream = secondary.stream().__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::binary(
            primary.execution_domain(),
            secondary.execution_domain(),
            ctx,
            "execution_resolver::test::reverse_fence_binary",
        )
        .unwrap();
        // Forward fence only at resolve time: the output stream waits on the
        // secondary's event; the secondary stream has no waits yet.
        assert_eq!(dependency_wait_call_count_for_stream(output_stream), 1);
        assert_eq!(dependency_wait_call_count_for_stream(secondary_stream), 0);

        drop(execution);
        // Reverse fence at drop: the secondary's retained stream waits on the
        // consumer-completion event recorded on the output stream, so its
        // stream-ordered deallocations cannot race the consuming kernel.
        assert_eq!(dependency_wait_call_count_for_stream(secondary_stream), 1);
        assert_eq!(dependency_wait_call_count_for_stream(output_stream), 1);
    }

    fn dropping_same_provenance_resolution_inserts_no_waits() {
        let _event_observation = lock_event_observation();
        let rt = runtime();
        let ctx = Exec::RETAINED;
        let stream = rt.stream().__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::binary(
            rt.execution_domain(),
            rt.execution_domain(),
            ctx,
            "execution_resolver::test::same_provenance_reverse_fence",
        )
        .unwrap();
        drop(execution);
        assert_eq!(dependency_wait_call_count_for_stream(stream), 0);
    }

    fn dropping_unary_override_resolution_fences_primary_stream() {
        let _event_observation = lock_event_observation();
        let rt = runtime();
        let override_stream = CudaStream::new().unwrap();
        let override_mr = DeviceMemoryResource::pool(TEST_POOL_BYTES, None).unwrap();
        let ctx = Exec::RETAINED.with_stream(&override_stream).with_memory_resource(&override_mr);
        let primary_stream = rt.stream().__debug_raw_handle_value();
        let output_stream = override_stream.__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::unary(
            rt.execution_domain(),
            ctx,
            "execution_resolver::test::reverse_fence_unary_override",
        )
        .unwrap();
        assert_eq!(dependency_wait_call_count_for_stream(output_stream), 1);
        assert_eq!(dependency_wait_call_count_for_stream(primary_stream), 0);

        drop(execution);
        assert_eq!(dependency_wait_call_count_for_stream(primary_stream), 1);
    }

    fn dropping_many_resolution_fences_each_distinct_input_stream_once() {
        let _event_observation = lock_event_observation();
        let primary = runtime();
        let secondary = runtime();
        let ctx = Exec::RETAINED;
        let secondary_stream = secondary.stream().__debug_raw_handle_value();

        reset_dependency_observation();
        let execution = ResolvedExecution::many(
            primary.execution_domain(),
            [secondary.execution_domain(), secondary.execution_domain()],
            ctx,
            "execution_resolver::test::reverse_fence_dedupe",
        )
        .unwrap();
        drop(execution);
        // The same input stream ordered twice still gets exactly one reverse
        // wait: registration deduplicates by stream identity.
        assert_eq!(dependency_wait_call_count_for_stream(secondary_stream), 1);
    }

    #[test]
    #[ignore = "requires at least 2 GPUs"]
    fn cross_device_secondary_is_rejected_with_public_classification() {
        assert!(DeviceId::count().unwrap() >= 2, "requires at least 2 GPUs");

        let device0 = DeviceId::new(0).unwrap();
        let device1 = DeviceId::new(1).unwrap();
        let primary = Runtime::builder().with_device(device0).build().unwrap();
        let secondary = Runtime::builder().with_device(device1).build().unwrap();
        let ctx = Exec::RETAINED;

        let result = ResolvedExecution::binary(
            primary.execution_domain(),
            secondary.execution_domain(),
            ctx,
            "execution_resolver::test::cross_device",
        );
        let err = match result {
            Ok(_) => panic!("cross-device resolver input unexpectedly succeeded"),
            Err(err) => err,
        };

        assert_eq!(err.code(), ErrorCode::CrossDeviceInputs);
        assert_eq!(err.operation(), "execution_resolver::test::cross_device");
        assert_eq!(err.fact_value(FactKeyName::LeftDevice), Some(&FactValue::DeviceId(1)));
        assert_eq!(err.fact_value(FactKeyName::RightDevice), Some(&FactValue::DeviceId(0)));
    }

    #[test]
    fn execution_resolver_cases() {
        run_named_cases(&[
            ("binary_orders_secondary_into_primary_output_domain", binary_orders_secondary_into_primary_output_domain),
            (
                "dropping_same_provenance_resolution_inserts_no_waits",
                dropping_same_provenance_resolution_inserts_no_waits,
            ),
            (
                "unary_with_borrowed_overrides_orders_primary_into_output_domain",
                unary_with_borrowed_overrides_orders_primary_into_output_domain,
            ),
            (
                "dropping_resolution_fences_cross_stream_inputs_after_consumer",
                dropping_resolution_fences_cross_stream_inputs_after_consumer,
            ),
            (
                "dropping_unary_override_resolution_fences_primary_stream",
                dropping_unary_override_resolution_fences_primary_stream,
            ),
            (
                "unary_uses_primary_domain_and_exec_options_without_overrides",
                unary_uses_primary_domain_and_exec_options_without_overrides,
            ),
            (
                "many_orders_each_secondary_into_primary_output_domain",
                many_orders_each_secondary_into_primary_output_domain,
            ),
            (
                "with_output_domain_orders_sources_into_explicit_output_domain",
                with_output_domain_orders_sources_into_explicit_output_domain,
            ),
            (
                "dropping_many_resolution_fences_each_distinct_input_stream_once",
                dropping_many_resolution_fences_each_distinct_input_stream_once,
            ),
        ]);
    }
}
