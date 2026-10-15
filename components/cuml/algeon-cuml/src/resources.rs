use std::{
    fmt,
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cudf::{
    RawCurrentResourceRouteLease, RawStreamReservationLease, Runtime, StreamReservationLease,
    runtime::{BindingExtensionInitError, BindingExtensionLease, DeviceEntryGuard, ExecutionBindingId},
};

use crate::error::{ErrorCode, FailurePhase, Result, check_status, error, from_cudf, from_interop};

const OP: &str = "CumlResources::for_runtime";
struct NativeHandle {
    raw: AtomicUsize,
    device: cudf::DeviceId,
    stream: usize,
    memory_resource: usize,
}

impl NativeHandle {
    fn create(device: cudf::DeviceId, stream: usize, mr: usize, owner: usize) -> Result<Self> {
        let _guard = DeviceEntryGuard::enter(device, OP).map_err(|e| from_cudf(OP, e))?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            cuml_sys::cuml_handle_create(
                device.ordinal() as i32,
                stream as *mut std::ffi::c_void,
                mr as *mut std::ffi::c_void,
                owner as *mut std::ffi::c_void,
                &mut raw,
            )
        };
        check_status(status, OP)?;
        if raw.is_null() {
            return Err(error(
                ErrorCode::InvariantViolation,
                FailurePhase::NativeCall,
                OP,
                "cuML returned a null handle after successful creation",
            ));
        }
        Ok(Self { raw: AtomicUsize::new(raw as usize), device, stream, memory_resource: mr })
    }
    fn raw(&self, op: &'static str) -> Result<cuml_sys::cuml_handle_t> {
        let raw = self.raw.load(Ordering::Acquire);
        if raw == 0 {
            return Err(error(ErrorCode::ResourcesClosed, FailurePhase::Preflight, op, "cuML resources are closed"));
        }
        Ok(raw as cuml_sys::cuml_handle_t)
    }
}

impl Drop for NativeHandle {
    fn drop(&mut self) {
        let raw = self.raw.swap(0, Ordering::AcqRel);
        if raw != 0 && DeviceEntryGuard::enter(self.device, "CumlResources::drop").is_ok() {
            unsafe {
                let _ = cuml_sys::cuml_handle_destroy(raw as cuml_sys::cuml_handle_t);
            }
        }
    }
}

pub struct CumlResources {
    runtime: Arc<Runtime>,
    native: BindingExtensionLease<NativeHandle>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CumlResources {
    pub fn for_runtime(runtime: Arc<Runtime>) -> Result<Self> {
        let binding = runtime.execution_binding().map_err(|e| from_interop(OP, e))?;
        let stream = binding.cuda_stream().as_raw() as usize;
        let mr = binding.memory_resource().as_raw() as usize;
        let manager = runtime.device_memory_manager().map_err(|e| from_cudf(OP, e))?;
        let manager_runtime = Runtime::builder()
            .with_device(runtime.device())
            .with_stream(runtime.stream().clone())
            .with_memory_resource(manager.resource().clone())
            .build()
            .map_err(|e| from_cudf(OP, e))?;
        let manager_binding = manager_runtime.execution_binding().map_err(|e| from_interop(OP, e))?;
        let current = manager_binding.memory_resource().as_raw() as usize;
        if mr != current {
            return Err(error(
                ErrorCode::MemoryResourceUnsupported,
                FailurePhase::Preflight,
                OP,
                "cuML requires the device memory manager current resource",
            )
            .with_detail("runtime_memory_resource", format!("0x{mr:x}"))
            .with_detail("current_memory_resource", format!("0x{current:x}")));
        }
        let native = runtime
            .get_or_try_init_binding_extension(OP, || {
                let manager = runtime.device_memory_manager().map_err(|error| from_cudf(OP, error))?;
                let owner = manager.resource().as_borrowed_sys().as_raw() as usize;
                NativeHandle::create(runtime.device(), stream, mr, owner)
            })
            .map_err(|e| match e {
                BindingExtensionInitError::Lifecycle(e) => from_cudf(OP, e),
                BindingExtensionInitError::Factory(e) => e,
            })?;
        let result = Self { runtime, native, _not_send_sync: PhantomData };
        result.validate_exact_binding(OP)?;
        Ok(result)
    }

    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }
    pub fn device(&self) -> cudf::DeviceId {
        self.runtime.device()
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.runtime.binding_id()
    }

    pub(crate) fn validate_exact_binding(&self, op: &'static str) -> Result<()> {
        let binding = self.runtime.execution_binding().map_err(|e| from_interop(op, e))?;
        let guard = self.native.enter(op).map_err(|e| from_cudf(op, e))?;
        let n = guard.value();
        if binding.device() != n.device.ordinal() as i32
            || binding.cuda_stream().as_raw() as usize != n.stream
            || binding.memory_resource().as_raw() as usize != n.memory_resource
        {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                op,
                "runtime binding differs from the cuML handle binding",
            ));
        }
        n.raw(op).map(|_| ())
    }

    /// Creates RF pool streams covered by the runtime stream's query-domain grant.
    /// When `reservation` is `None`, pool allocations route to the device manager's backend resource.
    pub fn training_streams(
        &self,
        count: NonZeroUsize,
        reservation: Option<&StreamReservationLease>,
    ) -> Result<TrainingStreams> {
        const OP: &str = "CumlResources::training_streams";
        self.validate_exact_binding(OP)?;
        if reservation.is_some_and(|lease| {
            lease.stream().identity() != self.runtime.stream().identity() || lease.manager().device() != self.device()
        }) {
            return Err(error(
                ErrorCode::ReservationCoverageMissing,
                FailurePhase::Preflight,
                OP,
                "reservation must cover the runtime stream on the same device",
            ));
        }
        let count = i32::try_from(count.get())
            .map_err(|_| error(ErrorCode::SizeOverflow, FailurePhase::Preflight, OP, "stream count exceeds i32"))?;
        let mut pool = std::ptr::null_mut();
        self.with_native(OP, |handle| {
            check_status(unsafe { cuml_sys::cuml_rf_training_streams_create(handle, count, &mut pool) }, OP)
        })?;
        let mut streams = TrainingStreams {
            members: Vec::with_capacity(count as usize),
            routes: Vec::with_capacity(count as usize),
            count: count as usize,
            pool,
            binding_id: self.binding_id(),
            runtime_stream: self.runtime.stream().identity(),
            device: self.device(),
            _not_send_sync: PhantomData,
        };
        let limit = reservation
            .map(|lease| {
                usize::try_from(lease.usage().map_err(|e| from_cudf(OP, e))?.allocation_limit_bytes).map_err(|_| {
                    error(ErrorCode::SizeOverflow, FailurePhase::Preflight, OP, "reservation limit exceeds usize")
                })
            })
            .transpose()?;
        let manager = self.runtime.device_memory_manager().map_err(|e| from_cudf(OP, e))?;
        // A sibling member needs an identity distinct from every other pool stream,
        // including streams created for another forest on the same query domain.
        static NEXT_MEMBER: AtomicUsize = AtomicUsize::new(1);
        for index in 0..count {
            let mut raw = std::ptr::null_mut();
            check_status(unsafe { cuml_sys::cuml_rf_training_streams_raw(pool, index, &mut raw) }, OP)?;
            if let Some(reservation) = reservation {
                let id = NEXT_MEMBER.fetch_add(1, Ordering::Relaxed) as u64;
                // SAFETY: the pool owns this CUDA stream until after members are cleared in Drop.
                let member =
                    unsafe { reservation.attach_sibling_raw_stream(raw, id, limit.expect("reservation limit")) }
                        .map_err(|e| from_cudf(OP, e))?;
                streams.members.push(member);
            } else {
                // SAFETY: the pool keeps the raw CUDA stream live until routes are cleared in Drop.
                let route = unsafe { manager.route_current_resource_raw(raw, manager.resource()) }
                    .map_err(|e| from_cudf(OP, e))?;
                streams.routes.push(route);
            }
        }
        Ok(streams)
    }

    pub(crate) fn with_native<T>(
        &self,
        op: &'static str,
        call: impl FnOnce(cuml_sys::cuml_handle_t) -> Result<T>,
    ) -> Result<T> {
        let binding = self.runtime.execution_binding().map_err(|e| from_interop(op, e))?;
        let _device = DeviceEntryGuard::enter(self.device(), op).map_err(|e| from_cudf(op, e))?;
        let guard = self.native.enter(op).map_err(|e| from_cudf(op, e))?;
        let n = guard.value();
        if binding.device() != n.device.ordinal() as i32
            || binding.cuda_stream().as_raw() as usize != n.stream
            || binding.memory_resource().as_raw() as usize != n.memory_resource
        {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                op,
                "runtime binding differs from the cuML handle binding",
            ));
        }
        let manager = self.runtime.device_memory_manager().map_err(|e| from_cudf(op, e))?;
        // cuML allocates through RMM's current resource; the RAFT handle's workspace
        // resource is not used. Keep this route installed through the native call.
        let _route =
            manager.route_current_resource(self.runtime.stream(), manager.resource()).map_err(|e| from_cudf(op, e))?;
        call(n.raw(op)?)
    }
}

impl fmt::Debug for CumlResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CumlResources")
            .field("device", &self.device())
            .field("binding_id", &self.binding_id())
            .finish_non_exhaustive()
    }
}

pub struct TrainingStreams {
    members: Vec<RawStreamReservationLease>,
    routes: Vec<RawCurrentResourceRouteLease>,
    count: usize,
    pool: cuml_sys::cuml_rf_training_streams_t,
    binding_id: ExecutionBindingId,
    runtime_stream: usize,
    device: cudf::DeviceId,
    _not_send_sync: PhantomData<*mut ()>,
}
impl TrainingStreams {
    /// Returns per-pool-stream reservation usage; standalone routes have no grant usage.
    pub fn usage(&self) -> Result<Vec<cudf::ReservationUsage>> {
        self.members.iter().map(|m| m.usage().map_err(|e| from_cudf("TrainingStreams::usage", e))).collect()
    }

    pub(crate) fn raw_for_fit(
        &self,
        resources: &CumlResources,
        op: &'static str,
    ) -> Result<cuml_sys::cuml_rf_training_streams_t> {
        resources.validate_exact_binding(op)?;
        if self.binding_id != resources.binding_id()
            || self.runtime_stream != resources.runtime().stream().identity()
            || self.device != resources.device()
            || self.members.len() + self.routes.len() != self.count
        {
            return Err(error(
                ErrorCode::ReservationCoverageMissing,
                FailurePhase::Preflight,
                op,
                "training pool is not covered by the current runtime reservation",
            ));
        }
        Ok(self.pool)
    }
}

impl Drop for TrainingStreams {
    fn drop(&mut self) {
        // Detach before the pool destroys its CUDA streams; a recycled raw stream
        // value must never inherit the previous member's allocation route.
        self.members.clear();
        self.routes.clear();
        if !self.pool.is_null() && DeviceEntryGuard::enter(self.device, "TrainingStreams::drop").is_ok() {
            unsafe {
                let _ = cuml_sys::cuml_rf_training_streams_destroy(self.pool);
            }
        }
    }
}
