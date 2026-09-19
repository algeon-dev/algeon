//! Unstable physical-memory-pressure support for integration tests.
//!
//! This helper first read-only verifies that the service installed its memory
//! manager, then deliberately allocates through one short-lived CUDA sidecar
//! resource. It neither installs nor wraps that manager, so allocator callbacks
//! and the bounded allocator-event ring retain their production ownership and
//! semantics while a test creates physical pressure.

use crate::{
    DeviceId, DeviceMemoryManager, DeviceMemoryResource, Runtime,
    core::{DeviceBuffer, Error, ErrorCode, Result},
};

/// Stable label describing how [`PhysicalMemoryPressureLease`] holds memory.
pub const PHYSICAL_MEMORY_PRESSURE_METHOD: &str = "cuda_on_sidecar_fixed_chunks";

const MEBIBYTE: u64 = 1024 * 1024;
const MINIMUM_RETAINED_FREE_BYTES: u64 = 256 * MEBIBYTE;
const ALLOCATION_GRANULARITY_BYTES: u64 = 2 * MEBIBYTE;
// cuda_on allocations can consume small per-buffer driver/resource metadata
// beyond requested bytes. Keep this fixed margin outside the 256 MiB reserve.
const SIDECAR_ALLOCATION_OVERHEAD_MARGIN_BYTES: u64 = 64 * MEBIBYTE;
const MINIMUM_CHUNK_BYTES: u64 = 8 * MEBIBYTE;
const MAXIMUM_CHUNKS: usize = 16;

/// Observed facts for one physical-memory-pressure lease.
///
/// `free_before_bytes` and `free_under_pressure_bytes` are CUDA snapshots,
/// not a reservation or residency claim. `free_after_release_bytes` and
/// `recovered_bytes` are set only by
/// [`PhysicalMemoryPressureLease::release_and_verify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalMemoryPressureFacts {
    /// Device on which the sidecar resource and all held buffers reside.
    pub device: DeviceId,
    /// The fixed sidecar mechanism used to create pressure.
    pub method: &'static str,
    /// CUDA free-memory snapshot immediately before creating the sidecar.
    pub free_before_bytes: u64,
    /// CUDA free-memory snapshot after all planned sidecar buffers are held.
    pub free_under_pressure_bytes: u64,
    /// Sum of the fixed, successfully allocated sidecar buffer sizes.
    pub bytes_held: u64,
    /// Free-memory target retained by the fixed plan before CUDA allocation
    /// granularity and concurrent-process effects.
    pub reserve_target_bytes: u64,
    /// CUDA free-memory snapshot after explicit release, when verified.
    pub free_after_release_bytes: Option<u64>,
    /// Positive CUDA free-memory recovery observed after explicit release.
    pub recovered_bytes: Option<u64>,
}

impl PhysicalMemoryPressureFacts {
    /// Return one allocation size that is strictly larger than the observed
    /// CUDA free memory while pressure is active.
    ///
    /// The caller remains responsible for proving its own allocator/domain
    /// contract. In particular, an allocator with cached pages may satisfy an
    /// allocation without making another physical CUDA allocation.
    pub fn request_bytes_above_pressure(&self) -> Result<usize> {
        let request = self.free_under_pressure_bytes.checked_add(ALLOCATION_GRANULARITY_BYTES).ok_or_else(|| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "testing::memory::PhysicalMemoryPressureFacts::request_bytes_above_pressure",
                "physical-pressure request size overflowed while adding allocator granularity",
            )
        })?;
        usize::try_from(request).map_err(|_| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "testing::memory::PhysicalMemoryPressureFacts::request_bytes_above_pressure",
                "physical-pressure request size does not fit this platform",
            )
            .with_detail("request_bytes", request.to_string())
        })
    }
}

/// RAII owner of a bounded physical-memory-pressure sidecar.
///
/// Establishment requires a manager already installed by the service under
/// test. The lease then creates an independent [`DeviceMemoryResource::cuda_on`]
/// resource and a dedicated [`Runtime`] / stream for its buffers. It does not
/// install, wrap, or configure the process manager, its callbacks, or its
/// allocator-event ring.
///
/// Always prefer [`Self::release_and_verify`] in a test so CUDA
/// synchronization and memory-recovery facts are observed. [`Drop`] is only a
/// device-aware best-effort fallback for unwinding paths.
pub struct PhysicalMemoryPressureLease {
    facts: PhysicalMemoryPressureFacts,
    sidecar: Option<PressureSidecar>,
}

struct PressureSidecar {
    runtime: Runtime,
    buffers: Vec<DeviceBuffer>,
}

#[derive(Debug)]
struct PressurePlan {
    bytes_held: u64,
    reserve_target_bytes: u64,
    chunks: Vec<usize>,
}

impl PhysicalMemoryPressureLease {
    /// Establish bounded physical CUDA memory pressure on `device`.
    ///
    /// The allocation plan is computed once from CUDA's observed free-memory
    /// snapshot. It leaves at least the 256 MiB hard reserve, plus a 2 MiB
    /// allocation-granularity allowance and a fixed 64 MiB sidecar-overhead
    /// margin. It performs no more than 16 allocations. A failed planned
    /// allocation is reported as an inability to establish safe pressure; this
    /// helper never allocates repeatedly until failure.
    ///
    /// # Errors
    ///
    /// Returns a structured error when the service has not already installed a
    /// manager for `device`, when available memory cannot support the fixed
    /// safe plan, or when the bounded sidecar plan cannot be fully established.
    pub fn establish(device: DeviceId) -> Result<Self> {
        Self::ensure_manager_is_already_installed(device)?;

        let free_before = device
            .available_memory()
            .map_err(|error| error.reanchor("testing::memory::PhysicalMemoryPressureLease::establish"))?
            .free_bytes;
        let plan = PressurePlan::from_free_bytes(free_before)?;
        let planned_bytes_held = plan.bytes_held;

        // This resource is deliberately independent from the process manager:
        // it creates actual CUDA pressure without changing manager ownership.
        let sidecar_resource = DeviceMemoryResource::cuda_on(device)
            .map_err(|error| error.reanchor("testing::memory::PhysicalMemoryPressureLease::establish"))?;
        let runtime = Runtime::builder()
            .with_device(device)
            .with_memory_resource(sidecar_resource)
            .build()
            .map_err(|error| error.reanchor("testing::memory::PhysicalMemoryPressureLease::establish"))?;

        let mut lease = Self {
            facts: PhysicalMemoryPressureFacts {
                device,
                method: PHYSICAL_MEMORY_PRESSURE_METHOD,
                free_before_bytes: free_before,
                free_under_pressure_bytes: free_before,
                bytes_held: 0,
                reserve_target_bytes: plan.reserve_target_bytes,
                free_after_release_bytes: None,
                recovered_bytes: None,
            },
            sidecar: Some(PressureSidecar { runtime, buffers: Vec::with_capacity(plan.chunks.len()) }),
        };

        for chunk_bytes in plan.chunks {
            let runtime =
                &lease.sidecar.as_ref().expect("new physical-memory-pressure lease retains its sidecar").runtime;
            match DeviceBuffer::new_in_runtime(runtime, chunk_bytes) {
                Ok(buffer) => lease
                    .sidecar
                    .as_mut()
                    .expect("new physical-memory-pressure lease retains its sidecar")
                    .buffers
                    .push(buffer),
                Err(error) => return Err(lease.establishment_failure(error, "planned_chunk_allocation_failed")),
            }
            lease.facts.bytes_held = lease
                .facts
                .bytes_held
                .checked_add(chunk_bytes as u64)
                .expect("fixed physical-memory-pressure plan cannot overflow its held-byte total");
        }
        debug_assert_eq!(lease.facts.bytes_held, planned_bytes_held);

        let runtime = &lease.sidecar.as_ref().expect("new physical-memory-pressure lease retains its sidecar").runtime;
        if let Err(error) = runtime.synchronize() {
            return Err(lease.establishment_failure(error, "sidecar_synchronization_failed"));
        }

        let free_under_pressure = match device.available_memory() {
            Ok(memory) => memory.free_bytes,
            Err(error) => return Err(lease.establishment_failure(error, "under_pressure_snapshot_failed")),
        };
        lease.facts.free_under_pressure_bytes = free_under_pressure;

        if free_under_pressure < MINIMUM_RETAINED_FREE_BYTES {
            let error = Error::new_with(
                ErrorCode::OutOfMemory,
                "testing::memory::PhysicalMemoryPressureLease::establish",
                "bounded sidecar plan left less than the required safe CUDA free-memory reserve",
            )
            .with_detail("reason", "safe_reserve_not_preserved")
            .with_detail("free_under_pressure_bytes", free_under_pressure.to_string())
            .with_detail("minimum_retained_free_bytes", MINIMUM_RETAINED_FREE_BYTES.to_string());
            return Err(lease.establishment_failure(error, "safe_reserve_not_preserved"));
        }

        if free_under_pressure >= free_before {
            let error = Error::new_with(
                ErrorCode::OutOfMemory,
                "testing::memory::PhysicalMemoryPressureLease::establish",
                "sidecar buffers did not reduce the observed CUDA free-memory snapshot",
            )
            .with_detail("reason", "physical_pressure_not_observed")
            .with_detail("free_before_bytes", free_before.to_string())
            .with_detail("free_under_pressure_bytes", free_under_pressure.to_string());
            return Err(lease.establishment_failure(error, "physical_pressure_not_observed"));
        }

        Ok(lease)
    }

    /// Return the current evidence snapshot for this lease.
    pub fn facts(&self) -> PhysicalMemoryPressureFacts {
        self.facts
    }

    /// Return the CUDA device on which this lease holds pressure.
    pub fn device(&self) -> DeviceId {
        self.facts.device
    }

    /// Release the sidecar, verify device-aware synchronization, and return
    /// recovery facts.
    ///
    /// This consuming method guarantees that the buffers are dropped before
    /// the dedicated runtime/resource and records a CUDA free-memory snapshot
    /// only after those owners have been released.
    pub fn release_and_verify(mut self) -> Result<PhysicalMemoryPressureFacts> {
        self.release_inner()?;
        Ok(self.facts)
    }

    fn ensure_manager_is_already_installed(device: DeviceId) -> Result<()> {
        let options = DeviceMemoryManager::options_for_device(device)
            .map_err(|error| error.reanchor("testing::memory::PhysicalMemoryPressureLease::establish"))?;
        if options.is_some() {
            return Ok(());
        }

        Err(Error::new_with(
            ErrorCode::ExecutionBindingUnavailable,
            "testing::memory::PhysicalMemoryPressureLease::establish",
            "physical pressure requires a device memory manager installed by the service under test",
        )
        .with_detail("device", device.ordinal().to_string())
        .with_detail("reason", "device_memory_manager_not_installed")
        .with_detail(
            "remedy",
            "install the service-owned device memory manager before establishing physical test pressure",
        ))
    }

    fn establishment_failure(&mut self, error: Error, reason: &'static str) -> Error {
        let cleanup_error = self.release_inner().err();
        let mut error = error
            .reanchor_if_needed("testing::memory::PhysicalMemoryPressureLease::establish")
            .with_detail("reason", reason)
            .with_detail("device", self.facts.device.ordinal().to_string())
            .with_detail("free_before_bytes", self.facts.free_before_bytes.to_string())
            .with_detail("bytes_held", self.facts.bytes_held.to_string())
            .with_detail("reserve_target_bytes", self.facts.reserve_target_bytes.to_string());
        if let Some(cleanup_error) = cleanup_error {
            error = error.with_detail("cleanup_error", cleanup_error.to_string());
        }
        error
    }

    fn release_inner(&mut self) -> Result<()> {
        let Some(PressureSidecar { runtime, buffers }) = self.sidecar.take() else {
            return Ok(());
        };

        let synchronization_error = runtime.synchronize().err();

        // The buffers must retire before their sidecar runtime releases the
        // resource that owns their exact execution binding.
        drop(buffers);
        // The sidecar runtime must release before sampling recovery so its
        // cuda_on resource no longer retains physical CUDA allocations.
        drop(runtime);

        let free_after_release = self
            .facts
            .device
            .available_memory()
            .map_err(|error| error.reanchor("testing::memory::PhysicalMemoryPressureLease::release_and_verify"))?
            .free_bytes;
        let recovered_bytes = free_after_release.saturating_sub(self.facts.free_under_pressure_bytes);
        self.facts.free_after_release_bytes = Some(free_after_release);
        self.facts.recovered_bytes = Some(recovered_bytes);

        if let Some(error) = synchronization_error {
            return Err(error
                .reanchor("testing::memory::PhysicalMemoryPressureLease::release_and_verify")
                .with_detail("free_after_release_bytes", free_after_release.to_string())
                .with_detail("recovered_bytes", recovered_bytes.to_string()));
        }

        if recovered_bytes == 0 {
            return Err(Error::new_with(
                ErrorCode::OutOfMemory,
                "testing::memory::PhysicalMemoryPressureLease::release_and_verify",
                "releasing sidecar buffers did not recover observed CUDA free memory",
            )
            .with_detail("reason", "physical_pressure_recovery_not_observed")
            .with_detail("free_under_pressure_bytes", self.facts.free_under_pressure_bytes.to_string())
            .with_detail("free_after_release_bytes", free_after_release.to_string()));
        }

        Ok(())
    }
}

impl Drop for PhysicalMemoryPressureLease {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

impl PressurePlan {
    fn from_free_bytes(free_before_bytes: u64) -> Result<Self> {
        let minimum_required = MINIMUM_RETAINED_FREE_BYTES
            .checked_add(ALLOCATION_GRANULARITY_BYTES)
            .and_then(|value| value.checked_add(SIDECAR_ALLOCATION_OVERHEAD_MARGIN_BYTES))
            .and_then(|value| value.checked_add(MINIMUM_CHUNK_BYTES))
            .expect("physical-memory-pressure constants fit u64");
        if free_before_bytes < minimum_required {
            return Err(Error::new_with(
                ErrorCode::OutOfMemory,
                "testing::memory::PhysicalMemoryPressureLease::establish",
                "insufficient CUDA free memory for a bounded physical-pressure sidecar and its safe reserve",
            )
            .with_detail("reason", "safe_pressure_unavailable")
            .with_detail("free_before_bytes", free_before_bytes.to_string())
            .with_detail("minimum_required_bytes", minimum_required.to_string())
            .with_detail("minimum_retained_free_bytes", MINIMUM_RETAINED_FREE_BYTES.to_string()));
        }

        let usable_bytes = free_before_bytes
            - MINIMUM_RETAINED_FREE_BYTES
            - ALLOCATION_GRANULARITY_BYTES
            - SIDECAR_ALLOCATION_OVERHEAD_MARGIN_BYTES;
        let bytes_held = usable_bytes / ALLOCATION_GRANULARITY_BYTES * ALLOCATION_GRANULARITY_BYTES;
        let reserve_target_bytes = free_before_bytes - bytes_held;
        let chunk_count =
            usize::try_from(bytes_held.div_ceil(MINIMUM_CHUNK_BYTES)).unwrap_or(usize::MAX).clamp(1, MAXIMUM_CHUNKS);
        let chunk_count_u64 = chunk_count as u64;
        let base_chunk_bytes = bytes_held / chunk_count_u64;
        let final_chunk_bytes = base_chunk_bytes + bytes_held % chunk_count_u64;
        let base_chunk_bytes = usize::try_from(base_chunk_bytes).map_err(|_| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "testing::memory::PhysicalMemoryPressureLease::establish",
                "bounded physical-pressure chunk size does not fit this platform",
            )
        })?;
        let final_chunk_bytes = usize::try_from(final_chunk_bytes).map_err(|_| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "testing::memory::PhysicalMemoryPressureLease::establish",
                "bounded physical-pressure final chunk size does not fit this platform",
            )
        })?;
        let mut chunks = vec![base_chunk_bytes; chunk_count];
        *chunks.last_mut().expect("bounded physical-pressure plan has at least one chunk") = final_chunk_bytes;

        Ok(Self { bytes_held, reserve_target_bytes, chunks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_pressure_plan_is_bounded_and_preserves_the_documented_reserve() {
        let free_before = 48 * 1024 * MEBIBYTE;
        let plan = PressurePlan::from_free_bytes(free_before).expect("safe plan");

        assert_eq!(SIDECAR_ALLOCATION_OVERHEAD_MARGIN_BYTES, 64 * MEBIBYTE);
        assert!(plan.chunks.len() <= MAXIMUM_CHUNKS);
        assert!(plan.chunks.iter().all(|chunk| *chunk > 0));
        assert_eq!(plan.chunks.iter().map(|chunk| *chunk as u64).sum::<u64>(), plan.bytes_held);
        assert!(
            plan.reserve_target_bytes
                >= MINIMUM_RETAINED_FREE_BYTES
                    + ALLOCATION_GRANULARITY_BYTES
                    + SIDECAR_ALLOCATION_OVERHEAD_MARGIN_BYTES
        );
        assert_eq!(plan.bytes_held % ALLOCATION_GRANULARITY_BYTES, 0);
    }

    #[test]
    fn fixed_pressure_plan_refuses_to_spend_the_safe_reserve() {
        let minimum_required = MINIMUM_RETAINED_FREE_BYTES
            + ALLOCATION_GRANULARITY_BYTES
            + SIDECAR_ALLOCATION_OVERHEAD_MARGIN_BYTES
            + MINIMUM_CHUNK_BYTES;
        let error = PressurePlan::from_free_bytes(minimum_required - 1).expect_err("insufficient memory must fail");

        assert_eq!(error.code(), ErrorCode::OutOfMemory);
        assert_eq!(error.operation(), "testing::memory::PhysicalMemoryPressureLease::establish");
        assert_eq!(error.detail_value("reason"), Some("safe_pressure_unavailable"));
    }

    #[test]
    fn request_above_pressure_adds_one_fixed_granularity() {
        let facts = PhysicalMemoryPressureFacts {
            device: DeviceId::DEFAULT,
            method: PHYSICAL_MEMORY_PRESSURE_METHOD,
            free_before_bytes: 0,
            free_under_pressure_bytes: 17 * MEBIBYTE,
            bytes_held: 0,
            reserve_target_bytes: 0,
            free_after_release_bytes: None,
            recovered_bytes: None,
        };

        assert_eq!(facts.request_bytes_above_pressure().unwrap(), (19 * MEBIBYTE) as usize);
    }
}
