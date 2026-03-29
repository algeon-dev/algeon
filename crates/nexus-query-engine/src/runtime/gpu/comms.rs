//! Reserved communication seam for future single-query multi-GPU execution.

/// Collective-communication facade for GPU query execution.
///
/// This trait exists so future replicated or sharded query placement can add
/// collective operations without changing every execution-context boundary.
pub trait GpuComms: Send + Sync {
    /// Returns whether this communication context contains exactly one device.
    fn is_single_device(&self) -> bool;
}

/// Communication context for today's single-device query placement.
#[derive(Debug, Default, Clone, Copy)]
pub struct SingleDeviceComms;

impl GpuComms for SingleDeviceComms {
    fn is_single_device(&self) -> bool {
        true
    }
}
