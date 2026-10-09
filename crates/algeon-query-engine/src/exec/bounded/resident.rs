use super::GpuDataFrame;

/// A frame together with the device bytes observed when it was produced.
///
/// `device_bytes` is a sizing input for later chunking and join decisions. It is
/// not an accounting record: the frame's allocations are charged to the attempt's
/// memory domain by the allocator, and the figure is not refreshed when the
/// frame is filtered or aliased.
pub(in crate::exec) struct ResidentGpuFrame {
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) device_bytes: u64,
}

impl ResidentGpuFrame {
    pub(in crate::exec) const fn with_device_bytes(frame: GpuDataFrame, device_bytes: u64) -> Self {
        Self { frame, device_bytes }
    }

    pub(in crate::exec) const fn device_bytes(&self) -> u64 {
        self.device_bytes
    }
}

/// A partial aggregate state the coordinator has accepted into merge state.
/// Same shape as any resident frame; the alias marks the ownership step.
pub(in crate::exec) type CommittedPartial = ResidentGpuFrame;
