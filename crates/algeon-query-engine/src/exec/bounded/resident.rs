use super::GpuDataFrame;

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

pub(in crate::exec) type CommittedPartial = ResidentGpuFrame;
