/// Host/device boundary transfer metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundaryMetrics {
    /// Number of GPU-resident input handoffs.
    pub gpu_resident_input_handoffs: u64,
    /// Number of host-to-device uploads.
    pub host_to_device_uploads: u64,
    /// Rows uploaded from host to device.
    pub host_to_device_uploaded_rows: u64,
    /// Bytes uploaded from host to device.
    pub host_to_device_uploaded_bytes: u64,
    /// Number of device-to-host materializations.
    pub device_to_host_materializations: u64,
    /// Rows materialized from device to host.
    pub device_to_host_materialized_rows: u64,
    /// Bytes materialized from device to host.
    pub device_to_host_materialized_bytes: u64,
    /// Number of device-to-device copies.
    pub device_to_device_copies: u64,
    /// Bytes copied device-to-device.
    pub device_to_device_copied_bytes: u64,
}

impl BoundaryMetrics {
    /// Records one GPU-resident input handoff.
    pub fn record_gpu_resident_input_handoff(&mut self) {
        self.gpu_resident_input_handoffs += 1;
    }

    /// Records one host-to-device upload.
    pub fn record_host_to_device_upload(&mut self, rows: u64, bytes: u64) {
        self.host_to_device_uploads += 1;
        self.host_to_device_uploaded_rows += rows;
        self.host_to_device_uploaded_bytes += bytes;
    }

    /// Records one device-to-host materialization.
    pub fn record_device_to_host_materialization(&mut self, rows: u64, bytes: u64) {
        self.device_to_host_materializations += 1;
        self.device_to_host_materialized_rows += rows;
        self.device_to_host_materialized_bytes += bytes;
    }

    /// Records one device-to-device copy.
    pub fn record_device_to_device_copy(&mut self, bytes: u64) {
        self.device_to_device_copies += 1;
        self.device_to_device_copied_bytes += bytes;
    }

    pub(super) fn merge_from(&mut self, other: BoundaryMetrics) {
        self.gpu_resident_input_handoffs =
            self.gpu_resident_input_handoffs.saturating_add(other.gpu_resident_input_handoffs);
        self.host_to_device_uploads = self.host_to_device_uploads.saturating_add(other.host_to_device_uploads);
        self.host_to_device_uploaded_rows =
            self.host_to_device_uploaded_rows.saturating_add(other.host_to_device_uploaded_rows);
        self.host_to_device_uploaded_bytes =
            self.host_to_device_uploaded_bytes.saturating_add(other.host_to_device_uploaded_bytes);
        self.device_to_host_materializations =
            self.device_to_host_materializations.saturating_add(other.device_to_host_materializations);
        self.device_to_host_materialized_rows =
            self.device_to_host_materialized_rows.saturating_add(other.device_to_host_materialized_rows);
        self.device_to_host_materialized_bytes =
            self.device_to_host_materialized_bytes.saturating_add(other.device_to_host_materialized_bytes);
        self.device_to_device_copies = self.device_to_device_copies.saturating_add(other.device_to_device_copies);
        self.device_to_device_copied_bytes =
            self.device_to_device_copied_bytes.saturating_add(other.device_to_device_copied_bytes);
    }
}
