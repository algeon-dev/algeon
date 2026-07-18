use crate::cudf_status_t;

unsafe extern "C" {
    /// Probe whether the cuFile driver can be opened by this process.
    pub fn cudf_gds_driver_available() -> cudf_status_t;
}
