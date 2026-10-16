use std::fmt;

/// Whether an allocation failure came from the device or from a reservation
/// limit. `Unknown` means the shim did not record a source; it is never
/// inferred from an out-of-memory status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CumlErrorFailureSource {
    Physical,
    ReservationLimit,
    Unknown,
}
impl fmt::Display for CumlErrorFailureSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Physical => "PHYSICAL",
            Self::ReservationLimit => "RESERVATION_LIMIT",
            Self::Unknown => "UNKNOWN",
        })
    }
}

/// Failure detail the shim keeps thread-locally after a failed call.
/// `algeon-cuml` copies it out as soon as the failed call returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CumlErrorDetail {
    pub status: u32,
    pub failure_source: CumlErrorFailureSource,
    pub domain_id: u64,
    pub member_id: u64,
    pub generation: u64,
    pub requested_bytes: u64,
    pub alignment: u64,
    pub cuda_error: i32,
    pub message: String,
}

impl CumlErrorDetail {
    fn from_raw(raw: &cuml_sys::cuml_error_detail_t, status: cuml_sys::cuml_status_t) -> Self {
        let len = raw.message.iter().position(|byte| *byte == 0).unwrap_or(raw.message.len());
        let message = raw.message[..len].iter().map(|byte| *byte as u8).collect::<Vec<_>>();
        let failure_source = match raw.failure_source {
            cuml_sys::CUML_ERROR_FAILURE_SOURCE_PHYSICAL => CumlErrorFailureSource::Physical,
            cuml_sys::CUML_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT => CumlErrorFailureSource::ReservationLimit,
            _ => CumlErrorFailureSource::Unknown,
        };
        Self {
            status,
            failure_source,
            domain_id: raw.domain_id,
            member_id: raw.member_id,
            generation: raw.generation,
            requested_bytes: raw.requested_bytes,
            alignment: raw.alignment,
            cuda_error: raw.cuda_error,
            message: String::from_utf8_lossy(&message).into_owned(),
        }
    }
}

pub(crate) fn read_ffi_error(status: cuml_sys::cuml_status_t) -> (String, Option<CumlErrorDetail>) {
    // The shim's thread-local detail pointer expires on its next call; copy it here.
    unsafe {
        let ptr = cuml_sys::cuml_last_error_detail();
        if ptr.is_null() {
            return (format!("cuML shim returned status {status}"), None);
        }
        let detail = CumlErrorDetail::from_raw(&*ptr, status);
        let message = if detail.message.is_empty() {
            format!("cuML shim returned status {status}")
        } else {
            detail.message.clone()
        };
        (message, Some(detail))
    }
}
