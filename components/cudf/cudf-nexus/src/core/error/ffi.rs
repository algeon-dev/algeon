//! Typed payload copied from the cuDF shim error boundary.

use std::fmt;

/// Allocation-failure provenance reported by the native shim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CudfErrorFailureSource {
    Physical,
    ReservationLimit,
    Unknown,
}

impl CudfErrorFailureSource {
    pub(crate) fn from_raw(value: cudf_sys::cudf_error_failure_source_t) -> Self {
        match value {
            cudf_sys::CUDF_ERROR_FAILURE_SOURCE_PHYSICAL => Self::Physical,
            cudf_sys::CUDF_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT => Self::ReservationLimit,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for CudfErrorFailureSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Physical => "PHYSICAL",
            Self::ReservationLimit => "RESERVATION_LIMIT",
            Self::Unknown => "UNKNOWN",
        })
    }
}

/// The self-contained native payload copied before another shim call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudfErrorDetail {
    pub status: u32,
    pub failure_source: CudfErrorFailureSource,
    pub domain_id: u64,
    pub member_id: u64,
    pub generation: u64,
    pub requested_bytes: u64,
    pub alignment: u64,
    /// CUDA runtime error code, or `-1` when unavailable at the boundary.
    pub cuda_error: i32,
    pub message: String,
}

impl CudfErrorDetail {
    pub(crate) fn from_raw(raw: &cudf_sys::cudf_error_detail_t, status: cudf_sys::cudf_status_t) -> Self {
        let message_len = raw.message.iter().position(|byte| *byte == 0).unwrap_or(raw.message.len());
        let message = raw.message[..message_len].iter().map(|byte| *byte as u8).collect::<Vec<_>>();
        Self {
            // The returned status is authoritative even when a validation
            // path populated only the legacy message setter in native code.
            status,
            failure_source: CudfErrorFailureSource::from_raw(raw.failure_source),
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
