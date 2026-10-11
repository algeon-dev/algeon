use cudf::CudfAllocationAttribution;

use super::{CumlErrorFailureSource, Error, ErrorCode};

pub fn allocation_attribution(error: &Error) -> CudfAllocationAttribution {
    if let Some(detail) = error.ffi_detail() {
        match detail.failure_source {
            CumlErrorFailureSource::ReservationLimit => return CudfAllocationAttribution::ConfirmedQueryLimit,
            CumlErrorFailureSource::Physical => return CudfAllocationAttribution::ConfirmedPhysical,
            CumlErrorFailureSource::Unknown => {},
        }
    }
    // An OOM status without a native failure source does not identify the
    // exhausted resource; never infer a query-limit or physical failure.
    if matches!(error.code(), ErrorCode::OutOfMemory | ErrorCode::HostOutOfMemory) {
        CudfAllocationAttribution::UnattributedAllocationFailure
    } else {
        CudfAllocationAttribution::NonAllocationFailure
    }
}
