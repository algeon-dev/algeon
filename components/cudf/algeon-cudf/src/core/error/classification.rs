//! Shared coarse classification for `cudf::Error`.
//!
//! This module owns the single canonical policy that maps a `cudf::Error`
//! into a coarse discriminant and into an allocation-failure attribution.
//! Downstream crates (`algeon-datafusion` and `query-engine`) keep only
//! a local-enum projection of these owner decisions and must not reimplement
//! the precedence.

use std::fmt;

use super::{
    Error,
    code::{ErrorCode, ErrorKind, ErrorStatus},
    ffi::CudfErrorFailureSource,
};

/// Coarse classification of a `cudf::Error` for downstream mapping.
///
/// Each variant corresponds to one branch of the canonical
/// `code_for_cudf_error` policy. Downstream map this to their own
/// `ErrorCode`/`QueryErrorCode` without reimplementing the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudfErrorClassification {
    SizeOverflow,
    HostMaterializationLimitExceeded,
    ResourceExhausted,
    TemporaryFailure,
    PersistentFailure,
    InvalidInput,
    Unsupported,
    DependencyFailure,
    InvariantViolation,
}

impl fmt::Display for CudfErrorClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SizeOverflow => "SizeOverflow",
            Self::HostMaterializationLimitExceeded => "HostMaterializationLimitExceeded",
            Self::ResourceExhausted => "ResourceExhausted",
            Self::TemporaryFailure => "TemporaryFailure",
            Self::PersistentFailure => "PersistentFailure",
            Self::InvalidInput => "InvalidInput",
            Self::Unsupported => "Unsupported",
            Self::DependencyFailure => "DependencyFailure",
            Self::InvariantViolation => "InvariantViolation",
        };
        f.write_str(name)
    }
}

/// Allocation-failure attribution owned by `cudf`.
///
/// This is the single canonical policy that separates a proven query-cap
/// rejection from physical exhaustion and from unattributed failures.
/// Downstream crates map each variant to their own limit/OOM/unknown codes
/// without reimplementing the precedence.
///
/// Only the typed FFI failure source proves the allocation origin. A budget
/// scope annotates every error that passes through it while its own status
/// reads `exceeded_or_lower_layer_rejected`, so the presence of a
/// [`super::BudgetExhaustionDetail`] or `budget_*` details is a scope usage
/// snapshot, never proof of a limit rejection, and is deliberately not
/// consulted here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudfAllocationAttribution {
    /// The native shim proved the failure hit the query reservation limit.
    ConfirmedQueryLimit,
    /// The native shim proved the failure is physical device exhaustion.
    ConfirmedPhysical,
    /// An allocation-typed failure without proven source. Callers must not
    /// report this as a reached query cap.
    UnattributedAllocationFailure,
    /// Not an allocation failure at all. A co-attached budget annotation, if
    /// any, describes the scope the error passed through, not its cause.
    NonAllocationFailure,
}

impl fmt::Display for CudfAllocationAttribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::ConfirmedQueryLimit => "ConfirmedQueryLimit",
            Self::ConfirmedPhysical => "ConfirmedPhysical",
            Self::UnattributedAllocationFailure => "UnattributedAllocationFailure",
            Self::NonAllocationFailure => "NonAllocationFailure",
        };
        f.write_str(name)
    }
}

/// Attribute the allocation origin of a `cudf::Error`.
///
/// Policy (in order):
/// 1. FFI `ReservationLimit` -> `ConfirmedQueryLimit`
/// 2. FFI `Physical` -> `ConfirmedPhysical`
/// 3. allocation-typed code (`OutOfMemory`, `ForeignResourceExhausted`) ->
///    `UnattributedAllocationFailure`
/// 4. otherwise -> `NonAllocationFailure`
///
/// Budget-exhaustion annotation presence never changes the outcome: a
/// dependency failure that merely traversed a budgeted scope stays
/// `NonAllocationFailure`, and an allocation failure without FFI provenance
/// stays `UnattributedAllocationFailure`.
pub fn allocation_attribution(error: &Error) -> CudfAllocationAttribution {
    if let Some(detail) = error.ffi_detail() {
        match detail.failure_source {
            CudfErrorFailureSource::ReservationLimit => {
                return CudfAllocationAttribution::ConfirmedQueryLimit;
            },
            CudfErrorFailureSource::Physical => return CudfAllocationAttribution::ConfirmedPhysical,
            CudfErrorFailureSource::Unknown => {},
        }
    }
    if is_allocation_code(error.code()) {
        CudfAllocationAttribution::UnattributedAllocationFailure
    } else {
        CudfAllocationAttribution::NonAllocationFailure
    }
}

fn is_allocation_code(code: ErrorCode) -> bool {
    matches!(code, ErrorCode::OutOfMemory | ErrorCode::ForeignResourceExhausted)
}

/// Classify a `cudf::Error` into its coarse discriminant.
///
/// Policy (in order):
/// 1. `ErrorCode::SizeOverflow` -> `SizeOverflow`
/// 2. `ErrorCode::HostMaterializationLimitExceeded` -> `HostMaterializationLimitExceeded`
/// 3. `kind == ResourceExhausted` -> `ResourceExhausted`
/// 4. `status == Temporary` -> `TemporaryFailure`
/// 5. `status == Persistent` -> `PersistentFailure`
/// 6. `kind == InvalidInput | NotFound` -> `InvalidInput`
/// 7. `kind == Unsupported` -> `Unsupported`
/// 8. `kind == DependencyFailure` -> `DependencyFailure`
/// 9. `kind == InvariantViolation | ExecutionContext` -> `InvariantViolation`
/// 10. otherwise -> `DependencyFailure`
///
/// Allocation failures keep their coarse `ResourceExhausted` discriminant
/// here; use [`allocation_attribution`] when the caller needs the proven
/// query-limit / physical / unattributed distinction.
pub fn classify_cudf_error(error: &Error) -> CudfErrorClassification {
    match error.code() {
        ErrorCode::SizeOverflow => return CudfErrorClassification::SizeOverflow,
        ErrorCode::HostMaterializationLimitExceeded => {
            return CudfErrorClassification::HostMaterializationLimitExceeded;
        },
        _ => {},
    }
    if error.kind() == ErrorKind::ResourceExhausted {
        return CudfErrorClassification::ResourceExhausted;
    }
    match error.status() {
        ErrorStatus::Temporary => return CudfErrorClassification::TemporaryFailure,
        ErrorStatus::Persistent => return CudfErrorClassification::PersistentFailure,
        ErrorStatus::Permanent => {},
    }
    match error.kind() {
        ErrorKind::InvalidInput | ErrorKind::NotFound => CudfErrorClassification::InvalidInput,
        ErrorKind::Unsupported => CudfErrorClassification::Unsupported,
        ErrorKind::DependencyFailure => CudfErrorClassification::DependencyFailure,
        ErrorKind::InvariantViolation | ErrorKind::ExecutionContext => CudfErrorClassification::InvariantViolation,
        _ => CudfErrorClassification::DependencyFailure,
    }
}

#[cfg(test)]
mod allocation_attribution_tests {
    use super::*;
    use crate::core::error::{BudgetExhaustionDetail, CudfErrorDetail};

    const TEST_OPERATION: &str = "cudf.test.allocation_attribution";

    fn ffi_detail_for_test(failure_source: CudfErrorFailureSource) -> CudfErrorDetail {
        CudfErrorDetail {
            status: 0,
            failure_source,
            domain_id: 0,
            member_id: 0,
            generation: 0,
            requested_bytes: 4096,
            alignment: 256,
            cuda_error: -1,
            message: "synthetic allocation failure".to_string(),
        }
    }

    fn budget_annotation_for_test() -> BudgetExhaustionDetail {
        BudgetExhaustionDetail::new(1024, 1024, 1024, 512, 4096, 1, "test_boundary", TEST_OPERATION)
    }

    fn annotated(error: Error) -> Error {
        error.with_budget_exhaustion_if_absent(budget_annotation_for_test())
    }

    #[test]
    fn reservation_limit_is_a_confirmed_query_limit_despite_budget_annotation() {
        let error = annotated(
            Error::new_with(ErrorCode::OutOfMemory, TEST_OPERATION, "reservation limit")
                .with_ffi_detail(ffi_detail_for_test(CudfErrorFailureSource::ReservationLimit)),
        );

        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::ConfirmedQueryLimit);
    }

    #[test]
    fn physical_failure_is_confirmed_physical_despite_budget_annotation() {
        let error = annotated(
            Error::new_with(ErrorCode::OutOfMemory, TEST_OPERATION, "physical oom")
                .with_ffi_detail(ffi_detail_for_test(CudfErrorFailureSource::Physical)),
        );

        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::ConfirmedPhysical);
    }

    #[test]
    fn unknown_allocation_source_with_scope_usage_stays_unattributed() {
        let error = annotated(
            Error::new_with(ErrorCode::OutOfMemory, TEST_OPERATION, "unknown allocation")
                .with_ffi_detail(ffi_detail_for_test(CudfErrorFailureSource::Unknown)),
        );

        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::UnattributedAllocationFailure);
    }

    #[test]
    fn allocation_without_any_provenance_stays_unattributed_despite_scope_usage() {
        let error = annotated(Error::new_with(ErrorCode::OutOfMemory, TEST_OPERATION, "plain oom"));

        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::UnattributedAllocationFailure);
    }

    #[test]
    fn foreign_resource_exhaustion_without_provenance_stays_unattributed() {
        let error =
            annotated(Error::new_with(ErrorCode::ForeignResourceExhausted, TEST_OPERATION, "foreign exhausted"));

        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::UnattributedAllocationFailure);
    }

    #[test]
    fn dependency_failure_through_a_budgeted_scope_is_not_an_allocation_failure() {
        let error = annotated(Error::new_with(ErrorCode::IoFailure, TEST_OPERATION, "dependency failure"));

        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::NonAllocationFailure);
    }

    #[test]
    fn size_overflow_is_not_an_allocation_failure() {
        let error = annotated(Error::new_with(ErrorCode::SizeOverflow, TEST_OPERATION, "size overflow"));

        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::NonAllocationFailure);
    }
}
