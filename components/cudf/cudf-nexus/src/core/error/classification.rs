//! Shared coarse classification for `cudf::Error`.
//!
//! This module owns the single canonical policy that maps a `cudf::Error`
//! into a coarse discriminant. Downstream crates (`datafusion-nexus` and
//! `nexus-query-engine`) keep only a local-enum projection of this
//! classification.

use std::fmt;

use super::Error;
use super::code::{ErrorCode, ErrorKind, ErrorStatus};

/// Coarse classification of a `cudf::Error` for downstream mapping.
///
/// Each variant corresponds to one branch of the canonical
/// `code_for_cudf_error` policy. Downstreams map this to their own
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
        ErrorKind::InvariantViolation | ErrorKind::ExecutionContext => {
            CudfErrorClassification::InvariantViolation
        },
        _ => CudfErrorClassification::DependencyFailure,
    }
}
