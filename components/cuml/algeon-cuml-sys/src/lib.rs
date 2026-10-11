//! Raw C bindings to the cuML native shim. Prefer the safe `cuml` crate.
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion); workspace code
//! uses this crate for its component contracts. See the
//! [API documentation boundary](https://algeon.dev/design#api-documentation).
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, clippy::all)]

include!("bindings.rs");

pub const EXPECTED_CUML_VERSION: (u16, u16, u16) = (26, 8, 0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeVersionError {
    QueryFailed,
    Incompatible { expected: (u16, u16, u16), actual: (u16, u16, u16) },
}

impl std::fmt::Display for RuntimeVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueryFailed => write!(f, "cuml_version_get failed"),
            Self::Incompatible { expected, actual } => write!(
                f,
                "cuML shim build version {}.{}.{} does not match required {}.{}.{}",
                actual.0, actual.1, actual.2, expected.0, expected.1, expected.2
            ),
        }
    }
}

impl std::error::Error for RuntimeVersionError {}

/// Validate the cuML version compiled into `libcuml_rust.so` from `CUML_VERSION_*` headers.
///
/// This reports the version the shim was built against. It does not query the
/// loaded `libcuml.so`. Build-support pins that library to the same cuML install
/// through rpath and direct NEEDED entries.
pub fn validate_runtime_version() -> Result<(u16, u16, u16), RuntimeVersionError> {
    let mut major = 0;
    let mut minor = 0;
    let mut patch = 0;
    let status = unsafe { cuml_version_get(&mut major, &mut minor, &mut patch) };
    if status != CUML_SUCCESS {
        return Err(RuntimeVersionError::QueryFailed);
    }
    let actual = (major as u16, minor as u16, patch as u16);
    if actual != EXPECTED_CUML_VERSION {
        return Err(RuntimeVersionError::Incompatible { expected: EXPECTED_CUML_VERSION, actual });
    }
    Ok(actual)
}

#[cfg(feature = "test-hooks")]
pub mod test_support;
