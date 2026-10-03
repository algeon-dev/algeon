//! Identity of one exact runtime execution binding.

use std::{
    fmt,
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::core::error::{Error, ErrorCode, Result};

static NEXT_EXECUTION_BINDING_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity for one exact runtime execution binding.
///
/// The identity changes whenever the CUDA stream or memory resource changes;
/// derivations that only add retained lifetimes keep it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExecutionBindingId(NonZeroU64);

impl ExecutionBindingId {
    pub(crate) fn next() -> Result<Self> {
        let value = NEXT_EXECUTION_BINDING_ID.fetch_add(1, Ordering::Relaxed);
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "runtime::ExecutionBindingId::next",
                "runtime execution binding identity space was exhausted",
            )
            .with_detail("binding_id", value.to_string())
        })
    }

    /// Return the stable non-zero numeric identity.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for ExecutionBindingId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
