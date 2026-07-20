//! Stable runtime-family identity and non-owning cache witnesses.

use std::{
    fmt,
    num::NonZeroU64,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::core::error::{Error, ErrorCode, Result};

static NEXT_RUNTIME_AFFINITY_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EXECUTION_BINDING_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity for one exact runtime execution binding.
///
/// Unlike [`RuntimeAffinityId`], which names a cache-compatible runtime
/// family, this identity changes whenever the CUDA stream or memory resource
/// changes. Derivations that only add retained lifetimes or adopt another
/// runtime family's affinity preserve their existing execution binding.
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

/// Stable identity shared by one base runtime and every runtime/output derived
/// from that base owner.
///
/// This identity is deliberately separate from execution workflow and bundle
/// provenance: changing a borrowed stream or memory resource does not change
/// the runtime family that owns an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuntimeAffinityId(NonZeroU64);

impl RuntimeAffinityId {
    fn next() -> Result<Self> {
        let value = NEXT_RUNTIME_AFFINITY_ID.fetch_add(1, Ordering::Relaxed);
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "runtime::RuntimeAffinityId::next",
                "runtime affinity identity space was exhausted",
            )
            .with_detail("affinity_id", value.to_string())
        })
    }

    /// Return the stable non-zero numeric identity.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for RuntimeAffinityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug)]
pub(crate) struct RuntimeAffinityOwner {
    id: RuntimeAffinityId,
}

impl RuntimeAffinityOwner {
    pub(crate) fn new() -> Result<Arc<Self>> {
        Ok(Arc::new(Self { id: RuntimeAffinityId::next()? }))
    }

    pub(crate) fn id(&self) -> RuntimeAffinityId {
        self.id
    }
}

/// Non-owning witness for a runtime family's stable affinity.
///
/// Cloning this value never keeps a runtime or GPU output alive. GPU outputs
/// do retain the underlying affinity owner through their execution domain, so
/// [`is_alive`](Self::is_alive) alone is not evidence that a counted
/// [`crate::Runtime`] owner still exists. Caches that require a current runtime
/// must retain a [`RuntimeLeaseAnchor`] as well.
#[derive(Debug, Clone)]
pub struct RuntimeAffinityAnchor {
    id: RuntimeAffinityId,
    owner: Weak<RuntimeAffinityOwner>,
}

impl RuntimeAffinityAnchor {
    pub(crate) fn new(owner: &Arc<RuntimeAffinityOwner>) -> Self {
        Self { id: owner.id(), owner: Arc::downgrade(owner) }
    }

    /// Stable identity carried by this non-owning witness.
    pub fn id(&self) -> RuntimeAffinityId {
        self.id
    }

    /// Whether a runtime or derived GPU output still retains this affinity.
    ///
    /// This does not prove that a counted runtime lease is current.
    pub fn is_alive(&self) -> bool {
        self.owner.strong_count() != 0
    }

    pub(crate) fn matches_owner(&self, owner: &Arc<RuntimeAffinityOwner>) -> bool {
        self.id == owner.id() && self.owner.ptr_eq(&Arc::downgrade(owner))
    }
}

#[derive(Debug)]
pub(crate) struct RuntimeLeaseOwner;

impl RuntimeLeaseOwner {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

/// Non-owning witness that a specific runtime-owner lineage is still current.
///
/// Unlike [`RuntimeAffinityAnchor`], this witness is kept alive only by
/// `Runtime` values (including clones and derived override runtimes), never by
/// scalar/column/table/data-frame outputs.
#[derive(Debug, Clone)]
pub struct RuntimeLeaseAnchor {
    affinity: RuntimeAffinityAnchor,
    owner: Weak<RuntimeLeaseOwner>,
}

impl RuntimeLeaseAnchor {
    pub(crate) fn new(affinity: RuntimeAffinityAnchor, owner: &Arc<RuntimeLeaseOwner>) -> Self {
        Self { affinity, owner: Arc::downgrade(owner) }
    }

    /// Affinity identity of the runtime owner represented by this lease.
    pub fn affinity_id(&self) -> RuntimeAffinityId {
        self.affinity.id()
    }

    /// Whether at least one counted `Runtime` value in this owner lineage is
    /// still alive.
    pub fn is_current(&self) -> bool {
        self.affinity.is_alive() && self.owner.strong_count() != 0
    }

    pub(crate) fn matches(
        &self,
        affinity_owner: &Arc<RuntimeAffinityOwner>,
        lease_owner: &Arc<RuntimeLeaseOwner>,
    ) -> bool {
        self.affinity.matches_owner(affinity_owner) && self.owner.ptr_eq(&Arc::downgrade(lease_owner))
    }
}
