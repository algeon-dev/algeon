//! Process-wide query-service memory-governor claims.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct GovernorClaimLease {
    _inner: Arc<GovernorClaimLeaseInner>,
}

#[derive(Debug)]
struct GovernorClaimLeaseInner {
    ordinals: Vec<usize>,
    slot_guarantee: SlotGuarantee,
    released: AtomicBool,
}

#[derive(Debug, Default)]
struct GovernorClaimState {
    optimistic_claims: usize,
    guaranteed_minimum: bool,
}

impl Drop for GovernorClaimLeaseInner {
    fn drop(&mut self) {
        self.release();
    }
}

impl GovernorClaimLeaseInner {
    fn release(&self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let claims = governor_claims();
        let Ok(mut claims) = claims.lock() else {
            eprintln!("[nexus-query-engine] warning: failed to release query-service governor claims");
            return;
        };
        for ordinal in &self.ordinals {
            let Some(state) = claims.get_mut(ordinal) else {
                continue;
            };
            match self.slot_guarantee {
                SlotGuarantee::GuaranteedMinimum => state.guaranteed_minimum = false,
                SlotGuarantee::Optimistic => state.optimistic_claims = state.optimistic_claims.saturating_sub(1),
            }
            if state.optimistic_claims == 0 && !state.guaranteed_minimum {
                claims.remove(ordinal);
            }
        }
    }
}

impl GovernorClaimLease {
    pub(super) fn release(&self) {
        self._inner.release();
    }
}

fn governor_claims() -> &'static Mutex<BTreeMap<usize, GovernorClaimState>> {
    static CLAIMS: OnceLock<Mutex<BTreeMap<usize, GovernorClaimState>>> = OnceLock::new();
    CLAIMS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(super) fn claim_governor_ordinals(ordinals: &[usize], slot_guarantee: SlotGuarantee) -> Result<GovernorClaimLease> {
    let claims = governor_claims();
    let mut claims = claims.lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.query_service.governor_claim",
            "query-service governor claim registry mutex was poisoned",
        )
    })?;
    if let Some(conflict) = ordinals
        .iter()
        .find(|ordinal| claims.get(ordinal).is_some_and(|state| governor_claim_conflicts(state, slot_guarantee)))
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.tenant.validate",
            "query-service slot guarantee cannot share an active device memory governor claim",
        )
        .with_fact("field", "slot_guarantee")
        .with_fact("reason", "shared_governor_guaranteed_minimum_conflict")
        .with_fact("device_ordinal", *conflict)
        .with_fact("device_ordinals", comma_separated_usize(ordinals))
        .with_fact("slot_guarantee", slot_guarantee.as_str()));
    }
    for ordinal in ordinals {
        let state = claims.entry(*ordinal).or_default();
        match slot_guarantee {
            SlotGuarantee::GuaranteedMinimum => state.guaranteed_minimum = true,
            SlotGuarantee::Optimistic => state.optimistic_claims = state.optimistic_claims.saturating_add(1),
        }
    }
    Ok(GovernorClaimLease {
        _inner: Arc::new(GovernorClaimLeaseInner {
            ordinals: ordinals.to_vec(),
            slot_guarantee,
            released: AtomicBool::new(false),
        }),
    })
}

fn governor_claim_conflicts(state: &GovernorClaimState, requested: SlotGuarantee) -> bool {
    match requested {
        SlotGuarantee::GuaranteedMinimum => state.guaranteed_minimum || state.optimistic_claims > 0,
        SlotGuarantee::Optimistic => state.guaranteed_minimum,
    }
}
