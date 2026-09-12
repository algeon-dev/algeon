use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use super::{ownership::Owner, *};

/// A live Flight SQL prepared statement.
///
/// JDBC clients such as DataGrip prepare a statement once and then re-run the
/// held handle. A DDL statement prepares without a result-set schema, so JDBC
/// clients classify it as an update and execute through `executeUpdate` (the
/// update channel). Some clients still re-run that same handle, so the second
/// execution would otherwise hit DataFusion's "already exists" / "not found"
/// catalog error. We deduplicate no-result-set DDL per handle: the first
/// execution runs, later executions of the *same* handle are no-ops.
/// Query / DML prepared statements set `no_result_ddl = false` and are never
/// deduplicated, so they can still be executed repeatedly.
///
/// Contract: a no-result-set DDL handle is applied **at most once**. This is
/// scoped to *one* handle; a fresh `CreatePreparedStatement` (a distinct
/// handle) re-runs the DDL and still surfaces a genuine duplicate, so an
/// independent client running `CREATE` against a pre-existing table is not
/// masked. The dedup targets the sequential double-send a single JDBC client
/// issues for one logical statement; it deliberately does not re-apply the DDL
/// if catalog state changes underneath a reused handle (e.g. the object is
/// dropped elsewhere and the same handle is executed again), and it does not
/// serialize two *concurrent* executions of the same handle.
#[derive(Debug)]
pub(super) struct PreparedStatementEntry {
    /// The prepared SQL text. Re-planned on each execution (planning is
    /// side-effect-free; only `execute_logical_plan` mutates the catalog).
    pub(super) sql: String,
    /// True iff the prepared plan is DDL (no result set). Classified once at
    /// prepare time so neither execution channel has to re-plan to decide
    /// whether dedup applies.
    pub(super) no_result_ddl: bool,
    pub(super) execution_domain: StatementExecutionDomain,
    /// Set after the first successful execution. Only consulted for
    /// `no_result_ddl` handles, to turn a repeat execution into a no-op.
    pub(super) executed: AtomicBool,
}

impl PreparedStatementEntry {
    pub(super) fn new(sql: String, no_result_ddl: bool, execution_domain: StatementExecutionDomain) -> Self {
        Self { sql, no_result_ddl, execution_domain, executed: AtomicBool::new(false) }
    }

    /// True when this is a no-result-set DDL handle that has already executed
    /// once; i.e. a repeat execution that must be treated as a no-op.
    pub(super) fn ddl_already_executed(&self) -> bool {
        self.no_result_ddl && self.executed.load(Ordering::Relaxed)
    }

    /// Record a successful execution. Only meaningful for `no_result_ddl`
    /// handles; a no-op otherwise, so callers can mark unconditionally.
    pub(super) fn mark_executed(&self) {
        if self.no_result_ddl {
            self.executed.store(true, Ordering::Relaxed);
        }
    }
}

/// Hard cap on live prepared-statement handles across all connections.
/// Bounds the worst case for clients that leak handles by disconnecting
/// without calling `ClosePreparedStatement`. Sized for DataGrip-class
/// clients (a few hundred concurrent handles); raise when a real workload
/// demands it. Hitting the cap returns `RESOURCE_EXHAUSTED` rather than
/// silently evicting another connection's prepared statement.
const MAX_PREPARED_STATEMENTS: usize = 1024;

/// Per-principal cap on live prepared-statement handles.
///
/// The global cap alone lets one authenticated user fill the whole table and
/// lock every other user out, because a leaked handle is never reclaimed.
/// Dividing the global cap so no single principal can consume more than a
/// quarter of it keeps one misbehaving client from denying service to the
/// rest while still leaving a normal JDBC client far more handles than it
/// opens.
///
/// This applies only to named principals. Under [`AuthMode::Disabled`] every
/// caller is the same anonymous owner, so a per-owner cap there would lower
/// the deployment's real limit from the global cap to this one without
/// isolating anybody from anybody.
const MAX_PREPARED_STATEMENTS_PER_PRINCIPAL: usize = MAX_PREPARED_STATEMENTS / 4;

/// Live prepared statements, with their handles' owners and the caps that
/// bound them.
///
/// The counts and the map are guarded together: the cap check and the insert
/// that consumes it are a single critical section, so concurrent
/// `CreatePreparedStatement` calls cannot each observe room below the cap and
/// then both insert past it.
#[derive(Debug, Default)]
pub(super) struct PreparedStatementStore {
    state: Mutex<PreparedStatementState>,
}

#[derive(Debug, Default)]
struct PreparedStatementState {
    entries: HashMap<String, Arc<PreparedStatementEntry>>,
    /// Live handle count per owner. An owner with no live handles is removed,
    /// so this map is bounded by the number of principals currently holding
    /// prepared statements rather than by the number ever seen.
    per_owner: HashMap<Owner, usize>,
}

impl PreparedStatementStore {
    /// Insert `handle` as owned by `owner`, or reject if either the global or
    /// the owner's cap is already consumed.
    pub(super) fn insert(
        &self,
        handle: String,
        owner: Owner,
        entry: PreparedStatementEntry,
    ) -> Result<(), PreparedStatementLimit> {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.entries.len() >= MAX_PREPARED_STATEMENTS {
            return Err(PreparedStatementLimit::Global);
        }
        if owner != Owner::Anonymous {
            let owned = state.per_owner.get(&owner).copied().unwrap_or(0);
            if owned >= MAX_PREPARED_STATEMENTS_PER_PRINCIPAL {
                return Err(PreparedStatementLimit::Principal);
            }
        }
        // A repeat insert of the same handle replaces its entry and must not
        // double-count the owner's quota. Handles are freshly minted UUIDs,
        // so this is a defensive branch on the only path that could otherwise
        // leak a count.
        if state.entries.insert(handle, Arc::new(entry)).is_none() {
            *state.per_owner.entry(owner).or_insert(0) += 1;
        }
        Ok(())
    }

    /// Resolve `handle` for a caller that has already been authorized.
    pub(super) fn get(&self, handle: &str) -> Option<Arc<PreparedStatementEntry>> {
        let state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.entries.get(handle).map(Arc::clone)
    }

    /// Remove `handle`, releasing its owner's quota. Returns whether a live
    /// handle was removed, so a close of an unknown handle stays a traceable
    /// no-op rather than an error.
    pub(super) fn remove(&self, handle: &str, owner: &Owner) -> bool {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.entries.remove(handle).is_none() {
            return false;
        }
        if let std::collections::hash_map::Entry::Occupied(mut count) = state.per_owner.entry(owner.clone()) {
            *count.get_mut() -= 1;
            if *count.get() == 0 {
                count.remove();
            }
        }
        true
    }

    /// Live handle count. Test-only hook for asserting that quota is released.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).entries.len()
    }
}

/// Which cap a rejected `CreatePreparedStatement` hit. The two are reported
/// distinctly so an operator can tell a server-wide shortage from one client
/// exhausting its own quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PreparedStatementLimit {
    Global,
    Principal,
}

impl PreparedStatementLimit {
    pub(super) const fn limit(self) -> usize {
        match self {
            Self::Global => MAX_PREPARED_STATEMENTS,
            Self::Principal => MAX_PREPARED_STATEMENTS_PER_PRINCIPAL,
        }
    }

    pub(super) const fn message(self) -> &'static str {
        match self {
            Self::Global => "too many live prepared statements on this server",
            Self::Principal => "too many live prepared statements for this user",
        }
    }
}

impl DatafusionAlgeonFlightSqlService {
    pub(super) fn dataset_schema_for_prepared_statement(
        &self,
        plan: &LogicalPlan,
    ) -> crate::server::adapter_error::Result<Schema> {
        match plan {
            LogicalPlan::Ddl(ddl) => {
                verify_update_ddl(
                    ddl,
                    "algeon_datafusion.server.flight.do_action_create_prepared_statement.verify_plan",
                )?;
                Ok(Schema::empty())
            },
            LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(Error::new(
                ErrorCode::UnsupportedOperator,
                "algeon_datafusion.server.flight.do_action_create_prepared_statement.verify_plan",
                "Flight SQL prepared DML and COPY statements are not supported",
            )),
            LogicalPlan::Statement(statement) => match statement {
                Statement::SetVariable(_) | Statement::ResetVariable(_) => Ok(Schema::empty()),
                Statement::TransactionStart(_) | Statement::TransactionEnd(_) => Err(Error::new(
                    ErrorCode::UnsupportedOperator,
                    "algeon_datafusion.server.flight.do_action_create_prepared_statement.verify_plan",
                    "Flight SQL transactions are not supported",
                )),
                Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => Err(Error::new(
                    ErrorCode::InvalidRequest,
                    "algeon_datafusion.server.flight.do_action_create_prepared_statement.verify_plan",
                    "SQL PREPARE, EXECUTE, and DEALLOCATE are not supported through Flight SQL prepared statements",
                )),
            },
            _ => Ok(plan.schema().as_arrow().clone()),
        }
    }

    pub(super) fn prepared_statement_has_result_set(&self, plan: &LogicalPlan) -> bool {
        !matches!(
            plan,
            LogicalPlan::Ddl(_) | LogicalPlan::Statement(Statement::SetVariable(_) | Statement::ResetVariable(_))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::auth::Principal;

    fn owner(name: &str) -> Owner {
        Owner::from_principal(Some(&Principal { username: name.to_string() }))
    }

    fn entry() -> PreparedStatementEntry {
        PreparedStatementEntry::new("SELECT 1".to_string(), false, StatementExecutionDomain::Data)
    }

    fn fill(store: &PreparedStatementStore, owner: &Owner, count: usize) {
        for index in 0..count {
            store.insert(format!("{owner:?}-{index}"), owner.clone(), entry()).expect("below both caps");
        }
    }

    #[test]
    fn principal_cap_does_not_starve_other_principals() {
        // One client exhausting its own quota must leave the rest of the
        // table available, which the global cap alone does not guarantee.
        let store = PreparedStatementStore::default();
        let hog = owner("mallory");
        fill(&store, &hog, MAX_PREPARED_STATEMENTS_PER_PRINCIPAL);

        assert_eq!(
            store.insert("one-too-many".to_string(), hog, entry()).unwrap_err(),
            PreparedStatementLimit::Principal
        );
        store.insert("alice-0".to_string(), owner("alice"), entry()).expect("another principal still has quota");
    }

    #[test]
    fn closing_a_handle_returns_quota_to_its_owner() {
        let store = PreparedStatementStore::default();
        let alice = owner("alice");
        fill(&store, &alice, MAX_PREPARED_STATEMENTS_PER_PRINCIPAL);
        assert!(store.insert("blocked".to_string(), alice.clone(), entry()).is_err());

        assert!(store.remove(&format!("{alice:?}-0"), &alice), "a live handle is removed");
        store.insert("reused".to_string(), alice, entry()).expect("the released slot is reusable");
    }

    #[test]
    fn global_cap_bounds_the_anonymous_owner() {
        // Without auth every caller is the same owner, so the per-principal
        // cap must not apply: the deployment's limit stays the global cap it
        // had before ownership existed.
        let store = PreparedStatementStore::default();
        for index in 0..MAX_PREPARED_STATEMENTS {
            store.insert(format!("anon-{index}"), Owner::Anonymous, entry()).expect("below the global cap");
        }
        assert_eq!(
            store.insert("overflow".to_string(), Owner::Anonymous, entry()).unwrap_err(),
            PreparedStatementLimit::Global
        );
    }

    #[test]
    fn removing_an_unknown_handle_is_a_no_op() {
        let store = PreparedStatementStore::default();
        assert!(!store.remove("never-created", &owner("alice")));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn concurrent_creates_cannot_exceed_the_cap() {
        // The pre-existing check read `len()` and inserted as two steps, so
        // racing creates could both observe room and both insert past the cap.
        let store = Arc::new(PreparedStatementStore::default());
        let start = Arc::new(std::sync::Barrier::new(8));
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|thread| {
                let store = Arc::clone(&store);
                let start = Arc::clone(&start);
                let accepted = Arc::clone(&accepted);
                std::thread::spawn(move || {
                    start.wait();
                    for index in 0..MAX_PREPARED_STATEMENTS {
                        if store.insert(format!("{thread}-{index}"), Owner::Anonymous, entry()).is_ok() {
                            accepted.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("worker thread");
        }
        assert_eq!(accepted.load(Ordering::Relaxed), MAX_PREPARED_STATEMENTS);
        assert_eq!(store.len(), MAX_PREPARED_STATEMENTS);
    }
}
