use super::*;

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
/// scoped to *one* handle — a fresh `CreatePreparedStatement` (a distinct
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
    /// Set after the first successful execution. Only consulted for
    /// `no_result_ddl` handles, to turn a repeat execution into a no-op.
    pub(super) executed: AtomicBool,
}

impl PreparedStatementEntry {
    pub(super) fn new(sql: String, no_result_ddl: bool) -> Self {
        Self { sql, no_result_ddl, executed: AtomicBool::new(false) }
    }

    /// True when this is a no-result-set DDL handle that has already executed
    /// once — i.e. a repeat execution that must be treated as a no-op.
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

impl DatafusionNexusFlightSqlService {
    pub(super) fn dataset_schema_for_prepared_statement(&self, plan: &LogicalPlan) -> crate::error::Result<Schema> {
        match plan {
            LogicalPlan::Ddl(ddl) => {
                verify_update_ddl(
                    ddl,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                )?;
                Ok(Schema::empty())
            },
            LogicalPlan::Dml(_) | LogicalPlan::Copy(_) => Err(Error::new(
                ErrorCode::UnsupportedOperator,
                "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                "Flight SQL prepared DML and COPY statements are not supported",
            )),
            LogicalPlan::Statement(statement) => match statement {
                Statement::SetVariable(_) | Statement::ResetVariable(_) => Ok(Schema::empty()),
                Statement::TransactionStart(_) | Statement::TransactionEnd(_) => Err(Error::new(
                    ErrorCode::UnsupportedOperator,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
                    "Flight SQL transactions are not supported",
                )),
                Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => Err(Error::new(
                    ErrorCode::InvalidRequest,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.verify_plan",
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
