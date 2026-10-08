//! Backend capability contract.
//!
//! Plan validation answers column binding, schema, null and ordering
//! semantics, and DAG invariants on its own. What it cannot answer is whether
//! an execution backend can *implement* the result with a correct kernel.
//! Those questions live here, and the engine supplies the RAPIDS answers.
//!
//! The contract is deliberately small: it exists to cut this crate's
//! dependency on a component, not to anticipate a second backend. Inputs and
//! outputs are query IR, Arrow types, and [`QueryError`], never a component
//! handle, expression, runtime context, or allocator.
//!
//! [`QueryError`]: crate::error::QueryError

use std::fmt;

use arrow::datatypes::DataType as ArrowDataType;

use crate::{
    error::Result,
    expr::{AggregateSpec, BinaryOp, Expr},
    plan::ProjectionOutput,
    policy::PlanningPolicy,
};

/// How a plan evaluates an expression, where the backend binding differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpressionUse {
    /// An ordinary row-wise value expression under SQL three-valued logic.
    Value,
    /// A filter predicate. Backends evaluate these with null-logical
    /// comparisons, so a predicate can bind where the same expression as a
    /// value would not, and vice versa.
    FilterPredicate,
}

/// Which aggregate lowering the plan asks the backend to prove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateBinding {
    /// One stage over raw input, grouped or global.
    SingleStage,
    /// The final merge over partial aggregate state.
    FinalMerge,
}

/// Execution-owner questions plan validation cannot answer from Arrow and IR
/// semantics alone.
///
/// An implementation must answer with the same component rules execution
/// binds with. A query that this contract accepts and a kernel then rejects is
/// a defect in the implementation, not a tolerated outcome.
///
/// `Send + Sync` because [`AnalysisContext`] crosses `async fn` boundaries in
/// engine and adapter pipeline stages, and those futures must stay `Send`.
pub trait BackendCapabilities: fmt::Debug + Send + Sync {
    /// Whether the backend has a physical representation for this Arrow type.
    ///
    /// Representable is not the same as usable everywhere: a timezone-carrying
    /// timestamp or a list column is representable but is not a group or
    /// comparison key. That restriction is semantic and stays with plan
    /// validation.
    fn supports_type(&self, data_type: &ArrowDataType) -> bool;

    /// The Arrow output type of the backend's binary kernel for these operands.
    ///
    /// An `Err` means the kernel has no binding for the operand pair, and
    /// validation reports it as an unsupported expression. Validation reaches
    /// this after its own temporal, interval and decimal rules have run; a
    /// declared `output_type` is accepted only for decimal division and is
    /// checked by those rules rather than inferred here.
    fn binary_output_type(&self, op: BinaryOp, left: &ArrowDataType, right: &ArrowDataType) -> Result<ArrowDataType>;

    /// Proves the backend can bind this expression for this use.
    ///
    /// The check is structural and runs without device memory or a stream.
    /// Its `Err` keeps the backend's own code and facts, which the validator
    /// forwards unchanged so the report names the kernel that refused.
    fn check_expression(&self, expr: &Expr, expression_use: ExpressionUse) -> Result<()>;

    /// Proves the backend can bind this projection output.
    fn check_projection_output(&self, output: &ProjectionOutput) -> Result<()>;

    /// Proves the backend can bind this aggregate, reporting the output column
    /// names the binding produces.
    ///
    /// The names are what plan validation compares against the declared output
    /// schema, so the proof and the schema check read one result instead of
    /// two independent derivations.
    fn aggregate_output_names(&self, spec: &AggregateSpec, binding: AggregateBinding) -> Result<Vec<String>>;
}

/// Carries the backend capability through one analysis or validation pass.
///
/// This is a short-lived borrow, not stored state: neither [`QueryPlan`] nor
/// the capability analysis nor a compiled artifact holds it, so plan IR and
/// proof never take on an implementation's lifetime. The one value that does
/// outlive the pass is `policy`, which the admission artifact records so
/// execution runs under the thresholds the plan was admitted with.
///
/// [`QueryPlan`]: crate::plan::QueryPlan
#[derive(Debug, Clone, Copy)]
pub struct AnalysisContext<'a> {
    backend: &'a dyn BackendCapabilities,
    /// Copied into the context so every pass and every error fact it emits
    /// reads one policy; `admit` overrides it with the policy it decides under.
    policy: PlanningPolicy,
}

impl<'a> AnalysisContext<'a> {
    /// Binds an analysis pass to one backend implementation under `policy`,
    /// the execution policy the plan is optimized, admitted, and executed
    /// with.
    pub const fn new(backend: &'a dyn BackendCapabilities, policy: PlanningPolicy) -> Self {
        Self { backend, policy }
    }

    /// The same pass under the execution policy the plan is optimized,
    /// admitted, and executed with.
    #[must_use]
    pub const fn with_policy(mut self, policy: PlanningPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The backend answering this pass's capability questions.
    pub(crate) const fn backend(&self) -> &'a dyn BackendCapabilities {
        self.backend
    }

    /// The execution policy this pass analyzes the plan under.
    pub const fn policy(&self) -> PlanningPolicy {
        self.policy
    }
}
