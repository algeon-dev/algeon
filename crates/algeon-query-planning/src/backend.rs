//! Backend capability contract.
//!
//! Plan validation resolves column binding, schemas, nullability, ordering semantics,
//! and DAG invariants independently. Backend capability checks evaluate whether an
//! execution backend provides physical kernel support for specific types, operations,
//! and aggregate bindings.
//!
//! The interface decouples the planning crate from concrete engine components.
//! Inputs and outputs operate entirely on query IR, Arrow types, and [`QueryError`];
//! component handles, expressions, runtime contexts, and allocators are excluded.
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

/// Execution capabilities verified through backend-specific rules rather than
/// generic Arrow or IR semantics.
///
/// Implementations must align capability validation with runtime kernel constraints.
/// An operation admitted by this contract and subsequently rejected by execution
/// constitutes a contract violation.
///
/// Requires `Send + Sync` so [`AnalysisContext`] can traverse asynchronous boundaries
/// in query planning and execution pipelines.
pub trait BackendCapabilities: fmt::Debug + Send + Sync {
    /// Indicates whether the backend supports a physical representation for the Arrow type.
    ///
    /// Physical support does not imply unrestricted semantic validity. For example,
    /// a timestamp with timezone or a list type may have a physical representation
    /// while remaining invalid as a grouping or comparison key. Semantic constraints
    /// are enforced during plan validation.
    fn supports_type(&self, data_type: &ArrowDataType) -> bool;

    /// Largest number of elements one backend column holds, counted at every
    /// level: the row count, and the child element count of each nested list.
    ///
    /// A column that would exceed it in one source read or one intermediate
    /// result cannot be represented. Source chunk planning sizes every read
    /// under this count.
    fn max_column_elements(&self) -> u64;

    /// The Arrow output type of the backend's binary kernel for these operands.
    ///
    /// An `Err` means the kernel has no binding for the operand pair, and
    /// validation reports it as an unsupported expression. Validation reaches
    /// this after its own temporal, interval and decimal rules have run; a
    /// declared `output_type` is accepted only for decimal division, and those
    /// rules check it. This method does not infer it.
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
    /// schema, so the proof and the schema check read one result. Neither
    /// derives the names independently.
    fn aggregate_output_names(&self, spec: &AggregateSpec, binding: AggregateBinding) -> Result<Vec<String>>;
}

/// Carries backend capabilities through one analysis or validation pass.
///
/// Analysis context represents a short-lived borrow. Neither [`QueryPlan`] nor
/// the capability analysis nor a compiled artifact retains it, ensuring plan IR
/// and proofs remain independent of backend lifetimes. The `policy` value
/// is recorded into the admission artifact so execution runs under the thresholds
/// established during plan admission.
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
