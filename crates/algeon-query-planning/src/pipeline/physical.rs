//! Physical specs the compiler records for each pipeline task.
//!
//! A spec holds the structural choices made for one task: how a Parquet
//! source is read, which strategies a join may run, which top-k an aggregate
//! fuses, and how a final sort runs. Choices that depend on the runtime
//! (chunk counts, lane availability, prefetch policy, observed sizes) stay in
//! the engine.

use crate::{
    capability::{
        JoinForm, JoinStrategy, JoinStrategyChoice, KeyedHashJoinStrategyInputFacts, KeyedHashJoinStrategyOptions,
        choose_join_strategy,
    },
    error::{QueryError, QueryErrorCode, Result},
    pipeline::model::{PipelineTaskKind, SortStrategy},
    plan::{JoinNode, NodeId},
};

/// What a task executes, with the spec of kinds that need one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskPhysical {
    /// Read a source using the compiled source-output contract.
    SourceRead(SourceReadPlan),
    /// Borrow an existing GPU-resident input.
    GpuResidentSource,
    /// Import host Arrow literal batches.
    LiteralBatchSource,
    /// Evaluate a unary operation such as filter or projection.
    Unary,
    /// Accumulate and finalize aggregate state with optional top-k fusion.
    Aggregate(AggregatePlan),
    /// Evaluate window expressions over the required input ordering.
    Window,
    /// Construct the resident join build state.
    JoinBuild(JoinPlan),
    /// Probe join state with the other input.
    JoinProbe(JoinPlan),
    /// Gather join matches into output columns.
    JoinMaterialize,
    /// Produce the final ordering under the compiled sort strategy.
    FinalSort(SortStrategy),
    /// Apply the plan's row limit.
    Limit,
    /// Combine the union's input resources.
    Union,
    /// Normalize graph edges before graph construction.
    EdgeNormalize,
    /// Execute the graph operation represented by the plan node.
    GraphAlgorithm,
    /// Execute the vector operation represented by the plan node.
    VectorAlgorithm,
    /// Execute the machine-learning operation represented by the plan node.
    MlAlgorithm,
    /// Materialize results at the declared output boundary.
    SinkMaterialize,
}

impl TaskPhysical {
    /// Returns the task-kind label corresponding to this physical specification.
    pub const fn kind(&self) -> PipelineTaskKind {
        match self {
            Self::SourceRead(_) => PipelineTaskKind::SourceRead,
            Self::GpuResidentSource => PipelineTaskKind::GpuResidentSource,
            Self::LiteralBatchSource => PipelineTaskKind::LiteralBatchSource,
            Self::Unary => PipelineTaskKind::Unary,
            Self::Aggregate(_) => PipelineTaskKind::Aggregate,
            Self::Window => PipelineTaskKind::Window,
            Self::JoinBuild(_) => PipelineTaskKind::JoinBuild,
            Self::JoinProbe(_) => PipelineTaskKind::JoinProbe,
            Self::JoinMaterialize => PipelineTaskKind::JoinMaterialize,
            Self::FinalSort(_) => PipelineTaskKind::FinalSort,
            Self::Limit => PipelineTaskKind::Limit,
            Self::Union => PipelineTaskKind::Union,
            Self::EdgeNormalize => PipelineTaskKind::EdgeNormalize,
            Self::GraphAlgorithm => PipelineTaskKind::GraphAlgorithm,
            Self::VectorAlgorithm => PipelineTaskKind::VectorAlgorithm,
            Self::MlAlgorithm => PipelineTaskKind::MlAlgorithm,
            Self::SinkMaterialize => PipelineTaskKind::SinkMaterialize,
        }
    }
}

/// How a local Parquet source task reads its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceReadPlan {
    /// Whether the read materializes a frame or remains deferred for consumers.
    pub output: SourceOutput,
    /// The source has dynamic-filter routes. Their producers run before this
    /// task, so a source read by its only consumer applies them here.
    pub dynamic_filter_routes: bool,
    /// The source feeds one join input directly with no dynamic-filter
    /// route, so a deferred read may start a prefetch. Whether it does is
    /// runtime policy.
    pub prefetch_eligible: bool,
}

impl SourceReadPlan {
    /// Read the whole source into one frame.
    pub const FRAME: Self =
        Self { output: SourceOutput::Frame, dynamic_filter_routes: false, prefetch_eligible: false };
}

/// The payload a local Parquet source task produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceOutput {
    /// Read the whole source into one frame.
    Frame,
    /// Read the source as a join leaf stream with the unary stages above it
    /// (`adjacent_stages`) and materialize it into one frame.
    JoinLeafFrame {
        /// Whether unary stages between source and consumer are applied during reading.
        adjacent_stages: bool,
    },
    /// Keep a join leaf read deferred for the consumers to pull.
    JoinLeafStream {
        /// Whether unary stages between source and consumer are applied during reading.
        adjacent_stages: bool,
    },
    /// Keep a read deferred for an aggregate to pull chunk by chunk.
    AggregateInputStream {
        /// Whether unary stages between source and consumer are applied during reading.
        adjacent_stages: bool,
        /// Structural facts used when selecting aggregate input chunk sizes.
        chunk_hints: AggregateInputHints,
    },
    /// Keep a read deferred for consumers that include both a join input and
    /// an aggregate.
    SharedJoinAggregateStream {
        /// Whether unary stages between source and consumer are applied during reading.
        adjacent_stages: bool,
        /// Structural facts used when selecting aggregate input chunk sizes.
        chunk_hints: AggregateInputHints,
    },
    /// Keep a read deferred through a chain of unary stages whose consumers
    /// take only frames. The last stage reads the source chunk by chunk,
    /// applies the stages to each chunk and concatenates the stage outputs in
    /// chunk order, so the whole decoded source is never resident at once.
    UnaryChainStream,
}

/// Structural facts an aggregate fed by a deferred source read sizes its
/// chunks from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateInputHints {
    /// Unary stages sit between the source and its consumers.
    pub adjacent_stages: bool,
    /// The fed aggregate groups by a plain string column.
    pub grouped_string_key: bool,
}

/// The strategies a join may run and the one planning chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinPlan {
    form: JoinForm,
    /// Strategy chosen from planning-time size facts; it sets the memory
    /// shape admission assumed and stays allowed through every narrowing.
    planned: JoinStrategy,
    /// Strategies the task's input, state and output forms admit. Starts as
    /// every strategy of `form` and is only ever narrowed.
    allowed: Vec<JoinStrategy>,
}

impl JoinPlan {
    /// A plan whose chosen strategy is one of the allowed ones.
    #[doc(hidden)]
    pub fn new(form: JoinForm, planned: JoinStrategy, allowed: Vec<JoinStrategy>) -> Result<Self> {
        if !allowed.contains(&planned) || allowed.iter().any(|strategy| !form.strategies().contains(strategy)) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.pipeline.join_plan",
                "a join plan allows a strategy its form cannot run or plans one it does not allow",
            )
            .with_fact("planned_strategy", planned.as_str())
            .with_fact("allowed_strategies", strategy_list(&allowed)));
        }
        Ok(Self { form, planned, allowed })
    }

    /// Every strategy of `form`, with `planned` chosen among them.
    pub(crate) fn for_form(form: JoinForm, planned: JoinStrategy) -> Self {
        Self { form, planned, allowed: form.strategies().to_vec() }
    }

    /// Keeps the allowed strategies `keep` accepts. The planned strategy set
    /// the contract the narrowing enforces, so it always stays.
    pub(crate) fn narrowed(&self, keep: impl Fn(JoinStrategy) -> bool) -> Self {
        let allowed = self.allowed.iter().copied().filter(|strategy| *strategy == self.planned || keep(*strategy));
        Self { form: self.form, planned: self.planned, allowed: allowed.collect() }
    }

    #[doc(hidden)]
    pub const fn form(&self) -> JoinForm {
        self.form
    }

    #[doc(hidden)]
    pub const fn planned(&self) -> JoinStrategy {
        self.planned
    }

    #[doc(hidden)]
    pub fn allowed(&self) -> &[JoinStrategy] {
        &self.allowed
    }

    /// Chooses the strategy for observed facts. The choice must stay within
    /// the strategies this plan proved executable for the task's input,
    /// state, and output forms; anything else is an invariant violation.
    #[doc(hidden)]
    pub fn refine(
        &self,
        join: &JoinNode,
        input: KeyedHashJoinStrategyInputFacts,
        options: &KeyedHashJoinStrategyOptions,
    ) -> Result<JoinStrategyChoice> {
        let choice = choose_join_strategy(self.form, join, input, options);
        if !self.allowed.contains(&choice.strategy()) {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.pipeline.join_plan",
                "observed facts chose a join strategy outside the compiled join plan",
            )
            .with_fact("planned_strategy", self.planned.as_str())
            .with_fact("chosen_strategy", choice.strategy().as_str())
            .with_fact("allowed_strategies", strategy_list(&self.allowed)));
        }
        Ok(choice)
    }
}

fn strategy_list(strategies: &[JoinStrategy]) -> String {
    strategies.iter().map(|strategy| strategy.as_str()).collect::<Vec<_>>().join(",")
}

/// What an aggregate task fuses into its execution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AggregatePlan {
    /// Fused top-k stages, or `None` when aggregation has no top-k fusion.
    pub topk: Option<TopKFusion>,
}

/// A fused top-k over an aggregate's output: the projections between them,
/// the sort, and the limit the sort fused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopKFusion {
    /// Projection nodes between the aggregate and the sort, each the sole
    /// consumer of its input; the aggregate applies them before the top-k.
    pub post_aggregate_projections: Vec<NodeId>,
    /// Sort whose `FusedTopK` strategy this fusion extends downward.
    pub sort_node: NodeId,
    /// Limit the sort fused; its fetch bounds the aggregate output.
    pub limit_node: NodeId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::ColumnRef,
        plan::{JoinKey, JoinOutputColumn, JoinSpec, JoinType},
    };

    fn inner_join() -> JoinNode {
        JoinNode {
            inputs: [NodeId::from_index_for_test(0), NodeId::from_index_for_test(1)],
            spec: JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")]),
        }
    }

    /// A small build with no probe size keeps the reusable hash table.
    fn small_build_facts() -> KeyedHashJoinStrategyInputFacts {
        KeyedHashJoinStrategyInputFacts {
            estimated_build_rows: Some(10),
            observed_build_rows: Some(10),
            build_side_bytes: Some(1024),
            probe_side_bytes: None,
            estimated_output_rows: None,
            estimated_output_bytes: None,
            build_key_unique: None,
            probe_reuse_count: 1,
            reusable_prebuild_supported: true,
        }
    }

    #[test]
    fn join_plan_refinement_never_leaves_the_allowed_strategies() {
        let join = inner_join();
        let options = KeyedHashJoinStrategyOptions::default();
        let full = JoinPlan::for_form(JoinForm::KeyedEqui, JoinStrategy::OneShotEqui);
        assert_eq!(
            full.refine(&join, small_build_facts(), &options).expect("an allowed refinement").strategy(),
            JoinStrategy::ReusableHashEqui
        );

        let narrowed = full.narrowed(|strategy| strategy != JoinStrategy::ReusableHashEqui);
        assert_eq!(narrowed.allowed(), [JoinStrategy::OneShotEqui, JoinStrategy::PartitionedHashEqui]);
        let error = narrowed.refine(&join, small_build_facts(), &options).expect_err("refinement outside the plan");
        assert_eq!(error.code(), QueryErrorCode::InternalInvariantViolation);
        assert_eq!(error.fact_value("chosen_strategy"), Some("reusable_hash_equi"));
        assert_eq!(error.fact_value("allowed_strategies"), Some("one_shot_equi,partitioned_hash_equi"));
    }

    #[test]
    fn join_plan_rejects_strategies_its_form_cannot_run() {
        let error = JoinPlan::new(
            JoinForm::KeyedEqui,
            JoinStrategy::MixedKeyedResidual,
            vec![JoinStrategy::MixedKeyedResidual],
        )
        .expect_err("a keyed equi-join cannot run the mixed residual strategy");
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        JoinPlan::new(JoinForm::KeyedEqui, JoinStrategy::OneShotEqui, vec![JoinStrategy::ReusableHashEqui])
            .expect_err("the planned strategy must be allowed");
    }
}
