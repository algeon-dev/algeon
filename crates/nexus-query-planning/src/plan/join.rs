//! Native join IR contracts.
//!
//! Join semantics, residual predicates, output projection, build-side choice,
//! null equality, and durable build-cache identity are represented separately so
//! optimizers and executors do not infer one contract from another.

use super::*;

/// Binary join operation.
///
/// `inputs[0]` is the semantic left input and `inputs[1]` is the semantic right
/// input. Physical build/probe choices are recorded separately in
/// [`JoinSpec::build_side`] so join semantics remain independent from executor
/// strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinNode {
    /// Semantic join inputs in `[left, right]` order.
    pub inputs: [NodeId; 2],
    /// Join semantics, predicates, projection, and build/cache choices.
    pub spec: JoinSpec,
}

/// Native join contract with keys, residual predicate, output projection, null
/// policy, and physical build/cache choices modeled separately.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinSpec {
    /// Logical join type.
    pub join_type: JoinType,
    /// Equality predicates matched by column pairs.
    ///
    /// Empty keys are valid for cross joins and keyless conditional joins when
    /// the selected executor path supports them.
    pub keys: Vec<JoinKey>,
    /// Optional predicate evaluated after equality matching.
    ///
    /// This is the place for non-equality join conditions. Keeping it separate
    /// lets the planner reason about hash keys and post-join filters
    /// independently.
    pub residual_predicate: Option<Expr>,
    /// Explicit output projection in result schema order.
    ///
    /// Executable native joins require this to be populated explicitly and to
    /// match the result schema order. An empty projection is only an incomplete
    /// builder default before a frontend sets the projection; execution
    /// validation rejects it.
    pub output_projection: Vec<JoinOutputColumn>,
    /// Null matching behavior for equality keys.
    pub null_equality: JoinNullEquality,
    /// Input chosen to build reusable/hash state.
    ///
    /// This is a physical execution choice. It does not redefine which input is
    /// semantic left or right; those remain [`JoinNode::inputs`].
    pub build_side: JoinSide,
    /// Input side whose order DataFusion preserves for exact statistical replay.
    ///
    /// DataFusion's `HashJoinExec` can preserve a different side from the native
    /// hash-build side. This side is only used for hidden order columns that
    /// make order-sensitive floating statistical aggregates bit-exact.
    pub datafusion_order_preserved_side: JoinSide,
    /// Durable cache identifier for reusable hash-join build state.
    ///
    /// When present, the executor may store and reuse the build artifact under
    /// this id for compatible joins in the same execution context. The id is a
    /// contract between frontend and executor: it must be non-empty and stable
    /// for the logical build input, key set, null-equality behavior, and other
    /// build-affecting choices.
    pub build_artifact_id: Option<HashJoinBuildArtifactId>,
    /// Output column name for the Boolean mark column in a `LeftMark` join.
    ///
    /// Must be `Some` when `join_type == LeftMark` and `None` otherwise.
    /// The mark column appears as the last field in the output schema.
    pub mark_column_name: Option<String>,
    /// A proven lookup relationship that bounds output rows by one input.
    ///
    /// The contract is only set by a lowering path that established at most one
    /// match per bounded row for the join keys. Generic joins leave this absent
    /// and retain the conservative Cartesian upper bound.
    #[doc(hidden)]
    pub cardinality_contract: Option<JoinCardinalityContract>,
    /// Output columns whose operator semantics prove that lookup gathers do not
    /// exceed the selected input column's total string payload bound.
    #[doc(hidden)]
    pub total_payload_bound_preserving_outputs: Vec<String>,
}

/// Cardinality contract for a join whose lookup side has at most one match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum JoinCardinalityContract {
    /// Every row on `bounded_side` produces at most one output row.
    AtMostOneMatchPerBoundedRow { bounded_side: JoinSide },
}

impl JoinSpec {
    /// Creates a join spec with default physical choices.
    ///
    /// Defaults are: no residual predicate, empty output projection,
    /// `NullsDoNotMatch`, right-side build, right-side DataFusion order
    /// preservation, and no reusable build artifact id.
    /// The empty output projection is an incomplete builder default; frontends
    /// must set an explicit projection before the join can execute.
    pub fn new(join_type: JoinType, keys: Vec<JoinKey>) -> Self {
        Self {
            join_type,
            keys,
            residual_predicate: None,
            output_projection: Vec::new(),
            null_equality: JoinNullEquality::NullsDoNotMatch,
            build_side: JoinSide::Right,
            datafusion_order_preserved_side: JoinSide::Right,
            build_artifact_id: None,
            mark_column_name: None,
            cardinality_contract: None,
            total_payload_bound_preserving_outputs: Vec::new(),
        }
    }

    /// Sets the non-equality join predicate.
    pub fn with_residual_predicate(mut self, predicate: Expr) -> Self {
        self.residual_predicate = Some(predicate);
        self
    }

    /// Sets an explicit result projection for the join.
    pub fn with_output_projection(mut self, output_projection: Vec<JoinOutputColumn>) -> Self {
        self.output_projection = output_projection;
        self
    }

    /// Sets null matching behavior for equality keys.
    pub fn with_null_equality(mut self, null_equality: JoinNullEquality) -> Self {
        self.null_equality = null_equality;
        self
    }

    /// Sets the physical build side without changing semantic left/right inputs.
    pub fn with_build_side(mut self, build_side: JoinSide) -> Self {
        self.build_side = build_side;
        self
    }

    /// Sets the side whose order should drive DataFusion-exact hidden ordering.
    pub fn with_datafusion_order_preserved_side(mut self, side: JoinSide) -> Self {
        self.datafusion_order_preserved_side = side;
        self
    }

    /// Assigns a durable reusable-build cache id.
    ///
    /// The value must identify the complete build artifact contract, not just a
    /// table name. Validation rejects empty ids.
    pub fn with_build_artifact_id(mut self, id: impl Into<String>) -> Self {
        self.build_artifact_id = Some(HashJoinBuildArtifactId::new(id));
        self
    }

    /// Sets the output column name for the Boolean mark column in a `LeftMark` join.
    pub fn with_mark_column_name(mut self, name: impl Into<String>) -> Self {
        self.mark_column_name = Some(name.into());
        self
    }

    /// Marks this join as having at most one match per row on `bounded_side`.
    pub fn with_at_most_one_match_per_bounded_row(mut self, bounded_side: JoinSide) -> Self {
        self.cardinality_contract = Some(JoinCardinalityContract::AtMostOneMatchPerBoundedRow { bounded_side });
        self
    }

    /// Marks one output whose total string payload remains bounded by its
    /// selected input column despite lookup-side value repetition.
    pub fn with_preserved_total_payload_bound_for_output(mut self, output_name: impl Into<String>) -> Self {
        let output_name = output_name.into();
        if !self.total_payload_bound_preserving_outputs.contains(&output_name) {
            self.total_payload_bound_preserving_outputs.push(output_name);
        }
        self
    }

    #[doc(hidden)]
    pub fn has_at_most_one_match_per_bounded_row(&self, bounded_side: JoinSide) -> bool {
        matches!(
            self.cardinality_contract,
            Some(JoinCardinalityContract::AtMostOneMatchPerBoundedRow { bounded_side: contract_side })
                if contract_side == bounded_side
        )
    }

    #[doc(hidden)]
    pub fn preserves_total_payload_bound_for_output(&self, output_name: &str) -> bool {
        self.total_payload_bound_preserving_outputs.iter().any(|candidate| candidate == output_name)
    }
}

/// Durable identifier for reusable hash-join build state.
///
/// Execution-context cache key for reusable hash-join build artifacts.
///
/// Frontends must change it when any build-affecting property changes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HashJoinBuildArtifactId(String);

impl HashJoinBuildArtifactId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns true when this id would be invalid for reusable build caching.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for HashJoinBuildArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One equality-key pair for a join.
///
/// `left` is resolved against the semantic left input and `right` against the
/// semantic right input. Non-equality conditions belong in
/// [`JoinSpec::residual_predicate`], not in this list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinKey {
    /// Column from the semantic left input.
    pub left: ColumnRef,
    /// Column from the semantic right input.
    pub right: ColumnRef,
}

impl JoinKey {
    pub fn new(left: ColumnRef, right: ColumnRef) -> Self {
        Self { left, right }
    }
}

/// One column selected by an explicit join output projection.
///
/// The projection list order is the result schema order. `output_name` is the
/// name exposed at the join output boundary and may differ from the input column
/// name to resolve duplicates or preserve frontend naming rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinOutputColumn {
    /// Which semantic input side provides the column.
    pub side: JoinSide,
    /// Column selected from that side.
    pub column: ColumnRef,
    /// Name assigned to the output field.
    pub output_name: String,
}

impl JoinOutputColumn {
    pub fn new(side: JoinSide, column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self { side, column, output_name: output_name.into() }
    }

    pub fn left(column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self::new(JoinSide::Left, column, output_name)
    }

    pub fn right(column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self::new(JoinSide::Right, column, output_name)
    }
}

impl fmt::Display for JoinOutputColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{} AS {}", self.side, self.column.name(), self.output_name)
    }
}

/// Logical join type represented by the native IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinType {
    /// Keep rows that match on the join condition from both sides.
    Inner,
    /// Keep every left row, with nullable right columns when unmatched.
    Left,
    /// Keep every row from both sides, with nullable columns for unmatched side.
    Full,
    /// Keep left rows that have a match; output only left-side columns.
    LeftSemi,
    /// Keep left rows that do not have a match; output only left-side columns.
    LeftAnti,
    /// Cartesian join; equality keys should be empty.
    Cross,
    /// Keep every left row; output left-side columns plus a Boolean mark column
    /// that is `true` when at least one matching right row exists.
    LeftMark,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Inner => "inner",
            Self::Left => "left",
            Self::Full => "full",
            Self::LeftSemi => "left_semi",
            Self::LeftAnti => "left_anti",
            Self::Cross => "cross",
            Self::LeftMark => "left_mark",
        })
    }
}

/// Null matching policy for join equality keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinNullEquality {
    /// Null key values do not match other nulls.
    NullsDoNotMatch,
    /// Null key values match other nulls.
    NullsMatch,
}

impl fmt::Display for JoinNullEquality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NullsDoNotMatch => "nulls_do_not_match",
            Self::NullsMatch => "nulls_match",
        })
    }
}

/// Semantic input side for a binary join.
///
/// In [`JoinNode::inputs`], left is index 0 and right is index 1. The same enum
/// is also used for physical build-side selection; in that context it names
/// which semantic side is built, not a new logical orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinSide {
    /// Semantic left input.
    Left,
    /// Semantic right input.
    Right,
}

impl fmt::Display for JoinSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Left => "left",
            Self::Right => "right",
        })
    }
}
