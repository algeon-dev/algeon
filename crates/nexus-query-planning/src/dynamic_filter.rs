//! DataFusion-free dynamic-filter IR, payloads, and diagnostics.
//!
//! Dynamic filters are runtime producer/consumer channels carried by native
//! plans. They let a selective producer such as a hash-join build side or
//! aggregate keyset narrow a downstream consumer such as a Parquet source or a
//! join probe before expensive work starts.

use std::{collections::BTreeMap, fmt};

use arrow::{
    array::{Array, Date32Array, Int32Array, Int64Array, UInt32Array, UInt64Array},
    datatypes::DataType,
};
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::{BinaryOp, Expr, LiteralValue},
    optimizer::{DimensionKeyFilter, KeyRange, parent_counts},
    parquet_stats::{I64StatisticsRange as I64Range, row_group_i64_range_for_root_column},
    plan::{JoinSide, NodeId, PlanNode, QueryPlan},
    source::{LocalParquetSource, SourceColumnRef},
};

/// Stable identifier for a dynamic-filter route inside one native plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DynamicFilterId(#[doc(hidden)] pub usize);

impl DynamicFilterId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for DynamicFilterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "df{}", self.0)
    }
}

/// Runtime ordering contract between a dynamic-filter producer and consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DynamicFilterReadiness {
    /// Consumer must wait until the producer payload is complete.
    ProducerCompleteBeforeConsumer,
}

impl DynamicFilterReadiness {
    /// Stable lower-case code used by diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProducerCompleteBeforeConsumer => "producer_complete_before_consumer",
        }
    }
}

impl fmt::Display for DynamicFilterReadiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dynamic-filter payload kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DynamicFilterKind {
    /// Inclusive min/max scalar range.
    Range,
    /// Exact small keyset that can be lowered into a bounded `IN (...)` predicate.
    SmallInList,
    /// Exact larger keyset applied by hash membership after decode or during join probe.
    HashSetContains,
}

impl DynamicFilterKind {
    /// Stable lower-case code used by diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Range => "range",
            Self::SmallInList => "small_in_list",
            Self::HashSetContains => "hash_set_contains",
        }
    }
}

impl fmt::Display for DynamicFilterKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a dynamic-filter consumer can apply the route payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DynamicFilterConsumerLowering {
    /// Payload can be translated to a Parquet reader predicate.
    ReaderPredicate,
    /// Payload can be merged with an existing source reader predicate.
    MergedReaderPredicate,
    /// Payload cannot be pushed into the reader but can be applied after decode.
    PostDecodeOnly,
    /// A static predicate mix prevents safe reader lowering for now.
    SkippedUnsupportedStaticPredicateMix,
}

impl DynamicFilterConsumerLowering {
    /// Stable lower-case code used by diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReaderPredicate => "reader_predicate",
            Self::MergedReaderPredicate => "merged_reader_predicate",
            Self::PostDecodeOnly => "post_decode_only",
            Self::SkippedUnsupportedStaticPredicateMix => "skipped_unsupported_static_predicate_mix",
        }
    }
}

impl fmt::Display for DynamicFilterConsumerLowering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Execution stage where a dynamic-filter consumer can first reduce work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DynamicFilterConsumerReductionStage {
    /// Local Parquet reader predicates can reduce materialized source rows and
    /// may also participate in row-group pruning.
    SourceReader,
    /// Local Parquet source applies an exact payload after decode.
    SourcePostDecode,
    /// Filter operator applies the payload before downstream join-probe work.
    FilterPostDecode,
    /// Hash-join probe applies the payload immediately before probe work.
    JoinProbePostDecode,
}

impl DynamicFilterConsumerReductionStage {
    /// Stable lower-case code used by diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SourceReader => "source_reader",
            Self::SourcePostDecode => "source_post_decode",
            Self::FilterPostDecode => "filter_post_decode",
            Self::JoinProbePostDecode => "join_probe_post_decode",
        }
    }
}

impl fmt::Display for DynamicFilterConsumerReductionStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Native producer for a dynamic-filter route.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DynamicFilterProducer {
    /// Hash-join build side produces column evidence for a downstream
    /// consumer. The producer columns are usually join keys, but may also be
    /// non-key build-frame columns when a later residual equality proves they
    /// constrain another source.
    HashJoinBuild {
        /// Join node that owns the hash build.
        join_node_id: NodeId,
        /// Semantic input side chosen as the physical build side.
        build_side: JoinSide,
        /// Build-side input node.
        build_input: NodeId,
    },
    /// Aggregate or distinct group keys produce an exact keyset.
    AggregateGroupKeys {
        /// Aggregate node producing distinct group keys.
        aggregate_node_id: NodeId,
    },
}

impl DynamicFilterProducer {
    /// Stable producer kind.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::HashJoinBuild { .. } => "hash_join_build",
            Self::AggregateGroupKeys { .. } => "aggregate_group_keys",
        }
    }

    /// Plan node primarily responsible for producing the route payload.
    pub const fn node_id(&self) -> NodeId {
        match self {
            Self::HashJoinBuild { join_node_id, .. } => *join_node_id,
            Self::AggregateGroupKeys { aggregate_node_id } => *aggregate_node_id,
        }
    }
}

/// Native consumer for a dynamic-filter route.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DynamicFilterConsumer {
    /// Local Parquet source consumes the filter for row-group pruning, reader
    /// predicate construction, or post-decode filtering.
    LocalParquetSource {
        /// Source node receiving the payload.
        source_node_id: NodeId,
    },
    /// Hash-join probe side consumes the filter before probe work.
    HashJoinProbe {
        /// Join node whose probe side can apply the payload.
        join_node_id: NodeId,
        /// Semantic input side used as probe.
        probe_side: JoinSide,
    },
    /// Filter operator applies the payload as an additional predicate.
    Filter {
        /// Filter node receiving the payload.
        filter_node_id: NodeId,
    },
}

impl DynamicFilterConsumer {
    /// Stable consumer kind.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::LocalParquetSource { .. } => "local_parquet_source",
            Self::HashJoinProbe { .. } => "hash_join_probe",
            Self::Filter { .. } => "filter",
        }
    }

    /// Plan node primarily responsible for consuming the route payload.
    pub const fn node_id(&self) -> NodeId {
        match self {
            Self::LocalParquetSource { source_node_id } => *source_node_id,
            Self::HashJoinProbe { join_node_id, .. } => *join_node_id,
            Self::Filter { filter_node_id } => *filter_node_id,
        }
    }
}

/// Producer/consumer route for one native dynamic filter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicFilterRoute {
    /// Plan-local dynamic-filter id.
    pub id: DynamicFilterId,
    /// Runtime producer.
    pub producer: DynamicFilterProducer,
    /// Runtime consumer.
    pub consumer: DynamicFilterConsumer,
    /// Producer columns that define the payload key.
    pub producer_columns: Vec<String>,
    /// Consumer columns that receive the payload.
    pub consumer_columns: Vec<String>,
    /// Payload kinds this route may produce.
    pub filter_kinds: Vec<DynamicFilterKind>,
    /// Consumer lowering selected during route discovery.
    pub consumer_lowering: DynamicFilterConsumerLowering,
    /// Runtime readiness contract.
    pub readiness: DynamicFilterReadiness,
}

impl DynamicFilterRoute {
    /// Creates a route with a producer-complete-before-consumer readiness contract.
    pub fn new(
        id: DynamicFilterId,
        producer: DynamicFilterProducer,
        consumer: DynamicFilterConsumer,
        producer_columns: Vec<String>,
        consumer_columns: Vec<String>,
        filter_kinds: Vec<DynamicFilterKind>,
        consumer_lowering: DynamicFilterConsumerLowering,
    ) -> Self {
        Self {
            id,
            producer,
            consumer,
            producer_columns,
            consumer_columns,
            filter_kinds,
            consumer_lowering,
            readiness: DynamicFilterReadiness::ProducerCompleteBeforeConsumer,
        }
    }

    /// Stage where this route can first reduce consumer-side work.
    pub const fn consumer_reduction_stage(&self) -> DynamicFilterConsumerReductionStage {
        match (&self.consumer, self.consumer_lowering) {
            (
                DynamicFilterConsumer::LocalParquetSource { .. },
                DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate,
            ) => DynamicFilterConsumerReductionStage::SourceReader,
            (DynamicFilterConsumer::LocalParquetSource { .. }, _) => {
                DynamicFilterConsumerReductionStage::SourcePostDecode
            },
            (DynamicFilterConsumer::Filter { .. }, _) => DynamicFilterConsumerReductionStage::FilterPostDecode,
            (DynamicFilterConsumer::HashJoinProbe { .. }, _) => {
                DynamicFilterConsumerReductionStage::JoinProbePostDecode
            },
        }
    }
}

/// Exact or approximate runtime payload produced for one dynamic-filter route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicFilter {
    /// No non-null key survived on the producer side, so the consumer can
    /// become an empty scan.
    Empty,
    /// Inclusive scalar range over an integer-like key.
    RangeI64 {
        /// Inclusive minimum producer key.
        min: i64,
        /// Inclusive maximum producer key.
        max: i64,
    },
    /// Exact small integer-like keyset plus its range.
    SmallInListI64 {
        /// Sorted unique values in the producer keyset.
        values: Vec<i64>,
        /// Inclusive minimum producer key.
        min: i64,
        /// Inclusive maximum producer key.
        max: i64,
    },
    /// Exact larger integer-like keyset applied after decode or at a join-probe
    /// boundary. Reader predicate lowering intentionally ignores this payload
    /// and uses only its range for row-group pruning.
    HashSetContainsI64 {
        /// Sorted unique values in the producer keyset.
        values: Vec<i64>,
        /// Inclusive minimum producer key.
        min: i64,
        /// Inclusive maximum producer key.
        max: i64,
    },
}

impl DynamicFilter {
    /// Builds an initial integer-like payload from host Arrow key values.
    pub fn from_i64_values(values: impl IntoIterator<Item = Option<i64>>, small_in_list_limit: usize) -> Self {
        let mut values = values.into_iter().flatten().collect::<Vec<_>>();
        if values.is_empty() {
            return Self::Empty;
        }
        values.sort_unstable();
        let min = values[0];
        let max = *values.last().expect("non-empty values have a max");
        values.dedup();
        if values_form_dense_i64_range(&values, min, max) {
            return Self::RangeI64 { min, max };
        }
        if values.len() <= small_in_list_limit {
            Self::SmallInListI64 { values, min, max }
        } else {
            Self::HashSetContainsI64 { values, min, max }
        }
    }

    /// Stable payload kind for diagnostics.
    pub const fn kind(&self) -> DynamicFilterKind {
        match self {
            Self::Empty | Self::RangeI64 { .. } => DynamicFilterKind::Range,
            Self::SmallInListI64 { .. } => DynamicFilterKind::SmallInList,
            Self::HashSetContainsI64 { .. } => DynamicFilterKind::HashSetContains,
        }
    }

    /// Inclusive integer-like min/max carried by this payload, when present.
    pub fn i64_range(&self) -> Option<(i64, i64)> {
        match self {
            Self::Empty => None,
            Self::RangeI64 { min, max }
            | Self::SmallInListI64 { min, max, .. }
            | Self::HashSetContainsI64 { min, max, .. } => Some((*min, *max)),
        }
    }

    /// Exact non-null key count when this payload carries one.
    pub fn value_count(&self) -> Option<usize> {
        match self {
            Self::Empty => Some(0),
            Self::RangeI64 { .. } => None,
            Self::SmallInListI64 { values, .. } | Self::HashSetContainsI64 { values, .. } => Some(values.len()),
        }
    }
}

fn values_form_dense_i64_range(values: &[i64], min: i64, max: i64) -> bool {
    let Some(span) = i128::from(max).checked_sub(i128::from(min)).and_then(|span| span.checked_add(1)) else {
        return false;
    };
    span == values.len() as i128
}

/// Route payload status recorded by runtime dynamic-filter channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicFilterPayload {
    /// The route produced a filter payload.
    Produced(DynamicFilter),
    /// The producer completed but no supported payload could be derived.
    Skipped {
        /// Stable reason code explaining why no payload was produced.
        reason: &'static str,
    },
}

/// Runtime set of dynamic filters keyed by plan-local route id.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicFilterSet {
    payloads: BTreeMap<DynamicFilterId, DynamicFilterPayload>,
}

impl DynamicFilterSet {
    /// Creates an empty runtime payload set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a produced route payload.
    pub fn insert(&mut self, id: DynamicFilterId, filter: DynamicFilter) {
        self.payloads.insert(id, DynamicFilterPayload::Produced(filter));
    }

    /// Records a completed route with no usable payload.
    pub fn insert_skipped(&mut self, id: DynamicFilterId, reason: &'static str) {
        self.payloads.insert(id, DynamicFilterPayload::Skipped { reason });
    }

    /// Returns true after the producer has completed for `id`, even when it
    /// skipped payload construction.
    pub fn is_ready(&self, id: DynamicFilterId) -> bool {
        self.payloads.contains_key(&id)
    }

    /// Looks up a produced route payload.
    pub fn produced(&self, id: DynamicFilterId) -> Option<&DynamicFilter> {
        match self.payloads.get(&id) {
            Some(DynamicFilterPayload::Produced(filter)) => Some(filter),
            Some(DynamicFilterPayload::Skipped { .. }) | None => None,
        }
    }

    /// Looks up the skipped reason for a completed route with no payload.
    pub fn skipped_reason(&self, id: DynamicFilterId) -> Option<&'static str> {
        match self.payloads.get(&id) {
            Some(DynamicFilterPayload::Skipped { reason }) => Some(*reason),
            Some(DynamicFilterPayload::Produced(_)) | None => None,
        }
    }
}

/// Runtime and route-discovery thresholds for native dynamic filters.
///
/// These values are shared by the optimizer and compiled execution so a route
/// admitted during planning is evaluated against the same skip and payload
/// thresholds at runtime. Keep this as one policy object before exposing any
/// subset through user-facing configuration. Fields are crate-visible only;
/// embedders should use the read-only accessors and `with_*` methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DynamicFilterExecutionOptions {
    /// Maximum exact keyset cardinality lowered into a reader `IN (...)` predicate.
    #[doc(hidden)]
    pub small_in_list_limit: usize,
    /// Runtime cap for producer rows when an exact host keyset would be materialized.
    #[doc(hidden)]
    pub max_build_rows: usize,
    /// Producer row threshold where local-Parquet consumers switch from exact
    /// host keysets to range-only payloads.
    #[doc(hidden)]
    pub large_local_parquet_range_producer_rows: usize,
    /// Local-Parquet consumer byte threshold below which large producer keysets
    /// are treated as unlikely to pay for their producer-before-consumer dependency.
    #[doc(hidden)]
    pub dimension_sized_consumer_bytes: u64,
    /// Selected row-group count below which a dimension-sized local-Parquet
    /// consumer has too little independent source work to justify a large keyset.
    #[doc(hidden)]
    pub tiny_local_parquet_consumer_row_groups: usize,
    /// Maximum range overlap ratio, in parts per million, for merging a runtime
    /// range payload into a reader predicate.
    #[doc(hidden)]
    pub max_range_reader_predicate_selectivity_ppm: u64,
    /// Maximum small-IN-list cardinality that may be deferred from reader
    /// predicate application into a post-decode filter when another predicate remains.
    #[doc(hidden)]
    pub max_deferred_small_in_list_reader_values: usize,
    /// Maximum unpruned hash-set payload cardinality accepted for source
    /// post-decode filtering.
    #[doc(hidden)]
    pub max_unpruned_hash_set_post_decode_values: usize,
    /// Minimum known source rows required before an unpruned hash-set payload is
    /// worth applying as a source post-decode filter.
    #[doc(hidden)]
    pub min_unpruned_hash_set_post_decode_source_rows: u64,
}

impl DynamicFilterExecutionOptions {
    const MAX_SELECTIVITY_PPM: u64 = 1_000_000;

    pub const DEFAULT_SMALL_IN_LIST_LIMIT: usize = 1024;
    pub const DEFAULT_MAX_BUILD_ROWS: usize = 1_000_000;
    pub const DEFAULT_LARGE_LOCAL_PARQUET_RANGE_PRODUCER_ROWS: usize = 4 * 1024;
    pub const DEFAULT_DIMENSION_SIZED_CONSUMER_BYTES: u64 = 256 * 1024 * 1024;
    pub const DEFAULT_TINY_LOCAL_PARQUET_CONSUMER_ROW_GROUPS: usize = 32;
    pub const DEFAULT_MAX_RANGE_READER_PREDICATE_SELECTIVITY_PPM: u64 = 950_000;
    pub const DEFAULT_MAX_DEFERRED_SMALL_IN_LIST_READER_VALUES: usize = 128;
    pub const DEFAULT_MAX_UNPRUNED_HASH_SET_POST_DECODE_VALUES: usize = 4096;
    pub const DEFAULT_MIN_UNPRUNED_HASH_SET_POST_DECODE_SOURCE_ROWS: u64 = 1_000_000;

    /// Maximum exact keyset cardinality lowered into a reader `IN (...)` predicate.
    pub fn small_in_list_limit(&self) -> usize {
        self.small_in_list_limit
    }

    /// Runtime cap for producer rows when an exact host keyset would be materialized.
    pub fn max_build_rows(&self) -> usize {
        self.max_build_rows
    }

    /// Producer row threshold where local-Parquet consumers switch from exact
    /// host keysets to range-only payloads.
    pub fn large_local_parquet_range_producer_rows(&self) -> usize {
        self.large_local_parquet_range_producer_rows
    }

    /// Local-Parquet consumer byte threshold below which large producer keysets
    /// are treated as unlikely to pay for themselves.
    pub fn dimension_sized_consumer_bytes(&self) -> u64 {
        self.dimension_sized_consumer_bytes
    }

    /// Selected row-group count below which a dimension-sized local-Parquet
    /// consumer has too little independent source work to justify a large keyset.
    pub fn tiny_local_parquet_consumer_row_groups(&self) -> usize {
        self.tiny_local_parquet_consumer_row_groups
    }

    /// Maximum range overlap ratio, in parts per million, for merging a runtime
    /// range payload into a reader predicate.
    pub fn max_range_reader_predicate_selectivity_ppm(&self) -> u64 {
        self.max_range_reader_predicate_selectivity_ppm
    }

    /// Maximum small-IN-list cardinality that may be deferred from reader
    /// predicate application into a post-decode filter when another predicate remains.
    pub fn max_deferred_small_in_list_reader_values(&self) -> usize {
        self.max_deferred_small_in_list_reader_values
    }

    /// Maximum unpruned hash-set payload cardinality accepted for source
    /// post-decode filtering.
    pub fn max_unpruned_hash_set_post_decode_values(&self) -> usize {
        self.max_unpruned_hash_set_post_decode_values
    }

    /// Minimum known source rows required before an unpruned hash-set payload is
    /// worth applying as a source post-decode filter.
    pub fn min_unpruned_hash_set_post_decode_source_rows(&self) -> u64 {
        self.min_unpruned_hash_set_post_decode_source_rows
    }

    /// Validates internal consistency for this dynamic-filter policy.
    ///
    /// Zero-valued thresholds remain accepted because this is still an expert
    /// Rust surface and callers may use zero to disable narrow sub-behaviors.
    /// The checks here only reject combinations that make planner and runtime
    /// contracts contradictory.
    pub fn validate(&self) -> Result<()> {
        if self.large_local_parquet_range_producer_rows != 0
            && self.max_build_rows != 0
            && self.large_local_parquet_range_producer_rows > self.max_build_rows
        {
            return Err(invalid_dynamic_filter_execution_option(
                "large_local_parquet_range_producer_rows",
                self.large_local_parquet_range_producer_rows,
                "max_build_rows",
                self.max_build_rows,
                "range_producer_rows_exceeds_runtime_build_cap",
            ));
        }
        if self.max_range_reader_predicate_selectivity_ppm > Self::MAX_SELECTIVITY_PPM {
            return Err(invalid_dynamic_filter_execution_option_max(
                "max_range_reader_predicate_selectivity_ppm",
                self.max_range_reader_predicate_selectivity_ppm,
                Self::MAX_SELECTIVITY_PPM,
                "selectivity_ppm_above_one_hundred_percent",
            ));
        }
        Ok(())
    }

    pub fn with_small_in_list_limit(mut self, limit: usize) -> Self {
        self.small_in_list_limit = limit;
        self
    }

    pub fn with_max_build_rows(mut self, rows: usize) -> Self {
        self.max_build_rows = rows;
        self
    }

    pub fn with_large_local_parquet_range_producer_rows(mut self, rows: usize) -> Self {
        self.large_local_parquet_range_producer_rows = rows;
        self
    }

    pub fn with_dimension_sized_consumer_bytes(mut self, bytes: u64) -> Self {
        self.dimension_sized_consumer_bytes = bytes;
        self
    }

    pub fn with_tiny_local_parquet_consumer_row_groups(mut self, row_groups: usize) -> Self {
        self.tiny_local_parquet_consumer_row_groups = row_groups;
        self
    }

    pub fn with_max_range_reader_predicate_selectivity_ppm(mut self, selectivity_ppm: u64) -> Self {
        self.max_range_reader_predicate_selectivity_ppm = selectivity_ppm;
        self
    }

    pub fn with_max_deferred_small_in_list_reader_values(mut self, values: usize) -> Self {
        self.max_deferred_small_in_list_reader_values = values;
        self
    }

    pub fn with_max_unpruned_hash_set_post_decode_values(mut self, values: usize) -> Self {
        self.max_unpruned_hash_set_post_decode_values = values;
        self
    }

    pub fn with_min_unpruned_hash_set_post_decode_source_rows(mut self, rows: u64) -> Self {
        self.min_unpruned_hash_set_post_decode_source_rows = rows;
        self
    }
}

fn invalid_dynamic_filter_execution_option(
    field: &'static str,
    value: usize,
    bound_field: &'static str,
    bound_value: usize,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.dynamic_filter.execution_options.validate",
        "dynamic-filter execution option is inconsistent with another policy field",
    )
    .with_fact("field", field)
    .with_fact("value", value)
    .with_fact("bound_field", bound_field)
    .with_fact("bound_value", bound_value)
    .with_fact("reason", reason)
}

fn invalid_dynamic_filter_execution_option_max(
    field: &'static str,
    value: u64,
    max_value: u64,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.dynamic_filter.execution_options.validate",
        "dynamic-filter execution option exceeds its valid maximum",
    )
    .with_fact("field", field)
    .with_fact("value", value)
    .with_fact("max_value", max_value)
    .with_fact("reason", reason)
}

impl Default for DynamicFilterExecutionOptions {
    fn default() -> Self {
        Self {
            small_in_list_limit: Self::DEFAULT_SMALL_IN_LIST_LIMIT,
            max_build_rows: Self::DEFAULT_MAX_BUILD_ROWS,
            large_local_parquet_range_producer_rows: Self::DEFAULT_LARGE_LOCAL_PARQUET_RANGE_PRODUCER_ROWS,
            dimension_sized_consumer_bytes: Self::DEFAULT_DIMENSION_SIZED_CONSUMER_BYTES,
            tiny_local_parquet_consumer_row_groups: Self::DEFAULT_TINY_LOCAL_PARQUET_CONSUMER_ROW_GROUPS,
            max_range_reader_predicate_selectivity_ppm: Self::DEFAULT_MAX_RANGE_READER_PREDICATE_SELECTIVITY_PPM,
            max_deferred_small_in_list_reader_values: Self::DEFAULT_MAX_DEFERRED_SMALL_IN_LIST_READER_VALUES,
            max_unpruned_hash_set_post_decode_values: Self::DEFAULT_MAX_UNPRUNED_HASH_SET_POST_DECODE_VALUES,
            min_unpruned_hash_set_post_decode_source_rows: Self::DEFAULT_MIN_UNPRUNED_HASH_SET_POST_DECODE_SOURCE_ROWS,
        }
    }
}

/// Maximum exact keyset cardinality lowered into a reader `IN (...)` predicate.
pub const DEFAULT_SMALL_IN_LIST_DYNAMIC_FILTER_LIMIT: usize =
    DynamicFilterExecutionOptions::DEFAULT_SMALL_IN_LIST_LIMIT;

/// Stable runtime skip reason for a large exact keyset whose local-Parquet
/// consumers are already small enough that payload materialization and footer
/// pruning work are not expected to pay for themselves.
pub const LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON: &str = "large_keyset_dimension_sized_local_parquet_consumer";

#[doc(hidden)]
pub fn dynamic_filter_consumer_source_feeds_downstream_work_fanout(plan: &QueryPlan, source_node_id: NodeId) -> bool {
    let parent_counts = parent_counts(plan);
    let mut current = source_node_id;
    let mut saw_work_boundary = false;
    loop {
        let parent_count = parent_counts[current.index()];
        if parent_count > 1 {
            return saw_work_boundary;
        }
        if parent_count == 0 {
            return false;
        }
        let Some(parent) = dynamic_filter_single_parent(plan, &parent_counts, current) else {
            return false;
        };
        let Some(parent_node) = plan.node(parent) else {
            return false;
        };
        if dynamic_filter_downstream_work_boundary(&parent_node.kind) {
            saw_work_boundary = true;
        }
        current = parent;
    }
}

fn dynamic_filter_single_parent(plan: &QueryPlan, parent_counts: &[usize], node_id: NodeId) -> Option<NodeId> {
    if parent_counts[node_id.index()] != 1 {
        return None;
    }
    plan.nodes().iter().find(|candidate| candidate.input_ids().contains(&node_id)).map(|candidate| candidate.id)
}

fn dynamic_filter_downstream_work_boundary(node: &PlanNode) -> bool {
    matches!(
        node,
        PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Join(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_)
    )
}

/// Extracts integer-like values from a host Arrow column for initial
/// dynamic-filter payload construction.
pub fn dynamic_filter_i64_values(array: &dyn Array) -> Option<Vec<Option<i64>>> {
    match array.data_type() {
        DataType::Int32 => {
            let array = array.as_any().downcast_ref::<Int32Array>()?;
            Some((0..array.len()).map(|row| (!array.is_null(row)).then(|| i64::from(array.value(row)))).collect())
        },
        DataType::Date32 => {
            let array = array.as_any().downcast_ref::<Date32Array>()?;
            Some((0..array.len()).map(|row| (!array.is_null(row)).then(|| i64::from(array.value(row)))).collect())
        },
        DataType::Int64 => {
            let array = array.as_any().downcast_ref::<Int64Array>()?;
            Some((0..array.len()).map(|row| (!array.is_null(row)).then(|| array.value(row))).collect())
        },
        DataType::UInt32 => {
            let array = array.as_any().downcast_ref::<UInt32Array>()?;
            Some((0..array.len()).map(|row| (!array.is_null(row)).then(|| i64::from(array.value(row)))).collect())
        },
        DataType::UInt64 => {
            let array = array.as_any().downcast_ref::<UInt64Array>()?;
            Some(
                (0..array.len())
                    .map(|row| (!array.is_null(row)).then(|| i64::try_from(array.value(row)).ok()).flatten())
                    .collect(),
            )
        },
        _ => None,
    }
}

/// Applies produced dynamic filters for one local Parquet source.
///
/// The returned source is a rewritten execution contract: row-group selections
/// are narrowed when footer statistics prove a non-overlap, and lowerable
/// payloads are merged into the reader predicate.
pub fn apply_dynamic_filters_to_local_parquet_source(
    source: &LocalParquetSource,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
) -> Result<LocalParquetSource> {
    apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter(source, routes, filter_set, |_, _| true)
}

#[doc(hidden)]
pub fn apply_dynamic_filters_to_local_parquet_source_with_reader_predicate_filter(
    source: &LocalParquetSource,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    mut should_merge_reader_predicate: impl FnMut(&DynamicFilterRoute, &DynamicFilter) -> bool,
) -> Result<LocalParquetSource> {
    let mut source = source.clone();
    for route in routes {
        let Some(filter) = filter_set.produced(route.id) else {
            continue;
        };
        let Some(column) = route.consumer_columns.first() else {
            continue;
        };
        let merge_reader_predicate = should_merge_reader_predicate(route, filter);
        apply_dynamic_filter_to_source(&mut source, route, column, filter, merge_reader_predicate)?;
    }
    Ok(source)
}

fn apply_dynamic_filter_to_source(
    source: &mut LocalParquetSource,
    route: &DynamicFilterRoute,
    column: &str,
    filter: &DynamicFilter,
    merge_reader_predicate: bool,
) -> Result<()> {
    if matches!(filter, DynamicFilter::Empty) {
        mark_source_row_groups_empty(source);
        source.scan_limit = Some(0);
        source.refresh_diagnostics();
        return Ok(());
    }
    if let Some(key_filter) = dynamic_filter_row_group_key_filter(filter) {
        prune_source_row_groups_by_key_filter(source, column, &key_filter)?;
    }
    if merge_reader_predicate
        && matches!(
            route.consumer_lowering,
            DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate
        )
        && let Some(predicate) = dynamic_filter_reader_predicate(source, column, filter)
        && source_reader_predicate_merge_is_supported(source, &predicate)
    {
        merge_source_reader_predicate(source, predicate);
    }
    Ok(())
}

/// Builds the set-aware row-group pruning filter carried by a dynamic-filter
/// payload. Range payloads reuse their `[min,max]` bounds as one run; small/large
/// keyset payloads coalesce their materialized
/// `values` into runs so a scattered keyset prunes the gaps a single range would
/// retain. Returns `None` when the payload carries nothing prunable.
fn dynamic_filter_row_group_key_filter(filter: &DynamicFilter) -> Option<DimensionKeyFilter> {
    match filter {
        DynamicFilter::Empty => None,
        DynamicFilter::RangeI64 { min, max } => {
            Some(DimensionKeyFilter::from_bounds(KeyRange { min: *min, max: *max }))
        },
        DynamicFilter::SmallInListI64 { values, .. } | DynamicFilter::HashSetContainsI64 { values, .. } => {
            DimensionKeyFilter::from_keys(values)
        },
    }
}

fn prune_source_row_groups_by_key_filter(
    source: &mut LocalParquetSource,
    column: &str,
    key_filter: &DimensionKeyFilter,
) -> Result<()> {
    let SourceColumnRef::File(column_index) = source.source_column_ref(column).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.dynamic_filter.source_pruning",
            "dynamic-filter source column is not represented by the local parquet source",
        )
        .with_fact("column", column)
    })?
    else {
        return Ok(());
    };

    let mut saw_statistics = false;
    let mut selected_total = 0usize;
    let mut retained_total = 0usize;
    let mut selections = Vec::with_capacity(source.files.len());

    for file in &source.files {
        let Some(path) = file.location.as_local_path() else {
            selections.push(file.row_groups.clone());
            continue;
        };
        let reader = SerializedFileReader::try_from(path).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.dynamic_filter.source_pruning",
                "failed to read parquet footer for dynamic-filter row-group pruning",
            )
            .with_source(error)
            .with_fact("path", path.display())
            .with_fact("column", column)
        })?;
        let metadata = reader.metadata();
        let selected = selected_or_all_row_groups(file.row_groups.as_ref(), metadata.num_row_groups());
        selected_total = selected_total.saturating_add(selected.len());
        let mut retained = Vec::new();
        for row_group_index in selected {
            let row_group = metadata.row_group(row_group_index);
            match row_group_i64_range_for_root_column(row_group, metadata.file_metadata().schema_descr(), column_index)
            {
                Some(row_group_range) => {
                    saw_statistics = true;
                    if key_filter.overlaps(KeyRange { min: row_group_range.min, max: row_group_range.max }) {
                        retained.push(row_group_index);
                    }
                },
                None => retained.push(row_group_index),
            }
        }
        let retained = if file.has_position_delete_files() { contiguous_row_group_cover(retained) } else { retained };
        retained_total = retained_total.saturating_add(retained.len());
        selections.push(Some(retained));
    }

    if !saw_statistics || retained_total >= selected_total {
        return Ok(());
    }
    if retained_total == 0 {
        for file in &mut source.files {
            file.row_groups = Some(Vec::new());
        }
        source.scan_limit = Some(0);
    } else {
        for (file, selection) in source.files.iter_mut().zip(selections) {
            if let Some(selection) = selection {
                file.row_groups = Some(selection);
            }
        }
    }
    source.refresh_diagnostics();
    Ok(())
}

fn mark_source_row_groups_empty(source: &mut LocalParquetSource) {
    for file in &mut source.files {
        file.row_groups = Some(Vec::new());
    }
}

fn selected_or_all_row_groups(selected: Option<&Vec<usize>>, row_group_count: usize) -> Vec<usize> {
    selected.cloned().unwrap_or_else(|| (0..row_group_count).collect())
}

fn contiguous_row_group_cover(row_groups: Vec<usize>) -> Vec<usize> {
    let Some(first) = row_groups.iter().copied().min() else {
        return row_groups;
    };
    let last = row_groups.iter().copied().max().expect("min proved non-empty row group selection");
    (first..=last).collect()
}

fn dynamic_filter_reader_predicate(source: &LocalParquetSource, column: &str, filter: &DynamicFilter) -> Option<Expr> {
    if matches!(filter, DynamicFilter::HashSetContainsI64 { .. }) {
        return None;
    }
    let predicate = dynamic_filter_predicate_for_schema(source.schema.as_ref(), column, filter)?;
    source.reader_predicate_is_supported(&predicate).then_some(predicate)
}

#[doc(hidden)]
pub fn dynamic_filter_reader_predicate_should_merge(
    source: &LocalParquetSource,
    column: &str,
    filter: &DynamicFilter,
    options: DynamicFilterExecutionOptions,
) -> Result<bool> {
    let Some(predicate) = dynamic_filter_reader_predicate(source, column, filter) else {
        return Ok(false);
    };
    if !source_reader_predicate_merge_is_supported(source, &predicate) {
        return Ok(false);
    }
    let DynamicFilter::RangeI64 { min, max } = filter else {
        return Ok(true);
    };
    let Some(source_range) = selected_source_i64_range(source, column)? else {
        return Ok(true);
    };
    let Some(overlap) = range_overlap_width(source_range, I64Range { min: *min, max: *max }) else {
        return Ok(true);
    };
    let Some(source_width) = range_width(source_range) else {
        return Ok(true);
    };
    Ok(overlap.saturating_mul(1_000_000)
        <= source_width.saturating_mul(u128::from(options.max_range_reader_predicate_selectivity_ppm)))
}

fn selected_source_i64_range(source: &LocalParquetSource, column: &str) -> Result<Option<I64Range>> {
    let SourceColumnRef::File(column_index) = source.source_column_ref(column).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.dynamic_filter.reader_predicate_selectivity",
            "dynamic-filter source column is not represented by the local parquet source",
        )
        .with_fact("column", column)
    })?
    else {
        return Ok(None);
    };

    let mut selected_range = None;
    for file in &source.files {
        let Some(path) = file.location.as_local_path() else {
            return Ok(None);
        };
        let reader = SerializedFileReader::try_from(path).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.dynamic_filter.reader_predicate_selectivity",
                "failed to read parquet footer for dynamic-filter reader-predicate selectivity",
            )
            .with_source(error)
            .with_fact("path", path.display())
            .with_fact("column", column)
        })?;
        let metadata = reader.metadata();
        for row_group_index in selected_or_all_row_groups(file.row_groups.as_ref(), metadata.num_row_groups()) {
            let row_group = metadata.row_group(row_group_index);
            let Some(row_group_range) =
                row_group_i64_range_for_root_column(row_group, metadata.file_metadata().schema_descr(), column_index)
            else {
                return Ok(None);
            };
            selected_range = Some(match selected_range {
                Some(existing) => merge_ranges(existing, row_group_range),
                None => row_group_range,
            });
        }
    }
    Ok(selected_range)
}

fn merge_ranges(left: I64Range, right: I64Range) -> I64Range {
    I64Range { min: left.min.min(right.min), max: left.max.max(right.max) }
}

fn range_width(range: I64Range) -> Option<u128> {
    i128::from(range.max).checked_sub(i128::from(range.min))?.checked_add(1)?.try_into().ok()
}

fn range_overlap_width(left: I64Range, right: I64Range) -> Option<u128> {
    let min = left.min.max(right.min);
    let max = left.max.min(right.max);
    (min <= max).then(|| range_width(I64Range { min, max })).flatten()
}

/// Builds a post-decode predicate for a dynamic-filter payload against an
/// already-materialized input schema.
///
/// This is intentionally schema-based rather than source-based so non-Parquet
/// consumers, such as a hash-join probe input, can use the same payload
/// semantics after decode.
pub fn dynamic_filter_predicate_for_schema(
    schema: &arrow::datatypes::Schema,
    column: &str,
    filter: &DynamicFilter,
) -> Option<Expr> {
    let data_type = schema.field_with_name(column).ok()?.data_type();
    let column_expr = Expr::column(column);
    match filter {
        DynamicFilter::Empty => {
            Some(Expr::binary(Expr::literal(LiteralValue::Int8(1)), BinaryOp::Eq, Expr::literal(LiteralValue::Int8(0))))
        },
        DynamicFilter::RangeI64 { min, max } => {
            let lower = Expr::binary(
                column_expr.clone(),
                BinaryOp::GtEq,
                Expr::literal(i64_literal_for_type(*min, data_type)?),
            );
            let upper =
                Expr::binary(column_expr, BinaryOp::LtEq, Expr::literal(i64_literal_for_type(*max, data_type)?));
            Some(Expr::binary(lower, BinaryOp::And, upper))
        },
        DynamicFilter::SmallInListI64 { values, .. } | DynamicFilter::HashSetContainsI64 { values, .. } => {
            let values = values
                .iter()
                .map(|value| i64_literal_for_type(*value, data_type).map(Expr::literal))
                .collect::<Option<Vec<_>>>()?;
            Some(Expr::in_list(column_expr, values))
        },
    }
}

fn merge_source_reader_predicate(source: &mut LocalParquetSource, predicate: Expr) {
    let predicate = match source.reader_predicate.take() {
        Some(existing) => Expr::binary(existing, BinaryOp::And, predicate),
        None => predicate,
    };
    source.reader_predicate = Some(predicate);
    source.refresh_diagnostics();
}

fn source_reader_predicate_merge_is_supported(source: &LocalParquetSource, predicate: &Expr) -> bool {
    if source.has_row_delete_files() {
        return false;
    }
    let predicate = match &source.reader_predicate {
        Some(existing) => Expr::binary(existing.clone(), BinaryOp::And, predicate.clone()),
        None => predicate.clone(),
    };
    source.reader_predicate_is_supported(&predicate)
}

fn i64_literal_for_type(value: i64, data_type: &DataType) -> Option<LiteralValue> {
    match data_type {
        DataType::Int32 => i32::try_from(value).ok().map(LiteralValue::Int32),
        DataType::Date32 => i32::try_from(value).ok().map(LiteralValue::Date32),
        DataType::Int64 => Some(LiteralValue::Int64(value)),
        DataType::UInt32 => u32::try_from(value).ok().map(LiteralValue::UInt32),
        DataType::UInt64 => u64::try_from(value).ok().map(LiteralValue::UInt64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_filter_execution_options_accept_zero_threshold_overrides() {
        DynamicFilterExecutionOptions::default().with_max_build_rows(0).validate().unwrap();
        DynamicFilterExecutionOptions::default().with_large_local_parquet_range_producer_rows(0).validate().unwrap();
    }

    #[test]
    fn dynamic_filter_execution_options_accept_independent_post_decode_limits() {
        DynamicFilterExecutionOptions::default()
            .with_max_build_rows(10)
            .with_large_local_parquet_range_producer_rows(10)
            .with_small_in_list_limit(1)
            .with_max_deferred_small_in_list_reader_values(2)
            .with_max_unpruned_hash_set_post_decode_values(11)
            .validate()
            .unwrap();
    }
}
