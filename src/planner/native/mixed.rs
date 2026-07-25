//! Narrow, property-checked relational mixed-execution boundary.
//!
//! This is deliberately not a fragment extractor. It recognizes one completed
//! root shape only: an optional transparent `CooperativeExec` around a CPU
//! `FilterExec` whose direct local-Parquet child can execute as one native
//! host-output leaf. The CPU filter is the final operator, so the native
//! leaf's `Incremental -> Final` emission strengthening cannot affect a
//! downstream DataFusion operator.

use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion::{
    datasource::{
        physical_plan::{FileScanConfigBuilder, FileSource, ParquetSource},
        source::DataSourceExec,
    },
    physical_expr::{
        EquivalenceProperties,
        expressions::{Column, LikeExpr, Literal},
    },
    physical_plan::{
        ExecutionPlan, Partitioning,
        coop::CooperativeExec,
        execution_plan::{Boundedness, EmissionType},
        filter::{FilterExec, FilterExecBuilder},
    },
};
use nexus_query_engine::{
    plan::{PlanNode, QueryPlan},
    source::SourceNode,
};

use crate::native_exec::NexusNativeExec;

/// A root-only CPU filter whose direct input was certified for replacement by
/// one native host-output leaf.
pub(super) struct CertifiedMixedFilterBoundary {
    root: Arc<dyn ExecutionPlan>,
    source: Arc<dyn ExecutionPlan>,
    source_schema: SchemaRef,
    has_cooperative_wrapper: bool,
}

impl CertifiedMixedFilterBoundary {
    /// Finds the one P2a boundary shape and proves its input-side contracts.
    pub(super) fn certify(plan: Arc<dyn ExecutionPlan>) -> Option<Self> {
        let (filter_plan, has_cooperative_wrapper) = if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
            (Arc::clone(cooperative.input()), true)
        } else {
            (Arc::clone(&plan), false)
        };
        let filter = filter_plan.downcast_ref::<FilterExec>()?;
        let source = Arc::clone(filter.input());
        let native_lowering_source = native_lowering_source(filter, source.as_ref())?;

        root_filter_is_eligible(filter, source.as_ref()).then(|| Self {
            root: plan,
            source_schema: source.schema(),
            source: native_lowering_source,
            has_cooperative_wrapper,
        })
    }

    /// The direct local-Parquet input to lower as the single native fragment.
    pub(super) fn source(&self) -> &Arc<dyn ExecutionPlan> {
        &self.source
    }

    /// Native wrappers at this boundary intentionally expose no inherited
    /// ordering, equivalence, or constraint facts.
    pub(super) fn native_equivalence_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.source_schema))
    }

    /// Validates that lowering produced exactly the source leaf this boundary
    /// certified, rather than a richer native fragment.
    pub(super) fn accepts_native_leaf(&self, native_plan: &QueryPlan) -> bool {
        native_plan.output_schema().as_ref() == self.source_schema.as_ref()
            && native_plan.nodes().len() == 1
            && matches!(
                native_plan.root_node().kind,
                PlanNode::Source(ref source) if matches!(source.as_ref(), SourceNode::LocalParquet(_))
            )
    }

    /// Rebuilds the CPU filter so DataFusion recomputes properties from the
    /// native child rather than retaining the source's incremental cache.
    pub(super) fn rebuild(&self, native_child: Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        let filter_plan = if self.has_cooperative_wrapper {
            self.root.downcast_ref::<CooperativeExec>()?.input()
        } else {
            &self.root
        };
        let filter = filter_plan.downcast_ref::<FilterExec>()?;
        if !native_leaf_is_certified(native_child.as_ref(), self.source_schema.as_ref()) {
            return None;
        }

        let filter = FilterExecBuilder::from(filter).with_input(native_child).build().ok()?;
        let plan: Arc<dyn ExecutionPlan> = if self.has_cooperative_wrapper {
            Arc::new(CooperativeExec::new(Arc::new(filter)))
        } else {
            Arc::new(filter)
        };
        is_certified_mixed_execution_plan(plan.as_ref()).then_some(plan)
    }
}

/// Returns true only for a P2a plan this module can fully re-certify.
///
/// The native optimizer uses this as an idempotence guard. Without it, a
/// repeated optimizer invocation would try to lower the retained CPU filter
/// again and overwrite the selected leaf report with a whole-plan miss.
pub(super) fn is_certified_mixed_execution_plan(plan: &dyn ExecutionPlan) -> bool {
    let (filter_plan, has_cooperative_wrapper) = if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        (cooperative.input().as_ref(), true)
    } else {
        (plan, false)
    };
    let Some(filter) = filter_plan.downcast_ref::<FilterExec>() else {
        return false;
    };
    let source = filter.input();
    let Some(native) = source.downcast_ref::<NexusNativeExec>() else {
        return false;
    };

    root_filter_has_deterministic_ilike(filter)
        && filter.fetch().is_none()
        && filter.projection().is_none()
        && filter.schema().as_ref() == source.schema().as_ref()
        && filter_properties_are_certified(filter)
        && native_leaf_is_certified(source.as_ref(), filter.schema().as_ref())
        && native_plan_is_single_local_parquet_leaf(native.native_plan())
        && (!has_cooperative_wrapper || cooperative_properties_are_certified(plan, filter))
}

fn root_filter_is_eligible(filter: &FilterExec, source: &dyn ExecutionPlan) -> bool {
    let Some(source_exec) = source.downcast_ref::<DataSourceExec>() else {
        return false;
    };
    let Some((file_scan, parquet_source)) = source_exec.downcast_to_file_source::<ParquetSource>() else {
        return false;
    };

    root_filter_has_deterministic_ilike(filter)
        && filter.fetch().is_none()
        && filter.projection().is_none()
        && source.fetch().is_none()
        && source.schema().as_ref() == filter.schema().as_ref()
        && simple_boundary_schema(source.schema().as_ref())
        && source_properties_are_certified(source)
        // This P2a family replaces only a stable local filesystem scan. The
        // broader native source lowering also admits remote Parquet, but a
        // remote object store has independent replay, credential, and
        // side-effect contracts that this root-only certificate does not own.
        && file_scan.object_store_url.as_str().starts_with("file://")
        && file_scan.file_groups.len() == 1
        && file_scan.constraints.is_empty()
        && file_scan.limit.is_none()
        && !file_scan.preserve_order
        && file_scan.output_ordering.is_empty()
        && file_scan.table_partition_cols().is_empty()
        && !file_scan.partitioned_by_file_group
        && file_scan.expr_adapter_factory.is_none()
        && crate::native::native_parquet_file_reader_factory_is_supported(parquet_source)
        // DataFusion retains a source-level copy of filters it can use for
        // Parquet pruning. P2a may clear that hint only when it exactly
        // matches the retained CPU ILIKE filter, which remains the sole
        // predicate evaluator after the source is replaced.
        && source_predicate_is_absent_or_matches_filter(parquet_source, filter)
        // ParquetSource always carries a source-level projection, including
        // for an ordinary unprojected scan. Allow only the canonical identity
        // form; a projection with different columns, aliases, or expressions
        // remains outside this P2a boundary.
        && file_source_projection_is_identity(file_scan.file_source().as_ref())
}

fn source_predicate_is_absent_or_matches_filter(parquet_source: &ParquetSource, filter: &FilterExec) -> bool {
    parquet_source.filter().is_none()
        || crate::native::scan_predicate_exactly_matches_filter(parquet_source, Some(filter))
}

/// Rebuild a P2a source without a source-level copy of the CPU predicate.
///
/// The caller has already certified that the retained root filter evaluates
/// the exact same expression. Keeping a second predicate on the native source
/// would require native support for that expression and would incorrectly
/// prevent a CPU-boundary rewrite for SQL plans that use Parquet pruning.
fn native_lowering_source(filter: &FilterExec, source: &dyn ExecutionPlan) -> Option<Arc<dyn ExecutionPlan>> {
    let source_exec = source.downcast_ref::<DataSourceExec>()?;
    let (file_scan, parquet_source) = source_exec.downcast_to_file_source::<ParquetSource>()?;
    if parquet_source.filter().is_none() {
        return Some(Arc::clone(filter.input()));
    }
    if !crate::native::scan_predicate_exactly_matches_filter(parquet_source, Some(filter)) {
        return None;
    }

    let projection_indices = file_scan.file_source().projection().map(|projection| projection.column_indices());
    let parquet_source = ParquetSource::new(file_scan.file_source().table_schema().clone());
    let mut builder =
        FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source) as Arc<dyn FileSource>);
    if let Some(projection_indices) = projection_indices {
        builder = builder.with_projection_indices(Some(projection_indices)).ok()?;
    }
    Some(DataSourceExec::from_data_source(builder.build()))
}

fn root_filter_has_deterministic_ilike(filter: &FilterExec) -> bool {
    let Some(like) = filter.predicate().downcast_ref::<LikeExpr>() else {
        return false;
    };
    let Some(column) = like.expr().downcast_ref::<Column>() else {
        return false;
    };
    let Some(pattern) = like.pattern().downcast_ref::<Literal>() else {
        return false;
    };

    !like.negated()
        && like.case_insensitive()
        && matches!(
            pattern.value(),
            datafusion::common::ScalarValue::Utf8(Some(_))
                | datafusion::common::ScalarValue::LargeUtf8(Some(_))
                | datafusion::common::ScalarValue::Utf8View(Some(_))
        )
        && filter
            .input()
            .schema()
            .fields()
            .get(column.index())
            .is_some_and(|field| field.name() == column.name() && field.data_type().is_string())
}

fn source_properties_are_certified(source: &dyn ExecutionPlan) -> bool {
    single_unknown_partition(source)
        && source.properties().output_ordering().is_none()
        && source.properties().boundedness == Boundedness::Bounded
        // DataSourceExec is incrementally emitting. The rebuilt root filter is
        // the only CPU consumer, and its native child is explicitly Final.
        && source.properties().emission_type == EmissionType::Incremental
        && canonical_empty_equivalence_properties(source.schema(), source.properties().equivalence_properties())
}

fn native_leaf_is_certified(native_child: &dyn ExecutionPlan, schema: &arrow::datatypes::Schema) -> bool {
    native_child.schema().as_ref() == schema
        && single_unknown_partition(native_child)
        && native_child.properties().output_ordering().is_none()
        && native_child.properties().boundedness == Boundedness::Bounded
        && native_child.properties().emission_type == EmissionType::Final
        && simple_boundary_schema(schema)
        && canonical_empty_equivalence_properties(
            native_child.schema(),
            native_child.properties().equivalence_properties(),
        )
}

fn filter_properties_are_certified(filter: &FilterExec) -> bool {
    single_unknown_partition(filter)
        && filter.properties().output_ordering().is_none()
        && filter.properties().boundedness == Boundedness::Bounded
        && filter.properties().emission_type == EmissionType::Final
        && canonical_empty_equivalence_properties(filter.schema(), filter.properties().equivalence_properties())
}

fn cooperative_properties_are_certified(plan: &dyn ExecutionPlan, filter: &FilterExec) -> bool {
    plan.schema().as_ref() == filter.schema().as_ref()
        && single_unknown_partition(plan)
        && plan.properties().output_ordering().is_none()
        && plan.properties().boundedness == Boundedness::Bounded
        && plan.properties().emission_type == EmissionType::Final
        && canonical_empty_equivalence_properties(plan.schema(), plan.properties().equivalence_properties())
}

fn canonical_empty_equivalence_properties(schema: SchemaRef, properties: &EquivalenceProperties) -> bool {
    properties.schema().as_ref() == schema.as_ref()
        && properties.eq_group().is_empty()
        && properties.oeq_class().is_empty()
        && properties.constants().is_empty()
        && properties.constraints().is_empty()
}

fn simple_boundary_schema(schema: &arrow::datatypes::Schema) -> bool {
    schema.fields().iter().all(|field| {
        matches!(
            field.data_type(),
            DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
        )
    })
}

fn single_unknown_partition(plan: &dyn ExecutionPlan) -> bool {
    matches!(plan.properties().output_partitioning(), Partitioning::UnknownPartitioning(1))
}

fn file_source_projection_is_identity(file_source: &dyn FileSource) -> bool {
    let Some(projection) = file_source.projection() else {
        return false;
    };
    let table_schema = file_source.table_schema().table_schema();
    projection.as_ref().len() == table_schema.fields().len()
        && projection.iter().enumerate().all(|(index, projection)| {
            let field = table_schema.field(index);
            projection.alias == field.name().as_str()
                && projection
                    .expr
                    .downcast_ref::<Column>()
                    .is_some_and(|column| column.index() == index && column.name() == field.name())
        })
}

fn native_plan_is_single_local_parquet_leaf(native_plan: &QueryPlan) -> bool {
    native_plan.nodes().len() == 1
        && matches!(
            native_plan.root_node().kind,
            PlanNode::Source(ref source) if matches!(source.as_ref(), SourceNode::LocalParquet(_))
        )
}
