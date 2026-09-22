use std::sync::Arc;

use arrow::datatypes::{DataType, Schema, SchemaRef};
#[cfg(feature = "cugraph")]
use arrow::{
    array::{ArrayRef, Int32Array, Int64Array},
    datatypes::Field,
    record_batch::RecordBatch,
};
#[cfg(feature = "cugraph")]
use datafusion::catalog::Session;
#[cfg(feature = "cugraph")]
use datafusion::datasource::{MemTable, TableProvider};
#[cfg(not(feature = "cugraph"))]
use datafusion::{
    common::tree_node::TreeNodeRecursion,
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, PlanProperties, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use datafusion::{
    common::{DataFusionError, Result},
    physical_plan::ExecutionPlan,
};

#[cfg(not(feature = "cugraph"))]
use super::CugraphAlgorithm;
use super::{
    CUGRAPH_EDGE_DESTINATION_COLUMN, CUGRAPH_EDGE_ID_COLUMN, CUGRAPH_EDGE_SOURCE_COLUMN, CugraphAlgorithmParams,
    CugraphAlgorithmSpec, CugraphVertexIdDomain, PredicateSourceColumnRole, PredicateSourceSpec, plan_error,
    validate_similarity_pair_table_schema, validate_vertex_id_domain_support_for_call, vertex_id_domain_for_call,
};
#[cfg(feature = "cugraph")]
use super::{CugraphBfsSource, CugraphBfsSourceVertex};
use crate::error::{Error, ErrorCode};

#[cfg(not(feature = "cugraph"))]
#[derive(Debug)]
pub(super) struct CugraphFeatureDisabledExec {
    algorithm: CugraphAlgorithm,
    cache: Arc<PlanProperties>,
}

/// Whether this plan can only terminate at the zero-child executor for an
/// unavailable cuGraph feature. DataFusion may add output wrappers above that
/// executor, but every path still ends at the same typed rejection before any
/// CPU or GPU data execution can begin.
#[cfg(not(feature = "cugraph"))]
pub(crate) fn is_feature_disabled_execution_plan(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<CugraphFeatureDisabledExec>().is_some() || {
        let children = plan.children();
        !children.is_empty() && children.into_iter().all(|child| is_feature_disabled_execution_plan(child.as_ref()))
    }
}

#[cfg(feature = "cugraph")]
pub(crate) fn is_feature_disabled_execution_plan(_plan: &dyn ExecutionPlan) -> bool {
    false
}

#[cfg(not(feature = "cugraph"))]
impl CugraphFeatureDisabledExec {
    pub(super) fn new(
        algorithm: CugraphAlgorithm,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
        _limit: Option<usize>,
    ) -> Result<Self> {
        let schema = match projection {
            Some(projection) => Arc::new(Schema::new(
                projection
                    .iter()
                    .map(|index| {
                        schema.fields().get(*index).cloned().ok_or_else(|| {
                            plan_error(format!("cuGraph output projection index {index} is out of range"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )),
            None => schema,
        };
        Ok(Self {
            algorithm,
            cache: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        })
    }
}

#[cfg(not(feature = "cugraph"))]
impl DisplayAs for CugraphFeatureDisabledExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CugraphFeatureDisabledExec: algorithm={}, reason=required_feature_disabled, required_feature=cugraph",
            self.algorithm.function_name()
        )
    }
}

#[cfg(not(feature = "cugraph"))]
impl ExecutionPlan for CugraphFeatureDisabledExec {
    fn name(&self) -> &str {
        "CugraphFeatureDisabledExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(plan_error("feature-disabled cuGraph SQL execution must not have input children"))
        }
    }

    fn execute(&self, _partition: usize, _task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        // Flight SQL has already issued a ticket when physical execution
        // starts. Keep the feature-gate error on that ticket's result stream
        // rather than rejecting stream construction, while still performing
        // no source scan, staging, or GPU work.
        let schema = self.schema();
        let error = super::cugraph_feature_disabled(self.algorithm);
        let stream = futures::stream::once(async move { Err(error) });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(feature = "cugraph")]
pub(super) async fn literal_bfs_source_plan(
    state: &dyn Session,
    spec: &CugraphAlgorithmSpec,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    let CugraphBfsSource::LiteralList(values) = &params.source else {
        return Ok(None);
    };
    if !values.first().is_some_and(|value| matches!(value, CugraphBfsSourceVertex::Int64(_))) {
        return Ok(None);
    }
    let values = values
        .iter()
        .map(|value| match value {
            CugraphBfsSourceVertex::Int64(value) => Ok(*value),
            CugraphBfsSourceVertex::Utf8(_) => Err(plan_error("numeric BFS source list contained a string value")),
        })
        .collect::<Result<Vec<_>>>()?;
    let (data_type, array): (DataType, ArrayRef) = match vertex_domain {
        CugraphVertexIdDomain::Int32 => {
            let values = values
                .into_iter()
                .map(|value| {
                    i32::try_from(value)
                        .map_err(|_| plan_error(format!("BFS source vertex {value} does not fit Int32 edge domain")))
                })
                .collect::<Result<Vec<_>>>()?;
            (DataType::Int32, Arc::new(Int32Array::from(values)) as ArrayRef)
        },
        CugraphVertexIdDomain::Int64 => (DataType::Int64, Arc::new(Int64Array::from(values)) as ArrayRef),
        CugraphVertexIdDomain::StringLogical => return Ok(None),
    };
    let schema = Arc::new(Schema::new(vec![Field::new("__source_vertex", data_type, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array])?;
    let table = MemTable::try_new(schema, vec![vec![batch]])?;
    table.scan(state, None, &[], None).await.map(Some)
}

pub(super) fn validate_edge_source_schema(
    spec: &CugraphAlgorithmSpec,
    schema: &SchemaRef,
) -> Result<CugraphVertexIdDomain> {
    let src = schema.field_with_name(CUGRAPH_EDGE_SOURCE_COLUMN).map_err(|_| {
        DataFusionError::from(Error::new(
            ErrorCode::FieldNotFound,
            "algeon_datafusion.cugraph_sql.edge_source_schema",
            "cuGraph edge child is missing canonical source column 'src'",
        ))
    })?;
    let dst = schema.field_with_name(CUGRAPH_EDGE_DESTINATION_COLUMN).map_err(|_| {
        DataFusionError::from(Error::new(
            ErrorCode::FieldNotFound,
            "algeon_datafusion.cugraph_sql.edge_source_schema",
            "cuGraph edge child is missing canonical destination column 'dst'",
        ))
    })?;
    let domain = vertex_id_domain_for_call(spec.algorithm, &spec.params, src.data_type(), dst.data_type())?;
    validate_vertex_id_domain_support_for_call(spec.algorithm, &domain)?;
    if spec.has_edge_id {
        let edge_id = schema.field_with_name(CUGRAPH_EDGE_ID_COLUMN).map_err(|_| {
            DataFusionError::from(Error::new(
                ErrorCode::FieldNotFound,
                "algeon_datafusion.cugraph_sql.edge_source_schema",
                "cuGraph edge child is missing canonical edge-id column 'edge_id'",
            ))
        })?;
        if !domain.accepts(edge_id.data_type()) {
            return Err(plan_error(format!("cuGraph edge-id column 'edge_id' must have dtype {}", domain.label())));
        }
    }
    spec.validate_logical_edge_id_capability(domain.is_string_logical())?;
    Ok(domain)
}

pub(super) fn validate_predicate_source_schema(
    spec: &CugraphAlgorithmSpec,
    source: &PredicateSourceSpec,
    schema: &SchemaRef,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<()> {
    if let CugraphAlgorithmParams::Similarity(params) = &spec.params {
        return validate_similarity_pair_table_schema(&params.vertex_pairs, schema, vertex_domain);
    }
    for role in &source.roles {
        let name = role.canonical_name();
        let field = schema
            .field_with_name(name)
            .map_err(|_| plan_error(format!("cuGraph side-input child is missing canonical column '{name}'")))?;
        match role {
            PredicateSourceColumnRole::Vertex
            | PredicateSourceColumnRole::Source
            | PredicateSourceColumnRole::Destination => {
                if !vertex_domain.accepts(field.data_type()) {
                    return Err(plan_error(format!(
                        "cuGraph side-input vertex column '{name}' must have dtype {}",
                        vertex_domain.label()
                    )));
                }
            },
            PredicateSourceColumnRole::EdgeId => {
                if vertex_domain.is_string_logical() || !vertex_domain.accepts(field.data_type()) {
                    return Err(plan_error(format!(
                        "cuGraph side-input edge-id column '{name}' requires the numeric edge-id domain"
                    )));
                }
            },
            PredicateSourceColumnRole::FloatValue => {
                if !matches!(field.data_type(), DataType::Float32 | DataType::Float64) {
                    return Err(plan_error(format!(
                        "cuGraph side-input value column '{name}' must have dtype Float32 or Float64"
                    )));
                }
            },
            PredicateSourceColumnRole::FirstVertex | PredicateSourceColumnRole::SecondVertex => {
                if !vertex_domain.accepts(field.data_type()) {
                    return Err(plan_error(format!(
                        "cuGraph side-input vertex column '{name}' must have dtype {}",
                        vertex_domain.label()
                    )));
                }
            },
            PredicateSourceColumnRole::Radius
            | PredicateSourceColumnRole::Mobility
            | PredicateSourceColumnRole::Mass
            | PredicateSourceColumnRole::X
            | PredicateSourceColumnRole::Y => {
                if field.data_type() != &DataType::Float32 {
                    return Err(plan_error(format!(
                        "cuGraph side-input value column '{name}' must have dtype Float32"
                    )));
                }
            },
        }
    }
    Ok(())
}
