use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::SchemaRef;
use cudf_query_engine::{PlanNode, QueryPlan, SourceNode};
use datafusion::{
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        PlanProperties,
        execution_plan::{Boundedness, EmissionType},
    },
};

use crate::error::Error;

pub(super) fn conservative_properties(schema: SchemaRef) -> Arc<PlanProperties> {
    // Deliberately conservative: one unknown, bounded, final partition and no ordering guarantees.
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        Partitioning::UnknownPartitioning(1),
        EmissionType::Final,
        Boundedness::Bounded,
    ))
}

pub(super) fn gpu_resident_source_names(plan: &QueryPlan) -> BTreeSet<String> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                SourceNode::LocalParquet(_) => None,
            },
            _ => None,
        })
        .collect()
}

pub(super) fn comma_separated<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    values.into_iter().collect::<Vec<_>>().join(",")
}

// Convert native query-runtime failures at the DataFusion boundary while preserving stable facts:
// `native_datafusion_phase` and `cpu_fallback_after_native_start=false`.
pub(super) fn native_query_runtime_error(
    operation: &'static str,
    message: impl Into<String>,
    source: cudf_query_engine::QueryError,
) -> Error {
    mark_native_runtime_error(Error::from_native(operation, message, source), "runtime_execution")
}

// Attach stable diagnostic facts used by callers to distinguish context setup from query runtime and
// to assert that native execution does not silently fall back to CPU once started.
pub(super) fn mark_native_runtime_error(error: Error, phase: &'static str) -> Error {
    error.with_fact("native_datafusion_phase", phase).with_fact("cpu_fallback_after_native_start", false)
}
