//! Shared helpers for the native execution wrappers.
//!
//! Each wrapper (`NativeCudfExec`, `NativeCudfComposedExec`,
//! `NativeCudfGpuProducerExec`) advertises the same conservative
//! `PlanProperties` and uses the same error tagging at the DataFusion runtime
//! boundary, so those concerns live here rather than being duplicated.

#![deny(missing_docs)]

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

/// Conservative `PlanProperties` shared by all native execution wrappers:
/// one unknown bounded partition, `Final` emission, no ordering guarantees.
pub(super) fn conservative_properties(schema: SchemaRef) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        Partitioning::UnknownPartitioning(1),
        EmissionType::Final,
        Boundedness::Bounded,
    ))
}

/// Collects `input_name`s of every GPU-resident source node in the plan.
///
/// Used to look up upstream DataFusion children that are expected to deliver
/// GPU-resident batches into the wrapper.
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

/// Joins iterator items with `,`. Used to render stable comma-separated lists
/// for plan-display strings.
pub(super) fn comma_separated<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    values.into_iter().collect::<Vec<_>>().join(",")
}

/// Maps a `QueryError` from native runtime execution into the adapter [`Error`]
/// while tagging the stable diagnostic facts callers assert on.
pub(super) fn native_query_runtime_error(
    operation: &'static str,
    message: impl Into<String>,
    source: cudf_query_engine::QueryError,
) -> Error {
    mark_native_runtime_error(Error::from_native(operation, message, source), "runtime_execution")
}

/// Tags an error with `native_datafusion_phase=<phase>` and
/// `cpu_fallback_after_native_start=false`.
///
/// Callers use these facts to distinguish context setup from query runtime,
/// and to assert that native execution does not silently fall back to CPU
/// once started.
pub(super) fn mark_native_runtime_error(error: Error, phase: &'static str) -> Error {
    error.with_fact("native_datafusion_phase", phase).with_fact("cpu_fallback_after_native_start", false)
}
