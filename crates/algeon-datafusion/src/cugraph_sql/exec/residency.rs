use std::sync::Arc;

use datafusion::datasource::source::DataSourceExec;

use super::{
    AlgeonNativeGpuProducerExec, CUGRAPH_EDGE_DESTINATION_COLUMN, CUGRAPH_EDGE_ID_COLUMN, CUGRAPH_EDGE_SOURCE_COLUMN,
    CUGRAPH_EDGE_WEIGHT_COLUMN, CugraphAlgorithmSpec, DataType, ExecutionPlan, GpuDataFrame, GraphAlgorithmSpec,
    GraphConstructionOptions, GraphConstructionPolicy, GraphEdgeColumnBindings, GraphResidencyDescriptor,
    GraphResidencyPlanAnalysis, GraphSourceFingerprintBypassReason, Result, Schema, as_algeon_native_composed_exec,
    as_algeon_native_exec, graph_residency_descriptor_for_edge_plan, is_logical_vertex_type, native_algorithm,
};
use crate::native::{
    FooterScope, graph_residency_descriptor_for_datafusion_local_parquet_scan, lower_physical_plan_to_native,
};

pub(super) fn cugraph_graph_residency_descriptor(
    spec: &CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    edge_normalize_inserted: bool,
) -> Result<Option<GraphResidencyDescriptor>> {
    if cugraph_spec_uses_logical_vertex_mapping(spec, edge_source.schema().as_ref()) {
        return Ok(None);
    }
    let native_spec = native_graph_spec_for_residency(spec, edge_normalize_inserted);
    Ok(cugraph_graph_residency_analysis(Arc::clone(&edge_source), &native_spec)?.descriptor().cloned())
}

pub(super) fn cugraph_spec_uses_logical_vertex_mapping(_spec: &CugraphAlgorithmSpec, edge_schema: &Schema) -> bool {
    let Ok(source_field) = edge_schema.field_with_name(CUGRAPH_EDGE_SOURCE_COLUMN) else {
        return true;
    };
    let Ok(destination_field) = edge_schema.field_with_name(CUGRAPH_EDGE_DESTINATION_COLUMN) else {
        return true;
    };
    is_logical_vertex_type(source_field.data_type()) || is_logical_vertex_type(destination_field.data_type())
}

pub(super) fn cugraph_graph_residency_analysis(
    edge_source: Arc<dyn ExecutionPlan>,
    native_spec: &GraphAlgorithmSpec,
) -> Result<GraphResidencyPlanAnalysis> {
    Ok(if let Some(native) = as_algeon_native_exec(edge_source.as_ref()) {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else if let Some(native) = as_algeon_native_composed_exec(edge_source.as_ref()) {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else if let Some(native) = edge_source.as_ref().downcast_ref::<AlgeonNativeGpuProducerExec>() {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else {
        cugraph_datafusion_edge_plan_residency_analysis(edge_source, native_spec)?
    })
}

pub(super) fn cugraph_datafusion_edge_plan_residency_analysis(
    edge_source: Arc<dyn ExecutionPlan>,
    native_spec: &GraphAlgorithmSpec,
) -> Result<GraphResidencyPlanAnalysis> {
    let _footers = FooterScope::enter();
    let direct_parquet_descriptor = edge_source
        .as_ref()
        .downcast_ref::<DataSourceExec>()
        .and_then(|source| graph_residency_descriptor_for_datafusion_local_parquet_scan(source, native_spec));
    let lowering = lower_physical_plan_to_native(edge_source)?;
    if lowering.plans.len() == 1 && lowering.selected_count() == 1 {
        return Ok(graph_residency_descriptor_for_edge_plan(&lowering.plans[0], native_spec));
    }
    Ok(direct_parquet_descriptor.map_or_else(
        || GraphResidencyPlanAnalysis::Bypass(GraphSourceFingerprintBypassReason::NonLocalParquetSource),
        GraphResidencyPlanAnalysis::Keyable,
    ))
}

pub(super) fn native_graph_spec_for_residency(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
) -> GraphAlgorithmSpec {
    let mut edge_columns = GraphEdgeColumnBindings::new(
        CUGRAPH_EDGE_SOURCE_COLUMN.to_string(),
        CUGRAPH_EDGE_DESTINATION_COLUMN.to_string(),
        spec.has_weight.then(|| CUGRAPH_EDGE_WEIGHT_COLUMN.to_string()),
    );
    if spec.has_edge_id {
        edge_columns = edge_columns.with_edge_id(CUGRAPH_EDGE_ID_COLUMN.to_string());
    }
    let algorithm = native_algorithm(spec.algorithm);
    GraphAlgorithmSpec::with_stable_schema(
        algorithm,
        edge_columns,
        GraphConstructionOptions {
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            policy: if edge_normalize_inserted {
                GraphConstructionPolicy::PythonCugraph
            } else {
                GraphConstructionPolicy::RawLibcugraph
            },
        },
        algorithm.definition().default_params(),
    )
    .with_edge_normalize_inserted(edge_normalize_inserted)
}

pub(super) fn attach_graph_residency(
    frame: GpuDataFrame,
    graph_residency: Option<&GraphResidencyDescriptor>,
    edge_schema: &Schema,
    spec: &CugraphAlgorithmSpec,
) -> GpuDataFrame {
    match graph_residency {
        Some(graph_residency) if graph_residency_matches_edge_schema(graph_residency, edge_schema, spec) => {
            frame.with_graph_residency(graph_residency.clone())
        },
        None | Some(_) => frame,
    }
}

pub(super) fn graph_residency_matches_edge_schema(
    graph_residency: &GraphResidencyDescriptor,
    edge_schema: &Schema,
    spec: &CugraphAlgorithmSpec,
) -> bool {
    let Ok(source_field) = edge_schema.field_with_name(CUGRAPH_EDGE_SOURCE_COLUMN) else {
        return false;
    };
    let Ok(destination_field) = edge_schema.field_with_name(CUGRAPH_EDGE_DESTINATION_COLUMN) else {
        return false;
    };
    if source_field.data_type() != &graph_residency.vertex_data_type
        || destination_field.data_type() != &graph_residency.vertex_data_type
    {
        return false;
    }
    graph_residency_optional_dtype_matches(
        graph_residency.weight_data_type.as_ref(),
        spec.has_weight.then_some(CUGRAPH_EDGE_WEIGHT_COLUMN),
        edge_schema,
    ) && graph_residency_optional_dtype_matches(
        graph_residency.edge_id_data_type.as_ref(),
        spec.has_edge_id.then_some(CUGRAPH_EDGE_ID_COLUMN),
        edge_schema,
    )
}

pub(super) fn graph_residency_optional_dtype_matches(
    expected: Option<&DataType>,
    column: Option<&str>,
    edge_schema: &Schema,
) -> bool {
    match expected {
        Some(expected) => column
            .and_then(|column| edge_schema.field_with_name(column).ok())
            .is_some_and(|field| field.data_type() == expected),
        None => true,
    }
}
