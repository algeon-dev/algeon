use super::*;
pub(super) fn cugraph_graph_residency_descriptor(
    spec: &CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    edge_normalize_inserted: bool,
) -> Option<GraphResidencyDescriptor> {
    if cugraph_spec_uses_logical_vertex_mapping(spec, edge_source.schema().as_ref()) {
        return None;
    }
    let native_spec = native_graph_spec_for_residency(spec, edge_normalize_inserted);
    cugraph_graph_residency_analysis(Arc::clone(&edge_source), &native_spec).descriptor().cloned()
}

pub(super) fn cugraph_spec_uses_logical_vertex_mapping(spec: &CugraphAlgorithmSpec, edge_schema: &Schema) -> bool {
    let Ok(source_field) = edge_schema.field_with_name(&spec.src_col) else {
        return true;
    };
    let Ok(destination_field) = edge_schema.field_with_name(&spec.dst_col) else {
        return true;
    };
    is_logical_vertex_type(source_field.data_type()) || is_logical_vertex_type(destination_field.data_type())
}

pub(super) fn cugraph_graph_residency_analysis(
    edge_source: Arc<dyn ExecutionPlan>,
    native_spec: &GraphAlgorithmSpec,
) -> GraphResidencyPlanAnalysis {
    if let Some(native) = as_nexus_native_exec(edge_source.as_ref()) {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else if let Some(native) = as_nexus_native_composed_exec(edge_source.as_ref()) {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else if let Some(native) = edge_source.as_ref().downcast_ref::<NexusNativeGpuProducerExec>() {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else {
        cugraph_datafusion_edge_plan_residency_analysis(edge_source, native_spec)
    }
}

pub(super) fn cugraph_datafusion_edge_plan_residency_analysis(
    edge_source: Arc<dyn ExecutionPlan>,
    native_spec: &GraphAlgorithmSpec,
) -> GraphResidencyPlanAnalysis {
    let direct_parquet_descriptor =
        edge_source.as_ref().downcast_ref::<datafusion::datasource::source::DataSourceExec>().and_then(|source| {
            crate::native::graph_residency_descriptor_for_datafusion_local_parquet_scan(source, native_spec)
        });
    let lowering = crate::native::lower_physical_plan_to_native(edge_source);
    if lowering.plans.len() == 1 && lowering.selected_count() == 1 {
        return graph_residency_descriptor_for_edge_plan(&lowering.plans[0], native_spec);
    }
    direct_parquet_descriptor.map_or_else(
        || GraphResidencyPlanAnalysis::Bypass(GraphSourceFingerprintBypassReason::NonLocalParquetSource),
        GraphResidencyPlanAnalysis::Keyable,
    )
}

pub(super) fn native_graph_spec_for_residency(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
) -> GraphAlgorithmSpec {
    let mut edge_columns =
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone());
    if let Some(edge_id) = &spec.edge_id_col {
        edge_columns = edge_columns.with_edge_id(edge_id.clone());
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
    let Ok(source_field) = edge_schema.field_with_name(&spec.src_col) else {
        return false;
    };
    let Ok(destination_field) = edge_schema.field_with_name(&spec.dst_col) else {
        return false;
    };
    if source_field.data_type() != &graph_residency.vertex_data_type
        || destination_field.data_type() != &graph_residency.vertex_data_type
    {
        return false;
    }
    graph_residency_optional_dtype_matches(
        graph_residency.weight_data_type.as_ref(),
        spec.weight_col.as_deref(),
        edge_schema,
    ) && graph_residency_optional_dtype_matches(
        graph_residency.edge_id_data_type.as_ref(),
        spec.edge_id_col.as_deref(),
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
