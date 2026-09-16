use super::{
    DataFrame, ExecutionContext, GpuDataFrame, GraphAlgorithmExecutionInputs, GraphAlgorithmExecutionResult, NodeId,
    QueryError, QueryErrorCode, Result, cudf_query_error, force_atlas2_params, map_cugraph_error,
    normalize_graph_result_schema, project_edge_list, run_with_cugraph_graph, table_graph_options,
};

pub(super) fn execute_force_atlas2(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let GraphAlgorithmExecutionInputs { edge_list: input, side_inputs } = inputs;
    let crate::plan::GraphAlgorithmParams::ForceAtlas2(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.force_atlas2",
            "ForceAtlas2 graph algorithm node carried non-ForceAtlas2 parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let fa2_params = force_atlas2_params(params)?;
    let vertex_attributes = params
        .vertex_attributes
        .as_ref()
        .map(|binding| force_atlas2_vertex_attribute_series(binding, &side_inputs))
        .transpose()?;
    let initial_positions = params
        .initial_positions
        .as_ref()
        .map(|binding| force_atlas2_initial_position_series(binding, &side_inputs))
        .transpose()?;
    let seed = params.seed;

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_force_atlas2",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.force_atlas2_rng",
                "failed to create cuGraph RNG state for ForceAtlas2",
            ))?;
            let result = match graph.vertex_dtype() {
                cugraph::types::DataType::Int32 => force_atlas2_with_side_inputs::<i32>(
                    entered,
                    graph,
                    &mut rng_state,
                    &fa2_params,
                    vertex_attributes.as_ref(),
                    initial_positions.as_ref(),
                )?,
                cugraph::types::DataType::Int64 => force_atlas2_with_side_inputs::<i64>(
                    entered,
                    graph,
                    &mut rng_state,
                    &fa2_params,
                    vertex_attributes.as_ref(),
                    initial_positions.as_ref(),
                )?,
                vertex_dtype => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "nexus_query_engine.graph_algorithm.force_atlas2",
                        "ForceAtlas2 graph vertex dtype is unsupported",
                    )
                    .with_fact("vertex_dtype", vertex_dtype));
                },
            };
            let algorithm_metadata = format!("seed={seed}");
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.force_atlas2_result",
                "failed to convert cuGraph ForceAtlas2 result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                cache_scope_peak_bytes: None,
            })
        },
    )
}

struct ForceAtlas2VertexAttributeSeries {
    vertex: cudf::Series,
    radius: Option<cudf::Series>,
    mobility: Option<cudf::Series>,
    mass: Option<cudf::Series>,
}

struct ForceAtlas2InitialPositionSeries {
    vertex: cudf::Series,
    x: cudf::Series,
    y: cudf::Series,
}

fn force_atlas2_vertex_attribute_series(
    binding: &crate::plan::GraphForceAtlas2VertexAttributesInput,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<ForceAtlas2VertexAttributeSeries> {
    let frame = force_atlas2_side_input_frame(side_inputs, binding.input, "vertex_attributes")?;
    Ok(ForceAtlas2VertexAttributeSeries {
        vertex: force_atlas2_side_input_column(frame, &binding.vertex, "vertex_attributes.vertex", None)?,
        radius: binding
            .radius
            .as_deref()
            .map(|column| {
                force_atlas2_side_input_column(frame, column, "vertex_attributes.radius", Some(cudf::TypeId::Float32))
            })
            .transpose()?,
        mobility: binding
            .mobility
            .as_deref()
            .map(|column| {
                force_atlas2_side_input_column(frame, column, "vertex_attributes.mobility", Some(cudf::TypeId::Float32))
            })
            .transpose()?,
        mass: binding
            .mass
            .as_deref()
            .map(|column| {
                force_atlas2_side_input_column(frame, column, "vertex_attributes.mass", Some(cudf::TypeId::Float32))
            })
            .transpose()?,
    })
}

fn force_atlas2_initial_position_series(
    binding: &crate::plan::GraphForceAtlas2InitialPositionsInput,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<ForceAtlas2InitialPositionSeries> {
    let frame = force_atlas2_side_input_frame(side_inputs, binding.input, "initial_positions")?;
    Ok(ForceAtlas2InitialPositionSeries {
        vertex: force_atlas2_side_input_column(frame, &binding.vertex, "initial_positions.vertex", None)?,
        x: force_atlas2_side_input_column(frame, &binding.x, "initial_positions.x", Some(cudf::TypeId::Float32))?,
        y: force_atlas2_side_input_column(frame, &binding.y, "initial_positions.y", Some(cudf::TypeId::Float32))?,
    })
}

fn force_atlas2_side_input_frame<'a>(
    side_inputs: &'a std::collections::BTreeMap<NodeId, GpuDataFrame>,
    input: NodeId,
    role: &'static str,
) -> Result<&'a DataFrame> {
    side_inputs.get(&input).map(GpuDataFrame::frame).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
            "ForceAtlas2 side input was not executed",
        )
        .with_fact("side_input_role", role)
        .with_fact("side_input_node", input)
    })
}

fn force_atlas2_side_input_column(
    frame: &DataFrame,
    column: &str,
    role: &'static str,
    expected_dtype: Option<cudf::TypeId>,
) -> Result<cudf::Series> {
    let series = frame.series(column).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
            "failed to resolve ForceAtlas2 side-input column",
            error,
        )
        .with_fact("side_input_role", role)
        .with_fact("column", column)
    })?;
    let dtype = series.dtype().map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
            "failed to inspect ForceAtlas2 side-input column dtype",
            error,
        )
        .with_fact("side_input_role", role)
        .with_fact("column", column)
    })?;
    if series.null_count() != 0 || expected_dtype.is_some_and(|expected| dtype.id() != expected) {
        return Err(QueryError::new(
            QueryErrorCode::GraphInputInvalid,
            "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
            "ForceAtlas2 side-input columns must be non-null and match their declared dtype",
        )
        .with_fact("side_input_role", role)
        .with_fact("column", column)
        .with_fact("null_rows", series.null_count())
        .with_fact("dtype", format!("{:?}", dtype.id())));
    }
    Ok(series)
}

fn force_atlas2_with_side_inputs<V>(
    entered: &cugraph::EnteredContext<'_>,
    graph: &mut cugraph::Graph<cugraph::Forward>,
    rng_state: &mut cugraph::random::RngState,
    params: &cugraph::layout::ForceAtlas2Params,
    attributes: Option<&ForceAtlas2VertexAttributeSeries>,
    positions: Option<&ForceAtlas2InitialPositionSeries>,
) -> Result<cugraph::layout::LayoutResult>
where
    V: cugraph::types::CugraphElement + cudf::CudfType,
{
    if let Some(attributes) = attributes {
        validate_force_atlas2_vertex_series::<V>(&attributes.vertex, "vertex_attributes.vertex")?;
    }
    if let Some(positions) = positions {
        validate_force_atlas2_vertex_series::<V>(&positions.vertex, "initial_positions.vertex")?;
    }

    let attribute_vertex_slice = attributes
        .map(|input| force_atlas2_device_slice::<V>(&input.vertex, "vertex_attributes.vertex"))
        .transpose()?;
    let attribute_vertex_view = attribute_vertex_slice
        .as_ref()
        .map(|slice| force_atlas2_device_view::<V>(slice, "vertex_attributes.vertex"))
        .transpose()?;
    let radius_slice = attributes
        .and_then(|input| input.radius.as_ref())
        .map(|series| force_atlas2_device_slice::<f32>(series, "vertex_attributes.radius"))
        .transpose()?;
    let radius_view = radius_slice
        .as_ref()
        .map(|slice| force_atlas2_device_view::<f32>(slice, "vertex_attributes.radius"))
        .transpose()?;
    let mobility_slice = attributes
        .and_then(|input| input.mobility.as_ref())
        .map(|series| force_atlas2_device_slice::<f32>(series, "vertex_attributes.mobility"))
        .transpose()?;
    let mobility_view = mobility_slice
        .as_ref()
        .map(|slice| force_atlas2_device_view::<f32>(slice, "vertex_attributes.mobility"))
        .transpose()?;
    let mass_slice = attributes
        .and_then(|input| input.mass.as_ref())
        .map(|series| force_atlas2_device_slice::<f32>(series, "vertex_attributes.mass"))
        .transpose()?;
    let mass_view = mass_slice
        .as_ref()
        .map(|slice| force_atlas2_device_view::<f32>(slice, "vertex_attributes.mass"))
        .transpose()?;
    let native_attributes = attributes.map(|_| cugraph::layout::ForceAtlas2VertexAttrs {
        radius: radius_view
            .as_ref()
            .map(|values| (attribute_vertex_view.as_ref().expect("attributes carry a vertex view"), values)),
        mobility: mobility_view
            .as_ref()
            .map(|values| (attribute_vertex_view.as_ref().expect("attributes carry a vertex view"), values)),
        mass: mass_view
            .as_ref()
            .map(|values| (attribute_vertex_view.as_ref().expect("attributes carry a vertex view"), values)),
    });

    let position_vertex_slice =
        positions.map(|input| force_atlas2_device_slice::<V>(&input.vertex, "initial_positions.vertex")).transpose()?;
    let position_vertex_view = position_vertex_slice
        .as_ref()
        .map(|slice| force_atlas2_device_view::<V>(slice, "initial_positions.vertex"))
        .transpose()?;
    let x_slice =
        positions.map(|input| force_atlas2_device_slice::<f32>(&input.x, "initial_positions.x")).transpose()?;
    let x_view =
        x_slice.as_ref().map(|slice| force_atlas2_device_view::<f32>(slice, "initial_positions.x")).transpose()?;
    let y_slice =
        positions.map(|input| force_atlas2_device_slice::<f32>(&input.y, "initial_positions.y")).transpose()?;
    let y_view =
        y_slice.as_ref().map(|slice| force_atlas2_device_view::<f32>(slice, "initial_positions.y")).transpose()?;
    let native_positions = positions.map(|_| cugraph::layout::InitialPositions {
        vertices: position_vertex_view.as_ref().expect("positions carry a vertex view"),
        x: x_view.as_ref().expect("positions carry an x view"),
        y: y_view.as_ref().expect("positions carry a y view"),
    });

    graph
        .force_atlas2_with(entered, rng_state, params, native_attributes.as_ref(), native_positions.as_ref())
        .map_err(map_cugraph_error("nexus_query_engine.graph_algorithm.force_atlas2", "cuGraph ForceAtlas2 failed"))
}

fn validate_force_atlas2_vertex_series<V: cudf::CudfType>(series: &cudf::Series, role: &'static str) -> Result<()> {
    let dtype = series.dtype().map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
            "failed to inspect ForceAtlas2 vertex side-input dtype",
            error,
        )
    })?;
    if dtype.id() != V::TYPE_ID
        || series.nunique().map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
                "failed to validate ForceAtlas2 side-input vertex uniqueness",
                error,
            )
        })? != series.len()
    {
        return Err(QueryError::new(
            QueryErrorCode::GraphInputInvalid,
            "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
            "ForceAtlas2 side-input vertices must be unique and match the graph vertex dtype",
        )
        .with_fact("side_input_role", role)
        .with_fact("rows", series.len())
        .with_fact("dtype", format!("{:?}", dtype.id())));
    }
    Ok(())
}

fn force_atlas2_device_slice<'a, T: cudf::CudfType>(
    series: &'a cudf::Series,
    role: &'static str,
) -> Result<cudf::core::DeviceSlice<'a>> {
    series.as_column().device_slice::<T>().map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
            "failed to borrow ForceAtlas2 side-input device slice",
            error,
        )
        .with_fact("side_input_role", role)
    })
}

fn force_atlas2_device_view<'a, T: cugraph::types::CugraphElement>(
    slice: &'a cudf::core::DeviceSlice<'a>,
    role: &'static str,
) -> Result<cugraph::DeviceArrayView<'a, T>> {
    cugraph::cudf_interop::device_array_view_from_slice::<T>(slice).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.force_atlas2_side_input",
        match role {
            "vertex_attributes.vertex" => "failed to wrap ForceAtlas2 attribute vertices",
            "vertex_attributes.radius" => "failed to wrap ForceAtlas2 radius values",
            "vertex_attributes.mobility" => "failed to wrap ForceAtlas2 mobility values",
            "vertex_attributes.mass" => "failed to wrap ForceAtlas2 mass values",
            "initial_positions.vertex" => "failed to wrap ForceAtlas2 initial-position vertices",
            "initial_positions.x" => "failed to wrap ForceAtlas2 initial x values",
            "initial_positions.y" => "failed to wrap ForceAtlas2 initial y values",
            _ => "failed to wrap ForceAtlas2 side-input values",
        },
    ))
}
