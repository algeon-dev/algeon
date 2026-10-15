use std::sync::Arc;

use arrow::array::Int64Array;
use query_planning::plan::{
    GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions, GraphEdgeColumnBindings,
    NativePlanCostReason, native_plan_cost_decision,
};

use super::*;

#[test]
fn plans_without_column_data_are_not_selected_by_cost_unless_native_is_required() {
    let literal_plan = |schema: Arc<Schema>| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
        builder.finish_with_host_sink(source).unwrap()
    };
    let no_columns = literal_plan(Arc::new(Schema::empty()));
    assert_eq!(native_plan_cost_decision(&no_columns, true), Some(NativePlanCostReason::NoColumnData));
    assert_eq!(native_plan_cost_decision(&no_columns, false), None);
    // A relation folded to zero rows carries no column values either, which
    // takes precedence over the host-literal passthrough rule.
    let no_rows = literal_plan(key_value_schema());
    assert_eq!(native_plan_cost_decision(&no_rows, true), Some(NativePlanCostReason::NoColumnData));

    // A GPU algorithm over the same empty input is GPU work.
    let edges = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(edges)));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::StronglyConnectedComponents,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::StronglyConnectedComponents(Default::default()),
    );
    let graph = builder.add_graph_algorithm(source, Vec::new(), spec).unwrap();
    let graph_plan = builder.finish_with_host_sink(graph).unwrap();
    assert_eq!(native_plan_cost_decision(&graph_plan, true), None);
}

#[test]
fn host_literal_passthrough_to_the_host_is_not_selected_by_cost_unless_native_is_required() {
    let schema = key_value_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![10, 20]))],
    )
    .unwrap();
    let literal_plan = |limit: Option<LimitSpec>| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder
            .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch.clone()])));
        let root = match limit {
            Some(limit) => builder.add_limit(source, limit).unwrap(),
            None => source,
        };
        builder.finish_with_host_sink(root).unwrap()
    };
    let passthrough = literal_plan(None);

    assert_eq!(native_plan_cost_decision(&passthrough, true), Some(NativePlanCostReason::HostLiteralPassthrough));
    assert_eq!(native_plan_cost_decision(&passthrough, false), None);
    // A GPU-resident result stays on the device for its consumer.
    let resident = passthrough.clone_with_gpu_resident_sink();
    assert_eq!(native_plan_cost_decision(&resident, true), None);
    let limited = literal_plan(Some(LimitSpec::fetch(1)));
    assert_eq!(native_plan_cost_decision(&limited, true), None);
}

fn key_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false), Field::new("v", DataType::Int64, false)]))
}
