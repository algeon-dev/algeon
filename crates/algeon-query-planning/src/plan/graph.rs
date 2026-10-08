//! Structural invariants every [`QueryPlan`](super::QueryPlan) value holds.
//!
//! A plan is constructed only through [`validated_plan_parts`]: node ids equal
//! their positions, every input precedes its consumer, the sink reaches every
//! node, and each node's output schema comes from one source. Filters, sorts,
//! limits, windows and edge normalization derive it from their input;
//! sources, projections and algorithm nodes take it from their payload;
//! aggregates and joins keep the schema their frontend declared; a union keeps
//! its declared canonical schema after checking its inputs against it by
//! position.

use std::sync::Arc;

use super::{
    EdgeNormalizeSpec, Field, NodeId, PlanNode, QueryNode, ResultSink, Schema, SchemaRef, UnionNode, WindowNode,
};
use crate::error::{QueryError, QueryErrorCode, Result};

const OPERATION: &str = "query_engine.plan.graph";

/// Checks the node list a rewrite produced before compaction indexes it: ids
/// equal positions, every input precedes its consumer, and `root` exists.
pub(super) fn check_node_order(nodes: &[QueryNode], root: NodeId) -> Result<()> {
    for (position, node) in nodes.iter().enumerate() {
        if node.id.index() != position {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "query plan node id must match its position",
            )
            .with_fact("node", node.id)
            .with_fact("position", position));
        }
        if let Some(input) = node.input_ids().iter().find(|input| input.index() >= position) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "query plan node input must precede its consumer",
            )
            .with_fact("consumer", node.id)
            .with_fact("input", *input));
        }
    }
    if root.index() >= nodes.len() {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, "query plan sink input does not exist")
            .with_fact("sink_input", root)
            .with_fact("node_count", nodes.len()));
    }
    Ok(())
}

/// Validates a complete node list and sink and derives every node's output
/// schema, in node order.
///
/// Order matters: `check_node_order` runs first so the schema pass may index
/// inputs by id and read a schema already derived; reachability and the sink
/// check run last because they read derived schemas. A node whose declared
/// schema differs from the derived one is overwritten, except for aggregates,
/// joins and unions, whose declaration is the contract. Every error carries
/// `InvalidPlan` and the offending node id as a fact.
pub(super) fn validated_plan_parts(mut nodes: Vec<QueryNode>, sink: &ResultSink) -> Result<Vec<QueryNode>> {
    check_node_order(&nodes, sink.input())?;
    for position in 0..nodes.len() {
        let (inputs, rest) = nodes.split_at_mut(position);
        let node = &mut rest[0];
        node.facts.output_schema = derive_output(node.id, &node.kind, Some(&node.facts.output_schema), inputs)?;
    }
    check_reachable(&nodes, sink.input())?;
    check_sink(&nodes, sink)?;
    Ok(nodes)
}

/// The output schema of node `id` given the nodes before it. Aggregates,
/// joins and unions return their `declared` schema.
pub(super) fn derive_output(
    id: NodeId,
    kind: &PlanNode,
    declared: Option<&SchemaRef>,
    inputs: &[QueryNode],
) -> Result<SchemaRef> {
    let input = |id: NodeId| inputs[id.index()].output_schema();
    let declared = || {
        declared.cloned().ok_or_else(|| {
            QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, "query plan node requires a declared output schema")
                .with_fact("node", id)
        })
    };
    Ok(match kind {
        PlanNode::Source(source) => Arc::clone(source.output_schema()),
        PlanNode::Filter(filter) => Arc::clone(input(filter.input)),
        PlanNode::Sort(sort) => Arc::clone(input(sort.input)),
        PlanNode::Limit(limit) => Arc::clone(input(limit.input)),
        PlanNode::Projection(projection) => Arc::clone(projection.output_schema()),
        PlanNode::Window(window) => window_output_schema(window, input(window.input)),
        PlanNode::EdgeNormalize(edge_normalize) => {
            edge_normalize_output_schema(input(edge_normalize.input), &edge_normalize.spec)?
        },
        PlanNode::GraphAlgorithm(graph) => {
            check_input_ordinals(id, graph.spec.params.side_input_ordinals(), 1, graph.inputs.len())?;
            Arc::clone(graph.output_schema())
        },
        PlanNode::VectorAlgorithm(vector) => {
            let ordinals = vector.spec.params.relation_bindings().map(|binding| binding.input);
            check_input_ordinals(id, ordinals, 0, vector.inputs.len())?;
            Arc::clone(vector.output_schema())
        },
        PlanNode::MlAlgorithm(ml) => {
            check_input_ordinals(
                id,
                ml.spec.params.relation_bindings().map(|binding| binding.input),
                0,
                ml.inputs.len(),
            )?;
            Arc::clone(ml.output_schema())
        },
        PlanNode::Aggregate(_) | PlanNode::Join(_) => declared()?,
        PlanNode::Union(union) => {
            let declared = declared()?;
            check_union_inputs(id, union, &declared, inputs)?;
            declared
        },
    })
}

/// An algorithm node's bindings name each of its inputs from position `first`
/// on exactly once, and name nothing else.
fn check_input_ordinals(id: NodeId, ordinals: impl Iterator<Item = usize>, first: usize, inputs: usize) -> Result<()> {
    let mut named = vec![false; inputs];
    for ordinal in ordinals {
        if ordinal < first || ordinal >= inputs || std::mem::replace(&mut named[ordinal], true) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "algorithm relation binding must name a distinct input of its node",
            )
            .with_fact("node_id", id)
            .with_fact("input_ordinal", ordinal)
            .with_fact("input_count", inputs));
        }
    }
    if inputs == 0 || named[first.min(inputs)..].iter().any(|named| !named) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OPERATION,
            "every algorithm input must be named by one relation binding",
        )
        .with_fact("node_id", id)
        .with_fact("input_count", inputs));
    }
    Ok(())
}

/// The window input's fields followed by one field per function.
pub(super) fn window_output_schema(window: &WindowNode, input: &SchemaRef) -> SchemaRef {
    let fields = input
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .chain(window.spec.functions.iter().map(|function| function.output().clone()))
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

/// The edge columns an edge normalization keeps, in role order.
pub(super) fn edge_normalize_output_schema(input_schema: &SchemaRef, spec: &EdgeNormalizeSpec) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(
        2 + usize::from(spec.edge_columns.weight.is_some()) + usize::from(spec.edge_columns.edge_id.is_some()),
    );
    fields.push(edge_normalize_field(input_schema, &spec.edge_columns.source, "source")?);
    fields.push(edge_normalize_field(input_schema, &spec.edge_columns.destination, "destination")?);
    if let Some(weight) = &spec.edge_columns.weight {
        fields.push(edge_normalize_field(input_schema, weight, "weight")?);
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        fields.push(edge_normalize_field(input_schema, edge_id, "edge_id")?);
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn edge_normalize_field(input_schema: &SchemaRef, column: &str, role: &'static str) -> Result<Field> {
    input_schema.field_with_name(column).cloned().map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.graph_algorithm.edge_normalize_schema",
            "edge-normalize column is not present in the input schema",
        )
        .with_fact("column", column)
        .with_fact("role", role)
        .with_source(error)
    })
}

/// A union's inputs match its declared schema by position: the same field
/// count and Arrow types, and no nullable input under a non-nullable output
/// field. Input field names do not matter; the declared schema names the
/// output.
fn check_union_inputs(id: NodeId, union: &UnionNode, declared: &SchemaRef, inputs: &[QueryNode]) -> Result<()> {
    if union.inputs.is_empty() {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, "union node requires at least one input")
            .with_fact("node_id", id));
    }
    for (input_index, input) in union.inputs.iter().enumerate() {
        let input_schema = inputs[input.index()].output_schema();
        if input_schema.fields().len() != declared.fields().len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "union input schema field count must match the union output schema",
            )
            .with_fact("node_id", id)
            .with_fact("input_index", input_index)
            .with_fact("input_node", *input)
            .with_fact("input_fields", input_schema.fields().len())
            .with_fact("output_fields", declared.fields().len()));
        }
        for (field_index, (input_field, output_field)) in
            input_schema.fields().iter().zip(declared.fields().iter()).enumerate()
        {
            if input_field.data_type() != output_field.data_type()
                || (input_field.is_nullable() && !output_field.is_nullable())
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    OPERATION,
                    "union input field must match the union output field type and nullability by ordinal",
                )
                .with_fact("node_id", id)
                .with_fact("input_index", input_index)
                .with_fact("input_node", *input)
                .with_fact("field_index", field_index)
                .with_fact("input_type", format!("{:?}", input_field.data_type()))
                .with_fact("output_type", format!("{:?}", output_field.data_type()))
                .with_fact("input_nullable", input_field.is_nullable())
                .with_fact("output_nullable", output_field.is_nullable()));
            }
        }
    }
    Ok(())
}

/// Every node must be reachable from the sink input. Walking in reverse
/// order suffices because inputs precede consumers, which `check_node_order`
/// established before this runs.
fn check_reachable(nodes: &[QueryNode], root: NodeId) -> Result<()> {
    let mut reachable = vec![false; nodes.len()];
    reachable[root.index()] = true;
    for node in nodes.iter().rev() {
        if reachable[node.id.index()] {
            for input in node.input_ids() {
                reachable[input.index()] = true;
            }
        }
    }
    match reachable.iter().position(|reachable| !reachable) {
        Some(index) => Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OPERATION,
            "query plan contains a node the sink does not reach",
        )
        .with_fact("node", nodes[index].id)
        .with_fact("sink_input", root)),
        None => Ok(()),
    }
}

/// A host sink names the root output: the same field count and Arrow types
/// by ordinal. Names, nullability and metadata are the frontend's.
pub(super) fn check_sink(nodes: &[QueryNode], sink: &ResultSink) -> Result<()> {
    let Some(schema) = sink.caller_visible_host_schema() else {
        return Ok(());
    };
    let root = &nodes[sink.input().index()];
    let root_fields = root.output_schema().fields();
    if schema.fields().len() != root_fields.len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.validate_sink",
            "host sink schema field count must match the root node output",
        )
        .with_fact("root_node", root.id)
        .with_fact("sink_fields", schema.fields().len())
        .with_fact("root_fields", root_fields.len()));
    }
    for (index, (sink_field, root_field)) in schema.fields().iter().zip(root_fields.iter()).enumerate() {
        if sink_field.data_type() != root_field.data_type() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.plan.validate_sink",
                "host sink Arrow type must match the root node output by ordinal",
            )
            .with_fact("root_node", root.id)
            .with_fact("field_index", index)
            .with_fact("sink_type", format!("{:?}", sink_field.data_type()))
            .with_fact("root_type", format!("{:?}", root_field.data_type())));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use crate::{
        error::QueryErrorCode,
        expr::{AggregateFunction, Expr, LimitSpec, SortExpr, SortSpec},
        plan::{NodeId, PlanNode, QueryPlanBuilder, RewrittenNodes, WindowFunction, WindowSpec},
        source::{GpuResidentSource, SourceNode},
    };

    fn source(builder: &mut QueryPlanBuilder, name: &str, fields: Vec<Field>) -> NodeId {
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(name, Arc::new(Schema::new(fields)))))
    }

    #[test]
    fn rewrite_rejects_a_forward_input_before_compaction_reads_it() {
        let mut builder = QueryPlanBuilder::new();
        let input = source(&mut builder, "input", vec![Field::new("v", DataType::Int64, false)]);
        let filter = builder.add_filter(input, Expr::is_not_null(Expr::column("v"))).unwrap();
        let plan = builder.finish_gpu_resident(filter).unwrap();
        let mut nodes = plan.nodes().to_vec();
        let PlanNode::Filter(filter) = &mut nodes[1].kind else { unreachable!("node 1 is the filter") };
        filter.input = NodeId(7);

        let error = plan.rewrite(RewrittenNodes::in_place(nodes, plan.root_id())).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.operation(), "query_engine.plan.graph");
        assert_eq!(error.fact_value("input"), Some("n7"));
    }

    #[test]
    fn row_preserving_and_window_nodes_derive_their_output_schema() {
        let metadata = HashMap::from([("origin".to_string(), "frontend".to_string())]);
        let mut builder = QueryPlanBuilder::new();
        let input = source(
            &mut builder,
            "input",
            vec![
                Field::new("k", DataType::Int32, false).with_metadata(metadata),
                Field::new("v", DataType::Int64, true),
            ],
        );
        let input_schema = Arc::clone(builder.node_output_schema(input).unwrap());
        let filter = builder.add_filter(input, Expr::is_not_null(Expr::column("v"))).unwrap();
        let sort =
            builder.add_sort(filter, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)])).unwrap();
        let limit = builder.add_limit(sort, LimitSpec::fetch(5)).unwrap();
        for node in [filter, sort, limit] {
            assert_eq!(builder.node_output_schema(node), Some(&input_schema));
        }

        let window = builder
            .add_window(
                limit,
                WindowSpec::new(
                    vec![Expr::column("k")],
                    vec![SortExpr::new(Expr::column("v"), false, false)],
                    vec![
                        WindowFunction::rank("r"),
                        WindowFunction::cumulative_aggregate(
                            AggregateFunction::Sum,
                            Expr::column("v"),
                            Field::new("running", DataType::Int64, true),
                        ),
                    ],
                ),
            )
            .unwrap();
        let fields = input_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain([Field::new("r", DataType::UInt64, false), Field::new("running", DataType::Int64, true)])
            .collect::<Vec<_>>();
        assert_eq!(builder.node_output_schema(window).unwrap().as_ref(), &Schema::new(fields));
    }

    #[test]
    fn union_output_keeps_its_declared_names_and_rejects_narrowed_nullability() {
        let mut builder = QueryPlanBuilder::new();
        let left = source(&mut builder, "left", vec![Field::new("a", DataType::Int64, false)]);
        let right = source(&mut builder, "right", vec![Field::new("b", DataType::Int64, true)]);
        let declared = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));
        let forward = builder.add_union(vec![left, right], Arc::clone(&declared)).unwrap();
        let reversed = builder.add_union(vec![right, left], Arc::clone(&declared)).unwrap();
        let without_first = builder.add_union(vec![right], Arc::clone(&declared)).unwrap();
        for union in [forward, reversed, without_first] {
            assert_eq!(builder.node_output_schema(union), Some(&declared));
        }

        let narrowed = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let error = builder.add_union(vec![left, right], narrowed).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("input_index"), Some("1"));
        let retyped = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, true)]));
        assert_eq!(builder.add_union(vec![left], retyped).unwrap_err().code(), QueryErrorCode::InvalidPlan);
    }
}
