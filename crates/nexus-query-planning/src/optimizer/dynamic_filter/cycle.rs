use super::{DynamicFilterConsumer, DynamicFilterProducer, DynamicFilterRoute, JoinSide, NodeId, PlanNode, QueryPlan};

pub(super) fn dynamic_filter_route_equivalent(left: &DynamicFilterRoute, right: &DynamicFilterRoute) -> bool {
    left.producer == right.producer
        && left.consumer == right.consumer
        && left.producer_columns == right.producer_columns
        && left.consumer_columns == right.consumer_columns
}

pub(super) fn dynamic_filter_route_would_create_node_cycle(
    plan: &QueryPlan,
    routes: &[DynamicFilterRoute],
    route: &DynamicFilterRoute,
) -> bool {
    dynamic_filter_endpoint_reaches_with_routes(
        plan,
        routes,
        dynamic_filter_consumer_endpoint(&route.consumer),
        dynamic_filter_producer_endpoint(&route.producer),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DynamicFilterDependencyEndpoint {
    NodeOutput(NodeId),
    JoinBuild(NodeId),
    JoinProbe(NodeId),
}

fn dynamic_filter_producer_endpoint(producer: &DynamicFilterProducer) -> DynamicFilterDependencyEndpoint {
    match producer {
        DynamicFilterProducer::HashJoinBuild { join_node_id, .. } => {
            DynamicFilterDependencyEndpoint::JoinBuild(*join_node_id)
        },
        DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } => {
            DynamicFilterDependencyEndpoint::NodeOutput(*aggregate_node_id)
        },
    }
}

fn dynamic_filter_consumer_endpoint(consumer: &DynamicFilterConsumer) -> DynamicFilterDependencyEndpoint {
    match consumer {
        DynamicFilterConsumer::LocalParquetSource { source_node_id } => {
            DynamicFilterDependencyEndpoint::NodeOutput(*source_node_id)
        },
        DynamicFilterConsumer::HashJoinProbe { join_node_id, .. } => {
            DynamicFilterDependencyEndpoint::JoinProbe(*join_node_id)
        },
        DynamicFilterConsumer::Filter { filter_node_id } => {
            DynamicFilterDependencyEndpoint::NodeOutput(*filter_node_id)
        },
    }
}

fn dynamic_filter_endpoint_reaches_with_routes(
    plan: &QueryPlan,
    routes: &[DynamicFilterRoute],
    from: DynamicFilterDependencyEndpoint,
    to: DynamicFilterDependencyEndpoint,
) -> bool {
    let mut stack = vec![from];
    let mut visited = std::collections::BTreeSet::new();
    while let Some(endpoint) = stack.pop() {
        if !visited.insert(endpoint) {
            continue;
        }
        if endpoint == to {
            return true;
        }
        push_dynamic_filter_dependency_successors(plan, routes, endpoint, &mut stack);
    }
    false
}

fn push_dynamic_filter_dependency_successors(
    plan: &QueryPlan,
    routes: &[DynamicFilterRoute],
    endpoint: DynamicFilterDependencyEndpoint,
    stack: &mut Vec<DynamicFilterDependencyEndpoint>,
) {
    for node in plan.nodes() {
        match &node.kind {
            PlanNode::Join(join) => {
                let build_input = match join.spec.build_side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                let probe_input = match join.spec.build_side {
                    JoinSide::Left => join.inputs[1],
                    JoinSide::Right => join.inputs[0],
                };
                match endpoint {
                    DynamicFilterDependencyEndpoint::NodeOutput(node_id) if node_id == build_input => {
                        stack.push(DynamicFilterDependencyEndpoint::JoinBuild(node.id));
                    },
                    DynamicFilterDependencyEndpoint::NodeOutput(node_id) if node_id == probe_input => {
                        stack.push(DynamicFilterDependencyEndpoint::JoinProbe(node.id));
                    },
                    DynamicFilterDependencyEndpoint::JoinBuild(node_id) if node_id == node.id => {
                        stack.push(DynamicFilterDependencyEndpoint::JoinProbe(node.id));
                    },
                    DynamicFilterDependencyEndpoint::JoinProbe(node_id) if node_id == node.id => {
                        stack.push(DynamicFilterDependencyEndpoint::NodeOutput(node.id));
                    },
                    _ => {},
                }
            },
            _ => {
                if let DynamicFilterDependencyEndpoint::NodeOutput(node_id) = endpoint
                    && node.input_ids().contains(&node_id)
                {
                    stack.push(DynamicFilterDependencyEndpoint::NodeOutput(node.id));
                }
            },
        }
    }
    for route in routes {
        if dynamic_filter_producer_endpoint(&route.producer) == endpoint {
            stack.push(dynamic_filter_consumer_endpoint(&route.consumer));
        }
    }
}
