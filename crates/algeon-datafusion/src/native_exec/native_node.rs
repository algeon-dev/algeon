//! The closed set of adapter nodes that run native GPU work.

use std::any::Any;

use datafusion::physical_plan::ExecutionPlan;

use super::{
    GpuResidentProducer, composed::AlgeonNativeComposedGpuProducerExec, graph::GraphExecutionPlan,
    producer::AlgeonNativeGpuProducerExec,
};
#[cfg(test)]
use super::{
    child_inputs::tests::FakeProducerExec,
    composed::tests::{FakeGpuResidentProducer, FakeHostOwnedGraphExec},
};
#[cfg(all(test, feature = "cugraph"))]
use crate::cugraph_sql::CountingGpuResidentProducer;
#[cfg(feature = "cugraph")]
use crate::cugraph_sql::CugraphAlgoExec;
use crate::{cuml_sql::CumlAlgoExec, cuvs_sql::CuvsAlgoExec};

/// One adapter node that produces GPU work, classified by its exact type.
pub(crate) enum NativeNode<'a> {
    /// A native fragment with a GPU-resident sink.
    Producer(&'a AlgeonNativeGpuProducerExec),
    /// A composed native fragment with a GPU-resident sink.
    ComposedProducer(&'a AlgeonNativeComposedGpuProducerExec),
    /// A cuVS or cuML relation function.
    RelationFunction(&'a dyn GpuResidentProducer),
    /// A cuGraph algorithm.
    #[cfg(feature = "cugraph")]
    Graph(&'a CugraphAlgoExec),
    /// A test double standing in for a frontend node.
    #[cfg(test)]
    TestDouble { producer: Option<&'a dyn GpuResidentProducer>, graph: Option<&'a dyn GraphExecutionPlan> },
}

/// Classifies `plan` by its exact type; wrappers are never followed here.
pub(crate) fn native_node(plan: &dyn ExecutionPlan) -> Option<NativeNode<'_>> {
    let any = plan as &dyn Any;
    if let Some(node) = any.downcast_ref::<AlgeonNativeGpuProducerExec>() {
        return Some(NativeNode::Producer(node));
    }
    if let Some(node) = any.downcast_ref::<AlgeonNativeComposedGpuProducerExec>() {
        return Some(NativeNode::ComposedProducer(node));
    }
    if let Some(node) = any.downcast_ref::<CuvsAlgoExec>() {
        return Some(NativeNode::RelationFunction(node));
    }
    if let Some(node) = any.downcast_ref::<CumlAlgoExec>() {
        return Some(NativeNode::RelationFunction(node));
    }
    #[cfg(feature = "cugraph")]
    if let Some(node) = any.downcast_ref::<CugraphAlgoExec>() {
        return Some(NativeNode::Graph(node));
    }
    #[cfg(test)]
    if let Some(node) = test_double(any) {
        return Some(node);
    }
    None
}

impl<'a> NativeNode<'a> {
    /// The node's GPU-resident output, when it produces one a native consumer
    /// can read.
    pub(crate) fn gpu_resident_producer(&self) -> Option<&'a dyn GpuResidentProducer> {
        match *self {
            Self::Producer(node) => Some(node),
            Self::ComposedProducer(node) => Some(node),
            Self::RelationFunction(node) => Some(node),
            #[cfg(feature = "cugraph")]
            Self::Graph(node) => node.gpu_resident_producer(),
            #[cfg(test)]
            Self::TestDouble { producer, .. } => producer,
        }
    }

    /// The node's graph execution, when it runs a graph algorithm.
    pub(crate) fn graph_execution_plan(&self) -> Option<&'a dyn GraphExecutionPlan> {
        match *self {
            Self::Producer(_) | Self::ComposedProducer(_) | Self::RelationFunction(_) => None,
            #[cfg(feature = "cugraph")]
            Self::Graph(node) => Some(node),
            #[cfg(test)]
            Self::TestDouble { graph, .. } => graph,
        }
    }
}

#[cfg(test)]
fn test_double(any: &dyn Any) -> Option<NativeNode<'_>> {
    if let Some(node) = any.downcast_ref::<FakeGpuResidentProducer>() {
        return Some(NativeNode::TestDouble { producer: Some(node), graph: node.graph_execution_plan() });
    }
    if let Some(node) = any.downcast_ref::<FakeHostOwnedGraphExec>() {
        return Some(NativeNode::TestDouble { producer: None, graph: Some(node) });
    }
    if let Some(node) = any.downcast_ref::<FakeProducerExec>() {
        return Some(NativeNode::TestDouble { producer: Some(node), graph: None });
    }
    #[cfg(feature = "cugraph")]
    if let Some(node) = any.downcast_ref::<CountingGpuResidentProducer>() {
        return Some(NativeNode::TestDouble { producer: Some(node), graph: Some(node) });
    }
    None
}
