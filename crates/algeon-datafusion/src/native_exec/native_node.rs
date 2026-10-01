//! The closed set of adapter nodes that run native GPU work.

use std::{any::Any, sync::Arc};

use datafusion::physical_plan::ExecutionPlan;

use super::{
    AlgeonNativeExec, GpuFunctionHostOccurrenceExec, GpuResidentProducer, admission::NativeAdmissionExec,
    graph::GraphExecutionPlan,
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
    /// A native execution unit, with host or GPU-resident output.
    Fragment(&'a AlgeonNativeExec),
    /// The admission node rooting a query's plan.
    Admission(&'a NativeAdmissionExec),
    /// A GPU function result's host materialization, bound to one occurrence.
    Occurrence(&'a GpuFunctionHostOccurrenceExec),
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
    if let Some(node) = any.downcast_ref::<AlgeonNativeExec>() {
        return Some(NativeNode::Fragment(node));
    }
    if let Some(node) = any.downcast_ref::<NativeAdmissionExec>() {
        return Some(NativeNode::Admission(node));
    }
    if let Some(node) = any.downcast_ref::<GpuFunctionHostOccurrenceExec>() {
        return Some(NativeNode::Occurrence(node));
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
            Self::Fragment(node) => (!node.materializes_host()).then_some(node as &dyn GpuResidentProducer),
            Self::Admission(_) | Self::Occurrence(_) => None,
            Self::RelationFunction(node) => Some(node),
            #[cfg(feature = "cugraph")]
            Self::Graph(node) => node.gpu_resident_producer(),
            #[cfg(test)]
            Self::TestDouble { producer, .. } => producer,
        }
    }

    /// The input of a root wrapper that adds no rows or GPU work of its own:
    /// the admission node and a GPU function occurrence.
    pub(crate) fn transparent_root_wrapper_input(&self) -> Option<&'a Arc<dyn ExecutionPlan>> {
        match *self {
            Self::Admission(node) => Some(node.input()),
            Self::Occurrence(node) => Some(node.input()),
            _ => None,
        }
    }

    /// The node's graph execution, when it runs a graph algorithm.
    pub(crate) fn graph_execution_plan(&self) -> Option<&'a dyn GraphExecutionPlan> {
        match *self {
            Self::Fragment(_) | Self::Admission(_) | Self::Occurrence(_) | Self::RelationFunction(_) => None,
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
