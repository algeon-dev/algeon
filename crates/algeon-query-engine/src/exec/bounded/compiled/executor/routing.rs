use query_planning::{
    pipeline::{PipelineTaskKind, ResidentDataForm, ResidentResourceId, ResourceSharing},
    plan::{NodeId, PlanNode},
    source::SourceNode,
};

use super::CompiledGpuTaskExecutor;

impl CompiledGpuTaskExecutor<'_> {
    /// The compiled contract for how this resource's consumers read it.
    pub(super) fn resource_sharing(&self, resource: ResidentResourceId) -> ResourceSharing {
        self.pipeline
            .resources()
            .iter()
            .find(|candidate| candidate.id() == resource)
            .map_or(ResourceSharing::SingleConsume, |resource| resource.sharing())
    }

    /// The strongest payload form the compiled contract lets this resource
    /// carry. The producer may still supply a weaker form.
    pub(super) fn resource_data_form(&self, resource: ResidentResourceId) -> ResidentDataForm {
        self.pipeline
            .resources()
            .iter()
            .find(|candidate| candidate.id() == resource)
            .map_or(ResidentDataForm::Frame, |resource| resource.data_form())
    }

    pub(super) fn local_parquet_source_node_for_resource(&self, resource: ResidentResourceId) -> Option<NodeId> {
        let producer = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource)?.producer();
        let source_node_id = self.pipeline.tasks().iter().find(|candidate| candidate.id() == producer)?.plan_node()?;
        self.plan.node(source_node_id).and_then(|node| match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Some(source_node_id),
            _ => None,
        })
    }

    // The materialize task preserves chunks only when the compiled contract
    // lets its output stay a stream and one consumer owns it.
    pub(super) fn join_probe_output_consumed_as_stream(&self, node_id: NodeId) -> bool {
        let Some(materialize) = self
            .pipeline
            .tasks()
            .iter()
            .find(|task| task.kind() == PipelineTaskKind::JoinMaterialize && task.plan_node() == Some(node_id))
        else {
            return false;
        };
        let [output] = materialize.output_resources() else {
            return false;
        };
        self.resource_data_form(*output) == ResidentDataForm::FrameStream
            && self.resource_sharing(*output) == ResourceSharing::SingleConsume
    }
}
