//! Symbolic native pipeline compiler model.
//!
//! This module is deliberately below the DataFusion adapter boundary. It models
//! generic pipeline tasks, dependencies, barriers, resident resources, sink
//! boundaries without naming bounded executor families or calling cuDF
//! operators.

mod compile;
mod contract;
mod display;
mod model;
mod physical;
mod shadow;
mod validate;

pub use compile::compile_pipeline;
pub(crate) use compile::{PipelineCompilation, compile_pipeline_outcome};
pub use model::{
    AcceptedInputForm, CompiledPipeline, OrderingScope, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind,
    PipelineEdge, PipelineEdgeId, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentDataForm,
    ResidentLifetime, ResidentResource, ResidentResourceId, ResourceOrdering, ResourceSharing, SinkBoundary,
    SinkBoundaryKind, SortStrategy,
};
pub use physical::{
    AggregateInputHints, AggregatePlan, JoinPlan, SourceOutput, SourceReadPlan, TaskPhysical, TopKFusion,
};
pub use shadow::{
    ShadowBytes, ShadowMemoryModels, ShadowPipelineEstimate, ShadowResourceLifetime, ShadowResourceModel,
    ShadowTaskEnvelope,
};
