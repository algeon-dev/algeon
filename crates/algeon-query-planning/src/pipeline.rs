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
mod shadow;
mod validate;

pub use compile::compile_pipeline;
pub(crate) use compile::{COMPILED_PIPELINE_GRAPH_NOT_CONSTRUCTIBLE, PipelineCompilation, compile_pipeline_outcome};
pub use model::{
    AcceptedInputForm, AggregateExecutionStage, CompiledPipeline, OrderingScope, PipelineBarrier, PipelineBarrierId,
    PipelineBarrierKind, PipelineEdge, PipelineEdgeId, PipelineEdgeKind, PipelineTask, PipelineTaskId,
    PipelineTaskKind, ResidentDataForm, ResidentLifetime, ResidentResource, ResidentResourceId, ResourceOrdering,
    ResourceSharing, SinkBoundary, SinkBoundaryKind, SortStrategy,
};
pub use shadow::{
    ShadowBytes, ShadowMemoryModels, ShadowPipelineEstimate, ShadowResourceLifetime, ShadowResourceModel,
    ShadowTaskEnvelope,
};
