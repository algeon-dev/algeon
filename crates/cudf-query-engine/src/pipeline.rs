//! Symbolic native pipeline compiler model.
//!
//! This module is deliberately below the DataFusion adapter boundary. It models
//! generic pipeline tasks, dependencies, barriers, resident resources, sink
//! boundaries, and reservation gates without naming bounded executor families or
//! calling cuDF operators.

mod compile;
mod display;
mod model;
mod validate;

pub use model::{
    CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
    PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
    ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
};
