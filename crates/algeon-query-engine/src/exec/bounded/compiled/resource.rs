use super::ResidentFrameStreamPayload;
use crate::{
    exec::bounded::{
        BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame,
    },
    plan::NodeId,
};

/// GPU-resident payload stored in the compiled scheduler resource table.
#[doc(hidden)]
pub struct CompiledGpuResource {
    pub(super) payload: CompiledGpuResourcePayload,
}

// Payloads encode which tasks may consume a value and whether it owns resident storage.
pub(super) enum CompiledGpuResourcePayload {
    ResidentFrame { frame: Box<ResidentGpuFrame>, applied_limit: Option<NodeId> },
    ResidentFrameStream(ResidentFrameStreamPayload),
    JoinLeafLocalParquetSource(Box<BoundedJoinLeafLocalParquetSourceStream>),
    PendingJoinLeafLocalParquetSource(Box<BoundedJoinLeafLocalParquetSourcePrefetch>),
}

impl CompiledGpuResource {
    pub(in crate::exec::bounded) fn new(frame: ResidentGpuFrame) -> Self {
        Self { payload: CompiledGpuResourcePayload::ResidentFrame { frame: Box::new(frame), applied_limit: None } }
    }

    pub(in crate::exec::bounded) fn frame_with_applied_limit(frame: ResidentGpuFrame, limit: NodeId) -> Self {
        Self {
            payload: CompiledGpuResourcePayload::ResidentFrame { frame: Box::new(frame), applied_limit: Some(limit) },
        }
    }

    pub(in crate::exec::bounded) fn join_leaf_local_parquet_source(
        stream: BoundedJoinLeafLocalParquetSourceStream,
    ) -> Self {
        Self { payload: CompiledGpuResourcePayload::JoinLeafLocalParquetSource(Box::new(stream)) }
    }

    pub(in crate::exec::bounded) fn pending_join_leaf_local_parquet_source(
        prefetch: BoundedJoinLeafLocalParquetSourcePrefetch,
    ) -> Self {
        Self { payload: CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(Box::new(prefetch)) }
    }

    pub(in crate::exec::bounded) fn frame_stream(frames: Vec<ResidentGpuFrame>) -> Self {
        Self { payload: CompiledGpuResourcePayload::ResidentFrameStream(ResidentFrameStreamPayload::new(frames)) }
    }

    pub(in crate::exec::bounded) fn frame_stream_payload(stream: ResidentFrameStreamPayload) -> Self {
        Self { payload: CompiledGpuResourcePayload::ResidentFrameStream(stream) }
    }

    /// Diagnostic name of the payload's form.
    pub(super) fn form_name(&self) -> &'static str {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { .. } => "frame",
            CompiledGpuResourcePayload::ResidentFrameStream(_) => "frame_stream",
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => "source_stream",
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => "pending_source_read",
        }
    }
}
