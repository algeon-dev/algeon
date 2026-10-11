use query_planning::plan::NodeId;

use super::ResidentFrameStreamPayload;
use crate::exec::bounded::{
    BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame,
};

/// GPU-resident payload stored in the compiled scheduler resource table.
///
/// The payload form is decided by the producing task and settled against the
/// resource's declared data form before it is stored; consumers accept the
/// forms their operator can read and reject the rest as a contract error.
pub struct CompiledGpuResource {
    pub(super) payload: CompiledGpuResourcePayload,
}

// Payloads encode which tasks may consume a value and whether it owns resident storage.
pub(super) enum CompiledGpuResourcePayload {
    /// One complete frame. `applied_limit` names the limit node a fused top-k
    /// already applied; the limit task then passes the frame through, because
    /// applying the limit again would skip the offset rows a second time.
    ResidentFrame { frame: Box<ResidentGpuFrame>, applied_limit: Option<NodeId> },
    /// Frames pulled on demand; needs the attempt to release.
    ResidentFrameStream(ResidentFrameStreamPayload),
    /// A planned source read that has not started. Holds no device memory, so
    /// it releases by `Drop`.
    JoinLeafLocalParquetSource(Box<BoundedJoinLeafLocalParquetSourceStream>),
    /// A source read already running on a pool worker. Release must join or
    /// abandon that worker through the attempt.
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
