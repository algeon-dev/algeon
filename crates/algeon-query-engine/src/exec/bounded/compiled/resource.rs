use super::ResidentFrameStreamPayload;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    exec::bounded::{
        BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame,
    },
    plan::NodeId,
    runtime::resident::GpuDataFrame,
};

const OPERATION: &str = "query_engine.exec.bounded.compiled.lane";

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}

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

    pub fn num_rows(&self) -> usize {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.num_rows(),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.num_rows() as usize,
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => 0,
        }
    }

    pub fn num_columns(&self) -> usize {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.num_columns(),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.num_columns(),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => 0,
        }
    }

    /// Snapshots a shared input for lane-claim scratch state.
    ///
    /// Lane-family pipelines currently resolve single-consumer inputs by move;
    /// this covers shared inputs without aliasing coordinator state, and fails
    /// closed on lazy or deferred payloads that cannot cross the boundary.
    pub(in crate::exec::bounded) fn snapshot_for_lane_claim(&self) -> Result<CompiledGpuResource> {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, applied_limit: None } => {
                Ok(CompiledGpuResource::new(ResidentGpuFrame {
                    frame: frame.frame.clone(),
                    device_bytes: frame.device_bytes(),
                }))
            },
            // A fused limit is a terminal-executor marker (pass-through vs. re-slicing); rebuilding
            // via `CompiledGpuResource::new` would drop it and risk re-applying the limit (double OFFSET).
            CompiledGpuResourcePayload::ResidentFrame { applied_limit: Some(limit), .. } => {
                Err(invalid("compiled lane task cannot snapshot a resident frame with a fused limit")
                    .with_fact("applied_limit_node_id", *limit))
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                let Some(alias) = stream.try_alias() else {
                    return Err(invalid("compiled lane task cannot snapshot a lazy shared frame stream"));
                };
                Ok(CompiledGpuResource::frame_stream_payload(alias))
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                Err(invalid("compiled lane task cannot share a deferred join leaf source"))
            },
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => {
                Err(invalid("compiled lane task cannot share a pending source read"))
            },
        }
    }

    pub(in crate::exec::bounded) fn clone_frame(&self) -> GpuDataFrame {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.clone(),
            CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                unreachable!("resident frame streams must be consumed by stream-aware compiled tasks")
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => {
                unreachable!("deferred join leaf sources must be consumed by join probe tasks")
            },
        }
    }

    pub(in crate::exec::bounded) fn reservation_bytes(&self) -> u64 {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.device_bytes(),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.live_reservation_bytes(),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => 0,
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => prefetch.reservation_bytes(),
        }
    }

    pub(super) fn as_join_leaf_local_parquet_source(&self) -> Option<&BoundedJoinLeafLocalParquetSourceStream> {
        match &self.payload {
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => Some(stream),
            CompiledGpuResourcePayload::ResidentFrame { .. }
            | CompiledGpuResourcePayload::ResidentFrameStream(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => None,
        }
    }

    pub(super) fn applied_limit(&self) -> Option<NodeId> {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { applied_limit, .. } => *applied_limit,
            CompiledGpuResourcePayload::ResidentFrameStream(_)
            | CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => None,
        }
    }
}
