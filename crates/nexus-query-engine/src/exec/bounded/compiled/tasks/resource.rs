use super::*;

/// GPU-resident payload stored in the compiled scheduler resource table.
#[doc(hidden)]
pub struct CompiledGpuResource {
    pub(super) payload: CompiledGpuResourcePayload,
}

// Resource-table variants with different ownership and consumption contracts.
pub(super) enum CompiledGpuResourcePayload {
    ResidentFrame { frame: Box<ResidentGpuFrame>, applied_limit: Option<NodeId> },
    ResidentFrameStream(ResidentFrameStreamPayload),
    JoinLeafLocalParquetSource(Box<BoundedJoinLeafLocalParquetSourceStream>),
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
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => 0,
        }
    }

    pub fn num_columns(&self) -> usize {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.num_columns(),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.num_columns(),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => 0,
        }
    }

    pub(in crate::exec::bounded) fn clone_frame(&self) -> GpuDataFrame {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.frame.clone(),
            CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                unreachable!("resident frame streams must be consumed by stream-aware compiled tasks")
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                unreachable!("deferred join leaf sources must be consumed by join probe tasks")
            },
        }
    }

    pub(in crate::exec::bounded) fn reservation_bytes(&self) -> u64 {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.reservation.bytes(),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.live_reservation_bytes(),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => 0,
        }
    }

    pub(in crate::exec::bounded) fn reservation(&self) -> &ActiveReservation {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => &frame.reservation,
            CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                unreachable!("resident frame streams do not expose a single reservation")
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                unreachable!("deferred join leaf sources do not own a resident reservation")
            },
        }
    }

    pub(super) fn as_join_leaf_local_parquet_source(&self) -> Option<&BoundedJoinLeafLocalParquetSourceStream> {
        match &self.payload {
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => Some(stream),
            CompiledGpuResourcePayload::ResidentFrame { .. } | CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                None
            },
        }
    }

    pub(super) fn applied_limit(&self) -> Option<NodeId> {
        match &self.payload {
            CompiledGpuResourcePayload::ResidentFrame { applied_limit, .. } => *applied_limit,
            CompiledGpuResourcePayload::ResidentFrameStream(_)
            | CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => None,
        }
    }
}
