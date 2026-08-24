use super::{
    BoundedJoinSideFrame, CompiledJoinIndices, GpuDataFrame, ResidentGpuFrame, release_bounded_join_side_frame,
};

/// Transaction boundary for native join materialization owners.
#[must_use]
pub(in crate::exec) struct JoinMaterializationTxn {
    sides: Vec<BoundedJoinSideFrame>,
    indices: Option<CompiledJoinIndices>,
}

impl JoinMaterializationTxn {
    pub(in crate::exec) fn new() -> Self {
        Self { sides: Vec::new(), indices: None }
    }

    pub(in crate::exec::bounded) fn with_sides(left: BoundedJoinSideFrame, right: BoundedJoinSideFrame) -> Self {
        let mut txn = Self::new();
        txn.push_side(left);
        txn.push_side(right);
        txn
    }

    pub(in crate::exec::bounded) fn push_side(&mut self, side: BoundedJoinSideFrame) {
        self.sides.push(side);
    }

    pub(in crate::exec::bounded) fn side(&self, index: usize) -> &BoundedJoinSideFrame {
        self.sides.get(index).expect("join materialization transaction side index should exist")
    }

    pub(in crate::exec::bounded) fn sides(&self) -> (&BoundedJoinSideFrame, &BoundedJoinSideFrame) {
        (self.side(0), self.side(1))
    }

    pub(in crate::exec::bounded::join::compiled) fn set_indices(&mut self, indices: CompiledJoinIndices) {
        assert!(self.indices.is_none(), "join materialization txn indices already set");
        self.indices = Some(indices);
    }

    pub(in crate::exec::bounded::join::compiled) fn indices(&self) -> &CompiledJoinIndices {
        self.indices.as_ref().expect("join materialization transaction indices should be set")
    }

    pub(in crate::exec::bounded::join::compiled) fn drop_indices(&mut self) {
        self.indices = None;
    }

    pub(in crate::exec) fn commit_frame(mut self, frame: GpuDataFrame, device_bytes: u64) -> ResidentGpuFrame {
        self.release_uncommitted();
        ResidentGpuFrame::with_device_bytes(frame, device_bytes)
    }

    pub(in crate::exec) fn rollback(mut self) {
        self.release_uncommitted();
    }

    fn release_uncommitted(&mut self) {
        self.indices = None;
        while let Some(side) = self.sides.pop() {
            release_bounded_join_side_frame(side);
        }
    }

    fn is_empty(&self) -> bool {
        self.sides.is_empty() && self.indices.is_none()
    }
}

impl Drop for JoinMaterializationTxn {
    fn drop(&mut self) {
        debug_assert!(
            self.is_empty(),
            "join materialization transaction dropped with unreleased resources; call rollback or commit"
        );
    }
}
