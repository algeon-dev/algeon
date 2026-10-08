use super::{
    BoundedJoinSideFrame, CompiledJoinIndices, GpuDataFrame, ResidentGpuFrame, release_bounded_join_side_frame,
};

/// Owns the side frames and gather maps of one join materialization until the owner ends it.
///
/// `with_sides` takes the left input (index 0) and the right input (index 1), and `set_indices`
/// adds the gather maps computed from them. The output frame is never stored here. Both
/// `commit_frame` and `rollback` release the gather maps first and then the sides from last to
/// first, before the caller's next allocation on the query domain; `commit_frame` additionally
/// returns the caller's output as a `ResidentGpuFrame`.
///
/// A rollback restores no operator state. The error that caused it propagates, and a retry is
/// the caller's decision, for example the partitioned driver that still owns the partition the
/// sides were sliced from. Every path must end in `commit_frame` or `rollback`; dropping a
/// transaction that still holds resources fails a `debug_assert!`.
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

    /// Stores the gather maps. A transaction holds one set; a second call panics.
    pub(in crate::exec::bounded::join::compiled) fn set_indices(&mut self, indices: CompiledJoinIndices) {
        assert!(self.indices.is_none(), "join materialization txn indices already set");
        self.indices = Some(indices);
    }

    pub(in crate::exec::bounded::join::compiled) fn indices(&self) -> &CompiledJoinIndices {
        self.indices.as_ref().expect("join materialization transaction indices should be set")
    }

    /// Releases the gather maps while keeping the sides. The one-shot residual path calls it once
    /// the candidate frame exists, which frees the maps before the residual filter allocates.
    pub(in crate::exec::bounded::join::compiled) fn drop_indices(&mut self) {
        self.indices = None;
    }

    /// Releases the gather maps and sides, then returns `frame` as the join output. `device_bytes`
    /// is the output size the caller observed.
    pub(in crate::exec) fn commit_frame(mut self, frame: GpuDataFrame, device_bytes: u64) -> ResidentGpuFrame {
        self.release_uncommitted();
        ResidentGpuFrame::with_device_bytes(frame, device_bytes)
    }

    /// Releases the gather maps and sides without producing an output.
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
