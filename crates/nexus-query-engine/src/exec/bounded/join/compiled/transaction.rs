use super::*;

/// Transaction boundary for bounded join materialization resources.
///
/// Join materialization often owns side-frame reservations, index handles, work
/// reservations, output reservations, and already-produced stream frames at the
/// same time. All normal success and error paths should either commit through
/// this transaction or roll it back so releases flow through
/// `ReservationTracker::release` and remain visible in the reservation
/// timeline. `ActiveReservation::Drop` is still a last-ditch capacity cleanup,
/// but it must not be the intended accounting path for join materialization.
#[must_use]
pub(in crate::exec) struct JoinMaterializationTxn {
    sides: Vec<BoundedJoinSideFrame>,
    gather_map_reservation: Option<ActiveReservation>,
    mixed_size_data_reservation: Option<ActiveReservation>,
    filtered_probe_state_reservation: Option<ActiveReservation>,
    indices: Option<CompiledJoinIndices>,
    output_reservation: Option<ActiveReservation>,
    stream_outputs: Vec<ResidentGpuFrame>,
}

impl JoinMaterializationTxn {
    pub(in crate::exec) fn new() -> Self {
        Self {
            sides: Vec::new(),
            gather_map_reservation: None,
            mixed_size_data_reservation: None,
            filtered_probe_state_reservation: None,
            indices: None,
            output_reservation: None,
            stream_outputs: Vec::new(),
        }
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

    pub(in crate::exec) fn set_gather_map_reservation(&mut self, reservation: ActiveReservation) {
        assert!(self.gather_map_reservation.is_none(), "join materialization txn gather reservation already set");
        self.gather_map_reservation = Some(reservation);
    }

    pub(in crate::exec) fn release_gather_map_reservation(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) {
        if let Some(reservation) = self.gather_map_reservation.take() {
            reservations.release(reservation, metrics);
        }
    }

    pub(in crate::exec) fn set_mixed_size_data_reservation(&mut self, reservation: ActiveReservation) {
        assert!(
            self.mixed_size_data_reservation.is_none(),
            "join materialization txn mixed-size-data reservation already set"
        );
        self.mixed_size_data_reservation = Some(reservation);
    }

    pub(in crate::exec) fn release_mixed_size_data_reservation(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) {
        if let Some(reservation) = self.mixed_size_data_reservation.take() {
            reservations.release(reservation, metrics);
        }
    }

    #[allow(dead_code)]
    pub(in crate::exec::bounded) fn set_filtered_probe_state_reservation(&mut self, reservation: ActiveReservation) {
        assert!(
            self.filtered_probe_state_reservation.is_none(),
            "join materialization txn filtered-probe-state reservation already set"
        );
        self.filtered_probe_state_reservation = Some(reservation);
    }

    pub(in crate::exec::bounded::join::compiled) fn set_indices(&mut self, indices: CompiledJoinIndices) {
        assert!(self.indices.is_none(), "join materialization txn indices already set");
        self.indices = Some(indices);
    }

    pub(in crate::exec::bounded::join::compiled) fn indices(&self) -> &CompiledJoinIndices {
        self.indices.as_ref().expect("join materialization transaction indices should be set")
    }

    pub(in crate::exec) fn set_output_reservation(&mut self, reservation: ActiveReservation) {
        assert!(self.output_reservation.is_none(), "join materialization txn output reservation already set");
        self.output_reservation = Some(reservation);
    }

    pub(in crate::exec) fn take_output_reservation(&mut self) -> ActiveReservation {
        self.output_reservation
            .take()
            .expect("join materialization transaction output reservation should be set before commit")
    }

    pub(in crate::exec) fn output_reservation_mut(&mut self) -> &mut ActiveReservation {
        self.output_reservation.as_mut().expect("join materialization transaction output reservation should be set")
    }

    #[allow(dead_code)]
    pub(in crate::exec::bounded) fn push_stream_output(&mut self, output: ResidentGpuFrame) {
        self.stream_outputs.push(output);
    }

    #[allow(dead_code)]
    pub(in crate::exec::bounded) fn extend_stream_outputs(&mut self, outputs: Vec<ResidentGpuFrame>) {
        self.stream_outputs.extend(outputs);
    }

    pub(in crate::exec) fn commit_frame(
        mut self,
        frame: GpuDataFrame,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> ResidentGpuFrame {
        let output_reservation = self.take_output_reservation();
        self.release_uncommitted(reservations, metrics);
        ResidentGpuFrame { frame, reservation: output_reservation }
    }

    #[allow(dead_code)]
    pub(in crate::exec::bounded) fn commit_stream(
        mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Vec<ResidentGpuFrame> {
        let outputs = std::mem::take(&mut self.stream_outputs);
        self.release_uncommitted(reservations, metrics);
        outputs
    }

    #[allow(dead_code)]
    pub(in crate::exec::bounded) fn commit_resources(
        mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) {
        self.release_uncommitted(reservations, metrics);
    }

    pub(in crate::exec) fn rollback(mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        self.release_uncommitted(reservations, metrics);
    }

    fn release_uncommitted(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        while let Some(output) = self.stream_outputs.pop() {
            drop(output.frame);
            reservations.release(output.reservation, metrics);
        }
        drop(self.indices.take());
        if let Some(reservation) = self.output_reservation.take() {
            reservations.release(reservation, metrics);
        }
        if let Some(reservation) = self.gather_map_reservation.take() {
            reservations.release(reservation, metrics);
        }
        if let Some(reservation) = self.filtered_probe_state_reservation.take() {
            reservations.release(reservation, metrics);
        }
        if let Some(reservation) = self.mixed_size_data_reservation.take() {
            reservations.release(reservation, metrics);
        }
        while let Some(side) = self.sides.pop() {
            release_bounded_join_side_frame(side, reservations, metrics);
        }
    }

    fn is_empty(&self) -> bool {
        self.sides.is_empty()
            && self.gather_map_reservation.is_none()
            && self.mixed_size_data_reservation.is_none()
            && self.filtered_probe_state_reservation.is_none()
            && self.indices.is_none()
            && self.output_reservation.is_none()
            && self.stream_outputs.is_empty()
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
