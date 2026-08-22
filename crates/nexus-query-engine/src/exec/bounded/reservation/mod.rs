use std::cell::RefCell;

use super::*;
use crate::{
    metrics::{
        CompiledTaskId, MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome,
        PlanNodeId, QueryId, ReservationProjectionLedgerHandle,
    },
    observability::{
        RuntimeEventContext, RuntimeExecutionObservationScope, RuntimeObservationPayload, RuntimeReservationAction,
        RuntimeReservationDisposition, RuntimeReservationId, RuntimeSpanId, TaskAttemptId,
    },
    runtime::{ExecutionContext, MemoryGrant, MemorySpaceId, gpu::ledger::MemoryGrantLiveReservation},
};

mod active;
mod diagnostics;
mod observation;
mod tracker;

#[cfg(test)]
mod tests;

pub(in crate::exec::bounded) use active::reconcile_work_reservation_to_output;
// Sibling modules resolve shared private items via `use super::*;`.
pub(in crate::exec) use active::{ActiveReservation, ActiveReservationSemantics, CommittedPartial, ResidentGpuFrame};
pub(in crate::exec) use diagnostics::ReservationDiagnosticScope;
use diagnostics::*;
use observation::*;
pub(in crate::exec) use tracker::ReservationTracker;
