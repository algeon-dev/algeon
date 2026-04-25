use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet, VecDeque},
    sync::Arc,
};

use super::*;
use crate::{
    adaptive_runtime::{AdaptiveMemoryHistory, AdaptiveTaskMemoryHistoryShape},
    metrics::{
        CompiledTaskId, MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome,
        PlanNodeId, QueryId, ReservationProjectionLedgerHandle,
    },
    mode::NativePipelineExecutionMode,
    observability::{
        RuntimeEventContext, RuntimeExecutionObservationScope, RuntimeObservationPayload, RuntimeReservationAction,
        RuntimeReservationDisposition, RuntimeReservationId, RuntimeSpanId, TaskAttemptId,
    },
    pipeline::CompiledPipeline,
    plan::QueryPlan,
    runtime::{
        ExecutionContext, MemoryGrant, MemorySpaceId, cache::local_parquet::*,
        gpu::governor::MemoryGrantLiveReservation,
    },
};

mod active;
mod diagnostics;
mod observation;
mod tracker;

#[cfg(test)]
mod tests;

pub(in crate::exec::bounded) use active::reconcile_work_reservation_to_output;
// Sibling modules resolve shared private items via `use super::*;`.
use active::*;
pub(in crate::exec) use active::{ActiveReservation, ActiveReservationSemantics, CommittedPartial, ResidentGpuFrame};
pub(in crate::exec) use diagnostics::ReservationDiagnosticScope;
use diagnostics::*;
use observation::*;
pub(in crate::exec) use tracker::ReservationTracker;
