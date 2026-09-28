//! Budget tokens for bounded execution boundaries.

use std::{borrow::Cow, collections::BTreeMap};

use super::{CudaStream, ExecutionOverrides, Runtime};
use crate::core::{
    error::{BudgetExhaustionDetail, Error, ErrorCode, Result},
    memory::{DeviceMemoryResource, ReservationUsage, StreamReservationLease},
};

/// Request for a bounded operation execution.
///
/// `budget_bytes` records the caller's stage request. A token over a shared
/// lease retains that request without changing the member's allocation limit.
/// The request does not predict temporary workspace allocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBudgetRequest {
    budget_bytes: usize,
    boundary: Cow<'static, str>,
}

impl ExecutionBudgetRequest {
    /// Create a request with an unnamed execution boundary.
    pub fn new(budget_bytes: usize) -> Self {
        Self { budget_bytes, boundary: Cow::Borrowed("unspecified") }
    }

    /// Attach a stable boundary label for diagnostics.
    pub fn with_boundary(mut self, boundary: impl Into<Cow<'static, str>>) -> Self {
        self.boundary = boundary.into();
        self
    }

    /// Bytes requested for this execution stage.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Diagnostic boundary label supplied by the caller.
    pub fn boundary(&self) -> &str {
        &self.boundary
    }
}

/// How a budget request was enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionBudgetEnforcement {
    /// Allocations are charged through a stream member under a query memory domain.
    QueryDomainMember,
}

impl ExecutionBudgetEnforcement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::QueryDomainMember => "query_domain_member",
        }
    }
}

/// Structured facts recorded for a budgeted execution attempt.
///
/// A report exists only for an operation that completed while its stream
/// reservation was active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBudgetReport {
    request: ExecutionBudgetRequest,
    member_limit_bytes: Option<usize>,
    allocator_high_water_bytes: Option<usize>,
    allocator_current_bytes: Option<usize>,
    allocator_peak_aligned_bytes: Option<usize>,
    retained_bytes: Option<usize>,
    output_bytes: Option<usize>,
    output_rows: Option<usize>,
    context_facts: BTreeMap<String, String>,
}

impl ExecutionBudgetReport {
    pub(crate) fn reservation_enforced(request: ExecutionBudgetRequest) -> Self {
        Self {
            request,
            member_limit_bytes: None,
            allocator_high_water_bytes: None,
            allocator_current_bytes: None,
            allocator_peak_aligned_bytes: None,
            retained_bytes: None,
            output_bytes: None,
            output_rows: None,
            context_facts: BTreeMap::new(),
        }
    }

    pub(crate) fn with_allocator_usage(mut self, usage: ReservationUsage) -> Self {
        self.member_limit_bytes = usize::try_from(usage.allocation_limit_bytes).ok();
        self.allocator_high_water_bytes = Some(usage.peak_bytes as usize);
        self.allocator_current_bytes = Some(usage.current_bytes as usize);
        self.allocator_peak_aligned_bytes = Some(usage.peak_aligned_bytes as usize);
        self
    }

    pub(crate) fn with_retained_bytes(mut self, retained_bytes: usize) -> Self {
        self.retained_bytes = Some(retained_bytes);
        self
    }

    pub(crate) fn with_output_bytes(mut self, output_bytes: usize) -> Self {
        self.output_bytes = Some(output_bytes);
        self
    }

    pub(crate) fn with_output_rows(mut self, output_rows: usize) -> Self {
        self.output_rows = Some(output_rows);
        self
    }

    pub(crate) fn with_context_facts<I, K, V>(mut self, facts: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: ToString,
    {
        self.context_facts.extend(facts.into_iter().map(|(key, value)| (key.into(), value.to_string())));
        self
    }

    pub fn request(&self) -> &ExecutionBudgetRequest {
        &self.request
    }

    pub fn enforcement(&self) -> ExecutionBudgetEnforcement {
        ExecutionBudgetEnforcement::QueryDomainMember
    }

    /// Allocator-enforced member limit, when its usage snapshot was available.
    pub fn member_limit_bytes(&self) -> Option<usize> {
        self.member_limit_bytes
    }

    pub fn allocator_high_water_bytes(&self) -> Option<usize> {
        self.allocator_high_water_bytes
    }

    pub fn allocator_current_bytes(&self) -> Option<usize> {
        self.allocator_current_bytes
    }

    pub fn allocator_peak_aligned_bytes(&self) -> Option<usize> {
        self.allocator_peak_aligned_bytes
    }

    pub fn retained_bytes(&self) -> Option<usize> {
        self.retained_bytes
    }

    pub fn output_bytes(&self) -> Option<usize> {
        self.output_bytes
    }

    pub fn output_rows(&self) -> Option<usize> {
        self.output_rows
    }

    pub fn context_facts(&self) -> &BTreeMap<String, String> {
        &self.context_facts
    }
}

/// Result wrapper for an operation that executed with a budget token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedExecutionOutcome<T> {
    value: T,
    report: ExecutionBudgetReport,
}

impl<T> BudgetedExecutionOutcome<T> {
    pub(crate) fn new(value: T, report: ExecutionBudgetReport) -> Self {
        Self { value, report }
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn report(&self) -> &ExecutionBudgetReport {
        &self.report
    }

    pub fn into_value(self) -> T {
        self.value
    }

    pub fn into_parts(self) -> (T, ExecutionBudgetReport) {
        (self.value, self.report)
    }
}

/// Execution-stage token sharing a query-domain stream member.
///
/// Every token owns a clone of the exact [`Runtime`] supplied to its
/// constructor, including its retained lifetimes. On drop, the lease clone
/// releases before the retained runtime. The final member owner detaches it.
#[derive(Debug)]
pub struct ExecutionBudgetToken {
    // Field order is load-bearing: detach the reservation and release its
    // query-domain resources before dropping the retained Runtime whose
    // domain owns the originating stream, MR, and retained query lifetimes.
    lease: StreamReservationLease,
    request: ExecutionBudgetRequest,
    _retained_runtime: Runtime,
}

impl ExecutionBudgetToken {
    /// Create a token sharing an existing query-domain stream member.
    ///
    /// The token retains the exact runtime and shares the member already
    /// attached to the lease's stream. Its stage request does not change the
    /// member's allocation limit or its parent domain grant. Dropping the
    /// token releases its lease clone; the member detaches when its final
    /// owner drops.
    pub fn from_shared_lease(
        runtime: &Runtime,
        request: ExecutionBudgetRequest,
        lease: StreamReservationLease,
    ) -> Self {
        Self { lease, request, _retained_runtime: runtime.clone() }
    }

    pub fn request(&self) -> &ExecutionBudgetRequest {
        &self.request
    }

    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        self.lease.memory_resource()
    }

    /// How this token's hard budget is enforced.
    pub fn enforcement(&self) -> ExecutionBudgetEnforcement {
        ExecutionBudgetEnforcement::QueryDomainMember
    }

    /// Stream that owns this token's active reservation.
    pub fn stream(&self) -> &CudaStream {
        self.lease.stream()
    }

    /// Borrow the stream reservation that enforces this token's allocation limit.
    pub fn stream_reservation(&self) -> &StreamReservationLease {
        &self.lease
    }

    /// Snapshot reservation usage counters.
    pub fn reservation_usage(&self) -> Result<ReservationUsage> {
        self.lease.usage()
    }

    /// Read the allocation limit enforced by this token's member lease.
    ///
    /// This can differ from the stage request. Reading the limit does not
    /// change the member or its parent domain.
    pub fn member_limit_bytes(&self) -> Result<usize> {
        let usage = self.lease.usage()?;
        usize::try_from(usage.allocation_limit_bytes).map_err(|_| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "runtime::budget::ExecutionBudgetToken::member_limit_bytes",
                "member allocation limit does not fit the platform allocation size",
            )
            .with_detail("allocation_limit_bytes", usage.allocation_limit_bytes.to_string())
        })
    }

    /// Borrow this token as an execution override for one operation.
    pub fn execution_overrides(&self) -> ExecutionOverrides<'_> {
        self.lease.execution_overrides()
    }

    pub(crate) fn reservation_enforced_report(&self) -> ExecutionBudgetReport {
        let report = ExecutionBudgetReport::reservation_enforced(self.request.clone());
        match self.lease.usage() {
            Ok(usage) => report.with_allocator_usage(usage),
            Err(_) => report,
        }
    }

    pub(crate) fn annotate_error(&self, err: Error, operation: &'static str) -> Error {
        let err = err
            .reanchor_if_needed(operation)
            .with_detail_if_absent("budget_operation", operation)
            .with_detail_if_absent("budget_boundary", self.request.boundary().to_string())
            .with_detail_if_absent("budget_enforcement", self.enforcement().as_str())
            .with_detail_if_absent("budget_status", "exceeded_or_lower_layer_rejected");
        match self.lease.usage() {
            Ok(usage) => {
                let Ok(member_limit_bytes) = usize::try_from(usage.allocation_limit_bytes) else {
                    return err;
                };
                let detail = BudgetExhaustionDetail::new(
                    member_limit_bytes,
                    usage.peak_bytes as usize,
                    usage.peak_aligned_bytes as usize,
                    usage.current_bytes as usize,
                    usage.largest_failed_allocation_bytes as usize,
                    usage.failed_allocation_count as usize,
                    self.request.boundary().to_string(),
                    operation,
                );
                err.with_budget_exhaustion_if_absent(detail)
                    .with_detail_if_absent("budget_bytes", member_limit_bytes.to_string())
                    .with_detail_if_absent("budget_peak_bytes", usage.peak_bytes.to_string())
                    .with_detail_if_absent("budget_current_bytes", usage.current_bytes.to_string())
                    .with_detail_if_absent("budget_peak_aligned_bytes", usage.peak_aligned_bytes.to_string())
                    .with_detail_if_absent("budget_current_aligned_bytes", usage.current_aligned_bytes.to_string())
                    .with_detail_if_absent("budget_failed_allocation_count", usage.failed_allocation_count.to_string())
                    .with_detail_if_absent(
                        "budget_largest_failed_allocation_bytes",
                        usage.largest_failed_allocation_bytes.to_string(),
                    )
            },
            Err(_) => err,
        }
    }
}
