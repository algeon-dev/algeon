//! Budget tokens for bounded execution boundaries.

use std::{borrow::Cow, collections::BTreeMap};

use super::{ExecutionOverrides, Runtime};
use crate::core::{
    error::{Error, Result},
    memory::{DeviceMemoryResource, DeviceMemoryResourceUsage},
};

/// Request for a bounded operation execution.
///
/// `budget_bytes` is enforced by wrapping an existing device memory resource
/// with an RMM limiting adaptor. It is not a claim that cudf-rs or libcudf can
/// preflight every temporary workspace allocation for the operation.
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

    /// Maximum live bytes allowed through the budget token's limiting memory
    /// resource.
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
    /// Allocations were routed through an RMM limiting resource adaptor.
    HardMemoryResourceLimit,
}

impl ExecutionBudgetEnforcement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HardMemoryResourceLimit => "hard_memory_resource_limit",
        }
    }
}

/// Outcome status for a budgeted execution attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionBudgetStatus {
    /// The operation completed while allocations were routed through the hard
    /// budget limit.
    HardEnforced,
}

impl ExecutionBudgetStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HardEnforced => "hard_enforced",
        }
    }
}

/// Structured facts recorded for a budgeted execution attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBudgetReport {
    request: ExecutionBudgetRequest,
    enforcement: ExecutionBudgetEnforcement,
    status: ExecutionBudgetStatus,
    allocator_high_water_bytes: Option<usize>,
    allocator_current_bytes: Option<usize>,
    allocator_peak_aligned_bytes: Option<usize>,
    retained_bytes: Option<usize>,
    output_bytes: Option<usize>,
    output_rows: Option<usize>,
    unresolved_contract: Option<Cow<'static, str>>,
    context_facts: BTreeMap<String, String>,
}

impl ExecutionBudgetReport {
    pub(crate) fn hard_enforced(request: ExecutionBudgetRequest) -> Self {
        Self {
            request,
            enforcement: ExecutionBudgetEnforcement::HardMemoryResourceLimit,
            status: ExecutionBudgetStatus::HardEnforced,
            allocator_high_water_bytes: None,
            allocator_current_bytes: None,
            allocator_peak_aligned_bytes: None,
            retained_bytes: None,
            output_bytes: None,
            output_rows: None,
            unresolved_contract: None,
            context_facts: BTreeMap::new(),
        }
    }

    pub(crate) fn with_allocator_usage(mut self, usage: DeviceMemoryResourceUsage) -> Self {
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

    pub(crate) fn with_unresolved_contract(mut self, unresolved_contract: &'static str) -> Self {
        self.unresolved_contract = Some(Cow::Borrowed(unresolved_contract));
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
        self.enforcement
    }

    pub fn status(&self) -> ExecutionBudgetStatus {
        self.status
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

    pub fn unresolved_contract(&self) -> Option<&str> {
        self.unresolved_contract.as_deref()
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

/// Hard execution-budget token backed by a budget-tracking memory resource.
#[derive(Debug, Clone)]
pub struct ExecutionBudgetToken {
    request: ExecutionBudgetRequest,
    memory_resource: DeviceMemoryResource,
}

impl ExecutionBudgetToken {
    /// Create a budget token by wrapping `upstream` in a hard-limiting,
    /// usage-tracking memory resource.
    pub fn new(upstream: &DeviceMemoryResource, request: ExecutionBudgetRequest) -> Result<Self> {
        let memory_resource = upstream.budget_tracking_limit(request.budget_bytes())?;
        Ok(Self { request, memory_resource })
    }

    /// Create a budget token using a runtime's default memory resource as the
    /// upstream allocator.
    pub fn from_runtime(runtime: &Runtime, request: ExecutionBudgetRequest) -> Result<Self> {
        Self::new(runtime.memory_resource(), request)
    }

    pub fn request(&self) -> &ExecutionBudgetRequest {
        &self.request
    }

    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        &self.memory_resource
    }

    /// Borrow this token as an execution override for one operation.
    pub fn execution_overrides(&self) -> ExecutionOverrides<'_> {
        ExecutionOverrides::new().with_memory_resource(&self.memory_resource)
    }

    pub(crate) fn hard_enforced_report(&self) -> ExecutionBudgetReport {
        let report = ExecutionBudgetReport::hard_enforced(self.request.clone());
        match self.memory_resource.budget_tracking_usage() {
            Ok(usage) => report.with_allocator_usage(usage),
            Err(_) => report,
        }
    }

    pub(crate) fn annotate_error(&self, err: Error, operation: &'static str) -> Error {
        let err = err
            .reanchor_if_needed(operation)
            .with_detail_if_absent("budget_operation", operation)
            .with_detail_if_absent("budget_boundary", self.request.boundary().to_string())
            .with_detail_if_absent("budget_bytes", self.request.budget_bytes().to_string())
            .with_detail_if_absent("budget_enforcement", ExecutionBudgetEnforcement::HardMemoryResourceLimit.as_str())
            .with_detail_if_absent("budget_status", "exceeded_or_lower_layer_rejected");
        match self.memory_resource.budget_tracking_usage() {
            Ok(usage) => err
                .with_detail_if_absent("budget_peak_bytes", usage.peak_bytes.to_string())
                .with_detail_if_absent("budget_current_bytes", usage.current_bytes.to_string())
                .with_detail_if_absent("budget_peak_aligned_bytes", usage.peak_aligned_bytes.to_string())
                .with_detail_if_absent("budget_current_aligned_bytes", usage.current_aligned_bytes.to_string())
                .with_detail_if_absent("budget_failed_allocation_count", usage.failed_allocation_count.to_string())
                .with_detail_if_absent(
                    "budget_largest_failed_allocation_bytes",
                    usage.largest_failed_allocation_bytes.to_string(),
                ),
            Err(_) => err,
        }
    }
}
