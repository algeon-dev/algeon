//! Native execution output assembly.

use arrow::record_batch::RecordBatch;

use crate::{metrics::QueryMetrics, plan::ResultSink, runtime::resident::GpuResultHandle};

/// Output residency requested by a query sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryOutputMode {
    /// Keep the query output as a runtime/device-affine GPU handle.
    GpuResident,
    /// Materialize the query output to a host Arrow record batch.
    MaterializeToHost,
}

impl QueryOutputMode {
    /// Derive the output mode requested by a query plan sink.
    pub fn for_sink(sink: &ResultSink) -> Self {
        match sink {
            ResultSink::GpuResident { .. } => Self::GpuResident,
            ResultSink::HostArrow { .. } => Self::MaterializeToHost,
        }
    }

    /// Stable lowercase label used in metrics and diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuResident => "gpu_resident",
            Self::MaterializeToHost => "materialize_to_host",
        }
    }
}

/// Output produced by native query execution.
///
/// The variant is driven by the query plan's sink. Host Arrow output has
/// crossed the device-to-host boundary; GPU-resident output remains bound to
/// its originating runtime and must be materialized explicitly if a caller
/// needs Arrow.
pub enum QueryOutput {
    /// GPU-resident result bound to its originating runtime/device.
    GpuResident(GpuResultHandle),
    /// Host Arrow result after explicit device-to-host materialization.
    HostArrow(RecordBatch),
}

impl QueryOutput {
    /// Output residency mode represented by this value.
    pub fn mode(&self) -> QueryOutputMode {
        match self {
            Self::GpuResident(_) => QueryOutputMode::GpuResident,
            Self::HostArrow(_) => QueryOutputMode::MaterializeToHost,
        }
    }

    /// Borrow the host Arrow batch when the result was materialized to host.
    pub fn as_host_arrow(&self) -> Option<&RecordBatch> {
        match self {
            Self::HostArrow(batch) => Some(batch),
            Self::GpuResident(_) => None,
        }
    }

    /// Borrow the GPU handle when the result stayed device-resident.
    pub fn as_gpu_resident(&self) -> Option<&GpuResultHandle> {
        match self {
            Self::GpuResident(frame) => Some(frame),
            Self::HostArrow(_) => None,
        }
    }
}

/// Native query output plus metrics collected during that execution.
///
/// Metrics ownership stays with the result so adapters can record them after
/// deciding whether to pass through a GPU handle or materialize host output.
pub struct QueryExecutionResult {
    output: QueryOutput,
    metrics: QueryMetrics,
}

impl QueryExecutionResult {
    /// Create an execution result with its collected metrics.
    pub fn new(output: QueryOutput, metrics: QueryMetrics) -> Self {
        Self { output, metrics }
    }

    /// Borrow the query output without consuming metrics.
    pub fn output(&self) -> &QueryOutput {
        &self.output
    }

    /// Borrow metrics collected during query execution.
    pub fn metrics(&self) -> &QueryMetrics {
        &self.metrics
    }

    /// Consume the result into output and metrics.
    pub fn into_parts(self) -> (QueryOutput, QueryMetrics) {
        (self.output, self.metrics)
    }
}
