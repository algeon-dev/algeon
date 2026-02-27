//! Internal flattening of [`NativeDataFusionRuntime`] enum variants into the
//! flat `(outcome, failure, output_*, metrics)` shape that report rows store.

use super::{
    NativeDataFusionExecutionMetrics, NativeDataFusionRuntime, NativeDataFusionRuntimeOutcome, tsv::stable_tsv_field,
};

#[derive(Debug, Clone)]
pub(super) struct NativeDataFusionRuntimeSummary {
    pub(super) started: bool,
    pub(super) outcome: NativeDataFusionRuntimeOutcome,
    pub(super) failure: Option<String>,
    pub(super) output_batches: Option<usize>,
    pub(super) output_rows: Option<usize>,
    pub(super) output_bytes: Option<usize>,
    pub(super) metrics: NativeDataFusionExecutionMetrics,
}

impl NativeDataFusionRuntimeSummary {
    pub(super) fn from_runtime(runtime: NativeDataFusionRuntime) -> Self {
        match runtime {
            NativeDataFusionRuntime::NotStarted => Self {
                started: false,
                outcome: NativeDataFusionRuntimeOutcome::NotStarted,
                failure: None,
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                metrics: NativeDataFusionExecutionMetrics::default(),
            },
            NativeDataFusionRuntime::Succeeded { output_batches, output_rows, output_bytes, metrics } => Self {
                started: true,
                outcome: NativeDataFusionRuntimeOutcome::Succeeded,
                failure: None,
                output_batches: Some(output_batches),
                output_rows: Some(output_rows),
                output_bytes: Some(output_bytes),
                metrics: NativeDataFusionExecutionMetrics::from_metrics(&metrics),
            },
            NativeDataFusionRuntime::Failed { message } => Self {
                started: true,
                outcome: NativeDataFusionRuntimeOutcome::Failed,
                failure: Some(stable_tsv_field(&message)),
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                metrics: NativeDataFusionExecutionMetrics::default(),
            },
        }
    }
}
