//! DataFusion statistics translated at the native output occurrence boundary.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::stats::Precision,
    physical_plan::{ExecutionPlan, StatisticsArgs, StatisticsContext},
};
use query_engine::plan::{
    AdvisoryOutputBinding, AdvisoryOutputStatistics, AdvisoryStatistic, AdvisoryUnknownReason, NodeId, QueryPlanBuilder,
};

use super::LoweredNativeInput;
use crate::native::normalize::{NormalizedNativeInput, NormalizedNativeLiteralSource};

pub(super) fn output_statistics(plan: &dyn ExecutionPlan) -> Option<AdvisoryOutputStatistics> {
    let statistics = StatisticsContext::new().compute(plan, &StatisticsArgs::new()).ok()?;
    Some(AdvisoryOutputStatistics {
        rows: lower_statistic(statistics.num_rows),
        bytes: lower_statistic(statistics.total_byte_size),
    })
}

fn lower_statistic(precision: Precision<usize>) -> AdvisoryStatistic {
    match precision {
        Precision::Exact(value) => AdvisoryStatistic::Exact(value as u64),
        Precision::Inexact(value) => AdvisoryStatistic::Inexact(value as u64),
        Precision::Absent => AdvisoryStatistic::Absent,
    }
}

pub(super) fn attach(builder: &mut QueryPlanBuilder, node: NodeId, schema: &SchemaRef, frontend: &dyn ExecutionPlan) {
    let (mut statistics, mut unknown_reason) = match output_statistics(frontend) {
        Some(statistics) => {
            let reason = (statistics.rows == AdvisoryStatistic::Absent
                || statistics.bytes == AdvisoryStatistic::Absent)
                .then_some(AdvisoryUnknownReason::MissingStatistics);
            (statistics, reason)
        },
        None => (absent(), Some(AdvisoryUnknownReason::StatisticsComputationFailed)),
    };
    // Lowering may prune columns or rename them. Renaming preserves byte layout;
    // projection to a different layout does not preserve frontend output bytes.
    let frontend_schema = frontend.schema();
    if schema.fields().len() != frontend_schema.fields().len()
        || schema.fields().iter().zip(frontend_schema.fields()).any(|(native, frontend)| {
            native.data_type() != frontend.data_type() || native.is_nullable() != frontend.is_nullable()
        })
    {
        statistics.bytes = AdvisoryStatistic::Absent;
        unknown_reason = Some(AdvisoryUnknownReason::OutputSchemaChanged);
    }
    builder.set_advisory_output(AdvisoryOutputBinding {
        node,
        schema: Arc::clone(schema),
        frontend: "DataFusion".to_owned(),
        frontend_version: datafusion::DATAFUSION_VERSION.to_owned(),
        statistics,
        unknown_reason,
    });
}

fn absent() -> AdvisoryOutputStatistics {
    AdvisoryOutputStatistics { rows: AdvisoryStatistic::Absent, bytes: AdvisoryStatistic::Absent }
}

pub(super) fn attach_input(
    builder: &mut QueryPlanBuilder,
    input: &NormalizedNativeInput<'_>,
    output: &LoweredNativeInput,
) {
    let frontend: &dyn ExecutionPlan = match input {
        NormalizedNativeInput::Filter(filter) => filter.filter,
        NormalizedNativeInput::Projection(projection) => projection.projection,
        NormalizedNativeInput::Aggregate(aggregate) => aggregate.statistics_output,
        NormalizedNativeInput::HashJoin(join) => join.join,
        NormalizedNativeInput::NestedLoopJoin(join) => join.join,
        NormalizedNativeInput::CrossJoin(_) => {
            builder.set_advisory_output(AdvisoryOutputBinding {
                node: output.node,
                schema: Arc::clone(&output.schema),
                frontend: "DataFusion".to_owned(),
                frontend_version: datafusion::DATAFUSION_VERSION.to_owned(),
                statistics: absent(),
                unknown_reason: Some(AdvisoryUnknownReason::CartesianFallback),
            });
            return;
        },
        NormalizedNativeInput::Source(source) => {
            if let Some(filter) = source.filter {
                filter
            } else {
                source.scan
            }
        },
        NormalizedNativeInput::LiteralSource(source) => match source {
            NormalizedNativeLiteralSource::PlaceholderRow(plan) => *plan,
            NormalizedNativeLiteralSource::Empty(plan) => *plan,
            NormalizedNativeLiteralSource::Memory(plan) => *plan,
        },
        NormalizedNativeInput::Limit(_)
        | NormalizedNativeInput::GpuResidentSource(_)
        | NormalizedNativeInput::Sort(_)
        | NormalizedNativeInput::Window(_)
        | NormalizedNativeInput::Union(_) => return,
    };
    attach(builder, output.node, &output.schema, frontend);
}
