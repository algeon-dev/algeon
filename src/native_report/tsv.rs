use std::{collections::BTreeMap, fmt::Write};

use datafusion::physical_plan::metrics::MetricsSet;

use super::*;

pub(super) fn format_join_diagnostics(diagnostics: &[NativeJoinDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, diagnostics) in diagnostics.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        let _ = write!(
            output,
            "operator={},join_type={},key_count={},key_shape={},residual_predicate={},null_equality={},build_side={},probe_side={}",
            diagnostics.operator,
            diagnostics.join_type,
            diagnostics.key_count,
            diagnostics.key_shape.replace(['\t', '\n', '\r', '|'], " "),
            diagnostics.residual_predicate,
            diagnostics.null_equality,
            diagnostics.build_side,
            diagnostics.probe_side
        );
    }
    output
}

pub(super) fn format_source_diagnostics(diagnostics: &[NativeSourceDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, diagnostics) in diagnostics.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        let rejected_features = if diagnostics.rejected_features.is_empty() {
            "none".to_string()
        } else {
            diagnostics.rejected_features.join(",")
        };
        let known_bytes = diagnostics.known_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        let selected_row_groups =
            diagnostics.selected_row_groups.map_or_else(|| "all".to_string(), |row_groups| row_groups.to_string());
        let _ = write!(
            output,
            "format={},direct_device_read={},file_count={},ranged_file_count={},known_bytes={},selected_row_groups={},read_columns={},projected_output_columns={},pure_filter_columns={},predicate={},byte_range={},partition_columns={},zero_column_count={},schema_mapping={},rejected_features={}",
            diagnostics.format,
            diagnostics.direct_device_read,
            diagnostics.file_count,
            diagnostics.ranged_file_count,
            known_bytes,
            selected_row_groups,
            diagnostics.read_column_count,
            diagnostics.projected_output_column_count,
            diagnostics.pure_filter_column_count,
            diagnostics.predicate_disposition,
            diagnostics.byte_range_disposition,
            diagnostics.partition_column_disposition,
            diagnostics.zero_column_count_disposition,
            diagnostics.schema_mapping_disposition,
            rejected_features,
        );
    }
    output
}

pub(super) fn format_bounded_join_reservation_facts(facts: &BTreeMap<String, String>) -> String {
    if facts.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, (key, value)) in facts.iter().enumerate() {
        if index > 0 {
            output.push(';');
        }
        let _ = write!(output, "{}={}", stable_tsv_field(key), stable_tsv_field(value));
    }
    output
}

pub(super) fn write_tpch_dataset_metadata(output: &mut String, metadata: Option<&TpchDatasetMetadata>) {
    let Some(metadata) = metadata else {
        return;
    };

    let _ = writeln!(output, "# tpch_dataset_root={}", metadata.root.display());
    let _ = writeln!(output, "# tpch_dataset_scale_factor={}", metadata_value(&metadata.scale_factor));
    let _ = writeln!(output, "# tpch_dataset_provenance={}", metadata_value(&metadata.provenance));
    let _ = writeln!(output, "# tpch_dataset_generation_command={}", metadata_value(&metadata.generation_command));
    let _ = writeln!(output, "# tpch_dataset_table_count={}", metadata.table_count);
    let _ = writeln!(output, "# tpch_dataset_file_count={}", metadata.file_count);
    let _ = writeln!(output, "# tpch_dataset_total_rows={}", metadata.total_rows);
    let _ = writeln!(output, "# tpch_dataset_total_bytes={}", metadata.total_bytes);
    let _ = writeln!(output, "# tpch_dataset_metadata_fingerprint={}", metadata.metadata_fingerprint);
    let _ = writeln!(
        output,
        "# tpch_dataset_content_fingerprint={}",
        metadata.content_fingerprint.as_deref().unwrap_or("not_computed")
    );
    let _ = writeln!(output, "# tpch_dataset_table_file_counts={}", metadata.table_file_counts());
    let _ = writeln!(output, "# tpch_dataset_table_row_counts={}", metadata.table_row_counts());
    let _ = writeln!(output, "# tpch_dataset_table_byte_sizes={}", metadata.table_byte_sizes());
    let _ = writeln!(output, "# tpch_dataset_table_row_group_counts={}", metadata.table_row_group_counts());
}

fn metadata_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

pub(super) fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

pub(super) fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

pub(super) fn optional_observed_metric_count(metrics: &MetricsSet, name: &str) -> Option<usize> {
    let observed_name = format!("{name}_observed");
    let observed = metrics.sum_by_name(&observed_name).map(|value| value.as_usize()).unwrap_or(0);
    (observed > 0).then(|| metric_count(metrics, name))
}
