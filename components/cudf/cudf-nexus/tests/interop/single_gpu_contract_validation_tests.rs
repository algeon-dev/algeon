//! Single-GPU interop validation tests aligned with the current interop
//! direction.
//!
//! These tests keep the validation consumer-neutral while still exercising the
//! graph-construction-shaped contract the current specs call for:
//!
//! - prepared export can describe one edge-list input without relying on
//!   private `cudf` ownership details
//! - deep-copy import remains a validated fallback when the consumer wants its
//!   own runtime-owned working set
//! - result handoff remains explicit and goes back through the same shared
//!   prepared-export boundary
use std::{collections::HashMap, ffi::c_void};

use cudf::{AggSpec, DataFrame, GroupByAgg, Runtime, testing::prelude::*};
use rapids_interop::{
    BufferBinding, ColumnLayout, CompletionAuthority, ExportProfile, LeaseCompletionResult, PreparedTableExport,
    ProducerReadiness, ReadStartAuthority, SemanticColumn,
};

use crate::common;

fn sample_edge_list(runtime: &Runtime) -> DataFrame {
    let src = runtime.series(&[1i32, 2, 1, 3, 2]).unwrap();
    let dst = runtime.series(&[2i32, 3, 3, 1, 1]).unwrap();
    DataFrame::from_series([("src", src), ("dst", dst)]).unwrap()
}

fn binding_map<'a>(bindings: &'a [BufferBinding<'a>]) -> HashMap<u32, &'a BufferBinding<'a>> {
    bindings.iter().map(|binding| (binding.id().get(), binding)).collect()
}

fn fixed_width_binding_ptr(column: &SemanticColumn, bindings: &HashMap<u32, &BufferBinding<'_>>) -> *const c_void {
    let values = match column.layout() {
        ColumnLayout::FixedWidth { values } => values,
        layout => panic!("expected fixed-width edge-list column, got {layout:?}"),
    };

    bindings
        .get(&values.get())
        .expect("prepared edge-list export is missing the fixed-width values binding")
        .bytes()
        .base_ptr()
}

fn readiness_device(readiness: &ProducerReadiness<'_>) -> i32 {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => stream.device(),
        ProducerReadiness::DeviceEventOrdered { event } => event.device(),
        ProducerReadiness::HostVisible => panic!("prepared device export must not report host-visible readiness"),
    }
}

fn sorted_out_degree_rows(frame: &DataFrame) -> Vec<(i32, i32)> {
    let src = frame.series("src").unwrap().to_vec_strict::<i32>().unwrap();
    let out_degree = frame.series("out_degree").unwrap().to_vec_strict::<i32>().unwrap();
    let mut rows: Vec<(i32, i32)> = src.into_iter().zip(out_degree).collect();
    rows.sort_by_key(|&(src, _)| src);
    rows
}

struct MockSingleGpuGraphInput {
    names: Vec<String>,
    num_edges: usize,
    device: i32,
    src_ptr: *const c_void,
    dst_ptr: *const c_void,
}

impl MockSingleGpuGraphInput {
    fn from_prepared_export(export: &PreparedTableExport<'_>) -> Self {
        assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedDeviceExport);
        assert_eq!(
            export
                .try_activate_read_lease(
                    ReadStartAuthority::SynchronousByReturn,
                    CompletionAuthority::SynchronousByReturn
                )
                .unwrap()
                .complete()
                .unwrap(),
            LeaseCompletionResult::FallbackSynchronized
        );

        let names = export.table().names().expect("graph-shaped prepared export must carry schema names").to_vec();
        let columns = export.columns();
        assert_eq!(columns.len(), 2, "graph input should expose exactly src/dst columns");
        let bindings = binding_map(export.bindings());

        Self {
            names,
            num_edges: export.table().num_rows(),
            device: readiness_device(export.readiness()),
            src_ptr: fixed_width_binding_ptr(&columns[0], &bindings),
            dst_ptr: fixed_width_binding_ptr(&columns[1], &bindings),
        }
    }
}

#[test]
fn single_gpu_prepared_export_supports_graph_construction_shaped_consumption() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let edges = sample_edge_list(&producer_runtime);
    let src_ptr = edges.series("src").unwrap().as_column().testing_view().unwrap().device_data_ptr().unwrap();
    let dst_ptr = edges.series("dst").unwrap().as_column().testing_view().unwrap().device_data_ptr().unwrap();
    let export = edges.try_prepare_device_table_export().unwrap();

    let graph_input = MockSingleGpuGraphInput::from_prepared_export(&export);

    assert_eq!(graph_input.names, vec!["src".to_string(), "dst".to_string()]);
    assert_eq!(graph_input.num_edges, 5);
    assert_eq!(graph_input.device, producer_runtime.device().ordinal() as i32);
    assert_eq!(graph_input.src_ptr, src_ptr);
    assert_eq!(graph_input.dst_ptr, dst_ptr);
}

#[test]
fn single_gpu_prepared_export_fallback_import_can_run_algorithm_and_handoff_result() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let edges = sample_edge_list(&producer_runtime);
    let export = edges.try_prepare_device_table_export().unwrap();
    let _event_lock = common::lock_event_observation();

    let consumer_runtime = common::sample_runtime();
    let imported = DataFrame::try_import_from_prepared_table_export_into_runtime(&export, &consumer_runtime).unwrap();
    assert_eq!(imported.testing_execution_domain_stream_value(), consumer_runtime.stream().testing_raw_handle_value());
    assert_eq!(imported.column_names(), vec!["src", "dst"]);

    let out_degree = imported
        .groupby(["src"])
        .agg(&[AggSpec::new("dst", GroupByAgg::Count { include_nulls: false }).with_alias("out_degree")])
        .unwrap();

    assert_eq!(
        out_degree.testing_execution_domain_stream_value(),
        consumer_runtime.stream().testing_raw_handle_value()
    );
    assert_eq!(sorted_out_degree_rows(&out_degree), vec![(1, 2), (2, 2), (3, 1)]);

    let result_export = out_degree.try_prepare_device_table_export().unwrap();
    assert_eq!(result_export.diagnostics().profile(), ExportProfile::PreparedDeviceExport);
    assert_eq!(
        result_export.table().names().unwrap().iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["src", "out_degree"]
    );

    let handoff_runtime = common::sample_runtime();
    let handed_off =
        DataFrame::try_import_from_prepared_table_export_into_runtime(&result_export, &handoff_runtime).unwrap();

    assert_eq!(handed_off.column_names(), vec!["src", "out_degree"]);
    assert_eq!(sorted_out_degree_rows(&handed_off), vec![(1, 2), (2, 2), (3, 1)]);
    assert_eq!(handed_off.testing_execution_domain_stream_value(), handoff_runtime.stream().testing_raw_handle_value());
}
