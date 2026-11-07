//! Crate-root workflow and execution-placement API checks.

use cudf::Exec;

#[test]
fn crate_root_doc_workflow_smoke() {
    use cudf::{DataFrame, Runtime};

    let rt = Runtime::builder().build().unwrap();
    let ids = rt.series(&[1i32, 2, 3]).unwrap();
    let scores = rt.series(&[10.0f64, 20.0, 30.0]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows(), 3);

    let score_series = df.series("score").unwrap();
    assert_eq!(score_series.on(Exec::RETAINED).sum_value::<f64>().unwrap(), 60.0);
    assert_eq!(score_series.on(Exec::RETAINED).mean_value().unwrap(), Some(20.0));
}

#[test]
fn execution_placement_summary_is_publicly_reexported() {
    use cudf::{DataFrame, ExecutionPlacement, Runtime};

    fn assert_summary(_: ExecutionPlacement) {}

    let rt = Runtime::builder().build().unwrap();
    let series = rt.series(&[1i32, 2, 3]).unwrap();
    let df = DataFrame::from_series([("id", series.clone())]).unwrap();

    let runtime_placement = rt.placement();
    let series_placement = series.placement();
    let frame_placement = df.placement();

    assert_summary(runtime_placement);
    assert_eq!(series_placement, runtime_placement);
    assert_eq!(frame_placement, runtime_placement);
    assert_eq!(frame_placement.device, rt.device());
    assert_ne!(frame_placement.stream_debug, 0);
    assert_ne!(frame_placement.memory_resource_debug, 0);
}
