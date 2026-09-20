use std::sync::Arc;

use cudf::{DenseMatrixLayout, Runtime, RuntimeBuilder, testing::device_copy};
use cuml::{CumlResources, DeviceMatrix, ErrorCode, LogisticRegressionModel, LogisticRegressionParams};

#[test]
fn logistic_regression_adopts_labels_from_another_runtime() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let other_runtime = RuntimeBuilder::default().with_device(runtime.device()).build().expect("second runtime");
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let features = [-4.0_f32, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let column = runtime.column_from_slice(&features).expect("features");
    let packed = runtime.pack_columns(&[&column], DenseMatrixLayout::ColumnMajor).expect("matrix");
    let packed = packed.validate().expect("finite features");
    let input = DeviceMatrix::from_validated(&packed).expect("matrix view");
    let values = [0_i32, 0, 0, 0, 1, 1, 1, 1];
    let local = runtime.series(&values).expect("local labels");
    let foreign = other_runtime.series(&values).expect("foreign labels");
    assert_ne!(local.placement().stream_debug, foreign.placement().stream_debug);
    let params = LogisticRegressionParams { max_iter: 200, ..Default::default() };
    let copies_before = device_copy::device_to_device_copy_count_for_test();
    let (baseline, _) = LogisticRegressionModel::fit(&resources, &input, &local, params).expect("baseline fit");
    assert_eq!(device_copy::device_to_device_copy_count_for_test() - copies_before, 0);
    let copies_before = device_copy::device_to_device_copy_count_for_test();
    let (adopted, _) = LogisticRegressionModel::fit(&resources, &input, &foreign, params).expect("foreign fit");
    assert_eq!(
        device_copy::device_to_device_copy_count_for_test() - copies_before,
        1,
        "foreign labels require a device copy"
    );
    assert_eq!(
        adopted.predict(&resources, &input).unwrap().to_vec_strict::<i32>().unwrap(),
        baseline.predict(&resources, &input).unwrap().to_vec_strict::<i32>().unwrap(),
    );
}

#[test]
fn logistic_regression_fit_and_predict_binary_and_multiclass() {
    assert!(cudf::DeviceId::count().expect("CUDA device count") > 0);
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let x0 = [-4.0_f32, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let x1 = x0.map(|value| value * 0.5);
    let first = runtime.column_from_slice(&x0).expect("feature column");
    let second = runtime.column_from_slice(&x1).expect("feature column");
    let packed =
        runtime.pack_columns(&[&first, &second], DenseMatrixLayout::ColumnMajor).expect("column-major feature matrix");
    let packed = packed.validate().expect("finite feature matrix");
    let input = DeviceMatrix::from_validated(&packed).expect("matrix view");
    let binary = cudf::Series::from_column(runtime.column_from_slice(&[0_i32, 0, 0, 0, 1, 1, 1, 1]).expect("labels"))
        .expect("label series");
    let (model, fit) = LogisticRegressionModel::fit(
        &resources,
        &input,
        &binary,
        LogisticRegressionParams { max_iter: 200, ..Default::default() },
    )
    .expect("binary logistic fit");
    assert_eq!(model.n_classes(), 2);
    assert!(fit.objective.is_finite());
    assert!(fit.iterations <= 200);
    let predictions = model.predict(&resources, &input).expect("binary predictions");
    assert_eq!(predictions.len(), x0.len());
    let predicted = predictions.to_vec_strict::<i32>().expect("Int32 predictions");
    assert_eq!(predicted, [0, 0, 0, 0, 1, 1, 1, 1]);

    let multi =
        cudf::Series::from_column(runtime.column_from_slice(&[0_i32, 0, 1, 1, 2, 2, 2, 2]).expect("multiclass labels"))
            .expect("multiclass label series");
    let (multi_model, multi_fit) = LogisticRegressionModel::fit(
        &resources,
        &input,
        &multi,
        LogisticRegressionParams { max_iter: 200, ..Default::default() },
    )
    .expect("multiclass logistic fit");
    assert_eq!(multi_model.n_classes(), 3);
    assert!(multi_fit.objective.is_finite());
    assert_eq!(multi_model.predict(&resources, &input).expect("multiclass predictions").len(), 8);

    for (values, expected, reason) in [
        ([0_i32, 0, 0, 0, 2, 2, 2, 2], ErrorCode::LabelOutOfRange, "labels_not_contiguous"),
        ([1_i32, 1, 1, 1, 2, 2, 2, 2], ErrorCode::LabelOutOfRange, "labels_not_contiguous"),
        ([0_i32; 8], ErrorCode::LabelOutOfRange, "single_class"),
    ] {
        let labels = runtime.series(&values).expect("labels");
        let failure = LogisticRegressionModel::fit(&resources, &input, &labels, LogisticRegressionParams::default())
            .err()
            .expect("invalid labels");
        assert_eq!(failure.code(), expected);
        assert_eq!(failure.detail_value("reason"), Some(reason));
        assert!(failure.detail_value("distinct_count").is_some());
    }
    let nullable = runtime
        .nullable_series(&[0_i32, 0, 0, 0, 1, 1, 1, 1], &[true, true, false, true, true, true, true, true])
        .unwrap();
    let failure = LogisticRegressionModel::fit(&resources, &input, &nullable, LogisticRegressionParams::default())
        .err()
        .expect("null labels");
    assert_eq!(failure.code(), ErrorCode::NullUnexpected);
    assert_eq!(failure.detail_value("reason"), Some("label_nulls"));
    let wrong_dtype = runtime.series(&[0_i64, 0, 0, 0, 1, 1, 1, 1]).unwrap();
    assert_eq!(
        LogisticRegressionModel::fit(&resources, &input, &wrong_dtype, LogisticRegressionParams::default())
            .err()
            .unwrap()
            .code(),
        ErrorCode::UnsupportedDtype
    );
}

// cuML's default logger sink (cuml/common/logger.hpp default_sink) writes to the file named by
// CUML_DEBUG_LOG_FILE, but only flushes that file's buffer on process exit; there is no API to
// force a flush mid-process. Run the fit in a child process (re-exec of this same test binary) so
// the parent can read a fully flushed log after the child exits.
const DEBUG_LOG_CHILD_ENV: &str = "CUML_LOGISTIC_DEBUG_LOG_CHILD";

#[test]
fn logistic_regression_fit_does_not_enable_debug_logging() {
    if std::env::var_os(DEBUG_LOG_CHILD_ENV).is_some() {
        run_logistic_fit_for_debug_log_test();
        return;
    }

    let log_path = std::env::temp_dir().join(format!("cuml-logistic-debug-log-{}.log", std::process::id()));
    let exe = std::env::current_exe().expect("test binary path");
    let output = std::process::Command::new(&exe)
        .args(["logistic_regression_fit_does_not_enable_debug_logging", "--exact"])
        .env(DEBUG_LOG_CHILD_ENV, "1")
        .env("CUML_DEBUG_LOG_FILE", &log_path)
        .output()
        .expect("spawn logistic fit child process");
    assert!(
        output.status.success(),
        "child logistic fit process failed: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let log = std::fs::read_to_string(&log_path).expect("QN solver wrote the cuML debug log file");
    let _ = std::fs::remove_file(&log_path);
    // Pattern is "[%6t][%H:%M:%S:%f][%-6l] %v" (see cuml/common/logger.hpp default_pattern), so a
    // debug or trace line contains "[debug ]" or "[trace ]" respectively.
    assert!(!log.contains("[debug "), "QN solver logged at debug level:\n{log}");
    assert!(!log.contains("[trace "), "QN solver logged at trace level:\n{log}");
}

fn run_logistic_fit_for_debug_log_test() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let features = [-4.0_f32, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
    let column = runtime.column_from_slice(&features).expect("features");
    let packed = runtime.pack_columns(&[&column], DenseMatrixLayout::ColumnMajor).expect("matrix");
    let packed = packed.validate().expect("finite features");
    let input = DeviceMatrix::from_validated(&packed).expect("matrix view");
    let labels = runtime.series(&[0_i32, 0, 0, 0, 1, 1, 1, 1]).expect("labels");
    LogisticRegressionModel::fit(
        &resources,
        &input,
        &labels,
        LogisticRegressionParams { max_iter: 20, ..Default::default() },
    )
    .expect("logistic fit");
}
