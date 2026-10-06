use std::sync::Arc;

use cudf::{DenseMatrixLayout, Runtime, RuntimeBuilder, Series, TypeId, testing::device_copy};
use cuml::{CumlResources, DeviceMatrix, ErrorCode, LogisticRegressionModel, LogisticRegressionParams, MatrixLayout};

fn coefficients(runtime: &Runtime, model: &LogisticRegressionModel) -> Vec<f32> {
    let buffer = model.coefficients();
    // SAFETY: the buffer is a live Float32 allocation bound to the runtime's execution binding.
    let column = unsafe {
        runtime
            .column_from_device_ptr(buffer.as_ptr().expect("coefficient pointer"), buffer.len(), TypeId::Float32)
            .expect("coefficient column")
    };
    Series::from_column(column).expect("coefficient series").to_vec_strict::<f32>().expect("Float32 coefficients")
}

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
        adopted.predict(&resources, &input).unwrap().class.to_vec_strict::<i32>().unwrap(),
        baseline.predict(&resources, &input).unwrap().class.to_vec_strict::<i32>().unwrap(),
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
    assert_eq!(predictions.class.len(), x0.len());
    assert_eq!(predictions.probability.dtype().expect("dtype").id(), TypeId::Float32);
    let predicted = predictions.class.to_vec_strict::<i32>().expect("Int32 predictions");
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
    assert_eq!(multi_model.predict(&resources, &input).expect("multiclass predictions").class.len(), 8);

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

/// Overlapping classes keep fitted probabilities away from 0 and 1, so the
/// comparison with the host decision function exercises the transformation.
#[test]
fn logistic_regression_probabilities_follow_the_decision_function() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let rows = 15_usize;
    let x0 = (0..rows).map(|i| (i as f32 - 7.0) * 0.4).collect::<Vec<_>>();
    let x1 = (0..rows).map(|i| ((i * 7) % 5) as f32 - 2.0).collect::<Vec<_>>();
    let packed = runtime
        .pack_columns(
            &[&runtime.column_from_slice(&x0).unwrap(), &runtime.column_from_slice(&x1).unwrap()],
            DenseMatrixLayout::ColumnMajor,
        )
        .expect("matrix");
    let packed = packed.validate().expect("finite features");
    let input = DeviceMatrix::from_validated(&packed).expect("matrix view");
    assert_eq!(input.layout(), MatrixLayout::ColumnMajor);
    let params = LogisticRegressionParams { max_iter: 500, ..Default::default() };
    let binary = [0_i32, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1];
    let three = [0_i32, 0, 1, 0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 1, 2];

    // Binary: one coefficient vector [w0, w1, intercept]; class 1 has probability sigmoid(z).
    let (model, _) =
        LogisticRegressionModel::fit(&resources, &input, &runtime.series(&binary).unwrap(), params).unwrap();
    let w = coefficients(&runtime, &model);
    let prediction = model.predict(&resources, &input).expect("binary prediction");
    let (class, probability) =
        (prediction.class.to_vec_strict::<i32>().unwrap(), prediction.probability.to_vec_strict::<f32>().unwrap());
    for row in 0..rows {
        let z = f64::from(w[0]) * f64::from(x0[row]) + f64::from(w[1]) * f64::from(x1[row]) + f64::from(w[2]);
        let positive = 1.0 / (1.0 + (-z).exp());
        let (expected_class, expected) = if z > 0.0 { (1, positive) } else { (0, 1.0 - positive) };
        assert_eq!(class[row], expected_class, "row {row}");
        assert!(
            (f64::from(probability[row]) - expected).abs() <= 1e-5,
            "row {row}: {} vs {expected}",
            probability[row]
        );
        assert!(probability[row] >= 0.5);
    }
    assert!(probability.iter().any(|value| *value < 0.99), "overlapping classes must not saturate: {probability:?}");

    // Softmax: coefficients are a 3 x 3 column-major block; the last column is the intercept.
    let (model, _) =
        LogisticRegressionModel::fit(&resources, &input, &runtime.series(&three).unwrap(), params).unwrap();
    let w = coefficients(&runtime, &model);
    let prediction = model.predict(&resources, &input).expect("softmax prediction");
    let (class, probability) =
        (prediction.class.to_vec_strict::<i32>().unwrap(), prediction.probability.to_vec_strict::<f32>().unwrap());
    for row in 0..rows {
        let z = (0..3)
            .map(|c| {
                f64::from(w[c]) * f64::from(x0[row]) + f64::from(w[3 + c]) * f64::from(x1[row]) + f64::from(w[6 + c])
            })
            .collect::<Vec<_>>();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected = 1.0 / z.iter().map(|value| (value - max).exp()).sum::<f64>();
        assert_eq!(i64::from(class[row]), z.iter().position(|value| *value == max).unwrap() as i64, "row {row}");
        assert!(
            (f64::from(probability[row]) - expected).abs() <= 1e-5,
            "row {row}: {} vs {expected}",
            probability[row]
        );
        assert!(probability[row] >= 1.0 / 3.0);
    }
    assert!(probability.iter().any(|value| *value < 0.99), "overlapping classes must not saturate: {probability:?}");
}

#[test]
fn logistic_regression_predicts_a_single_row() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let pack = |values: &[f32]| {
        let column = runtime.column_from_slice(values).expect("features");
        runtime.pack_columns(&[&column], DenseMatrixLayout::ColumnMajor).expect("matrix")
    };
    let training = pack(&[-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]);
    let training = training.validate().expect("finite features");
    let labels = runtime.series(&[0_i32, 0, 0, 0, 1, 1, 1, 1]).expect("labels");
    let (model, _) = LogisticRegressionModel::fit(
        &resources,
        &DeviceMatrix::from_validated(&training).unwrap(),
        &labels,
        LogisticRegressionParams { max_iter: 200, ..Default::default() },
    )
    .expect("fit");
    let single = pack(&[3.5]);
    let single = single.validate().expect("finite features");
    let prediction =
        model.predict(&resources, &DeviceMatrix::from_validated(&single).unwrap()).expect("one-row predict");
    assert_eq!(prediction.class.to_vec_strict::<i32>().unwrap(), [1]);
    assert!(prediction.probability.to_vec_strict::<f32>().unwrap()[0] > 0.5);
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
