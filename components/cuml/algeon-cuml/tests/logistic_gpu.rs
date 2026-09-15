use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use cudf::{BoundaryKind, BoundaryObservation, DenseMatrixLayout, Runtime, RuntimeBuilder};
use cuml::{CumlResources, DeviceMatrix, ErrorCode, LogisticRegressionModel, LogisticRegressionParams};

#[test]
fn logistic_regression_adopts_labels_from_another_runtime() {
    let copies = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&copies);
    let runtime = Arc::new(
        RuntimeBuilder::default()
            .with_boundary_observer(move |observation: BoundaryObservation| {
                if observation.kind == BoundaryKind::DeviceToDeviceCopy {
                    observed.fetch_add(1, Ordering::Relaxed);
                }
            })
            .build()
            .expect("runtime"),
    );
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
    let (baseline, _) = LogisticRegressionModel::fit(&resources, &input, &local, params).expect("baseline fit");
    assert_eq!(copies.load(Ordering::Relaxed), 0);
    let (adopted, _) = LogisticRegressionModel::fit(&resources, &input, &foreign, params).expect("foreign fit");
    assert_eq!(copies.load(Ordering::Relaxed), 1, "foreign labels require a device copy");
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
