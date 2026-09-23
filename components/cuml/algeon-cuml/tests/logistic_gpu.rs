use std::sync::Arc;

use cudf::{DenseMatrixLayout, Runtime};
use cuml::{CumlResources, DeviceMatrix, LogisticRegressionModel, LogisticRegressionParams};

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
    let packed = packed.validate_finite().expect("finite feature matrix");
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
}
