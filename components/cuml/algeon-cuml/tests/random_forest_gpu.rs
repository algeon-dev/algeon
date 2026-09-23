use std::{num::NonZeroUsize, sync::Arc};

use cudf::{CudfAllocationAttribution, DenseMatrixLayout, QueryMemoryDomainOptions, Runtime, TypeId};
use cuml::{
    CumlResources, DeviceMatrix, RandomForestClassifier, RandomForestParams, RandomForestRegressor, SplitCriterion,
};

#[test]
fn classifier_and_regressor_fit_predict_and_validate_binding() {
    assert!(cudf::DeviceId::count().expect("CUDA device count") > 0);
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let rows = 48;
    let features = (0..rows).map(|i| i as f32 - 24.0).collect::<Vec<_>>();
    let flat = runtime.column_from_slice(&features).expect("flat features");
    let packed = runtime.pack_flat_column(&flat, rows, 1, DenseMatrixLayout::RowMajor).expect("row-major matrix");
    let packed = packed.validate_finite().expect("finite inputs");
    let input = DeviceMatrix::from_validated(&packed).expect("device matrix");
    let labels = features.iter().map(|x| i32::from(*x >= 0.0)).collect::<Vec<_>>();
    let label_series =
        cudf::Series::from_column(runtime.column_from_slice(&labels).expect("labels")).expect("label series");
    let stream_count = NonZeroUsize::new(1).expect("nonzero stream count");
    let standalone_pool = resources.training_streams(stream_count, None).expect("unreserved pool backend route");
    let standalone = RandomForestClassifier::fit(
        &resources,
        &standalone_pool,
        &input,
        &label_series,
        RandomForestParams { n_trees: 4, max_depth: 8, ..Default::default() },
    )
    .expect("fit on unreserved runtime");
    assert_eq!(standalone.predict(&resources, &input).expect("predict without reservation").len(), rows);
    drop(standalone);
    drop(standalone_pool);
    let manager = runtime.device_memory_manager().expect("device memory manager");
    let domain = manager
        .query_memory_domain(
            1_000_001,
            QueryMemoryDomainOptions::new(512 * 1024 * 1024).with_ceiling_bytes(512 * 1024 * 1024),
        )
        .expect("query memory domain");
    let reservation =
        domain.reserve_stream(runtime.stream(), 1_000_002, 512 * 1024 * 1024).expect("runtime stream reservation");
    let stream_lease =
        resources.training_streams(stream_count, Some(&reservation)).expect("reservation-covered RF streams");
    let classifier = RandomForestClassifier::fit(
        &resources,
        &stream_lease,
        &input,
        &label_series,
        RandomForestParams { n_trees: 16, max_depth: 8, split_criterion: SplitCriterion::Gini, ..Default::default() },
    )
    .expect("classifier fit");
    let pool_usage = stream_lease.usage().expect("pool stream usage");
    assert!(pool_usage[0].total_allocations > 0, "forest training must allocate on the reserved pool stream");
    assert_eq!(pool_usage[0].current_bytes, 0, "native training must release pool allocations before returning");
    let prediction = classifier.predict(&resources, &input).expect("classifier prediction");
    assert_eq!(prediction.dtype().expect("dtype").id(), TypeId::Int32);
    assert_eq!(prediction.to_vec_strict::<i32>().expect("class predictions"), labels);

    let targets = features.iter().map(|x| 2.0 * x + 1.0).collect::<Vec<_>>();
    let target_series =
        cudf::Series::from_column(runtime.column_from_slice(&targets).expect("targets")).expect("target series");
    let regressor = RandomForestRegressor::fit(
        &resources,
        &stream_lease,
        &input,
        &target_series,
        RandomForestParams { n_trees: 24, max_depth: 8, split_criterion: SplitCriterion::Mse, ..Default::default() },
    )
    .expect("regressor fit");
    let predicted = regressor.predict(&resources, &input).expect("regression prediction");
    let values = predicted.to_vec_strict::<f32>().expect("float predictions");
    let mse =
        values.iter().zip(&targets).map(|(actual, expected)| (actual - expected).powi(2)).sum::<f32>() / rows as f32;
    assert!(mse < 12.0, "training MSE was {mse}");

    let other_runtime = Arc::new(Runtime::new().expect("second runtime"));
    let other_resources = CumlResources::for_runtime(Arc::clone(&other_runtime)).expect("second resources");
    let other_manager = other_runtime.device_memory_manager().expect("second device memory manager");
    let other_domain = other_manager
        .query_memory_domain(
            1_000_003,
            QueryMemoryDomainOptions::new(512 * 1024 * 1024).with_ceiling_bytes(512 * 1024 * 1024),
        )
        .expect("second query memory domain");
    let other_reservation = other_domain
        .reserve_stream(other_runtime.stream(), 1_000_004, 512 * 1024 * 1024)
        .expect("second runtime stream reservation");
    let wrong_reservation = match resources
        .training_streams(NonZeroUsize::new(1).expect("nonzero stream count"), Some(&other_reservation))
    {
        Ok(_) => panic!("reservation from a different runtime must fail"),
        Err(error) => error,
    };
    assert_eq!(wrong_reservation.code(), cuml::ErrorCode::ReservationCoverageMissing);
    let other_flat = other_runtime.column_from_slice(&features).expect("second flat features");
    let other_packed = other_runtime
        .pack_flat_column(&other_flat, rows, 1, DenseMatrixLayout::RowMajor)
        .expect("second row-major matrix");
    let other_packed = other_packed.validate_finite().expect("second finite inputs");
    let other_input = DeviceMatrix::from_validated(&other_packed).expect("second device matrix");
    let mismatch = match classifier.predict(&other_resources, &other_input) {
        Ok(_) => panic!("different binding must fail"),
        Err(error) => error,
    };
    assert_eq!(mismatch.code(), cuml::ErrorCode::RuntimeBindingMismatch);
}

#[test]
fn training_pool_allocation_failure_retains_query_limit_attribution() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let values = (0..128).map(|i| i as f32 - 64.0).collect::<Vec<_>>();
    let flat = runtime.column_from_slice(&values).expect("feature column");
    let packed = runtime.pack_flat_column(&flat, values.len(), 1, DenseMatrixLayout::RowMajor).expect("packed input");
    let input = packed.validate_finite().expect("finite input");
    let input = DeviceMatrix::from_validated(&input).expect("device matrix");
    let labels = values.iter().map(|v| i32::from(*v >= 0.0)).collect::<Vec<_>>();
    let labels =
        cudf::Series::from_column(runtime.column_from_slice(&labels).expect("label column")).expect("label series");
    let manager = runtime.device_memory_manager().expect("manager");
    let domain = manager
        .query_memory_domain(1_000_007, QueryMemoryDomainOptions::new(4096).with_ceiling_bytes(4096))
        .expect("domain");
    let reservation = domain.reserve_stream(runtime.stream(), 1_000_008, 4096).expect("reservation");
    let streams =
        resources.training_streams(NonZeroUsize::new(1).unwrap(), Some(&reservation)).expect("reserved training pool");
    let error = match RandomForestClassifier::fit(
        &resources,
        &streams,
        &input,
        &labels,
        RandomForestParams { n_trees: 8, max_depth: 8, ..Default::default() },
    ) {
        Ok(_) => panic!("training must exceed tiny query grant"),
        Err(error) => error,
    };
    assert_eq!(error.code(), cuml::ErrorCode::OutOfMemory);
    assert_eq!(error.phase(), cuml::FailurePhase::NativeCall);
    assert_eq!(cuml::allocation_attribution(&error), CudfAllocationAttribution::ConfirmedQueryLimit);
    assert_eq!(error.ffi_detail().expect("native allocation detail").domain_id, 1_000_007);
    assert!(streams.usage().expect("member usage")[0].total_allocations > 0);
}

#[test]
fn predict_rejects_column_major_and_fit_rejects_nonfinite_target() {
    assert!(cudf::DeviceId::count().expect("CUDA device count") > 0);
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let first = runtime.column_from_slice(&[-3.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).expect("feature");
    let packed = runtime.pack_columns(&[&first], DenseMatrixLayout::ColumnMajor).expect("column-major inputs");
    let packed = packed.validate_finite().expect("finite inputs");
    let input = DeviceMatrix::from_validated(&packed).expect("device matrix");
    let labels = cudf::Series::from_column(runtime.column_from_slice(&[0_i32, 0, 0, 0, 1, 1, 1, 1]).expect("labels"))
        .expect("label series");
    let manager = runtime.device_memory_manager().expect("device memory manager");
    let domain = manager
        .query_memory_domain(
            1_000_005,
            QueryMemoryDomainOptions::new(512 * 1024 * 1024).with_ceiling_bytes(512 * 1024 * 1024),
        )
        .expect("query memory domain");
    let reservation =
        domain.reserve_stream(runtime.stream(), 1_000_006, 512 * 1024 * 1024).expect("runtime stream reservation");
    let streams = resources
        .training_streams(NonZeroUsize::new(1).expect("nonzero stream count"), Some(&reservation))
        .expect("reservation-covered training streams");
    let classifier = RandomForestClassifier::fit(
        &resources,
        &streams,
        &input,
        &labels,
        RandomForestParams { n_trees: 4, max_depth: 8, ..Default::default() },
    )
    .expect("fit classifier");
    assert!(streams.usage().expect("pool usage")[0].total_allocations > 0);
    let error = match classifier.predict(&resources, &input) {
        Ok(_) => panic!("predict layout is row-major only"),
        Err(error) => error,
    };
    assert_eq!(error.code(), cuml::ErrorCode::UnsupportedLayout);

    let bad_targets = cudf::Series::from_column(
        runtime.column_from_slice(&[0.0_f32, 1.0, 2.0, 3.0, f32::NAN, 5.0, 6.0, 7.0]).expect("targets"),
    )
    .expect("target series");
    let error = match RandomForestRegressor::fit(
        &resources,
        &streams,
        &input,
        &bad_targets,
        RandomForestParams { split_criterion: SplitCriterion::Mse, ..Default::default() },
    ) {
        Ok(_) => panic!("nonfinite target must fail preflight"),
        Err(error) => error,
    };
    assert_eq!(error.code(), cuml::ErrorCode::NonFiniteInput);
}
