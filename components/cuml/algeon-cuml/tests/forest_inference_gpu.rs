use std::{fs, path::PathBuf, sync::Arc};

use cudf::{CudfAllocationAttribution, DenseMatrixLayout, PackedMatrix, QueryMemoryDomainOptions, Runtime};
use cuml::{
    CumlResources, DeviceMatrix, ErrorCode, FailurePhase, ForestInferenceModel, ForestModelFormat, ForestOutput,
    ForestOutputKind, ForestPrecision, inspect_forest_model,
};
use cuml_sys::{CUML_SUCCESS, test_support::cuml_test_forest_gtil_predict};

const TOLERANCE: f64 = 1e-5;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forest").join(name)
}

fn csv(name: &str) -> Vec<Vec<f64>> {
    fs::read_to_string(fixture_path(name))
        .unwrap()
        .lines()
        .map(|line| line.split(',').map(|value| value.parse::<f64>().unwrap()).collect())
        .collect()
}

/// Fixture inputs are written as shortest round-trip Float32 decimals.
fn input(name: &str) -> Vec<Vec<f32>> {
    fs::read_to_string(fixture_path(&format!("{name}.input.csv")))
        .unwrap()
        .lines()
        .map(|line| line.split(',').map(|value| value.parse::<f32>().unwrap()).collect())
        .collect()
}

fn checkpoint(model: &str, format: ForestModelFormat) -> Vec<u8> {
    inspect_forest_model(&fs::read(fixture_path(model)).unwrap(), format).expect(model).1
}

fn pack(runtime: &Runtime, rows: &[Vec<f32>], precision: ForestPrecision) -> PackedMatrix {
    let columns = rows[0].len();
    let flat = rows.iter().flatten().copied().collect::<Vec<f32>>();
    let column = match precision {
        ForestPrecision::Single => runtime.column_from_slice(&flat),
        ForestPrecision::Double => runtime.column_from_slice(&flat.iter().map(|v| f64::from(*v)).collect::<Vec<_>>()),
    }
    .unwrap();
    runtime.pack_flat_column(&column, rows.len(), columns, DenseMatrixLayout::RowMajor).unwrap()
}

fn gtil(checkpoint: &[u8], rows: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let columns = rows[0].len();
    let flat = rows.iter().flatten().map(|v| f64::from(*v)).collect::<Vec<_>>();
    let mut output = vec![0.0; rows.len() * 64];
    let mut written = 0;
    let status = unsafe {
        cuml_test_forest_gtil_predict(
            checkpoint.as_ptr(),
            checkpoint.len(),
            flat.as_ptr(),
            rows.len() as i32,
            columns as i32,
            output.as_mut_ptr(),
            output.len(),
            &mut written,
        )
    };
    assert_eq!(status, CUML_SUCCESS);
    let width = written / rows.len();
    output[..written].chunks(width).map(<[f64]>::to_vec).collect()
}

fn resources() -> (Arc<Runtime>, CumlResources) {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    (runtime, resources)
}

/// Runs a model over its fixture input and returns row-major outputs.
fn predict(resources: &CumlResources, model: &ForestInferenceModel, rows: &[Vec<f32>]) -> cuml::Result<Vec<Vec<f64>>> {
    let packed = pack(resources.runtime(), rows, model.info().precision);
    let validated = packed.validate().unwrap();
    let matrix = match model.info().precision {
        ForestPrecision::Single => DeviceMatrix::from_validated(&validated)?,
        ForestPrecision::Double => DeviceMatrix::from_validated_float64(&validated)?,
    };
    let series = model.predict(resources, &matrix)?.into_series()?;
    let columns = series
        .iter()
        .map(|series| match model.output_kind() {
            ForestOutputKind::ClassIndex { .. } => {
                series.to_vec_strict::<i32>().unwrap().into_iter().map(f64::from).collect::<Vec<_>>()
            },
            _ => series.to_vec_strict::<f32>().unwrap().into_iter().map(f64::from).collect(),
        })
        .collect::<Vec<_>>();
    Ok((0..rows.len()).map(|row| columns.iter().map(|column| column[row]).collect()).collect())
}

fn assert_close(actual: &[Vec<f64>], expected: &[Vec<f64>], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} row count");
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.len(), expected.len(), "{label} row {row} width");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= TOLERANCE * e.abs().max(1.0), "{label} row {row}: {a} != {e}");
        }
    }
}

fn argmax_lowest(row: &[f64]) -> f64 {
    let mut best = 0;
    for (index, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = index;
        }
    }
    best as f64
}

#[test]
fn scores_match_source_predictor_and_gtil() {
    let (_runtime, resources) = resources();
    for (name, model, format) in [
        ("xgb_binary_logistic", "xgb_binary_logistic.json", ForestModelFormat::XgboostJson),
        ("xgb_multiclass_softprob", "xgb_multiclass_softprob.json", ForestModelFormat::XgboostJson),
        ("xgb_multiclass_softprob", "xgb_multiclass_softprob_ubj.ubj", ForestModelFormat::XgboostUbjson),
        ("lgb_regression_double", "lgb_regression_double.txt", ForestModelFormat::LightGbm),
    ] {
        let checkpoint = checkpoint(model, format);
        let mut forest = ForestInferenceModel::import(&resources, &checkpoint, ForestOutput::Scores).expect(model);
        let rows = input(name);
        let actual = predict(&resources, &forest, &rows).expect(model);
        assert_close(&actual, &csv(&format!("{name}.expected.csv")), model);
        assert_close(&actual, &gtil(&checkpoint, &rows), model);
        forest.close().expect("close");
    }
}

#[test]
fn double_threshold_forest_keeps_float32_inputs_on_the_exact_branch() {
    let (_runtime, resources) = resources();
    let checkpoint = checkpoint("lgb_threshold_boundary.txt", ForestModelFormat::LightGbm);
    let forest = ForestInferenceModel::import(&resources, &checkpoint, ForestOutput::Scores).unwrap();
    assert_eq!(forest.info().precision, ForestPrecision::Double);
    let rows = input("lgb_threshold_boundary");
    assert_eq!(rows[0], vec![1.0_f32]);
    // The threshold 1 - 2^-26 rounds to 1.0 in Float32. A single-precision forest
    // would route input 1.0 to the left leaf (-1); the expected output is the right leaf (+1).
    let actual = predict(&resources, &forest, &rows).unwrap();
    assert_eq!(actual, csv("lgb_threshold_boundary.expected.csv"));
    assert_eq!(actual[0], vec![1.0]);
}

#[test]
fn class_index_reads_the_first_slot_of_each_strided_row_and_breaks_ties_low() {
    let (_runtime, resources) = resources();
    for name in ["xgb_multiclass_softmax", "xgb_multiclass_tie"] {
        let checkpoint = checkpoint(&format!("{name}.json"), ForestModelFormat::XgboostJson);
        let forest = ForestInferenceModel::import(&resources, &checkpoint, ForestOutput::ClassIndex).unwrap();
        assert_eq!(forest.info().native_output_stride, 3);
        assert_eq!(forest.output_kind(), ForestOutputKind::ClassIndex { classes: 3 });
        let rows = input(name);
        let actual = predict(&resources, &forest, &rows).unwrap();
        assert_eq!(actual, csv(&format!("{name}.expected.csv")), "{name}");
        let oracle = gtil(&checkpoint, &rows).iter().map(|row| vec![argmax_lowest(row)]).collect::<Vec<_>>();
        assert_eq!(actual, oracle, "{name}");
    }
    let tie = csv("xgb_multiclass_tie.expected.csv");
    assert!(tie.iter().all(|row| row == &vec![1.0]), "tied classes 1 and 2 resolve to 1");
}

#[test]
fn predict_rejects_inputs_outside_the_model_contract() {
    let (runtime, resources) = resources();
    let single = checkpoint("xgb_binary_logistic.json", ForestModelFormat::XgboostJson);
    let mut forest = ForestInferenceModel::import(&resources, &single, ForestOutput::Scores).unwrap();
    let rows = input("xgb_binary_logistic");

    let narrow = rows.iter().map(|row| row[..3].to_vec()).collect::<Vec<_>>();
    let packed = pack(&runtime, &narrow, ForestPrecision::Single);
    let validated = packed.validate().unwrap();
    let error = forest.predict(&resources, &DeviceMatrix::from_validated(&validated).unwrap()).err().unwrap();
    assert_eq!(error.code(), ErrorCode::ShapeMismatch);

    let packed = pack(&runtime, &rows, ForestPrecision::Double);
    let validated = packed.validate().unwrap();
    let error = forest.predict(&resources, &DeviceMatrix::from_validated_float64(&validated).unwrap()).err().unwrap();
    assert_eq!(error.code(), ErrorCode::UnsupportedDtype);

    let columns = (0..4)
        .map(|feature| runtime.column_from_slice(&rows.iter().map(|row| row[feature]).collect::<Vec<_>>()).unwrap())
        .collect::<Vec<_>>();
    let packed = runtime.pack_columns(&columns.iter().collect::<Vec<_>>(), DenseMatrixLayout::ColumnMajor).unwrap();
    let validated = packed.validate().unwrap();
    let error = forest.predict(&resources, &DeviceMatrix::from_validated(&validated).unwrap()).err().unwrap();
    assert_eq!(error.code(), ErrorCode::UnsupportedLayout);

    let error = ForestInferenceModel::import(&resources, &single, ForestOutput::ClassIndex).err().unwrap();
    assert_eq!(error.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(error.phase(), FailurePhase::NativeCall);

    forest.close().unwrap();
    forest.close().expect("close is idempotent");
    let packed = pack(&runtime, &rows, ForestPrecision::Single);
    let validated = packed.validate().unwrap();
    let error = forest.predict(&resources, &DeviceMatrix::from_validated(&validated).unwrap()).err().unwrap();
    assert_eq!(error.code(), ErrorCode::ModelClosed);
}

#[test]
fn import_allocates_in_the_query_domain_and_reports_limit_failures() {
    const DOMAIN_ID: u64 = 74_101;
    const MEMBER_ID: u64 = 74_102;
    let runtime = Arc::new(Runtime::builder().with_device_memory_manager().build().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let checkpoint = checkpoint("xgb_multiclass_softprob.json", ForestModelFormat::XgboostJson);
    let manager = runtime.device_memory_manager().expect("device memory manager");

    {
        const GRANT: usize = 64 * 1024 * 1024;
        let domain = manager
            .query_memory_domain(DOMAIN_ID, QueryMemoryDomainOptions::new(GRANT).with_ceiling_bytes(GRANT))
            .expect("query memory domain");
        let reservation = domain.reserve_stream(runtime.stream(), MEMBER_ID, GRANT).expect("stream reservation");
        let mut forest = ForestInferenceModel::import(&resources, &checkpoint, ForestOutput::Scores).unwrap();
        assert!(reservation.usage().unwrap().current_bytes > 0, "device forest is charged to the query domain");
        forest.close().unwrap();
        runtime.stream().synchronize().unwrap();
        assert_eq!(reservation.usage().unwrap().current_bytes, 0, "close releases the device forest");
    }

    const TINY: usize = 256;
    let domain = manager
        .query_memory_domain(DOMAIN_ID + 2, QueryMemoryDomainOptions::new(TINY).with_ceiling_bytes(TINY))
        .expect("tiny query memory domain");
    let reservation = domain.reserve_stream(runtime.stream(), MEMBER_ID + 2, TINY).expect("tiny reservation");
    let error = ForestInferenceModel::import(&resources, &checkpoint, ForestOutput::Scores)
        .err()
        .expect("model storage exceeds the grant");
    assert_eq!(error.code(), ErrorCode::OutOfMemory);
    assert_eq!(cuml::allocation_attribution(&error), CudfAllocationAttribution::ConfirmedQueryLimit);
    assert_eq!(error.ffi_detail().expect("allocation provenance").domain_id, DOMAIN_ID + 2);
    runtime.stream().synchronize().unwrap();
    assert_eq!(reservation.usage().unwrap().current_bytes, 0, "a failed import leaves no device bytes");
}
