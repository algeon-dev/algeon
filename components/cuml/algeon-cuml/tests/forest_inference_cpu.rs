use std::{fs, path::PathBuf};

use cuml::{
    ErrorCode, ForestModelFormat, ForestModelInfo, ForestOutput, ForestOutputKind, ForestPrecision,
    inspect_forest_model,
};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forest").join(name);
    fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn info(num_features: usize, num_trees: usize, stride: usize, precision: ForestPrecision) -> ForestModelInfo {
    ForestModelInfo { num_features, num_trees, native_output_stride: stride, precision }
}

#[test]
fn inspect_reports_importer_facts_for_each_source_format() {
    let cases = [
        ("xgb_binary_logistic.json", ForestModelFormat::XgboostJson, 4, 1, ForestPrecision::Single),
        ("xgb_multiclass_softprob.json", ForestModelFormat::XgboostJson, 4, 3, ForestPrecision::Single),
        ("xgb_multiclass_softprob_ubj.ubj", ForestModelFormat::XgboostUbjson, 4, 3, ForestPrecision::Single),
        ("lgb_regression_double.txt", ForestModelFormat::LightGbm, 4, 1, ForestPrecision::Double),
        ("lgb_threshold_boundary.txt", ForestModelFormat::LightGbm, 1, 1, ForestPrecision::Double),
    ];
    for (name, format, features, stride, precision) in cases {
        let (inspected, checkpoint) = inspect_forest_model(&fixture(name), format).expect(name);
        assert_eq!(
            (inspected.num_features, inspected.native_output_stride, inspected.precision),
            (features, stride, precision),
            "{name}"
        );
        assert!(inspected.num_trees > 0, "{name}");
        let (reloaded, _) =
            inspect_forest_model(&checkpoint, ForestModelFormat::TreeliteCheckpoint).expect("checkpoint round trip");
        assert_eq!(reloaded, inspected, "{name} checkpoint must preserve importer facts");
    }
    let (json, _) =
        inspect_forest_model(&fixture("xgb_multiclass_softprob.json"), ForestModelFormat::XgboostJson).unwrap();
    let (ubjson, _) =
        inspect_forest_model(&fixture("xgb_multiclass_softprob_ubj.ubj"), ForestModelFormat::XgboostUbjson).unwrap();
    assert_eq!(json, ubjson);
}

#[test]
fn output_kind_follows_the_requested_output_and_stride() {
    let binary = info(4, 5, 1, ForestPrecision::Single);
    let multiclass = info(4, 15, 3, ForestPrecision::Single);
    assert_eq!(binary.output_kind(ForestOutput::Scores).unwrap(), ForestOutputKind::Score);
    assert_eq!(multiclass.output_kind(ForestOutput::Scores).unwrap(), ForestOutputKind::Vector(3));
    assert_eq!(multiclass.output_kind(ForestOutput::ClassIndex).unwrap(), ForestOutputKind::ClassIndex { classes: 3 });
    assert_eq!(multiclass.output_kind(ForestOutput::ClassIndex).unwrap().width(), 1);
    assert_eq!(binary.output_kind(ForestOutput::ClassIndex).unwrap_err().code(), ErrorCode::ForeignInvalidArgument);
}

#[test]
fn inspect_rejects_models_the_importer_cannot_execute() {
    let multi_target = inspect_forest_model(&fixture("xgb_multi_target.json"), ForestModelFormat::XgboostJson)
        .expect_err("nvForest rejects multi-target models");
    assert_eq!(multi_target.code(), ErrorCode::ForeignInvalidArgument);
    assert!(multi_target.message().contains("multi-target"), "{multi_target}");

    let corrupt = inspect_forest_model(&fixture("corrupt.json"), ForestModelFormat::XgboostJson)
        .expect_err("truncated JSON is rejected");
    assert_eq!(corrupt.code(), ErrorCode::ForeignInvalidArgument);
    assert!(corrupt.message().starts_with("XGBoost JSON parse: "), "{corrupt}");

    let wrong_format = inspect_forest_model(&fixture("xgb_binary_logistic.json"), ForestModelFormat::LightGbm)
        .expect_err("format is not inferred from content");
    assert_eq!(wrong_format.code(), ErrorCode::ForeignInvalidArgument);

    assert_eq!(inspect_forest_model(&[], ForestModelFormat::XgboostJson).unwrap_err().code(), ErrorCode::EmptyInput);
}
