use static_assertions::assert_not_impl_any;

assert_not_impl_any!(cuvs::CuvsResources: Send, Sync);
assert_not_impl_any!(cuvs::BruteForceIndex<'static>: Send, Sync);
assert_not_impl_any!(cuvs::KMeansModel: Send, Sync);
assert_not_impl_any!(cuvs::PreparedTensorLease<'static, 'static, 'static, f32>: Send, Sync);
assert_not_impl_any!(cuvs::PreparedValidatedTensorLease<'static, 'static, 'static>: Send, Sync);

#[test]
fn kmeans_safe_layer_uses_only_the_pinned_calls() {
    let source = include_str!("../src/kmeans.rs");
    for required in ["cuvsKMeansParamsCreate", "cuvsKMeansParamsDestroy", "cuvsKMeansFit", "cuvsKMeansPredict"] {
        assert!(source.contains(required), "safe KMeans layer must call {required}");
    }
}

#[test]
fn brute_force_validates_native_outputs_before_success() {
    let source = include_str!("../src/brute_force.rs");
    assert!(source.contains("neighbors.validate_native_index_range(0, indexed_rows, operation)?"));
    assert!(source.contains("distances.validate_native_input(operation)?"));
}

#[test]
fn pairwise_and_selection_use_pinned_calls_and_validate_outputs() {
    let pairwise = include_str!("../src/pairwise.rs");
    assert!(pairwise.contains("cuvsPairwiseDistance("));
    assert!(pairwise.contains("out.validate_native_input(operation)"));
    assert!(pairwise.contains("validate_matrix_binding(resources, x, operation)?"));
    assert!(pairwise.contains("validate_matrix_binding(resources, y, operation)?"));

    let selection = include_str!("../src/select_k.rs");
    assert!(selection.contains("cuvsSelectKBatched("));
    assert!(selection.contains("out_positions.validate_native_index_range("));
    assert!(selection.contains("out_values.validate_native_input(operation)"));
}

#[test]
fn kmeans_revalidates_native_centroids_before_model_construction() {
    let source = include_str!("../src/kmeans.rs");
    assert!(source.contains("centroids.validate_native_output_finite(resources, operation)?"));
    assert!(!source.contains("mark_native_output_finite"));
}

#[test]
fn pca_validates_every_native_float_output_before_adoption() {
    let source = include_str!("../src/pca.rs");
    for field in [
        "transformed",
        "components",
        "explained_variance",
        "explained_variance_ratio",
        "singular_values",
        "mean",
        "noise_variance",
    ] {
        assert!(
            source.contains(&format!("self.{field}.validate_native_input(operation)?")),
            "PCA must finite-validate {field} before returning success"
        );
    }
}

#[test]
fn algorithm_boundaries_reject_binding_and_device_mismatches() {
    let brute_force = include_str!("../src/brute_force.rs");
    assert!(brute_force.contains("matrix.binding_id() != resources.binding_id()"));
    assert!(brute_force.contains("matrix.device() != resources.device().ordinal() as i32"));

    let pca = include_str!("../src/pca.rs");
    assert!(pca.contains("input.binding_id() != resources.binding_id()"));
    assert!(pca.contains("input.device() != resources.device().ordinal() as i32"));
}
