use cuvs::{
    BruteForceIndex, CudaStream, DeviceMatrixF32, DistanceMetric, KMeansInit, KMeansModel, KMeansParams, MatrixLayout,
    PcaFitTransform, PcaParams, Resources, device_count,
};

fn require_gpu() {
    assert!(device_count().expect("query CUDA device count") > 0);
}

#[test]
fn resources_can_set_stream() {
    require_gpu();
    let mut resources = Resources::new().expect("resources");
    let stream = CudaStream::new().expect("stream");
    stream.set_on(&mut resources).expect("set stream");
}

#[test]
fn brute_force_knn_overfetch_drops_self_for_ten_points() {
    require_gpu();
    let resources = Resources::new().expect("resources");
    let points =
        [0.0_f32, 0.0, 1.0, 0.0, 3.0, 0.0, 6.0, 0.0, 10.0, 0.0, 15.0, 0.0, 21.0, 0.0, 28.0, 0.0, 36.0, 0.0, 45.0, 0.0];
    let matrix = DeviceMatrixF32::from_host(10, 2, MatrixLayout::RowMajor, &points).expect("matrix");
    let index = BruteForceIndex::build(&resources, &matrix.as_view(), DistanceMetric::L2Expanded).expect("build");
    let result = index.search_overfetch_drop_self(&resources, &matrix.as_view(), 3).expect("search");

    assert_eq!(result.n_queries, 10);
    assert_eq!(result.k, 3);
    for row in 0..10 {
        let neighbors = result.row_neighbors(row);
        assert_eq!(neighbors.len(), 3);
        assert!(!neighbors.contains(&(row as i64)), "row {row} should not contain itself: {neighbors:?}");
        assert!(neighbors.iter().all(|&idx| (0..10).contains(&idx)));
        assert!(result.row_distances(row).iter().all(|distance| distance.is_finite()));
    }
    assert_eq!(result.row_neighbors(0), &[1, 2, 3]);
    assert_eq!(result.row_neighbors(9), &[8, 7, 6]);
}

#[test]
fn kmeans_fit_predict_three_clusters() {
    require_gpu();
    let resources = Resources::new().expect("resources");
    let data =
        [0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 11.0, 11.0, 10.0, -10.0, 10.0, -10.0, 11.0, -9.0, 10.0];
    let matrix = DeviceMatrixF32::from_host(9, 2, MatrixLayout::RowMajor, &data).expect("matrix");
    let mut params = KMeansParams::new(3);
    params.init = KMeansInit::Array;
    params.max_iter = 100;
    params.tol = 1e-6;
    let initial_centroids = [0.0_f32, 0.0, 10.0, 10.0, -10.0, 10.0];

    let model = KMeansModel::fit(&resources, &matrix.as_view(), &params, Some(&initial_centroids)).expect("fit");
    let prediction = model.predict(&resources, &matrix.as_view(), &params).expect("predict");

    assert_eq!(prediction.labels.len(), 9);
    assert!(model.fit_result().n_iter > 0);
    assert!(model.fit_result().inertia.is_finite());
    assert!(prediction.inertia.is_finite());

    for cluster in prediction.labels.chunks_exact(3) {
        assert_eq!(cluster[0], cluster[1]);
        assert_eq!(cluster[1], cluster[2]);
    }
    let mut distinct = prediction.labels.chunks_exact(3).map(|cluster| cluster[0]).collect::<Vec<_>>();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 3, "expected one label per cluster group");
}

#[test]
fn pca_fit_transform_reduces_to_two_dimensions() {
    require_gpu();
    let resources = Resources::new().expect("resources");
    let rows = 6;
    let cols = 3;
    let col_major = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 4.0, 3.0, 6.0, 5.0, -1.0, 0.5, 2.0, 3.5, 5.0, 6.5];
    let matrix = DeviceMatrixF32::from_host(rows, cols, MatrixLayout::ColumnMajor, &col_major).expect("matrix");
    let params = PcaParams::new(2);

    let output = PcaFitTransform::fit_transform(&resources, &matrix.as_view(), &params).expect("pca");
    let transformed = output.transformed_to_host().expect("transformed");

    assert_eq!(output.rows(), rows);
    assert_eq!(output.n_components(), 2);
    assert_eq!(transformed.len(), rows * 2);
    assert!(transformed.iter().all(|value| value.is_finite()));
    assert_eq!(output.explained_variance().len(), 2);
    assert!(output.explained_variance().iter().all(|value| value.is_finite()));
    assert!(transformed.iter().any(|value| value.abs() > 1e-5), "transformed PCA output should not be all zeros");
}
