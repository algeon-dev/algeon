use std::sync::Arc;

use cudf::{DenseMatrixLayout, Runtime, core::types::TypeId};
use cuml::{ClusterSelectionMethod, CumlResources, DeviceMatrix, HdbscanFit, HdbscanParams};

#[test]
fn fit_finds_three_clusters_and_marks_outlier_as_noise() {
    assert!(cudf::DeviceId::count().expect("CUDA device count") > 0);
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let centers = [[0.0_f32, 0.0_f32], [20.0, 20.0], [-20.0, 20.0]];
    let mut host = Vec::with_capacity(25 * 2);
    for (cluster, center) in centers.into_iter().enumerate() {
        for point in 0..8 {
            host.extend([
                center[0] + (point % 4) as f32 * 0.04,
                center[1] + (point / 4) as f32 * 0.04 + cluster as f32 * 0.01,
            ]);
        }
    }
    host.extend([1000.0, -1000.0]);
    let rows = host.len() / 2;
    let buffer = runtime.device_buffer_from_host_data(&host).expect("upload training data");
    let values = runtime.column_from_device_buffer(buffer, host.len(), TypeId::Float32).expect("training column");
    let packed = runtime.pack_flat_column(&values, rows, 2, DenseMatrixLayout::RowMajor).expect("pack training matrix");
    let validated = packed.validate().expect("finite training matrix");
    let input = DeviceMatrix::from_validated(&validated).expect("device matrix");
    let output = HdbscanFit::fit(
        &resources,
        &input,
        HdbscanParams {
            min_samples: 3,
            min_cluster_size: 5,
            cluster_selection_method: ClusterSelectionMethod::Eom,
            ..HdbscanParams::default()
        },
    )
    .expect("fit HDBSCAN");

    assert_eq!(output.rows(), rows);
    assert_eq!(output.output.labels.len(), rows);
    assert_eq!(output.output.probabilities.len(), rows);
    assert_eq!(output.output.core_distances.len(), rows);
    assert_eq!(output.output.single_linkage_children.len(), 2 * (rows - 1));
    assert_eq!(output.output.single_linkage_sizes.len(), rows - 1);
    assert_eq!(output.output.single_linkage_distances.len(), rows - 1);
    assert_eq!(output.output.mst_sources.len(), rows - 1);
    assert_eq!(output.output.mst_destinations.len(), rows - 1);
    assert_eq!(output.output.mst_weights.len(), rows - 1);

    assert_eq!(output.n_clusters(), 3);
    let labels = cudf::Series::from_column(output.output.labels.into_column().expect("validated labels"))
        .expect("label series")
        .to_vec_strict::<i64>()
        .expect("Int64 labels");
    let clusters = labels.iter().copied().filter(|label| *label >= 0).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(clusters.len(), 3, "expected three non-noise clusters, got labels {labels:?}");
    assert_eq!(labels.last(), Some(&-1), "isolated outlier should be noise, got labels {labels:?}");
    let probabilities =
        cudf::Series::from_column(output.output.probabilities.into_column().expect("validated probabilities"))
            .expect("probability series")
            .to_vec_strict::<f32>()
            .expect("Float32 probabilities");
    assert!(probabilities.iter().all(|value| (0.0..=1.0).contains(value)));
    let distances =
        cudf::Series::from_column(output.output.core_distances.into_column().expect("validated core distances"))
            .expect("core-distance series")
            .to_vec_strict::<f32>()
            .expect("Float32 core distances");
    assert!(distances.iter().all(|distance| *distance >= 0.0));
}
