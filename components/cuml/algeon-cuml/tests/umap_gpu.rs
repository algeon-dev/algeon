use std::{
    ffi::{c_char, c_int, c_void},
    sync::Arc,
};

use cudf::{DenseMatrixLayout, Runtime, core::types::TypeId, testing::run_named_cases};
use cuml::{CumlResources, DeviceMatrix, ErrorCode, MatrixLayout, OwnedFloat32Matrix, UmapModel, UmapParams};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

fn cuda_memcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32 {
    const RTLD_NOW: c_int = 2;
    let library = b"libcudart.so\0";
    let symbol = b"cudaMemcpy\0";
    let handle = unsafe { dlopen(library.as_ptr().cast(), RTLD_NOW) };
    assert!(!handle.is_null(), "could not load libcudart.so");
    let address = unsafe { dlsym(handle, symbol.as_ptr().cast()) };
    assert!(!address.is_null(), "libcudart.so does not export cudaMemcpy");
    let copy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32 =
        unsafe { std::mem::transmute(address) };
    let status = unsafe { copy(dst, src, count, kind) };
    assert_eq!(unsafe { dlclose(handle) }, 0, "could not close libcudart.so");
    status
}

fn fit_and_transform_preserve_the_two_dimensional_embedding_shape() {
    let runtime = Arc::new(Runtime::builder().build().expect("GPU runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let rows = 24;
    let columns = 3;
    let host: Vec<f32> = (0..rows * columns).map(|i| ((i * 17 % 29) as f32) / 11.0).collect();
    let buffer = runtime.device_buffer_from_host_data(&host).expect("upload training data");
    let values = runtime.column_from_device_buffer(buffer, host.len(), TypeId::Float32).expect("training column");
    let packed =
        runtime.pack_flat_column(&values, rows, columns, DenseMatrixLayout::RowMajor).expect("pack training matrix");
    let validated = packed.validate().expect("finite training matrix");
    let input = DeviceMatrix::from_validated(&validated).expect("device matrix");
    let mut owned =
        OwnedFloat32Matrix::new(&resources, rows, columns, MatrixLayout::RowMajor).expect("owned training allocation");
    let status = cuda_memcpy(
        owned.as_mut_ptr().expect("owned training pointer"),
        validated.as_ptr().expect("validated training pointer"),
        host.len() * std::mem::size_of::<f32>(),
        3,
    );
    assert_eq!(status, 0, "device-to-device copy failed with CUDA status {status}");
    let mut model = UmapModel::fit(
        &resources,
        owned,
        UmapParams {
            n_neighbors: 5,
            n_components: 2,
            n_epochs: 20,
            learning_rate: 0.8,
            min_dist: 0.2,
            spread: 1.2,
            set_op_mix_ratio: 0.75,
            local_connectivity: 1.5,
            repulsion_strength: 1.25,
            negative_sample_rate: 4,
            transform_queue_size: 3.0,
            random_state: 7,
            ..UmapParams::default()
        },
    )
    .expect("fit UMAP");
    assert_eq!((model.embedding().rows(), model.embedding().columns()), (rows, 2));
    let transformed = model.transform(&resources, &input).expect("transform fitted rows");
    assert_eq!((transformed.rows(), transformed.columns()), (rows, 2));
    assert_eq!(transformed.layout(), MatrixLayout::RowMajor);
    assert_eq!(transformed.binding_id(), resources.binding_id());
    let mut fitted_values = vec![0.0_f32; rows * 2];
    let mut transformed_values = vec![0.0_f32; rows * 2];
    assert_eq!(
        cuda_memcpy(
            fitted_values.as_mut_ptr().cast(),
            model.embedding().as_ptr().expect("embedding pointer"),
            fitted_values.len() * std::mem::size_of::<f32>(),
            2,
        ),
        0,
        "embedding download failed",
    );
    assert_eq!(
        cuda_memcpy(
            transformed_values.as_mut_ptr().cast(),
            transformed.as_ptr().expect("transform pointer"),
            transformed_values.len() * std::mem::size_of::<f32>(),
            2,
        ),
        0,
        "transform download failed",
    );
    assert!(fitted_values.iter().all(|value| value.is_finite()));
    assert!(transformed_values.iter().all(|value| value.is_finite()));
    let mean_squared_distance = fitted_values
        .iter()
        .zip(&transformed_values)
        .map(|(fitted, transformed)| (fitted - transformed).powi(2))
        .sum::<f32>()
        / fitted_values.len() as f32;
    assert!(mean_squared_distance < 2.0, "self-transform MSE was {mean_squared_distance}");
}

fn assert_parameters_rejected(parameters: UmapParams) {
    let runtime = Arc::new(Runtime::builder().build().expect("GPU runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let owned = OwnedFloat32Matrix::new(&resources, 8, 3, MatrixLayout::RowMajor).expect("training allocation");
    let error = UmapModel::fit(&resources, owned, parameters).err().expect("invalid UMAP parameters");
    assert_eq!(error.code(), ErrorCode::ForeignInvalidArgument);
}

fn rejects_zero_neighbors() {
    assert_parameters_rejected(UmapParams { n_neighbors: 0, ..UmapParams::default() });
}

fn rejects_neighbor_count_equal_to_training_rows() {
    assert_parameters_rejected(UmapParams { n_neighbors: 8, ..UmapParams::default() });
}

fn rejects_zero_components() {
    assert_parameters_rejected(UmapParams { n_components: 0, ..UmapParams::default() });
}

fn rejects_epoch_count_outside_native_range() {
    assert_parameters_rejected(UmapParams { n_epochs: usize::MAX, ..UmapParams::default() });
}

fn rejects_non_finite_learning_rate() {
    assert_parameters_rejected(UmapParams { learning_rate: f32::NAN, ..UmapParams::default() });
}

fn rejects_negative_min_dist() {
    assert_parameters_rejected(UmapParams { min_dist: -0.1, ..UmapParams::default() });
}

fn rejects_zero_spread() {
    assert_parameters_rejected(UmapParams { spread: 0.0, ..UmapParams::default() });
}

fn rejects_set_op_mix_ratio_above_one() {
    assert_parameters_rejected(UmapParams { set_op_mix_ratio: 1.1, ..UmapParams::default() });
}

fn rejects_nonpositive_optimizer_parameters() {
    assert_parameters_rejected(UmapParams { negative_sample_rate: 0, ..UmapParams::default() });
    assert_parameters_rejected(UmapParams { transform_queue_size: 0.0, ..UmapParams::default() });
}

#[test]
fn umap_cases() {
    let cases: &[(&str, fn())] = &[
        (
            "fit_and_transform_preserve_the_two_dimensional_embedding_shape",
            fit_and_transform_preserve_the_two_dimensional_embedding_shape,
        ),
        ("rejects_zero_neighbors", rejects_zero_neighbors),
        ("rejects_neighbor_count_equal_to_training_rows", rejects_neighbor_count_equal_to_training_rows),
        ("rejects_zero_components", rejects_zero_components),
        ("rejects_epoch_count_outside_native_range", rejects_epoch_count_outside_native_range),
        ("rejects_non_finite_learning_rate", rejects_non_finite_learning_rate),
        ("rejects_negative_min_dist", rejects_negative_min_dist),
        ("rejects_zero_spread", rejects_zero_spread),
        ("rejects_set_op_mix_ratio_above_one", rejects_set_op_mix_ratio_above_one),
        ("rejects_nonpositive_optimizer_parameters", rejects_nonpositive_optimizer_parameters),
    ];
    run_named_cases(cases);
}
