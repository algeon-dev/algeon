use std::{
    ffi::{c_char, c_int, c_void},
    sync::Arc,
};

use cudf::{
    DenseMatrixLayout, Runtime,
    core::{Column, types::TypeId},
};
use cuml::{CumlResources, DeviceBuffer, DeviceMatrix, OwnedFloat32Matrix, PcaModel, PcaParams, TsvdModel, TsvdParams};

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

fn series_values(runtime: &Runtime, buffer: &DeviceBuffer<f32>) -> Vec<f32> {
    // The runtime shares the cuML execution binding, so the Series copy follows the output writes.
    let column = unsafe {
        runtime
            .column_from_device_ptr(buffer.as_ptr().expect("device buffer pointer"), buffer.len(), TypeId::Float32)
            .expect("copy validated output to a cuDF column")
    };
    cudf::Series::from_column(column).expect("output series").to_vec_strict::<f32>().expect("Float32 output values")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite(), "non-finite value: {actual}");
        assert!((actual - expected).abs() <= tolerance, "{actual} differs from {expected}");
    }
}

fn output_column(runtime: &Runtime, buffer: &DeviceBuffer<f32>) -> Column {
    unsafe {
        runtime
            .column_from_device_ptr(buffer.as_ptr().expect("device buffer pointer"), buffer.len(), TypeId::Float32)
            .expect("copy transformed output to a cuDF column")
    }
}

fn copy_packed_input(resources: &CumlResources, packed: &cudf::ValidatedPackedMatrix<'_>) -> OwnedFloat32Matrix {
    let mut input =
        OwnedFloat32Matrix::new(resources, packed.rows(), packed.columns(), cuml::MatrixLayout::ColumnMajor)
            .expect("owned input allocation");
    let source = packed.as_ptr().expect("packed input pointer");
    let destination = input.as_mut_ptr().expect("owned input pointer");
    let bytes = packed.rows() * packed.columns() * std::mem::size_of::<f32>();
    let status = cuda_memcpy(destination, source, bytes, 3);
    assert_eq!(status, 0, "device-to-device copy failed with CUDA status {status}");
    input
}

#[test]
fn pca_transform_preserves_borrowed_matrix_bits() {
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let first = runtime.column_from_slice(&[0.1_f32, 0.3, 0.7, 1.1, 1.7, 2.3, 2.9, 3.1]).expect("first feature");
    let second = runtime.column_from_slice(&[0.2_f32, 0.4, 0.8, 1.2, 1.8, 2.4, 2.8, 3.2]).expect("second feature");
    let packed = runtime.pack_columns(&[&first, &second], DenseMatrixLayout::ColumnMajor).expect("column-major input");
    let validated = packed.validate().expect("finite input");
    let matrix = DeviceMatrix::from_validated(&validated).expect("borrowed matrix");
    let mut owned = copy_packed_input(&resources, &validated);
    let model = PcaModel::fit(&resources, &mut owned, PcaParams::new(1)).expect("PCA fit");
    let mut before = [0.0_f32; 16];
    let mut after = [0.0_f32; 16];
    let bytes = before.len() * std::mem::size_of::<f32>();
    assert_eq!(cuda_memcpy(before.as_mut_ptr().cast(), matrix.as_ptr(), bytes, 2), 0);
    model.transform(&resources, &matrix).expect("PCA transform");
    assert_eq!(cuda_memcpy(after.as_mut_ptr().cast(), matrix.as_ptr(), bytes, 2), 0);
    assert_eq!(before.map(f32::to_bits), after.map(f32::to_bits), "a borrowed input must remain bitwise unchanged");
}

#[test]
fn pca_and_truncated_svd_fit_transform_and_transform_rank_one_data() {
    assert!(cudf::DeviceId::count().expect("CUDA device count") > 0);
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
    let x = [-3.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let a = runtime.column_from_slice(&x).expect("column a");
    let b = runtime.column_from_slice(&x.map(|v| v * 2.0)).expect("column b");
    let c = runtime.column_from_slice(&x.map(|v| -v)).expect("column c");
    let packed = runtime.pack_columns(&[&a, &b, &c], DenseMatrixLayout::ColumnMajor).expect("column-major matrix");
    let packed = packed.validate().expect("finite matrix");
    let matrix = DeviceMatrix::from_validated(&packed).expect("matrix view");

    let mut pca_input = copy_packed_input(&resources, &packed);
    let (pca, pca_fit_output) =
        PcaModel::fit_transform(&resources, &mut pca_input, PcaParams::new(1)).expect("PCA fit_transform");
    assert_eq!(pca.n_features(), 3);
    assert_eq!(pca.n_components(), 1);
    assert_eq!(pca.explained_variance_ratio().len(), 1);
    let pca_ratio = series_values(&runtime, pca.explained_variance_ratio());
    assert!(pca_ratio[0] > 0.99, "rank-one input should explain nearly all variance: {pca_ratio:?}");
    assert_eq!(pca_fit_output.len(), x.len());
    assert!(pca.transform(&resources, &matrix).expect("PCA transform").len() == x.len());

    let mut tsvd_input = copy_packed_input(&resources, &packed);
    let (tsvd, tsvd_fit_output) =
        TsvdModel::fit_transform(&resources, &mut tsvd_input, TsvdParams::new(1)).expect("TSVD fit_transform");
    assert_eq!(tsvd.n_features(), 3);
    assert_eq!(tsvd.n_components(), 1);
    assert_eq!(tsvd.explained_variance_ratio().len(), 1);
    let tsvd_ratio = series_values(&runtime, tsvd.explained_variance_ratio());
    assert!(tsvd_ratio[0] > 0.99, "rank-one input should explain nearly all variance: {tsvd_ratio:?}");
    assert_eq!(tsvd_fit_output.len(), x.len());
    assert_eq!(tsvd.transform(&resources, &matrix).expect("TSVD transform").len(), x.len());

    let mut tsvd_fit_input = copy_packed_input(&resources, &packed);
    let fitted_tsvd = TsvdModel::fit(&resources, &mut tsvd_fit_input, TsvdParams::new(1)).expect("TSVD fit");
    let fit_ratio = series_values(&runtime, fitted_tsvd.explained_variance_ratio());
    let fit_variance = series_values(&runtime, fitted_tsvd.explained_variance());
    let fit_singular_values = series_values(&runtime, fitted_tsvd.singular_values());
    assert!(fit_ratio.iter().all(|value| value.is_finite()));
    assert!(fit_variance.iter().all(|value| value.is_finite()));
    assert!(fit_singular_values.iter().all(|value| value.is_finite()));
    assert_close(&fit_ratio, &tsvd_ratio, 1e-5);
    assert_close(&fit_variance, &series_values(&runtime, tsvd.explained_variance()), 1e-5);
    assert_close(&fit_singular_values, &series_values(&runtime, tsvd.singular_values()), 1e-5);

    let pca_transformed = pca.transform(&resources, &matrix).expect("PCA transform for inverse");
    let pca_latent_column = output_column(&runtime, &pca_transformed);
    let pca_latent_columns = [&pca_latent_column];
    let pca_latent_packed =
        runtime.pack_columns(&pca_latent_columns, DenseMatrixLayout::ColumnMajor).expect("pack PCA latent output");
    let pca_latent = pca_latent_packed.validate().expect("finite PCA latent matrix");
    let pca_latent_matrix = DeviceMatrix::from_validated(&pca_latent).expect("PCA latent matrix");
    let pca_reconstructed = pca.inverse_transform(&resources, &pca_latent_matrix).expect("PCA inverse_transform");
    assert_close(
        &series_values(&runtime, &pca_reconstructed),
        &[
            -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 3.0, 2.0, 1.0, 0.0,
            -1.0, -2.0, -3.0, -4.0,
        ],
        1e-4,
    );

    let tsvd_transformed = tsvd.transform(&resources, &matrix).expect("TSVD transform for inverse");
    let tsvd_latent_column = output_column(&runtime, &tsvd_transformed);
    let tsvd_latent_columns = [&tsvd_latent_column];
    let tsvd_latent_packed =
        runtime.pack_columns(&tsvd_latent_columns, DenseMatrixLayout::ColumnMajor).expect("pack TSVD latent output");
    let tsvd_latent = tsvd_latent_packed.validate().expect("finite TSVD latent matrix");
    let tsvd_latent_matrix = DeviceMatrix::from_validated(&tsvd_latent).expect("TSVD latent matrix");
    let tsvd_reconstructed = tsvd.inverse_transform(&resources, &tsvd_latent_matrix).expect("TSVD inverse_transform");
    assert_close(
        &series_values(&runtime, &tsvd_reconstructed),
        &[
            -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 3.0, 2.0, 1.0, 0.0,
            -1.0, -2.0, -3.0, -4.0,
        ],
        1e-4,
    );
}
