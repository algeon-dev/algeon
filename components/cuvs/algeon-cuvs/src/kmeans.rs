use std::{ptr, ptr::NonNull};

use crate::{
    brute_force::{matrix_layout, validate_matrix_binding},
    device::{DeviceBuffer, MatrixLayout, OwnedFloat32Matrix, ValidatedDeviceTensor},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, invalid_input, invalid_state, require_nonzero},
    resources::CuvsResources,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansInit {
    KMeansPlusPlus,
    Random,
    Array,
}

impl KMeansInit {
    fn raw(self) -> cuvs_sys::cuvsKMeansInitMethod {
        match self {
            KMeansInit::KMeansPlusPlus => cuvs_sys::cuvsKMeansInitMethod::KMeansPlusPlus,
            KMeansInit::Random => cuvs_sys::cuvsKMeansInitMethod::Random,
            KMeansInit::Array => cuvs_sys::cuvsKMeansInitMethod::Array,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansMetric {
    L2Expanded,
    L2SqrtExpanded,
}

impl KMeansMetric {
    fn raw(self) -> cuvs_sys::cuvsDistanceType {
        match self {
            Self::L2Expanded => cuvs_sys::cuvsDistanceType::L2Expanded,
            Self::L2SqrtExpanded => cuvs_sys::cuvsDistanceType::L2SqrtExpanded,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeansParams {
    pub n_clusters: usize,
    pub max_iter: i32,
    pub tol: f64,
    pub n_init: i32,
    pub init: KMeansInit,
    pub metric: KMeansMetric,
}

impl KMeansParams {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 100,
            tol: 1e-4,
            n_init: 1,
            init: KMeansInit::KMeansPlusPlus,
            metric: KMeansMetric::L2Expanded,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KMeansFit {
    pub inertia: f64,
    pub n_iter: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct KMeansPrediction {
    pub rows: usize,
    pub inertia: f64,
}

#[derive(Debug)]
pub struct KMeansModel {
    resources: CuvsResources,
    params: KMeansParams,
    centroids: OwnedFloat32Matrix,
    n_features: usize,
    fit: KMeansFit,
}

struct ParamsHandle {
    raw: Option<NonNull<cuvs_sys::cuvsKMeansParams>>,
}

impl ParamsHandle {
    fn new(params: &KMeansParams, operation: &'static str) -> Result<Self> {
        validate_params(params, operation)?;
        let mut raw = ptr::null_mut();
        unsafe { check_cuvs(cuvs_sys::cuvsKMeansParamsCreate(&mut raw), operation)? };
        let raw = NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned null KMeans params"))?;
        unsafe {
            let native = raw.as_ptr();
            (*native).metric = params.metric.raw();
            (*native).n_clusters = params.n_clusters as i32;
            (*native).init = params.init.raw();
            (*native).max_iter = params.max_iter;
            (*native).tol = params.tol;
            (*native).n_init = params.n_init;
            (*native).hierarchical = false;
            (*native).device_buffer_samples = 0;
            (*native).init_size = 0;
        }
        Ok(Self { raw: Some(raw) })
    }

    fn as_ptr(&self, operation: &'static str) -> Result<cuvs_sys::cuvsKMeansParams_t> {
        self.raw.map(NonNull::as_ptr).ok_or_else(|| invalid_state(operation, "KMeans params have already been closed"))
    }

    fn close(&mut self, operation: &'static str) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        let result = unsafe { check_cuvs(cuvs_sys::cuvsKMeansParamsDestroy(raw.as_ptr()), operation) };
        if result.is_err() {
            self.raw = Some(raw);
        }
        result
    }
}

impl Drop for ParamsHandle {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe {
                let _ = cuvs_sys::cuvsKMeansParamsDestroy(raw.as_ptr());
            }
        }
    }
}

impl KMeansModel {
    pub fn fit(
        resources: &CuvsResources,
        data: &ValidatedDeviceTensor<'_>,
        params: KMeansParams,
        initial_centroids: Option<DeviceBuffer<f32>>,
    ) -> Result<Self> {
        let operation = "KMeansModel::fit";
        validate_matrix_binding(resources, data, operation)?;
        validate_params(&params, operation)?;
        let n_features = data.columns()?;
        let initial_centroids = initial_centroids
            .map(|centroids| {
                OwnedFloat32Matrix::from_buffer(
                    resources,
                    centroids,
                    params.n_clusters,
                    n_features,
                    MatrixLayout::RowMajor,
                    operation,
                )
            })
            .transpose()?;
        Self::fit_with_initial_centroids(resources, data, params, initial_centroids)
    }

    /// Fit KMeans with an optional owned row-major centroid relation.
    ///
    /// Consuming the relation provides the exclusive mutable ownership cuVS
    /// requires for `KMeansInit::Array`, including when the centroids came from
    /// a cudf packed table.
    pub fn fit_with_initial_centroids(
        resources: &CuvsResources,
        data: &ValidatedDeviceTensor<'_>,
        params: KMeansParams,
        initial_centroids: Option<OwnedFloat32Matrix>,
    ) -> Result<Self> {
        let operation = "KMeansModel::fit_with_initial_centroids";
        validate_matrix_binding(resources, data, operation)?;
        if matrix_layout(data, operation)? != MatrixLayout::RowMajor {
            return Err(invalid_input(operation, "KMeans input must be contiguous row-major"));
        }
        validate_params(&params, operation)?;
        let rows = data.rows()?;
        validate_row_count(rows, operation)?;
        let n_features = data.columns()?;
        validate_feature_count(n_features, operation)?;
        if params.n_clusters > rows {
            return Err(invalid_input(
                operation,
                format!("n_clusters ({}) must be <= rows ({rows})", params.n_clusters),
            ));
        }
        if matches!(params.init, KMeansInit::Array) != initial_centroids.is_some() {
            return Err(invalid_input(
                operation,
                "initial centroids must be supplied exactly when init is KMeansInit::Array",
            ));
        }
        let mut centroids = match initial_centroids {
            Some(centroids) => {
                if !centroids.is_finite_validated() {
                    return Err(invalid_input(operation, "initial centroids must carry finite-input validation"));
                }
                if centroids.rows() != params.n_clusters || centroids.columns() != n_features {
                    return Err(invalid_input(
                        operation,
                        format!(
                            "initial centroid shape [{}, {}] must equal [{}, {n_features}]",
                            centroids.rows(),
                            centroids.columns(),
                            params.n_clusters
                        ),
                    ));
                }
                if centroids.layout() != MatrixLayout::RowMajor {
                    return Err(invalid_input(operation, "initial centroids must be a contiguous row-major relation"));
                }
                if centroids.binding_id() != resources.binding_id() {
                    return Err(invalid_input(
                        operation,
                        "initial centroids and cuVS resources must use the same exact runtime binding",
                    )
                    .with_context("matrix_binding_id", centroids.binding_id())
                    .with_context("resource_binding_id", resources.binding_id()));
                }
                centroids
            },
            None => OwnedFloat32Matrix::new(resources, params.n_clusters, n_features, MatrixLayout::RowMajor)?,
        };
        let mut centroids_view = centroids.as_tensor_mut(resources)?;
        let mut inertia = 0.0;
        let mut n_iter = 0;
        resources.with_native(operation, |native| {
            let mut params_handle = ParamsHandle::new(&params, operation)?;
            let mut data_tensor = DlpackTensor::from_const(data.as_tensor());
            let mut centroids_tensor = DlpackTensor::from_mut(&mut centroids_view);
            let call_result = unsafe {
                check_cuvs(
                    cuvs_sys::cuvsKMeansFit(
                        native,
                        params_handle.as_ptr(operation)?,
                        data_tensor.as_mut_ptr(),
                        ptr::null_mut(),
                        centroids_tensor.as_mut_ptr(),
                        &mut inertia,
                        &mut n_iter,
                    ),
                    operation,
                )
            };
            finish_params_call(call_result, params_handle.close(operation))
        })?;
        if !inertia.is_finite() || inertia < 0.0 {
            return Err(invalid_state(operation, "native KMeans fit returned non-finite or negative inertia"));
        }
        if n_iter < 0 || n_iter > params.max_iter {
            return Err(invalid_state(
                operation,
                format!("native KMeans fit returned invalid iteration count {n_iter}"),
            ));
        }
        centroids.validate_native_output_finite(resources, operation)?;

        Ok(Self { resources: resources.retain(), params, centroids, n_features, fit: KMeansFit { inertia, n_iter } })
    }

    pub fn predict_into(
        &self,
        data: &ValidatedDeviceTensor<'_>,
        labels: &mut DeviceBuffer<i32>,
    ) -> Result<KMeansPrediction> {
        let operation = "KMeansModel::predict_into";
        validate_matrix_binding(&self.resources, data, operation)?;
        if matrix_layout(data, operation)? != MatrixLayout::RowMajor {
            return Err(invalid_input(operation, "KMeans input must be contiguous row-major"));
        }
        if data.columns()? != self.n_features {
            return Err(invalid_input(
                operation,
                format!("input columns {} must match model features {}", data.columns()?, self.n_features),
            ));
        }
        validate_feature_count(self.n_features, operation)?;
        let rows = data.rows()?;
        validate_row_count(rows, operation)?;
        if labels.len() != rows {
            return Err(invalid_input(
                operation,
                format!("labels contain {} elements, expected one per input row ({rows})", labels.len()),
            ));
        }

        let mut labels_view = labels.as_tensor_mut(&self.resources, &[rows], None, 0)?;
        let centroids_view = self.centroids.as_validated_tensor(&self.resources)?;
        let mut inertia = 0.0;
        self.resources.with_native(operation, |native| {
            let mut params_handle = ParamsHandle::new(&self.params, operation)?;
            let mut data_tensor = DlpackTensor::from_const(data.as_tensor());
            let mut centroids_tensor = DlpackTensor::from_const(centroids_view.as_tensor());
            let mut labels_tensor = DlpackTensor::from_mut(&mut labels_view);
            let call_result = unsafe {
                check_cuvs(
                    cuvs_sys::cuvsKMeansPredict(
                        native,
                        params_handle.as_ptr(operation)?,
                        data_tensor.as_mut_ptr(),
                        ptr::null_mut(),
                        centroids_tensor.as_mut_ptr(),
                        labels_tensor.as_mut_ptr(),
                        false,
                        &mut inertia,
                    ),
                    operation,
                )
            };
            finish_params_call(call_result, params_handle.close(operation))
        })?;
        if !inertia.is_finite() || inertia < 0.0 {
            return Err(invalid_state(operation, "native KMeans prediction returned non-finite or negative inertia"));
        }
        Ok(KMeansPrediction { rows, inertia })
    }

    pub fn params(&self) -> &KMeansParams {
        &self.params
    }

    pub fn fit_result(&self) -> KMeansFit {
        self.fit
    }

    pub fn centroids(&self) -> &OwnedFloat32Matrix {
        &self.centroids
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

fn validate_params(params: &KMeansParams, operation: &'static str) -> Result<()> {
    require_nonzero(params.n_clusters, operation, "n_clusters")?;
    if params.n_clusters > i32::MAX as usize {
        return Err(invalid_input(operation, "n_clusters exceeds the cuVS i32 parameter range"));
    }
    if params.max_iter <= 0 {
        return Err(invalid_input(operation, "max_iter must be greater than zero"));
    }
    if params.n_init <= 0 {
        return Err(invalid_input(operation, "n_init must be greater than zero"));
    }
    if !params.tol.is_finite() || params.tol < 0.0 {
        return Err(invalid_input(operation, "tol must be finite and non-negative"));
    }
    Ok(())
}

fn validate_row_count(rows: usize, operation: &'static str) -> Result<()> {
    if rows > i32::MAX as usize {
        return Err(invalid_input(operation, "KMeans row count exceeds the cuVS i32 range"));
    }
    Ok(())
}

fn validate_feature_count(columns: usize, operation: &'static str) -> Result<()> {
    if columns > i32::MAX as usize {
        return Err(invalid_input(operation, "KMeans feature count exceeds the cuVS prediction i32 range"));
    }
    Ok(())
}

fn finish_params_call(call: Result<()>, close: Result<()>) -> Result<()> {
    match (call, close) {
        (Err(error), _) => Err(error),
        (Ok(()), close) => close,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_count_accepts_cuvs_i32_maximum() {
        validate_row_count(i32::MAX as usize, "test").expect("i32::MAX rows are representable");
    }

    #[test]
    fn row_count_rejects_one_past_cuvs_i32_maximum() {
        let error = validate_row_count(i32::MAX as usize + 1, "test").expect_err("row count must fit i32");
        assert!(error.message().contains("row count exceeds"));
    }

    #[test]
    fn feature_count_rejects_one_past_the_prediction_i32_maximum() {
        let error =
            validate_feature_count(i32::MAX as usize + 1, "test").expect_err("feature count must fit prediction i32");
        assert!(error.message().contains("feature count exceeds"));
    }
}
