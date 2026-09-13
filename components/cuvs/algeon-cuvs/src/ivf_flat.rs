use std::{ptr, ptr::NonNull};

use crate::{
    brute_force::{DistanceMetric, KnnSearchOutput, matrix_layout, validate_matrix_binding},
    device::{DeviceBuffer, MatrixLayout, ValidatedDeviceTensor},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, from_cudf, invalid_input, invalid_state, require_nonzero},
    resources::CuvsResources,
};

#[derive(Debug, Clone, Copy)]
pub struct IvfFlatIndexParams {
    pub metric: DistanceMetric,
    pub n_lists: u32,
    pub kmeans_n_iters: u32,
    pub kmeans_trainset_fraction: f64,
}

impl IvfFlatIndexParams {
    pub fn new(metric: DistanceMetric, n_lists: u32) -> Self {
        Self { metric, n_lists, kmeans_n_iters: 20, kmeans_trainset_fraction: 0.5 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IvfFlatSearchParams {
    pub n_probes: u32,
}

impl Default for IvfFlatSearchParams {
    fn default() -> Self {
        Self { n_probes: 20 }
    }
}

struct ParamsHandle<T> {
    raw: Option<NonNull<T>>,
    destroy: unsafe extern "C" fn(*mut T) -> cuvs_sys::cuvsError_t,
}

impl<T> ParamsHandle<T> {
    fn new(
        create: unsafe extern "C" fn(*mut *mut T) -> cuvs_sys::cuvsError_t,
        destroy: unsafe extern "C" fn(*mut T) -> cuvs_sys::cuvsError_t,
        operation: &'static str,
    ) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe { check_cuvs(create(&mut raw), operation)? };
        Ok(Self {
            raw: Some(NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned null IVF-Flat params"))?),
            destroy,
        })
    }

    fn as_ptr(&self) -> *mut T {
        self.raw.expect("params remain live during native call").as_ptr()
    }

    fn close(&mut self, operation: &'static str) -> Result<()> {
        let Some(raw) = self.raw.take() else { return Ok(()) };
        let result = unsafe { check_cuvs((self.destroy)(raw.as_ptr()), operation) };
        if result.is_err() {
            self.raw = Some(raw);
        }
        result
    }
}

impl<T> Drop for ParamsHandle<T> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe {
                let _ = (self.destroy)(raw.as_ptr());
            }
        }
    }
}

/// Query-local owned IVF-Flat index. Native build copies the indexed vectors.
pub struct IvfFlatIndex {
    raw: Option<NonNull<cuvs_sys::cuvsIvfFlatIndex>>,
    resources: CuvsResources,
    rows: usize,
    columns: usize,
    n_lists: u32,
}

impl IvfFlatIndex {
    pub fn build(
        resources: &CuvsResources,
        dataset: &ValidatedDeviceTensor<'_>,
        params: IvfFlatIndexParams,
    ) -> Result<Self> {
        let operation = "IvfFlatIndex::build";
        validate_matrix_binding(resources, dataset, operation)?;
        if matrix_layout(dataset, operation)? != MatrixLayout::RowMajor {
            return Err(invalid_input(operation, "IVF-Flat dataset must be contiguous row-major"));
        }
        let rows = dataset.rows()?;
        let columns = dataset.columns()?;
        require_nonzero(rows, operation, "dataset rows")?;
        if params.n_lists == 0 || params.n_lists as usize > rows {
            return Err(invalid_input(
                operation,
                format!("n_lists ({}) must be in 1..=dataset rows ({rows})", params.n_lists),
            ));
        }
        if params.kmeans_n_iters == 0
            || !params.kmeans_trainset_fraction.is_finite()
            || !(0.0..=1.0).contains(&params.kmeans_trainset_fraction)
            || params.kmeans_trainset_fraction == 0.0
        {
            return Err(invalid_input(
                operation,
                "kmeans_n_iters must be positive and kmeans_trainset_fraction must be in (0, 1]",
            ));
        }
        if params.metric == DistanceMetric::CosineExpanded && columns < 2 {
            return Err(invalid_input(operation, "cosine IVF-Flat requires at least two dimensions"));
        }
        resources.with_native(operation, |native| {
            let mut config = ParamsHandle::new(
                cuvs_sys::cuvsIvfFlatIndexParamsCreate,
                cuvs_sys::cuvsIvfFlatIndexParamsDestroy,
                operation,
            )?;
            unsafe {
                let config_ptr = config.as_ptr();
                (*config_ptr).metric = params.metric.raw();
                (*config_ptr).n_lists = params.n_lists;
                (*config_ptr).kmeans_n_iters = params.kmeans_n_iters;
                (*config_ptr).kmeans_trainset_fraction = params.kmeans_trainset_fraction;
            }
            let mut raw = ptr::null_mut();
            unsafe { check_cuvs(cuvs_sys::cuvsIvfFlatIndexCreate(&mut raw), operation)? };
            let raw =
                NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned a null IVF-Flat index"))?;
            let mut tensor = DlpackTensor::from_const(dataset.as_tensor());
            let built = unsafe {
                check_cuvs(
                    cuvs_sys::cuvsIvfFlatBuild(native, config.as_ptr(), tensor.as_mut_ptr(), raw.as_ptr()),
                    operation,
                )
            };
            let closed = config.close(operation);
            if let Err(error) = built.and(closed) {
                unsafe {
                    let _ = cuvs_sys::cuvsIvfFlatIndexDestroy(raw.as_ptr());
                }
                return Err(error);
            }
            Ok(Self { raw: Some(raw), resources: resources.retain(), rows, columns, n_lists: params.n_lists })
        })
    }

    /// Search outputs are `[queries.rows(), k]` row-major. Missing neighbors have ordinal `i64::MAX`.
    pub fn search_into(
        &self,
        queries: &ValidatedDeviceTensor<'_>,
        k: usize,
        params: IvfFlatSearchParams,
        neighbors: &mut DeviceBuffer<i64>,
        distances: &mut DeviceBuffer<f32>,
    ) -> Result<KnnSearchOutput> {
        let operation = "IvfFlatIndex::search_into";
        require_nonzero(k, operation, "k")?;
        if k > self.rows || k > u32::MAX as usize {
            return Err(invalid_input(
                operation,
                format!("k ({k}) must be <= indexed rows ({}) and fit u32", self.rows),
            ));
        }
        if params.n_probes == 0 || params.n_probes > self.n_lists {
            return Err(invalid_input(
                operation,
                format!("n_probes ({}) must be in 1..=n_lists ({})", params.n_probes, self.n_lists),
            ));
        }
        validate_matrix_binding(&self.resources, queries, operation)?;
        if matrix_layout(queries, operation)? != MatrixLayout::RowMajor || queries.columns()? != self.columns {
            return Err(invalid_input(
                operation,
                "IVF-Flat queries must be contiguous row-major with the indexed dimension",
            ));
        }
        let n_queries = queries.rows()?;
        let output_len = n_queries
            .checked_mul(k)
            .ok_or_else(|| invalid_input(operation, "search output element count overflowed usize"))?;
        if neighbors.len() != output_len || distances.len() != output_len {
            return Err(invalid_input(
                operation,
                format!("neighbors and distances must each contain n_queries * k ({output_len}) elements"),
            ));
        }
        let mut neighbors_view = neighbors.as_tensor_mut(&self.resources, &[n_queries, k], None, 0)?;
        let mut distances_view = distances.as_tensor_mut(&self.resources, &[n_queries, k], None, 0)?;
        let raw = self.raw.ok_or_else(|| invalid_state(operation, "the IVF-Flat index has been closed"))?;
        self.resources.with_native(operation, |native| {
            let mut config = ParamsHandle::new(
                cuvs_sys::cuvsIvfFlatSearchParamsCreate,
                cuvs_sys::cuvsIvfFlatSearchParamsDestroy,
                operation,
            )?;
            unsafe {
                (*config.as_ptr()).n_probes = params.n_probes;
            }
            let mut query = DlpackTensor::from_const(queries.as_tensor());
            let mut neighbor = DlpackTensor::from_mut(&mut neighbors_view);
            let mut distance = DlpackTensor::from_mut(&mut distances_view);
            let filter = cuvs_sys::cuvsFilter { addr: 0, type_: cuvs_sys::cuvsFilterType::NO_FILTER };
            let result = unsafe {
                check_cuvs(
                    cuvs_sys::cuvsIvfFlatSearch(
                        native,
                        config.as_ptr(),
                        raw.as_ptr(),
                        query.as_mut_ptr(),
                        neighbor.as_mut_ptr(),
                        distance.as_mut_ptr(),
                        filter,
                    ),
                    operation,
                )
            };
            result.and(config.close(operation))
        })?;
        let indexed_rows =
            i64::try_from(self.rows).map_err(|_| invalid_input(operation, "indexed row count exceeds Int64"))?;
        // cuVS local top-k can return an in-range placeholder ID with an infinite distance.
        neighbors
            .inner_mut()
            .normalize_missing_neighbor_indices(distances.inner(), i64::MAX)
            .map_err(|error| from_cudf(operation, error))?;
        neighbors.validate_native_index_range_or_sentinel(0, indexed_rows, i64::MAX, operation)?;
        distances.validate_native_distances_for_valid_neighbors(neighbors, i64::MAX, operation)?;
        Ok(KnnSearchOutput { n_queries, k })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Report native destruction errors; a failed destruction is retried on drop.
    pub fn close(mut self) -> Result<()> {
        self.close_inner("IvfFlatIndex::close")
    }

    fn close_inner(&mut self, operation: &'static str) -> Result<()> {
        let Some(raw) = self.raw.take() else { return Ok(()) };
        let result = self.resources.with_native(operation, |_| unsafe {
            check_cuvs(cuvs_sys::cuvsIvfFlatIndexDestroy(raw.as_ptr()), operation)
        });
        if result.is_err() {
            self.raw = Some(raw);
        }
        result
    }
}

impl Drop for IvfFlatIndex {
    fn drop(&mut self) {
        let _ = self.close_inner("IvfFlatIndex::drop");
    }
}

impl std::fmt::Debug for IvfFlatIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IvfFlatIndex")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}
