use std::{marker::PhantomData, ptr, ptr::NonNull};

use crate::{
    brute_force::{DistanceMetric, KnnSearchOutput, matrix_layout, validate_matrix_binding},
    device::{DeviceBuffer, MatrixLayout, ValidatedDeviceTensor},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, from_cudf, invalid_input, invalid_state, require_nonzero},
    params_handle::ParamsHandle,
    resources::CuvsResources,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CagraBuildAlgo {
    IvfPq,
    NnDescent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CagraSearchAlgo {
    Auto,
    SingleCta,
    MultiCta,
    MultiKernel,
}

#[derive(Debug, Clone, Copy)]
pub struct CagraIndexParams {
    pub metric: DistanceMetric,
    pub graph_degree: usize,
    pub intermediate_graph_degree: usize,
    pub build_algo: CagraBuildAlgo,
}

impl CagraIndexParams {
    pub fn new(metric: DistanceMetric) -> Self {
        Self { metric, graph_degree: 64, intermediate_graph_degree: 128, build_algo: CagraBuildAlgo::IvfPq }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CagraSearchParams {
    pub itopk_size: usize,
    pub search_width: usize,
    pub max_iterations: usize,
    pub search_algo: CagraSearchAlgo,
}

impl Default for CagraSearchParams {
    fn default() -> Self {
        Self { itopk_size: 64, search_width: 1, max_iterations: 0, search_algo: CagraSearchAlgo::Auto }
    }
}

// cagra.cpp:1959-1994 allocates IVF-PQ graph params on create but destroys them
// according to the current build_algo. NN-descent must restore IVF_PQ on every exit.
unsafe extern "C" fn destroy_index_params(params: *mut cuvs_sys::cuvsCagraIndexParams) -> cuvs_sys::cuvsError_t {
    unsafe {
        (*params).build_algo = cuvs_sys::cuvsCagraGraphBuildAlgo::IVF_PQ;
        cuvs_sys::cuvsCagraIndexParamsDestroy(params)
    }
}

struct DatasetHandle(NonNull<cuvs_sys::cuvsDataset>);

impl Drop for DatasetHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = cuvs_sys::cuvsDatasetDestroy(self.0.as_ptr());
        }
    }
}

/// Query-local CAGRA index with a device-padded dataset. Rows already aligned
/// to 16 bytes use a borrowed padded view; other rows use an owning padded copy.
/// Native search requires a device-padded index (components/cuvs/cuvs/c/src/neighbors/cagra.cpp:1843).
pub struct CagraIndex<'data> {
    raw: Option<NonNull<cuvs_sys::cuvsCagraIndex>>,
    dataset: DatasetHandle,
    resources: CuvsResources,
    rows: usize,
    columns: usize,
    _dataset_borrow: PhantomData<&'data f32>,
}

impl<'data> CagraIndex<'data> {
    pub fn build(
        resources: &CuvsResources,
        dataset: &ValidatedDeviceTensor<'data>,
        params: CagraIndexParams,
    ) -> Result<Self> {
        let operation = "CagraIndex::build";
        validate_matrix_binding(resources, dataset, operation)?;
        if matrix_layout(dataset, operation)? != MatrixLayout::RowMajor {
            return Err(invalid_input(operation, "CAGRA dataset must be contiguous row-major"));
        }
        let rows = dataset.rows()?;
        let columns = dataset.columns()?;
        require_nonzero(rows, operation, "dataset rows")?;
        if rows > u32::MAX as usize {
            return Err(invalid_input(operation, "CAGRA graph indices require at most UInt32::MAX dataset rows"));
        }
        if params.graph_degree == 0
            || params.graph_degree >= rows
            || params.intermediate_graph_degree <= params.graph_degree
            || params.intermediate_graph_degree >= rows
        {
            return Err(invalid_input(
                operation,
                format!(
                    "graph_degree must be in 1..dataset rows ({rows}) and intermediate_graph_degree must be greater than graph_degree and less than dataset rows"
                ),
            ));
        }
        if params.metric == DistanceMetric::L2SqrtExpanded
            || (params.metric == DistanceMetric::CosineExpanded && params.build_algo == CagraBuildAlgo::NnDescent)
        {
            return Err(invalid_input(operation, "unsupported CAGRA metric and build algorithm combination"));
        }
        resources.with_native(operation, |native| {
            let mut config = ParamsHandle::new(
                cuvs_sys::cuvsCagraIndexParamsCreate,
                destroy_index_params,
                operation,
                "CAGRA params",
            )?;
            unsafe {
                let ptr = config.as_ptr();
                (*ptr).metric = params.metric.raw();
                (*ptr).graph_degree = params.graph_degree;
                (*ptr).intermediate_graph_degree = params.intermediate_graph_degree;
                (*ptr).build_algo = match params.build_algo {
                    CagraBuildAlgo::IvfPq => cuvs_sys::cuvsCagraGraphBuildAlgo::IVF_PQ,
                    CagraBuildAlgo::NnDescent => cuvs_sys::cuvsCagraGraphBuildAlgo::NN_DESCENT,
                };
            }
            let mut tensor = DlpackTensor::from_const(dataset.as_tensor());
            let mut padded = ptr::null_mut();
            // common.hpp:1068-1078,1157-1165 rejects an owning copy when the
            // device row width already has the required 16-byte alignment.
            unsafe {
                if columns.is_multiple_of(4) {
                    check_cuvs(
                        cuvs_sys::cuvsDatasetMakePaddedView(native, tensor.as_mut_ptr(), &mut padded),
                        operation,
                    )?;
                } else {
                    check_cuvs(
                        cuvs_sys::cuvsDatasetMakePadded(
                            native,
                            tensor.as_mut_ptr(),
                            cuvs_sys::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                            &mut padded,
                        ),
                        operation,
                    )?;
                }
            };
            let padded = DatasetHandle(
                NonNull::new(padded).ok_or_else(|| invalid_state(operation, "cuVS returned a null padded dataset"))?,
            );
            let mut raw = ptr::null_mut();
            unsafe { check_cuvs(cuvs_sys::cuvsCagraIndexCreate(&mut raw), operation)? };
            let raw = NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned a null CAGRA index"))?;
            let built = unsafe {
                check_cuvs(
                    cuvs_sys::cuvsCagraBuild(native, config.as_ptr(), padded.0.as_ptr(), raw.as_ptr()),
                    operation,
                )
            };
            let closed = config.close(operation);
            if let Err(error) = built.and(closed) {
                unsafe {
                    let _ = cuvs_sys::cuvsCagraIndexDestroy(raw.as_ptr());
                }
                return Err(error);
            }
            Ok(Self {
                raw: Some(raw),
                dataset: padded,
                resources: resources.retain(),
                rows,
                columns,
                _dataset_borrow: PhantomData,
            })
        })
    }

    /// Search outputs are `[queries.rows(), k]`. Missing neighbors have ordinal `i64::MAX`.
    pub fn search_into(
        &self,
        queries: &ValidatedDeviceTensor<'_>,
        k: usize,
        params: CagraSearchParams,
        neighbors: &mut DeviceBuffer<i64>,
        distances: &mut DeviceBuffer<f32>,
    ) -> Result<KnnSearchOutput> {
        let operation = "CagraIndex::search_into";
        require_nonzero(k, operation, "k")?;
        if k > self.rows
            || k > u32::MAX as usize
            || params.itopk_size < k
            || params.search_width == 0
            || (params.search_algo == CagraSearchAlgo::SingleCta && params.itopk_size > 1024)
        {
            return Err(invalid_input(
                operation,
                "k must not exceed dataset rows or itopk_size; search_width must be positive and single_cta itopk_size must not exceed 1024",
            ));
        }
        validate_matrix_binding(&self.resources, queries, operation)?;
        if matrix_layout(queries, operation)? != MatrixLayout::RowMajor || queries.columns()? != self.columns {
            return Err(invalid_input(
                operation,
                "CAGRA queries must be contiguous row-major with the indexed dimension",
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
        let raw = self.raw.ok_or_else(|| invalid_state(operation, "the CAGRA index has been closed"))?;
        self.resources.with_native(operation, |native| {
            let mut config = ParamsHandle::new(
                cuvs_sys::cuvsCagraSearchParamsCreate,
                cuvs_sys::cuvsCagraSearchParamsDestroy,
                operation,
                "CAGRA search params",
            )?;
            unsafe {
                let ptr = config.as_ptr();
                (*ptr).itopk_size = params.itopk_size;
                (*ptr).search_width = params.search_width;
                (*ptr).max_iterations = params.max_iterations;
                (*ptr).algo = match params.search_algo {
                    CagraSearchAlgo::Auto => cuvs_sys::cuvsCagraSearchAlgo::AUTO,
                    CagraSearchAlgo::SingleCta => cuvs_sys::cuvsCagraSearchAlgo::SINGLE_CTA,
                    CagraSearchAlgo::MultiCta => cuvs_sys::cuvsCagraSearchAlgo::MULTI_CTA,
                    CagraSearchAlgo::MultiKernel => cuvs_sys::cuvsCagraSearchAlgo::MULTI_KERNEL,
                };
            }
            let mut query = DlpackTensor::from_const(queries.as_tensor());
            let mut neighbor = DlpackTensor::from_mut(&mut neighbors_view);
            let mut distance = DlpackTensor::from_mut(&mut distances_view);
            let filter = cuvs_sys::cuvsFilter { addr: 0, type_: cuvs_sys::cuvsFilterType::NO_FILTER };
            let result = unsafe {
                check_cuvs(
                    cuvs_sys::cuvsCagraSearch(
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
        // Native CAGRA writes UINT32_MAX and FLT_MAX for exhausted graph slots,
        // including when the requested output index type is Int64.
        neighbors
            .inner_mut()
            .replace_int64_value_and_set_paired_float32_infinite(distances.inner_mut(), u32::MAX as i64, i64::MAX)
            .map_err(|error| from_cudf(operation, error))?;
        neighbors
            .inner_mut()
            .fill_int64_where_float32_infinite(distances.inner(), i64::MAX)
            .map_err(|error| from_cudf(operation, error))?;
        neighbors.validate_native_index_range_or_sentinel(
            0,
            i64::try_from(self.rows).map_err(|_| invalid_input(operation, "indexed row count exceeds Int64"))?,
            i64::MAX,
            operation,
        )?;
        distances.validate_native_distances_for_valid_neighbors(neighbors, i64::MAX, operation)?;
        Ok(KnnSearchOutput { n_queries, k })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Reports native destruction errors; failed destruction is retried on drop.
    pub fn close(mut self) -> Result<()> {
        self.close_inner("CagraIndex::close")
    }

    fn close_inner(&mut self, operation: &'static str) -> Result<()> {
        let Some(raw) = self.raw.take() else { return Ok(()) };
        let result = self.resources.with_native(operation, |_| unsafe {
            check_cuvs(cuvs_sys::cuvsCagraIndexDestroy(raw.as_ptr()), operation)
        });
        if result.is_err() {
            self.raw = Some(raw);
        }
        result
    }
}

impl Drop for CagraIndex<'_> {
    fn drop(&mut self) {
        let _ = self.close_inner("CagraIndex::drop");
    }
}

impl std::fmt::Debug for CagraIndex<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CagraIndex")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("dataset", &self.dataset.0)
            .finish_non_exhaustive()
    }
}
