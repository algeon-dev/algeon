use std::{marker::PhantomData, ptr};

use crate::{
    device::{DeviceBuffer, DeviceMatrixViewF32, MatrixLayout},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, invalid_input, invalid_state, require_nonzero},
    resources::Resources,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    L2Expanded,
    L2SqrtExpanded,
    CosineExpanded,
    InnerProduct,
}

impl DistanceMetric {
    pub(crate) fn raw(self) -> cuvs_sys::cuvsDistanceType {
        match self {
            DistanceMetric::L2Expanded => cuvs_sys::cuvsDistanceType::L2Expanded,
            DistanceMetric::L2SqrtExpanded => cuvs_sys::cuvsDistanceType::L2SqrtExpanded,
            DistanceMetric::CosineExpanded => cuvs_sys::cuvsDistanceType::CosineExpanded,
            DistanceMetric::InnerProduct => cuvs_sys::cuvsDistanceType::InnerProduct,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnnSearchResult {
    pub neighbors: Vec<i64>,
    pub distances: Vec<f32>,
    pub n_queries: usize,
    pub k: usize,
}

impl KnnSearchResult {
    pub fn row_neighbors(&self, row: usize) -> &[i64] {
        let start = row * self.k;
        &self.neighbors[start..start + self.k]
    }

    pub fn row_distances(&self, row: usize) -> &[f32] {
        let start = row * self.k;
        &self.distances[start..start + self.k]
    }
}

#[derive(Debug)]
pub struct BruteForceIndex<'data> {
    raw: cuvs_sys::cuvsBruteForceIndex_t,
    rows: usize,
    cols: usize,
    _dataset: PhantomData<&'data f32>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'data> BruteForceIndex<'data> {
    pub fn build(resources: &Resources, dataset: &DeviceMatrixViewF32<'data>, metric: DistanceMetric) -> Result<Self> {
        let operation = "BruteForceIndex::build";
        if dataset.layout() != MatrixLayout::RowMajor && dataset.layout() != MatrixLayout::ColumnMajor {
            return Err(invalid_input(operation, "dataset must be row-major or column-major"));
        }

        let mut raw = ptr::null_mut();
        unsafe { check_cuvs(cuvs_sys::cuvsBruteForceIndexCreate(&mut raw), operation)? };
        if raw.is_null() {
            return Err(invalid_state(operation, "cuVS returned a null brute-force index"));
        }

        let mut dataset_tensor = dataset.to_dlpack()?;
        let build_status = unsafe {
            cuvs_sys::cuvsBruteForceBuild(resources.raw(), dataset_tensor.as_mut_ptr(), metric.raw(), 0.0, raw)
        };
        if let Err(error) = check_cuvs(build_status, operation) {
            unsafe {
                let _ = cuvs_sys::cuvsBruteForceIndexDestroy(raw);
            }
            return Err(error);
        }

        Ok(Self { raw, rows: dataset.rows(), cols: dataset.cols(), _dataset: PhantomData, _not_send_sync: PhantomData })
    }

    pub fn search(
        &self,
        resources: &Resources,
        queries: &DeviceMatrixViewF32<'_>,
        k: usize,
    ) -> Result<KnnSearchResult> {
        let operation = "BruteForceIndex::search";
        require_nonzero(k, operation, "k")?;
        if k > self.rows {
            return Err(invalid_input(operation, format!("k ({k}) must be <= indexed rows ({})", self.rows)));
        }
        if queries.cols() != self.cols {
            return Err(invalid_input(
                operation,
                format!("query cols {} must match index cols {}", queries.cols(), self.cols),
            ));
        }

        let neighbors_device = DeviceBuffer::<i64>::new(queries.rows() * k)?;
        let distances_device = DeviceBuffer::<f32>::new(queries.rows() * k)?;

        let mut queries_tensor = queries.to_dlpack()?;
        let mut neighbors_tensor = DlpackTensor::device_matrix_i64(neighbors_device.as_mut_ptr(), queries.rows(), k);
        let mut distances_tensor =
            DlpackTensor::device_matrix_f32(distances_device.as_mut_ptr(), queries.rows(), k, MatrixLayout::RowMajor)?;
        let prefilter = cuvs_sys::cuvsFilter { addr: 0, type_: cuvs_sys::cuvsFilterType::NO_FILTER };

        unsafe {
            check_cuvs(
                cuvs_sys::cuvsBruteForceSearch(
                    resources.raw(),
                    self.raw,
                    queries_tensor.as_mut_ptr(),
                    neighbors_tensor.as_mut_ptr(),
                    distances_tensor.as_mut_ptr(),
                    prefilter,
                ),
                operation,
            )?;
        }

        Ok(KnnSearchResult {
            neighbors: neighbors_device.to_vec()?,
            distances: distances_device.to_vec()?,
            n_queries: queries.rows(),
            k,
        })
    }

    /// Search the same dataset used to build the index, overfetching one
    /// neighbor and dropping the per-row diagonal self-match.
    ///
    /// `queries` must be the indexed dataset in the same row order.
    pub fn search_overfetch_drop_self(
        &self,
        resources: &Resources,
        queries: &DeviceMatrixViewF32<'_>,
        k: usize,
    ) -> Result<KnnSearchResult> {
        let operation = "BruteForceIndex::search_overfetch_drop_self";
        require_nonzero(k, operation, "k")?;
        if k >= self.rows {
            return Err(invalid_input(
                operation,
                format!("k ({k}) must be < indexed rows ({}) when dropping self-neighbors", self.rows),
            ));
        }
        let overfetch = k + 1;
        let raw = self.search(resources, queries, overfetch)?;
        let mut neighbors = Vec::with_capacity(queries.rows() * k);
        let mut distances = Vec::with_capacity(queries.rows() * k);

        for row in 0..queries.rows() {
            let mut kept = 0;
            for rank in 0..overfetch {
                let offset = row * overfetch + rank;
                if raw.neighbors[offset] == row as i64 {
                    continue;
                }
                neighbors.push(raw.neighbors[offset]);
                distances.push(raw.distances[offset]);
                kept += 1;
                if kept == k {
                    break;
                }
            }
            if kept != k {
                return Err(invalid_state(
                    operation,
                    format!("query row {row} had only {kept} non-self neighbors after overfetching {overfetch}"),
                ));
            }
        }

        Ok(KnnSearchResult { neighbors, distances, n_queries: queries.rows(), k })
    }
}

impl Drop for BruteForceIndex<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = cuvs_sys::cuvsBruteForceIndexDestroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}
