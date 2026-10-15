use std::{marker::PhantomData, ptr, ptr::NonNull};

use crate::{
    device::{DeviceBuffer, DeviceElement, MatrixLayout, ValidatedDeviceTensor},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, invalid_input, invalid_state, require_nonzero},
    resources::CuvsResources,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnnSearchOutput {
    pub n_queries: usize,
    pub k: usize,
}

/// A brute-force index that retains the dataset borrow and exact runtime binding.
pub struct BruteForceIndex<'data> {
    raw: Option<NonNull<cuvs_sys::cuvsBruteForceIndex>>,
    resources: CuvsResources,
    rows: usize,
    columns: usize,
    _dataset: PhantomData<&'data f32>,
}

impl<'data> BruteForceIndex<'data> {
    /// Builds an index over `dataset`, a contiguous row-major or column-major
    /// matrix on the same exact runtime binding and CUDA device as `resources`.
    ///
    /// The index borrows `dataset` for `'data` and retains `resources`, so the
    /// dataset memory stays valid until the index is closed or dropped. A
    /// failed native build destroys the partial index before the error is
    /// returned.
    pub fn build(
        resources: &CuvsResources,
        dataset: &ValidatedDeviceTensor<'data, f32>,
        metric: DistanceMetric,
    ) -> Result<Self> {
        let operation = "BruteForceIndex::build";
        validate_matrix_binding(resources, dataset, operation)?;
        let rows = dataset.rows()?;
        let columns = dataset.columns()?;
        matrix_layout(dataset, operation)?;

        resources.with_native(operation, |native| {
            let mut raw = ptr::null_mut();
            unsafe { check_cuvs(cuvs_sys::cuvsBruteForceIndexCreate(&mut raw), operation)? };
            let raw = NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned a null index"))?;
            let mut dataset_tensor = DlpackTensor::from_const(dataset.as_tensor());
            let status = unsafe {
                cuvs_sys::cuvsBruteForceBuild(native, dataset_tensor.as_mut_ptr(), metric.raw(), 0.0, raw.as_ptr())
            };
            if let Err(error) = check_cuvs(status, operation) {
                unsafe {
                    let _ = cuvs_sys::cuvsBruteForceIndexDestroy(raw.as_ptr());
                }
                return Err(error);
            }
            Ok(Self { raw: Some(raw), resources: resources.retain(), rows, columns, _dataset: PhantomData })
        })
    }

    /// Searches `queries` for their `k` nearest indexed rows and writes the
    /// results into `neighbors` and `distances`.
    ///
    /// `k` must be nonzero and at most the indexed row count. `queries` must
    /// have the index's column count and share its binding and device. Both
    /// output buffers must contain exactly `n_queries * k` elements, laid out
    /// as `[n_queries, k]` row-major. Success is returned only after the
    /// native outputs pass validation on the buffers' execution binding: every
    /// neighbor index lies in `0..rows()` and every distance is finite.
    pub fn search_into(
        &self,
        queries: &ValidatedDeviceTensor<'_, f32>,
        k: usize,
        neighbors: &mut DeviceBuffer<i64>,
        distances: &mut DeviceBuffer<f32>,
    ) -> Result<KnnSearchOutput> {
        let operation = "BruteForceIndex::search_into";
        require_nonzero(k, operation, "k")?;
        if k > self.rows {
            return Err(invalid_input(operation, format!("k ({k}) must be <= indexed rows ({})", self.rows)));
        }
        validate_matrix_binding(&self.resources, queries, operation)?;
        matrix_layout(queries, operation)?;
        let n_queries = queries.rows()?;
        if queries.columns()? != self.columns {
            return Err(invalid_input(
                operation,
                format!("query columns {} must match index columns {}", queries.columns()?, self.columns),
            ));
        }
        let output_len = n_queries
            .checked_mul(k)
            .ok_or_else(|| invalid_input(operation, "search output element count overflowed usize"))?;
        if neighbors.len() != output_len || distances.len() != output_len {
            return Err(invalid_input(
                operation,
                format!(
                    "neighbors and distances must each contain n_queries * k ({output_len}) elements; found {} and {}",
                    neighbors.len(),
                    distances.len()
                ),
            ));
        }

        let mut neighbors_view = neighbors.as_tensor_mut(&self.resources, &[n_queries, k], None, 0)?;
        let mut distances_view = distances.as_tensor_mut(&self.resources, &[n_queries, k], None, 0)?;
        let raw = self.raw.ok_or_else(|| invalid_state(operation, "the brute-force index has been closed"))?;
        self.resources.with_native(operation, |native| {
            let mut queries_tensor = DlpackTensor::from_const(queries.as_tensor());
            let mut neighbors_tensor = DlpackTensor::from_mut(&mut neighbors_view);
            let mut distances_tensor = DlpackTensor::from_mut(&mut distances_view);
            let prefilter = cuvs_sys::cuvsFilter { addr: 0, type_: cuvs_sys::cuvsFilterType::NO_FILTER };
            unsafe {
                check_cuvs(
                    cuvs_sys::cuvsBruteForceSearch(
                        native,
                        raw.as_ptr(),
                        queries_tensor.as_mut_ptr(),
                        neighbors_tensor.as_mut_ptr(),
                        distances_tensor.as_mut_ptr(),
                        prefilter,
                    ),
                    operation,
                )
            }
        })?;
        let indexed_rows = i64::try_from(self.rows)
            .map_err(|_| invalid_input(operation, "indexed row count exceeds the Int64 neighbor range"))?;
        neighbors.validate_native_index_range(0, indexed_rows, operation)?;
        distances.validate_native_input(operation)?;
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
        self.close_inner("BruteForceIndex::close")
    }

    fn close_inner(&mut self, operation: &'static str) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        let result = self.resources.with_native(operation, |_| unsafe {
            check_cuvs(cuvs_sys::cuvsBruteForceIndexDestroy(raw.as_ptr()), operation)
        });
        if result.is_err() {
            self.raw = Some(raw);
        }
        result
    }
}

impl Drop for BruteForceIndex<'_> {
    fn drop(&mut self) {
        let _ = self.close_inner("BruteForceIndex::drop");
    }
}

impl std::fmt::Debug for BruteForceIndex<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BruteForceIndex")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("closed", &self.raw.is_none())
            .finish_non_exhaustive()
    }
}

pub(crate) fn validate_matrix_binding<T: DeviceElement>(
    resources: &CuvsResources,
    matrix: &ValidatedDeviceTensor<'_, T>,
    operation: &'static str,
) -> Result<()> {
    resources.validate_exact_binding(operation)?;
    if matrix.binding_id() != resources.binding_id() {
        return Err(invalid_input(operation, "matrix and cuVS resources must use the same exact runtime binding")
            .with_context("tensor_binding_id", matrix.binding_id())
            .with_context("resource_binding_id", resources.binding_id()));
    }
    if matrix.device() != resources.device().ordinal() as i32 {
        return Err(invalid_input(operation, "matrix and cuVS resources must use the same CUDA device")
            .with_context("tensor_device", matrix.device())
            .with_context("resource_device", resources.device().ordinal()));
    }
    let _ = matrix.rows()?;
    let _ = matrix.columns()?;
    Ok(())
}

pub(crate) fn matrix_layout<T: DeviceElement>(
    matrix: &ValidatedDeviceTensor<'_, T>,
    operation: &'static str,
) -> Result<MatrixLayout> {
    let rows = matrix.rows()? as i64;
    let columns = matrix.columns()? as i64;
    match matrix.strides() {
        None => Ok(MatrixLayout::RowMajor),
        Some([row_stride, column_stride]) if *row_stride == columns && *column_stride == 1 => {
            Ok(MatrixLayout::RowMajor)
        },
        Some([row_stride, column_stride]) if *row_stride == 1 && *column_stride == rows => {
            Ok(MatrixLayout::ColumnMajor)
        },
        _ => Err(invalid_input(operation, "cuVS matrices must be contiguous row-major or column-major tensors")),
    }
}
