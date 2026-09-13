use crate::{
    device::{DeviceBuffer, DeviceMatrix, MatrixLayout},
    error::{ErrorCode, FailurePhase, Result, check_status, error, invalid_input},
    resources::CumlResources,
};

const FIT: &str = "Hdbscan::fit";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ClusterSelectionMethod {
    Eom = 0,
    Leaf = 1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdbscanParameters {
    pub min_samples: usize,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub cluster_selection_epsilon: f32,
    pub allow_single_cluster: bool,
    pub cluster_selection_method: ClusterSelectionMethod,
    pub alpha: f32,
}

impl Default for HdbscanParameters {
    fn default() -> Self {
        Self {
            min_samples: 5,
            min_cluster_size: 5,
            max_cluster_size: 0,
            cluster_selection_epsilon: 0.0,
            allow_single_cluster: false,
            cluster_selection_method: ClusterSelectionMethod::Eom,
            alpha: 1.0,
        }
    }
}

pub struct HdbscanOutputBuffers {
    pub labels: DeviceBuffer<i64>,
    pub probabilities: DeviceBuffer<f32>,
    pub core_distances: DeviceBuffer<f32>,
    pub single_linkage_children: DeviceBuffer<i64>,
    pub single_linkage_sizes: DeviceBuffer<i64>,
    pub single_linkage_distances: DeviceBuffer<f32>,
    pub mst_sources: DeviceBuffer<i64>,
    pub mst_destinations: DeviceBuffer<i64>,
    pub mst_weights: DeviceBuffer<f32>,
}

impl HdbscanOutputBuffers {
    pub fn new(resources: &CumlResources, rows: usize) -> Result<Self> {
        if rows < 2 {
            return Err(invalid_input(FIT, "output buffers need at least two rows"));
        }
        // cuML's linkage and MST buffers hold one edge per merge; linkage
        // records two child node indices for each of those edges.
        let linkage_edges = rows - 1;
        let child_count = linkage_edges
            .checked_mul(2)
            .ok_or_else(|| invalid_input(FIT, "single-linkage child buffer size overflows"))?;
        Ok(Self {
            labels: DeviceBuffer::new(resources, rows)?,
            probabilities: DeviceBuffer::new(resources, rows)?,
            core_distances: DeviceBuffer::new(resources, rows)?,
            single_linkage_children: DeviceBuffer::new(resources, child_count)?,
            single_linkage_sizes: DeviceBuffer::new(resources, linkage_edges)?,
            single_linkage_distances: DeviceBuffer::new(resources, linkage_edges)?,
            mst_sources: DeviceBuffer::new(resources, linkage_edges)?,
            mst_destinations: DeviceBuffer::new(resources, linkage_edges)?,
            mst_weights: DeviceBuffer::new(resources, linkage_edges)?,
        })
    }
}

pub struct HdbscanFit {
    pub output: HdbscanOutputBuffers,
    n_rows: usize,
    n_clusters: usize,
}

impl HdbscanFit {
    pub fn rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn fit(resources: &CumlResources, input: &DeviceMatrix<'_>, parameters: HdbscanParameters) -> Result<Self> {
        Hdbscan::fit(resources, input, parameters)
    }
}

pub struct Hdbscan;

pub type HdbscanParams = HdbscanParameters;

impl Hdbscan {
    pub fn fit(
        resources: &CumlResources,
        input: &DeviceMatrix<'_>,
        parameters: HdbscanParameters,
    ) -> Result<HdbscanFit> {
        validate_input(resources, input, parameters)?;
        resources.validate_exact_binding(FIT)?;
        let rows = i32::try_from(input.rows()).map_err(|_| invalid_input(FIT, "row count exceeds cuML limits"))?;
        let columns =
            i32::try_from(input.columns()).map_err(|_| invalid_input(FIT, "feature count exceeds cuML limits"))?;
        let min_samples =
            i32::try_from(parameters.min_samples).map_err(|_| invalid_input(FIT, "min_samples exceeds cuML limits"))?;
        let min_cluster_size = i32::try_from(parameters.min_cluster_size)
            .map_err(|_| invalid_input(FIT, "min_cluster_size exceeds cuML limits"))?;
        let max_cluster_size = i32::try_from(parameters.max_cluster_size)
            .map_err(|_| invalid_input(FIT, "max_cluster_size exceeds cuML limits"))?;
        let mut n_clusters = 0_i32;
        let mut output = HdbscanOutputBuffers::new(resources, input.rows())?;
        let labels = output.labels.as_mut_ptr()?;
        let probabilities = output.probabilities.as_mut_ptr()?;
        let core_distances = output.core_distances.as_mut_ptr()?;
        let children = output.single_linkage_children.as_mut_ptr()?;
        let sizes = output.single_linkage_sizes.as_mut_ptr()?;
        let distances = output.single_linkage_distances.as_mut_ptr()?;
        let mst_sources = output.mst_sources.as_mut_ptr()?;
        let mst_destinations = output.mst_destinations.as_mut_ptr()?;
        let mst_weights = output.mst_weights.as_mut_ptr()?;
        resources.with_native(FIT, |handle| {
            let status = unsafe {
                cuml_sys::cuml_hdbscan_fit(
                    handle,
                    input.as_ptr().cast(),
                    rows,
                    columns,
                    min_samples,
                    min_cluster_size,
                    max_cluster_size,
                    parameters.cluster_selection_epsilon,
                    i32::from(parameters.allow_single_cluster),
                    parameters.cluster_selection_method as i32,
                    parameters.alpha,
                    &mut n_clusters,
                    labels.cast(),
                    probabilities.cast(),
                    core_distances.cast(),
                    children.cast(),
                    sizes.cast(),
                    distances.cast(),
                    mst_sources.cast(),
                    mst_destinations.cast(),
                    mst_weights.cast(),
                )
            };
            check_status(status, FIT)
        })?;
        let n_clusters = usize::try_from(n_clusters).map_err(|_| {
            error(
                ErrorCode::InvariantViolation,
                FailurePhase::OutputValidation,
                FIT,
                "native cluster count is negative",
            )
        })?;
        validate_output(&mut output, input.rows(), n_clusters)?;
        Ok(HdbscanFit { output, n_rows: input.rows(), n_clusters })
    }
}

fn validate_input(resources: &CumlResources, input: &DeviceMatrix<'_>, parameters: HdbscanParameters) -> Result<()> {
    if input.rows() < 2 || input.columns() == 0 {
        return Err(invalid_input(FIT, "input needs at least two rows and one feature"));
    }
    if input.layout() != MatrixLayout::RowMajor {
        return Err(error(
            ErrorCode::UnsupportedLayout,
            FailurePhase::Preflight,
            FIT,
            "HDBSCAN requires row-major float32 input",
        ));
    }
    if input.binding_id() != resources.binding_id() {
        return Err(error(
            ErrorCode::RuntimeBindingMismatch,
            FailurePhase::Preflight,
            FIT,
            "input must use the resource binding",
        ));
    }
    if parameters.min_samples == 0 || parameters.min_samples >= input.rows() {
        return Err(invalid_input(FIT, "min_samples must be in 1..input rows"));
    }
    if parameters.min_cluster_size < 2 || parameters.min_cluster_size > input.rows() {
        return Err(invalid_input(FIT, "min_cluster_size must be in 2..=input rows"));
    }
    if parameters.max_cluster_size > input.rows() {
        return Err(invalid_input(FIT, "max_cluster_size must be zero or no greater than input rows"));
    }
    if !parameters.cluster_selection_epsilon.is_finite() || parameters.cluster_selection_epsilon < 0.0 {
        return Err(invalid_input(FIT, "cluster_selection_epsilon must be finite and non-negative"));
    }
    if !parameters.alpha.is_finite() || parameters.alpha <= 0.0 {
        return Err(invalid_input(FIT, "alpha must be finite and positive"));
    }
    i32::try_from(input.rows()).map_err(|_| invalid_input(FIT, "row count exceeds cuML limits"))?;
    i32::try_from(input.columns()).map_err(|_| invalid_input(FIT, "feature count exceeds cuML limits"))?;
    i32::try_from(parameters.min_samples).map_err(|_| invalid_input(FIT, "min_samples exceeds cuML limits"))?;
    i32::try_from(parameters.min_cluster_size)
        .map_err(|_| invalid_input(FIT, "min_cluster_size exceeds cuML limits"))?;
    i32::try_from(parameters.max_cluster_size)
        .map_err(|_| invalid_input(FIT, "max_cluster_size exceeds cuML limits"))?;
    Ok(())
}

fn validate_output(output: &mut HdbscanOutputBuffers, rows: usize, n_clusters: usize) -> Result<()> {
    if rows != output.labels.len() {
        return Err(error(
            ErrorCode::InvariantViolation,
            FailurePhase::OutputValidation,
            FIT,
            "output row count does not match labels extent",
        ));
    }
    if n_clusters > rows {
        return Err(error(
            ErrorCode::InvariantViolation,
            FailurePhase::OutputValidation,
            FIT,
            "native cluster count exceeds row count",
        ));
    }
    let max_cluster_label = i64::try_from(rows - 1)
        .map_err(|_| error(ErrorCode::SizeOverflow, FailurePhase::OutputValidation, FIT, "row count exceeds i64"))?;
    // Linkage children can name internal nodes; MST endpoints name input rows only.
    let max_linkage_node = max_cluster_label.checked_mul(2).ok_or_else(|| {
        error(ErrorCode::SizeOverflow, FailurePhase::OutputValidation, FIT, "linkage node range overflows")
    })?;
    output.labels.validate_range(-1, max_cluster_label + 1, FIT).map_err(|e| e.with_detail("output", "labels"))?;
    output.probabilities.validate_finite(FIT)?;
    output.core_distances.validate_finite(FIT)?;
    output
        .single_linkage_children
        .validate_range(0, max_linkage_node + 1, FIT)
        .map_err(|e| e.with_detail("output", "children"))?;
    output
        .single_linkage_sizes
        .validate_range(1, max_cluster_label + 2, FIT)
        .map_err(|e| e.with_detail("output", "sizes"))?;
    output.single_linkage_distances.validate_finite(FIT)?;
    output
        .mst_sources
        .validate_range(0, max_cluster_label + 1, FIT)
        .map_err(|e| e.with_detail("output", "mst_sources"))?;
    output
        .mst_destinations
        .validate_range(0, max_cluster_label + 1, FIT)
        .map_err(|e| e.with_detail("output", "mst_destinations"))?;
    output.mst_weights.validate_finite(FIT)?;
    Ok(())
}
