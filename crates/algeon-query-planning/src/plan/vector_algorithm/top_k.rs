use super::common::VectorRelationBinding;

/// Per-row vector selection parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorTopKParams {
    /// Relation supplying one dense vector per result row group.
    pub input: VectorRelationBinding,
    /// Number of values selected from each input vector.
    pub k: usize,
    /// Select the smallest values when true, or the largest when false.
    pub select_min: bool,
}
