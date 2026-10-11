use std::fmt;

use crate::plan::DenseVectorBinding;

/// Row identity emitted alongside vector algorithm results.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorRowIdBinding {
    /// Use the zero-based row ordinal of the evaluated relation.
    RowIndex,
    /// Use a non-null integral column from the evaluated relation.
    Column(String),
}

impl VectorRowIdBinding {
    /// Construct a column-backed row identity.
    pub fn column(name: impl Into<String>) -> Self {
        Self::Column(name.into())
    }

    /// Stable diagnostic label.
    pub(crate) fn diagnostic_label(&self) -> &str {
        match self {
            Self::RowIndex => "row_index",
            Self::Column(name) => name,
        }
    }
}

/// Binds one input relation of a vector algorithm to its row identity and dense vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorRelationBinding {
    /// Position of the relation among the algorithm node inputs.
    pub input: usize,
    /// Row identity emitted by relational results.
    pub row_id: VectorRowIdBinding,
    /// Ordered dense vector binding.
    pub vector: DenseVectorBinding,
}

impl VectorRelationBinding {
    /// Construct a vector relation binding.
    pub fn new(
        input: usize,
        row_id: VectorRowIdBinding,
        feature_columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self { input, row_id, vector: DenseVectorBinding::columns(feature_columns) }
    }

    /// Construct a binding backed by one list/vector column.
    pub fn new_vector_column(input: usize, row_id: VectorRowIdBinding, column: impl Into<String>) -> Self {
        Self { input, row_id, vector: DenseVectorBinding::vector_column(column) }
    }
}

/// Distance metric used by exact kNN and K-means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorDistanceMetric {
    /// Expanded squared L2 distance.
    L2Expanded,
    /// Square root of expanded L2 distance.
    L2SqrtExpanded,
    /// Expanded cosine distance.
    CosineExpanded,
    /// Inner-product distance.
    InnerProduct,
}

impl VectorDistanceMetric {
    /// Stable diagnostic label.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::L2Expanded => "l2_expanded",
            Self::L2SqrtExpanded => "l2_sqrt_expanded",
            Self::CosineExpanded => "cosine_expanded",
            Self::InnerProduct => "inner_product",
        }
    }
}

impl fmt::Display for VectorDistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
