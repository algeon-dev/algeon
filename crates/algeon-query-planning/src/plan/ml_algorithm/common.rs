use crate::plan::{DenseVectorBinding, VectorRowIdBinding};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Columns and input position supplying one dense ML relation.
pub struct MlRelationBinding {
    /// Zero-based position in the ML node's relation inputs.
    pub input: usize,
    /// Logical row identifier preserved in the result.
    pub row_id: VectorRowIdBinding,
    /// Ordered scalar columns or one vector column supplying dense features.
    pub features: DenseVectorBinding,
    /// Supervised training target column; `None` for input and prediction relations.
    pub label: Option<String>,
}

impl MlRelationBinding {
    /// Binds ordered scalar feature columns, with no label.
    ///
    /// Names are retained without schema validation. Set `label` separately for a
    /// supervised training relation.
    pub fn new(
        input: usize,
        row_id: VectorRowIdBinding,
        features: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self { input, row_id, features: DenseVectorBinding::columns(features), label: None }
    }

    /// Binds a single list or fixed-size-list feature column, with no label.
    ///
    /// The binding retains the name without checking its schema or vector width.
    pub fn new_vector_column(input: usize, row_id: VectorRowIdBinding, column: impl Into<String>) -> Self {
        Self { input, row_id, features: DenseVectorBinding::vector_column(column), label: None }
    }
}
