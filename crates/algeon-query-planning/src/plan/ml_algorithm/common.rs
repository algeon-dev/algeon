use crate::plan::{DenseVectorBinding, VectorRowIdBinding};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlRelationBinding {
    pub input: usize,
    pub row_id: VectorRowIdBinding,
    pub features: DenseVectorBinding,
    pub label: Option<String>,
}

impl MlRelationBinding {
    pub fn new(
        input: usize,
        row_id: VectorRowIdBinding,
        features: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self { input, row_id, features: DenseVectorBinding::columns(features), label: None }
    }

    pub fn new_vector_column(input: usize, row_id: VectorRowIdBinding, column: impl Into<String>) -> Self {
        Self { input, row_id, features: DenseVectorBinding::vector_column(column), label: None }
    }
}
