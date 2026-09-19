use datafusion::common::Result;
use serde_json::{Map, Value};

use super::{
    CuvsDenseVectorBinding, CuvsOperation, CuvsOutputContract, CuvsRelationRole, canonicalize_cuvs_options,
    error::invalid_call,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CuvsVectorRelationBinding {
    relation_role: CuvsRelationRole,
    id: String,
    vector: CuvsDenseVectorBinding,
}

impl CuvsVectorRelationBinding {
    pub fn relation_role(&self) -> CuvsRelationRole {
        self.relation_role
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn vector(&self) -> &CuvsDenseVectorBinding {
        &self.vector
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CuvsValidatedOptions {
    canonical_json: String,
    bindings: Vec<CuvsVectorRelationBinding>,
    knn_k: Option<u32>,
    kmeans_clusters: Option<u32>,
    pca_components: Option<u32>,
}

impl CuvsValidatedOptions {
    pub fn canonical_json(&self) -> &str {
        &self.canonical_json
    }

    pub fn bindings(&self) -> &[CuvsVectorRelationBinding] {
        &self.bindings
    }

    pub fn knn_k(&self) -> Option<u32> {
        self.knn_k
    }

    pub fn kmeans_clusters(&self) -> Option<u32> {
        self.kmeans_clusters
    }

    pub fn pca_components(&self) -> Option<u32> {
        self.pca_components
    }
}

pub(super) fn validate_options(operation: CuvsOperation, raw: &str) -> Result<CuvsValidatedOptions> {
    let canonical = canonicalize_cuvs_options(operation, raw)
        .map_err(|error| invalid_call(operation, error.code(), error.message()))?;
    let object = canonical.as_object().expect("canonical cuVS options are always a JSON object");
    let bindings =
        operation.descriptor().relation_roles.iter().copied().map(|role| binding(object, role)).collect::<Vec<_>>();
    let option_u32 = |name: &str| object.get(name).and_then(Value::as_u64).map(|value| value as u32);
    let knn_k = (operation.descriptor().output_contract == CuvsOutputContract::BruteForceKnn)
        .then(|| option_u32("k").expect("canonical kNN options contain k"));
    let kmeans_clusters = (operation.descriptor().output_contract == CuvsOutputContract::KMeansAssignments)
        .then(|| option_u32("n_clusters").expect("canonical KMeans options contain n_clusters"));
    let pca_components = (operation.descriptor().output_contract == CuvsOutputContract::PcaComponents)
        .then(|| option_u32("n_components").expect("canonical PCA options contain n_components"));
    Ok(CuvsValidatedOptions { canonical_json: canonical.to_string(), bindings, knn_k, kmeans_clusters, pca_components })
}

fn binding(object: &Map<String, Value>, role: CuvsRelationRole) -> CuvsVectorRelationBinding {
    let name = role.as_str();
    let binding =
        object.get(name).and_then(Value::as_object).expect("canonical cuVS relation binding must be an object");
    let id = binding
        .get("id")
        .and_then(Value::as_str)
        .expect("canonical cuVS relation binding must contain an id")
        .to_string();
    let vector_object = binding
        .get("vector")
        .and_then(Value::as_object)
        .expect("canonical cuVS relation binding must contain a vector object");
    let vector = if let Some(columns) = vector_object.get("columns") {
        CuvsDenseVectorBinding::Columns(
            columns
                .as_array()
                .expect("canonical cuVS columns binding must be an array")
                .iter()
                .map(|column| column.as_str().expect("canonical cuVS vector column must be a string").to_string())
                .collect(),
        )
    } else {
        CuvsDenseVectorBinding::VectorColumn(
            vector_object
                .get("column")
                .and_then(Value::as_str)
                .expect("canonical cuVS vector-column binding must contain a column")
                .to_string(),
        )
    };
    CuvsVectorRelationBinding { relation_role: role, id, vector }
}
