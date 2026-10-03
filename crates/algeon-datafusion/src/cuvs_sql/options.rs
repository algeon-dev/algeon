use datafusion::logical_expr::LogicalPlan;
use serde_json::{Map, Value};

use super::{
    CuvsOperation, CuvsOutputContract, CuvsRelationRole, DenseVectorBinding, GpuOptionKind, canonicalize_cuvs_options,
    error::{invalid_call, invalid_option},
};
use crate::{error::Result, gpu_functions::dense_relation::infer_dense_relation_binding};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CuvsVectorRelationBinding {
    relation_role: CuvsRelationRole,
    id: String,
    vector: DenseVectorBinding,
}

impl CuvsVectorRelationBinding {
    pub fn relation_role(&self) -> CuvsRelationRole {
        self.relation_role
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn vector(&self) -> &DenseVectorBinding {
        &self.vector
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CuvsValidatedOptions {
    canonical_json: String,
    bindings: Vec<CuvsVectorRelationBinding>,
    knn_k: Option<u32>,
    top_k_k: Option<u32>,
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

    pub fn top_k_k(&self) -> Option<u32> {
        self.top_k_k
    }

    pub fn kmeans_clusters(&self) -> Option<u32> {
        self.kmeans_clusters
    }

    pub fn pca_components(&self) -> Option<u32> {
        self.pca_components
    }

    pub(super) fn named_values(&self, operation: CuvsOperation) -> Map<String, Value> {
        let canonical: Value =
            serde_json::from_str(&self.canonical_json).expect("validated cuVS options contain canonical JSON");
        let canonical = canonical.as_object().expect("validated cuVS options are an object");
        operation
            .descriptor()
            .options
            .iter()
            .filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
            .filter_map(|option| canonical.get(option.name).cloned().map(|value| (option.name.to_string(), value)))
            .collect()
    }
}

pub(super) fn validate_options(operation: CuvsOperation, raw: &str) -> Result<CuvsValidatedOptions> {
    let canonical = canonicalize_cuvs_options(operation, raw)
        .map_err(|error| invalid_option(operation, error.code(), error.argument(), error.message()))?;
    let object = canonical.as_object().expect("canonical cuVS options are always a JSON object");
    let bindings =
        operation.descriptor().relation_roles.iter().copied().map(|role| binding(object, role)).collect::<Vec<_>>();
    let option_u32 = |name: &str| object.get(name).and_then(Value::as_u64).map(|value| value as u32);
    let knn_k = (operation.descriptor().output_contract == CuvsOutputContract::KnnNeighbors)
        .then(|| option_u32("k").expect("canonical kNN options contain k"));
    let top_k_k =
        (operation == CuvsOperation::TopK).then(|| option_u32("k").expect("canonical Top-K options contain k"));
    let kmeans_clusters = (operation.descriptor().output_contract == CuvsOutputContract::KMeansAssignments)
        .then(|| option_u32("n_clusters").expect("canonical KMeans options contain n_clusters"));
    let pca_components = (operation.descriptor().output_contract == CuvsOutputContract::PcaComponents)
        .then(|| option_u32("n_components").expect("canonical PCA options contain n_components"));
    Ok(CuvsValidatedOptions {
        canonical_json: canonical.to_string(),
        bindings,
        knn_k,
        top_k_k,
        kmeans_clusters,
        pca_components,
    })
}

pub(super) fn options_from_named_values(
    operation: CuvsOperation,
    inputs: &[LogicalPlan],
    mut values: Map<String, Value>,
) -> Result<String> {
    let roles = operation.descriptor().relation_roles;
    if inputs.len() != roles.len() {
        return Err(invalid_call(
            operation,
            "invalid_relation_count",
            format!("{} requires {} relation argument(s)", operation.descriptor().sql_function_name, roles.len()),
        ));
    }
    for (role, input) in roles.iter().copied().zip(inputs) {
        if values.contains_key(role.as_str()) {
            return Err(invalid_call(
                operation,
                "unknown_argument",
                format!("cuVS relation role '{}' cannot appear in the value options object", role.as_str()),
            ));
        }
        values.insert(role.as_str().to_string(), relation_binding(operation, role, input)?);
    }
    Ok(Value::Object(values).to_string())
}

fn relation_binding(operation: CuvsOperation, role: CuvsRelationRole, input: &LogicalPlan) -> Result<Value> {
    infer_dense_relation_binding(
        input,
        role.as_str(),
        "cuVS",
        || super::error::field_not_found(operation, role.as_str(), "id"),
        |reason, message| invalid_call(operation, reason, message),
    )
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
        DenseVectorBinding::Columns(
            columns
                .as_array()
                .expect("canonical cuVS columns binding must be an array")
                .iter()
                .map(|column| column.as_str().expect("canonical cuVS vector column must be a string").to_string())
                .collect(),
        )
    } else {
        DenseVectorBinding::VectorColumn(
            vector_object
                .get("column")
                .and_then(Value::as_str)
                .expect("canonical cuVS vector-column binding must contain a column")
                .to_string(),
        )
    };
    CuvsVectorRelationBinding { relation_role: role, id, vector }
}
