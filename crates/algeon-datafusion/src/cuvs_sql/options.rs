use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use query_engine::plan::{
    VectorAlgorithmParams, VectorBruteForceKnnParams, VectorCagraBuildAlgo, VectorCagraParams, VectorCagraSearchAlgo,
    VectorDistanceMetric, VectorIvfFlatParams, VectorIvfPqCodebookKind, VectorIvfPqParams, VectorKMeansInit,
    VectorKMeansMetric, VectorKMeansParams, VectorPairwiseDistanceParams, VectorPcaParams, VectorPcaSolver,
    VectorRelationBinding, VectorRowIdBinding, VectorTopKParams,
};
use serde_json::{Map, Value};

use super::{
    CuvsOperation, CuvsRelationRole, DenseVectorBinding, canonicalize_cuvs_options,
    error::{ERRORS, invalid_option},
};
use crate::{
    error::Result,
    gpu_functions::dense_relation::{CanonicalOptions, DenseRelationArgument, DenseRelationFunction, LabelColumn},
};

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

/// Options of one cuVS call, validated and parsed once. The canonical JSON
/// is the call's identity and serialized form; validation and lowering read
/// the typed parameters.
#[derive(Debug, Clone)]
pub struct CuvsValidatedOptions {
    canonical_json: String,
    value_options: Map<String, Value>,
    bindings: Vec<CuvsVectorRelationBinding>,
    params: VectorAlgorithmParams,
}

impl CuvsValidatedOptions {
    pub fn canonical_json(&self) -> &str {
        &self.canonical_json
    }

    pub fn bindings(&self) -> &[CuvsVectorRelationBinding] {
        &self.bindings
    }

    /// The native vector algorithm parameters. Their relation `n` reads the
    /// `n`th binding.
    pub fn params(&self) -> &VectorAlgorithmParams {
        &self.params
    }

    /// The value options by name, without relation bindings.
    pub(super) fn value_options(&self) -> &Map<String, Value> {
        &self.value_options
    }
}

// The canonical JSON determines every other field, so it is the identity.
impl PartialEq for CuvsValidatedOptions {
    fn eq(&self, other: &Self) -> bool {
        self.canonical_json == other.canonical_json
    }
}

impl Eq for CuvsValidatedOptions {}

impl Hash for CuvsValidatedOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_json.hash(state);
    }
}

impl PartialOrd for CuvsValidatedOptions {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CuvsValidatedOptions {
    fn cmp(&self, other: &Self) -> Ordering {
        self.canonical_json.cmp(&other.canonical_json)
    }
}

pub(super) fn validate_options(operation: CuvsOperation, raw: &str) -> Result<CuvsValidatedOptions> {
    let canonical = canonicalize_cuvs_options(operation, raw)
        .map_err(|error| invalid_option(operation, error.code(), error.argument(), error.message()))?;
    let options = dense_relation_function(operation).canonical_options(&canonical);
    let bindings = operation
        .descriptor()
        .relation_roles
        .iter()
        .map(|&relation_role| {
            let binding = options.relation_binding(relation_role.as_str())?;
            Ok(CuvsVectorRelationBinding { relation_role, id: binding.id, vector: binding.vector })
        })
        .collect::<Result<Vec<_>>>()?;
    let params = algorithm_params(operation, &options, &bindings)?;
    let value_options = options.value_options(operation.descriptor().options);
    Ok(CuvsValidatedOptions { canonical_json: Value::Object(canonical).to_string(), value_options, bindings, params })
}

const DISTANCE_METRICS: [(&str, VectorDistanceMetric); 4] = [
    ("l2_expanded", VectorDistanceMetric::L2Expanded),
    ("l2_sqrt_expanded", VectorDistanceMetric::L2SqrtExpanded),
    ("cosine", VectorDistanceMetric::CosineExpanded),
    ("inner_product", VectorDistanceMetric::InnerProduct),
];

fn algorithm_params(
    operation: CuvsOperation,
    options: &CanonicalOptions<'_>,
    bindings: &[CuvsVectorRelationBinding],
) -> Result<VectorAlgorithmParams> {
    let relation = |ordinal: usize| {
        let binding = &bindings[ordinal];
        let row_id = VectorRowIdBinding::column(binding.id());
        match binding.vector() {
            DenseVectorBinding::Columns(columns) => {
                VectorRelationBinding::new(ordinal, row_id, columns.iter().cloned())
            },
            DenseVectorBinding::VectorColumn(column) => {
                VectorRelationBinding::new_vector_column(ordinal, row_id, column.clone())
            },
        }
    };
    Ok(match operation {
        CuvsOperation::BruteForceKnn => VectorAlgorithmParams::BruteForceKnn(VectorBruteForceKnnParams::new(
            relation(0),
            relation(1),
            options.usize("k")?,
            options.choice("metric", &DISTANCE_METRICS)?,
        )),
        CuvsOperation::IvfFlat => {
            let mut params = VectorIvfFlatParams::new(
                relation(0),
                relation(1),
                options.usize("k")?,
                options.choice("metric", &DISTANCE_METRICS)?,
                options.u32("n_lists")?,
            );
            params.n_probes = options.u32("n_probes")?;
            params.kmeans_n_iters = options.u32("kmeans_n_iters")?;
            params.kmeans_trainset_fraction = options.f64("kmeans_trainset_fraction")?;
            VectorAlgorithmParams::IvfFlat(params)
        },
        CuvsOperation::IvfPq => {
            let mut params = VectorIvfPqParams::new(
                relation(0),
                relation(1),
                options.usize("k")?,
                options.choice("metric", &DISTANCE_METRICS)?,
                options.u32("n_lists")?,
            );
            params.n_probes = options.u32("n_probes")?;
            params.kmeans_n_iters = options.u32("kmeans_n_iters")?;
            params.kmeans_trainset_fraction = options.f64("kmeans_trainset_fraction")?;
            params.pq_dim = options.u32("pq_dim")?;
            params.pq_bits = options.u32("pq_bits")?;
            params.codebook_kind = options.choice(
                "codebook_kind",
                &[
                    ("per_subspace", VectorIvfPqCodebookKind::PerSubspace),
                    ("per_cluster", VectorIvfPqCodebookKind::PerCluster),
                ],
            )?;
            VectorAlgorithmParams::IvfPq(params)
        },
        CuvsOperation::Cagra => {
            let mut params = VectorCagraParams::new(
                relation(0),
                relation(1),
                options.usize("k")?,
                options.choice("metric", &DISTANCE_METRICS)?,
            );
            params.graph_degree = options.u32("graph_degree")?;
            params.intermediate_graph_degree = options.u32("intermediate_graph_degree")?;
            params.build_algo = options.choice(
                "build_algo",
                &[("ivf_pq", VectorCagraBuildAlgo::IvfPq), ("nn_descent", VectorCagraBuildAlgo::NnDescent)],
            )?;
            params.itopk_size = options.u32("itopk_size")?;
            params.search_width = options.u32("search_width")?;
            params.max_iterations = options.u32("max_iterations")?;
            params.search_algo = options.choice(
                "search_algo",
                &[
                    ("auto", VectorCagraSearchAlgo::Auto),
                    ("single_cta", VectorCagraSearchAlgo::SingleCta),
                    ("multi_cta", VectorCagraSearchAlgo::MultiCta),
                    ("multi_kernel", VectorCagraSearchAlgo::MultiKernel),
                ],
            )?;
            VectorAlgorithmParams::Cagra(params)
        },
        CuvsOperation::PairwiseDistance => VectorAlgorithmParams::PairwiseDistance(VectorPairwiseDistanceParams {
            x: relation(0),
            y: relation(1),
            metric: options.choice("metric", &DISTANCE_METRICS)?,
        }),
        CuvsOperation::TopK => VectorAlgorithmParams::TopK(VectorTopKParams {
            input: relation(0),
            k: options.usize("k")?,
            select_min: options.choice("select", &[("min", true), ("max", false)])?,
        }),
        CuvsOperation::KMeans => {
            let mut params = VectorKMeansParams::new(relation(0), options.usize("n_clusters")?);
            params.metric = options.choice(
                "metric",
                &[
                    ("l2_expanded", VectorKMeansMetric::L2Expanded),
                    ("l2_sqrt_expanded", VectorKMeansMetric::L2SqrtExpanded),
                ],
            )?;
            params.max_iterations = options.u32("max_iter")?;
            params.tolerance = options.f64("tol")?;
            params.n_init = options.u32("n_init")?;
            params.init = options.choice(
                "init",
                &[("kmeans++", VectorKMeansInit::KMeansPlusPlus), ("random", VectorKMeansInit::Random)],
            )?;
            VectorAlgorithmParams::KMeans(params)
        },
        CuvsOperation::Pca => {
            let mut params = VectorPcaParams::new(relation(0), options.usize("n_components")?);
            params.whiten = options.bool("whiten")?;
            params.solver = options.choice(
                "solver",
                &[("cov_eig_dq", VectorPcaSolver::CovEigDq), ("cov_eig_jacobi", VectorPcaSolver::CovEigJacobi)],
            )?;
            params.tolerance = options.f64("tol")?;
            params.n_iterations = options.u32("n_iterations")?;
            params.flip_signs_based_on_u = options.bool("flip_signs_based_on_u")?;
            VectorAlgorithmParams::Pca(params)
        },
    })
}

/// The cuVS operation as a dense-relation function. No cuVS relation trains
/// on a label, so a `label` column is a feature.
pub(super) fn dense_relation_function(operation: CuvsOperation) -> DenseRelationFunction {
    let descriptor = operation.descriptor();
    DenseRelationFunction {
        errors: &ERRORS,
        sql_function_name: descriptor.sql_function_name,
        relations: descriptor
            .relation_roles
            .iter()
            .map(|role| DenseRelationArgument { role: role.as_str(), label: LabelColumn::Feature })
            .collect(),
        options: descriptor.options,
    }
}
