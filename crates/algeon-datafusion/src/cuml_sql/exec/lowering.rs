use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result};
use query_engine::plan::{
    MlAlgorithm, MlAlgorithmParams, MlAlgorithmSpec, MlClusterSelectionMethod, MlHdbscanParams,
    MlLogisticRegressionParams, MlPcaParams, MlPcaSolver, MlRandomForestParams, MlRelationBinding, MlSplitCriterion,
    MlTsvdParams, MlUmapInit, MlUmapParams, QueryPlan, QueryPlanBuilder, VectorRowIdBinding,
};

use crate::{
    cuml_sql::{CumlOperation, CumlValidatedOptions, DenseVectorBinding, error::invalid_call},
    error::{Error, ErrorCode},
    native_exec::{
        child_inputs::NativeChildBinding,
        relation_function::{RelationFunctionSink, add_projected_relation_source},
    },
};

pub(super) fn lower_cuml_query_plan(
    operation: CumlOperation,
    options: &CumlValidatedOptions,
    children: &[NativeChildBinding],
    output_schema: SchemaRef,
    sink: RelationFunctionSink,
) -> Result<QueryPlan> {
    if children.len() != options.bindings().len() {
        return Err(invalid_call(
            operation,
            "physical_child_count_mismatch",
            "cuML physical child count does not match its validated relation bindings",
        ));
    }
    let mut builder = QueryPlanBuilder::new();
    let mut relations = Vec::with_capacity(children.len());
    for (child, binding) in children.iter().zip(options.bindings()) {
        if child.role != binding.relation_role().as_str() {
            return Err(invalid_call(
                operation,
                "physical_child_role_mismatch",
                "cuML physical child order does not match semantic relation order",
            ));
        }
        let input = add_projected_relation_source(&mut builder, child);
        let relation = match binding.features() {
            DenseVectorBinding::Columns(columns) => {
                MlRelationBinding::new(input, VectorRowIdBinding::column(binding.id()), columns.iter().cloned())
            },
            DenseVectorBinding::VectorColumn(column) => {
                MlRelationBinding::new_vector_column(input, VectorRowIdBinding::column(binding.id()), column.clone())
            },
        };
        let mut relation = relation;
        relation.label = binding.label().map(str::to_owned);
        relations.push(relation);
    }
    let canonical: serde_json::Value = serde_json::from_str(options.canonical_json()).map_err(|error| {
        DataFusionError::from(
            Error::new(
                ErrorCode::Internal,
                "algeon_datafusion.cuml_sql.lower",
                "validated cuML options were not valid canonical JSON",
            )
            .with_source(error),
        )
    })?;
    let decomposition = || {
        let solver = match canonical["solver"].as_str().expect("validated cuML solver") {
            "cov_eig_dq" => MlPcaSolver::CovEigDq,
            "cov_eig_jacobi" => MlPcaSolver::CovEigJacobi,
            _ => unreachable!("canonical solver is registry validated"),
        };
        let tolerance = canonical["tol"].as_f64().expect("validated cuML tolerance");
        let n_iterations = canonical["n_iterations"].as_u64().expect("validated cuML iterations") as u32;
        (solver, tolerance, n_iterations)
    };
    let (algorithm, params) = match operation {
        CumlOperation::Pca => {
            let (solver, tolerance, n_iterations) = decomposition();
            let mut params = MlPcaParams::new(relations.remove(0), options.n_components().expect("PCA components"));
            params.solver = solver;
            params.tolerance = tolerance;
            params.n_iterations = n_iterations;
            params.whiten = canonical["whiten"].as_bool().expect("validated cuML whiten");
            (MlAlgorithm::Pca, MlAlgorithmParams::Pca(params))
        },
        CumlOperation::Tsvd => {
            let (solver, tolerance, n_iterations) = decomposition();
            let mut params = MlTsvdParams::new(relations.remove(0), options.n_components().expect("TSVD components"));
            params.solver = solver;
            params.tolerance = tolerance;
            params.n_iterations = n_iterations;
            (MlAlgorithm::Tsvd, MlAlgorithmParams::Tsvd(params))
        },
        CumlOperation::Hdbscan => {
            let mut params = MlHdbscanParams::new(relations.remove(0));
            params.min_samples = canonical["min_samples"].as_u64().unwrap() as usize;
            params.min_cluster_size = canonical["min_cluster_size"].as_u64().unwrap() as usize;
            params.max_cluster_size = canonical["max_cluster_size"].as_u64().unwrap() as usize;
            params.cluster_selection_epsilon = canonical["cluster_selection_epsilon"].as_f64().unwrap();
            params.alpha = canonical["alpha"].as_f64().unwrap();
            params.allow_single_cluster = canonical["allow_single_cluster"].as_bool().unwrap();
            params.cluster_selection_method = match canonical["cluster_selection_method"].as_str() {
                Some("eom") => MlClusterSelectionMethod::Eom,
                Some("leaf") => MlClusterSelectionMethod::Leaf,
                _ => return Err(invalid_call(operation, "invalid_enum_option", "invalid cluster selection method")),
            };
            (MlAlgorithm::Hdbscan, MlAlgorithmParams::Hdbscan(params))
        },
        CumlOperation::LogisticRegression => {
            let mut params = MlLogisticRegressionParams::new(relations.remove(0), relations.remove(0));
            params.penalty_l1 = canonical["penalty_l1"].as_f64().unwrap();
            params.penalty_l2 = canonical["penalty_l2"].as_f64().unwrap();
            params.max_iter = canonical["max_iter"].as_u64().unwrap() as usize;
            params.linesearch_max_iter = canonical["linesearch_max_iter"].as_u64().unwrap() as usize;
            params.lbfgs_memory = canonical["lbfgs_memory"].as_u64().unwrap() as usize;
            params.fit_intercept = canonical["fit_intercept"].as_bool().unwrap();
            params.penalty_normalized = canonical["penalty_normalized"].as_bool().unwrap();
            params.grad_tol = canonical["grad_tol"].as_f64().unwrap();
            params.change_tol = canonical["change_tol"].as_f64().unwrap();
            (MlAlgorithm::LogisticRegression, MlAlgorithmParams::LogisticRegression(params))
        },
        CumlOperation::RandomForestClassifier | CumlOperation::RandomForestRegressor => {
            let mut params = MlRandomForestParams::new(relations.remove(0), relations.remove(0));
            params.n_trees = canonical["n_trees"].as_u64().unwrap() as usize;
            params.max_depth = canonical["max_depth"].as_u64().unwrap() as i32;
            params.max_leaves =
                canonical.get("max_leaves").and_then(serde_json::Value::as_u64).map(|value| value as u32);
            params.max_features = canonical["max_features"].as_f64().unwrap();
            params.max_samples = canonical["max_samples"].as_f64().unwrap();
            params.bootstrap = canonical["bootstrap"].as_bool().unwrap();
            params.max_n_bins = canonical["max_n_bins"].as_u64().unwrap() as usize;
            params.min_samples_leaf = canonical["min_samples_leaf"].as_u64().unwrap() as usize;
            params.min_samples_split = canonical["min_samples_split"].as_u64().unwrap() as usize;
            params.min_impurity_decrease = canonical["min_impurity_decrease"].as_f64().unwrap();
            params.max_batch_size = canonical["max_batch_size"].as_u64().unwrap() as usize;
            params.seed = canonical["seed"].as_u64().unwrap();
            if operation == CumlOperation::RandomForestRegressor {
                params.criterion = None;
                (MlAlgorithm::RandomForestRegressor, MlAlgorithmParams::RandomForestRegressor(params))
            } else {
                params.criterion = Some(match canonical["split_criterion"].as_str() {
                    Some("gini") => MlSplitCriterion::Gini,
                    Some("entropy") => MlSplitCriterion::Entropy,
                    _ => unreachable!("validated RF split criterion"),
                });
                (MlAlgorithm::RandomForestClassifier, MlAlgorithmParams::RandomForestClassifier(params))
            }
        },
        CumlOperation::Umap => {
            let mut params = MlUmapParams::new(relations.remove(0));
            params.n_components = options.n_components().expect("UMAP components");
            params.n_neighbors = canonical["n_neighbors"].as_u64().unwrap() as usize;
            params.n_epochs = canonical["n_epochs"].as_u64().unwrap() as usize;
            params.learning_rate = canonical["learning_rate"].as_f64().unwrap();
            params.min_dist = canonical["min_dist"].as_f64().unwrap();
            params.spread = canonical["spread"].as_f64().unwrap();
            params.set_op_mix_ratio = canonical["set_op_mix_ratio"].as_f64().unwrap();
            params.local_connectivity = canonical["local_connectivity"].as_f64().unwrap();
            params.repulsion_strength = canonical["repulsion_strength"].as_f64().unwrap();
            params.negative_sample_rate = canonical["negative_sample_rate"].as_u64().unwrap() as usize;
            params.init = match canonical["init"].as_str() {
                Some("spectral") => MlUmapInit::Spectral,
                Some("random") => MlUmapInit::Random,
                _ => return Err(invalid_call(operation, "invalid_enum_option", "invalid UMAP initialization")),
            };
            params.random_state = canonical["random_state"].as_u64().unwrap();
            (MlAlgorithm::Umap, MlAlgorithmParams::Umap(params))
        },
    };
    let root = builder
        .try_add_ml_algorithm(MlAlgorithmSpec::new(algorithm, params, Arc::clone(&output_schema)))
        .map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.cuml_sql.lower",
                "failed to lower cuML SQL into native ML IR",
                error,
            ))
        })?;
    Ok(match sink {
        RelationFunctionSink::HostArrow => builder.finish_with_host_sink_schema(root, output_schema),
        RelationFunctionSink::DeferredHostArrow => {
            builder.finish_with_host_sink_schema(root, output_schema).clone_with_deferred_host_materialization_sink()
        },
        RelationFunctionSink::GpuResident => builder.finish_gpu_resident(root),
    })
}
