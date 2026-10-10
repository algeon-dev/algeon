#[cfg(feature = "test-hooks")]
use std::sync::{
    Mutex, PoisonError,
    atomic::{AtomicUsize, Ordering},
};
use std::{collections::BTreeMap, num::NonZeroUsize};

use cudf::{
    DenseMatrixLayout, Exec, ExecutionBudgetRequest, PackedMatrix, Runtime, Series, TypeId,
    interop::arrow_host::cudf_dtype_from_arrow_type,
};
use cuml::{
    ClusterSelectionMethod, CumlResources, DeviceMatrix, ForestInferenceModel, ForestOutput, ForestOutputKind,
    ForestPrecision, HdbscanFit, HdbscanParams, LogisticRegressionModel, LogisticRegressionParams, OwnedFloat32Matrix,
    PcaSolver, RandomForestClassifier, RandomForestParams, RandomForestRegressor, SplitCriterion, TsvdModel,
    TsvdParams, UmapInit, UmapModel, UmapParams,
};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{
        MlAlgorithm, MlAlgorithmNode, MlAlgorithmParams, MlClusterSelectionMethod, MlDecompositionSolver,
        MlForestModelFacts, MlForestModelSnapshot, MlForestOutputContract, MlForestPrecision, MlForestPredictParams,
        MlHdbscanParams, MlLogisticRegressionParams, MlRandomForestParams, MlRelationBinding, MlSplitCriterion,
        MlTsvdParams, MlUmapInit, MlUmapParams, NodeId, QueryPlan, VectorRowIdBinding, ml_classification_output_schema,
        ml_embedding_output_schema, ml_forest_output_schema, ml_hdbscan_output_schema, ml_regression_output_schema,
        vector_pca_output_schema,
    },
};
use query_runtime::{
    native_error::cudf_query_error,
    observability::{
        RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
    },
    runtime::ExecutionContext,
};

use super::assembly::{assemble_components, assemble_ml_output};
use crate::{
    exec::{
        GpuDataFrame,
        dense_relation::{
            DenseRelationInputs, ML_OPERATIONS, RelationColumns, matrix_bytes_for, pack_dense_relation_for,
            relation_element_type_for, relation_input_for, require_null_free_relation,
        },
    },
    native_error::{cudf_unsupported_type_error, cuml_query_error},
};

const EXECUTE: &str = "query_engine.ml_algorithm.execute";
const NATIVE: &str = "query_engine.ml_algorithm.native";
/// Element type every cuML fit and prediction reads. Packing narrows a
/// Float64 relation to it; forest prediction instead requires it at the input
/// and widens on the device when the model has double-precision thresholds.
const FEATURE_ELEMENT: TypeId = TypeId::Float32;

fn invalid(node_id: NodeId, reason: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::CudfInvalidInput, EXECUTE, message)
        .with_fact("node_id", node_id)
        .with_fact("reason", reason)
}

pub(in crate::exec) fn validate_ml_algorithm_execution(
    plan: &QueryPlan,
    node_id: NodeId,
    ml: &MlAlgorithmNode,
) -> Result<()> {
    if !ml.spec.params.supports_algorithm(ml.spec.algorithm) {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, EXECUTE, "ML node payload does not match its inputs")
            .with_fact("node_id", node_id)
            .with_fact("reason", "algorithm_parameter_mismatch"));
    }
    let (binding, components) = match &ml.spec.params {
        MlAlgorithmParams::Tsvd(params) => (&params.input, Some(params.n_components)),
        MlAlgorithmParams::Hdbscan(params) => (&params.input, None),
        MlAlgorithmParams::Umap(params) => (&params.input, Some(params.n_components)),
        MlAlgorithmParams::ForestPredict(params) => (&params.input, None),
        MlAlgorithmParams::LogisticRegression(params) => (&params.predict, None),
        MlAlgorithmParams::RandomForestClassifier(params) | MlAlgorithmParams::RandomForestRegressor(params) => {
            (&params.predict, None)
        },
    };
    let VectorRowIdBinding::Column(id) = &binding.row_id else {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, EXECUTE, "ML input requires a row ID")
            .with_fact("node_id", node_id)
            .with_fact("reason", "missing_logical_row_id_column"));
    };
    let schema = plan
        .node(ml.inputs[binding.input])
        .ok_or_else(|| {
            QueryError::new(QueryErrorCode::InvalidPlan, EXECUTE, "ML input node is missing")
                .with_fact("node_id", node_id)
                .with_fact("reason", "missing_evaluated_relation")
        })?
        .output_schema();
    let id_field = schema.field_with_name(id).map_err(|_| {
        QueryError::new(QueryErrorCode::InvalidPlan, EXECUTE, "ML ID column is missing")
            .with_fact("node_id", node_id)
            .with_fact("reason", "missing_row_id_column")
    })?;
    let id_type = id_field.data_type().clone();
    let expected = match &ml.spec.params {
        MlAlgorithmParams::LogisticRegression(_) | MlAlgorithmParams::RandomForestClassifier(_) => {
            ml_classification_output_schema(id_type)
        },
        MlAlgorithmParams::RandomForestRegressor(_) => ml_regression_output_schema(id_type),
        MlAlgorithmParams::ForestPredict(params) => ml_forest_output_schema(id_type, params.model.output()),
        MlAlgorithmParams::Hdbscan(_) => ml_hdbscan_output_schema(id_type),
        MlAlgorithmParams::Umap(_) => ml_embedding_output_schema(id_type, components.expect("UMAP dimension")),
        MlAlgorithmParams::Tsvd(_) => vector_pca_output_schema(id_type, components.expect("decomposition dimension")),
    };
    if ml.output_schema().as_ref() != expected.as_ref() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            EXECUTE,
            "ML output schema differs from the components contract",
        )
        .with_fact("node_id", node_id)
        .with_fact("reason", "unexpected_output_schema"));
    }
    Ok(())
}

pub(in crate::exec) fn execute_ml_algorithm(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    ml: &MlAlgorithmNode,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
) -> Result<GpuDataFrame> {
    context.check_cancellation(EXECUTE)?;
    validate_ml_algorithm_execution(plan, node_id, ml)?;
    let inputs = DenseRelationInputs::new(&ml.inputs, inputs);
    let binding = match &ml.spec.params {
        MlAlgorithmParams::LogisticRegression(params) => &params.predict,
        MlAlgorithmParams::RandomForestClassifier(params) | MlAlgorithmParams::RandomForestRegressor(params) => {
            &params.predict
        },
        _ => ml.spec.params.relation_bindings().next().expect("ML input"),
    };
    if relation_input_for(&ML_OPERATIONS, inputs, binding.input, node_id, "input")?.num_rows() == 0 {
        return empty_ml_output(context, ml);
    }
    for binding in ml.spec.params.relation_bindings() {
        require_null_free_relation(
            &ML_OPERATIONS,
            inputs,
            node_id,
            ml.spec.algorithm.as_str(),
            RelationColumns {
                role: ml.spec.params.relation_role(binding.input).as_str(),
                input: binding.input,
                row_id: &binding.row_id,
                vector: &binding.features,
                label: binding.label.as_deref(),
            },
        )?;
    }
    let bytes = context.device_budget_bytes();
    let bytes = usize::try_from(bytes).map_err(|_| {
        QueryError::new(QueryErrorCode::InvalidPlan, EXECUTE, "ML execution budget exceeds usize")
            .with_fact("budget_bytes", bytes)
    })?;
    let domain_token =
        context.execution_budget_token(ExecutionBudgetRequest::new(bytes).with_boundary("ml_algorithm"))?;
    let runtime = context.shared_runtime()?;
    let resources = CumlResources::for_runtime(runtime).map_err(|error| {
        cuml_query_error(EXECUTE, "failed to create query-local cuML resources", error).with_fact("node_id", node_id)
    })?;
    let version = cuml_sys::validate_runtime_version().map_err(|error| {
        QueryError::new(QueryErrorCode::CudfDependencyFailure, EXECUTE, "failed to validate native cuML version")
            .with_fact("node_id", node_id)
            .with_source(error)
    })?;
    match &ml.spec.params {
        MlAlgorithmParams::Hdbscan(params) => execute_row_major_ml(
            context,
            plan,
            node_id,
            ml,
            inputs,
            (&resources, version),
            RowMajorParams::Hdbscan(params),
        ),
        MlAlgorithmParams::LogisticRegression(params) => {
            execute_logistic_regression(context, plan, node_id, inputs, &resources, version, params)
        },
        MlAlgorithmParams::RandomForestClassifier(params) | MlAlgorithmParams::RandomForestRegressor(params) => {
            execute_random_forest(
                context,
                plan,
                node_id,
                inputs,
                (&resources, version, &domain_token),
                params,
                ml.spec.algorithm,
            )
        },
        MlAlgorithmParams::Umap(params) => execute_row_major_ml(
            context,
            plan,
            node_id,
            ml,
            inputs,
            (&resources, version),
            RowMajorParams::Umap(params),
        ),
        MlAlgorithmParams::Tsvd(params) => execute_tsvd_ml(context, plan, node_id, inputs, &resources, version, params),
        MlAlgorithmParams::ForestPredict(params) => {
            execute_forest_predict(context, plan, node_id, inputs, &resources, params)
        },
    }
}

fn execute_tsvd_ml(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    inputs: DenseRelationInputs<'_>,
    resources: &CumlResources,
    version: (u16, u16, u16),
    params: &MlTsvdParams,
) -> Result<GpuDataFrame> {
    let binding = &params.input;
    let algorithm = MlAlgorithm::Tsvd;
    let n_components = params.n_components;
    let solver = params.solver;
    let tolerance = params.tolerance;
    let n_iterations = params.n_iterations;
    let input = relation_input_for(&ML_OPERATIONS, inputs, binding.input, node_id, "input")?;
    let rows = input.num_rows();
    if rows < 2 {
        return Err(invalid(node_id, "insufficient_input_rows", "ML input requires at least two rows")
            .with_fact("input_rows", rows));
    }
    let identity = RuntimeOperatorStageIdentity::new()
        .with_node_id(node_id.to_string())
        .with_ml_algorithm(algorithm.as_str())
        .with_execution_binding_id(resources.binding_id().get())
        .with_device_ordinal(resources.device().ordinal())
        .with_native_version(version.0, version.1, version.2);
    let (packed, dimension) = pack_dense_relation_for(
        &ML_OPERATIONS,
        context,
        resources.runtime(),
        plan,
        node_id,
        "input",
        algorithm.as_str(),
        inputs.node_id(binding.input),
        &binding.features,
        input,
        FEATURE_ELEMENT,
        DenseMatrixLayout::ColumnMajor,
        identity.clone(),
        RuntimeOperatorStageKind::MlPack,
    )?;
    if dimension < 2 {
        return Err(invalid(node_id, "insufficient_feature_dimension", "ML input requires two feature dimensions")
            .with_fact("dimension", dimension));
    }
    if n_components > dimension {
        return Err(invalid(
            node_id,
            "component_count_exceeds_feature_dimension",
            "ML components exceed input dimension",
        )
        .with_fact("n_components", n_components)
        .with_fact("dimension", dimension));
    }
    let tolerance = tolerance as f32;
    if !tolerance.is_finite() {
        return Err(invalid(node_id, "tolerance_out_of_float32_range", "ML tolerance exceeds Float32 range"));
    }
    let identity = identity
        .with_vector_relation_role("input")
        .with_vector_dimension(dimension)
        .with_matrix_layout("column_major")
        .with_input_rows(rows as u64);
    let mut matrix = OwnedFloat32Matrix::from_packed(resources, packed)
        .map_err(|error| cuml_query_error(ML_OPERATIONS.pack, "failed to adopt packed ML matrix", error))?;
    let solver_name = solver.as_str();
    let solver = match solver {
        MlDecompositionSolver::CovEigDq => PcaSolver::CovEigDq,
        MlDecompositionSolver::CovEigJacobi => PcaSolver::CovEigJacobi,
    };
    let input_bytes = matrix_bytes_for::<f32>(&ML_OPERATIONS, rows, dimension, node_id, "input")?;
    let output_bytes = matrix_bytes_for::<f32>(&ML_OPERATIONS, rows, n_components, node_id, "output")?;
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlNative,
        identity.clone().with_strategy_label(solver_name).with_input_rows(rows as u64).with_input_bytes(input_bytes),
    );
    let transformed = (|| {
        context.check_cancellation(NATIVE)?;
        let mut params = TsvdParams::new(n_components);
        params.solver = solver;
        params.tolerance = tolerance;
        params.iterations = n_iterations;
        let (_model, output) = TsvdModel::fit_transform(resources, &mut matrix, params)
            .map_err(|error| cuml_query_error(NATIVE, "failed to fit and transform cuML TSVD", error))?;
        context.check_cancellation(NATIVE)?;
        Ok(output)
    })();
    let transformed = match transformed {
        Ok(output) => {
            guard.finish(Some(rows as u64), Some(output_bytes));
            output
        },
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlAssemble,
        identity,
    );
    let assembled = (|| {
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        let flat = transformed.into_column().map_err(|error| {
            cuml_query_error(ML_OPERATIONS.assemble, "failed to transfer ML component buffer", error)
        })?;
        let columns = resources
            .runtime()
            .unpack_dense_matrix(flat, rows, n_components, DenseMatrixLayout::ColumnMajor)
            .map_err(|error| cudf_query_error(ML_OPERATIONS.assemble, "failed to split ML component columns", error))?;
        let output = assemble_components(context, input, binding, columns)?;
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        Ok(output)
    })();
    match assembled {
        Ok(output) => {
            let bytes = match output.device_allocated_size_bytes() {
                Ok(bytes) => bytes,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            guard.finish(Some(rows as u64), Some(bytes));
            Ok(output)
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

fn empty_ml_output(context: &ExecutionContext, ml: &MlAlgorithmNode) -> Result<GpuDataFrame> {
    let runtime = context.runtime()?;
    let columns = ml
        .output_schema()
        .fields()
        .iter()
        .map(|field| {
            let dtype = cudf_dtype_from_arrow_type(field.data_type())
                .map_err(|error| cudf_unsupported_type_error(EXECUTE, "invalid empty ML output type", error))?;
            let series = runtime
                .empty_series(dtype)
                .map_err(|error| cudf_query_error(EXECUTE, "failed to create empty ML output", error))?;
            Ok((field.name().clone(), series))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GpuDataFrame::new(
        cudf::DataFrame::from_series(columns)
            .map_err(|error| cudf_query_error(EXECUTE, "failed to create empty ML frame", error))?,
    ))
}

fn checked_f32(value: f64, node_id: NodeId, option: &'static str) -> Result<f32> {
    let converted = value as f32;
    if !converted.is_finite() || (value > 0.0 && converted == 0.0) {
        return Err(invalid(node_id, "option_out_of_float32_range", "ML option exceeds Float32 range")
            .with_fact("option", option));
    }
    Ok(converted)
}

fn execute_logistic_regression(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    inputs: DenseRelationInputs<'_>,
    resources: &CumlResources,
    version: (u16, u16, u16),
    params: &MlLogisticRegressionParams,
) -> Result<GpuDataFrame> {
    let training = relation_input_for(&ML_OPERATIONS, inputs, params.training.input, node_id, "training")?;
    let predict = relation_input_for(&ML_OPERATIONS, inputs, params.predict.input, node_id, "predict")?;
    let rows = training.num_rows();
    if rows < 2 {
        return Err(invalid(node_id, "insufficient_training_rows", "training requires at least two rows")
            .with_fact("training_rows", rows));
    }
    let identity = RuntimeOperatorStageIdentity::new()
        .with_node_id(node_id.to_string())
        .with_ml_algorithm("logistic_regression")
        .with_execution_binding_id(resources.binding_id().get())
        .with_device_ordinal(resources.device().ordinal())
        .with_native_version(version.0, version.1, version.2)
        .with_matrix_layout("row_major");
    let (training_packed, training_dimension) = pack_supervised_relation(
        context,
        plan,
        node_id,
        inputs,
        (resources, &identity),
        (training, &params.training, "training"),
        "logistic_regression",
    )?;
    let label = params.training.label.as_deref().expect("validated training label");
    let labels = training
        .frame()
        .series(label)
        .map_err(|error| cudf_query_error(EXECUTE, "failed to read training label", error))?;
    let training_validated =
        training_packed.validate().map_err(|error| cudf_query_error(NATIVE, "invalid training matrix", error))?;
    let training_matrix = DeviceMatrix::from_validated(&training_validated)
        .map_err(|error| cuml_query_error(NATIVE, "invalid training matrix", error))?;
    let fit_params = LogisticRegressionParams {
        penalty_l1: params.penalty_l1,
        penalty_l2: params.penalty_l2,
        max_iter: params.max_iter,
        linesearch_max_iter: params.linesearch_max_iter,
        lbfgs_memory: params.lbfgs_memory,
        fit_intercept: params.fit_intercept,
        penalty_normalized: params.penalty_normalized,
        grad_tol: params.grad_tol,
        change_tol: params.change_tol,
    };
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlNative,
        identity
            .clone()
            .with_vector_relation_role("training")
            .with_vector_dimension(training_dimension)
            .with_input_rows(rows as u64)
            .with_input_bytes(matrix_bytes_for::<f32>(&ML_OPERATIONS, rows, training_dimension, node_id, "training")?),
    );
    let fit = (|| {
        context.check_cancellation(NATIVE)?;
        let fit = LogisticRegressionModel::fit(resources, &training_matrix, &labels, fit_params)
            .map_err(|error| cuml_query_error(NATIVE, "failed to fit cuML logistic regression", error))?;
        context.check_cancellation(NATIVE)?;
        Ok(fit)
    })();
    let model = match fit {
        Ok(result) => {
            let loss = if result.0.n_classes() == 2 { "qn_logistic" } else { "qn_softmax" };
            let convergence = if result.1.converged { "converged" } else { "max_iter" };
            guard.finish_with_strategy_label(format!("{loss}:{convergence}"), None, None);
            result.0
        },
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    context.check_cancellation(NATIVE)?;
    let (predict_packed, predict_dimension) = pack_supervised_relation(
        context,
        plan,
        node_id,
        inputs,
        (resources, &identity),
        (predict, &params.predict, "predict"),
        "logistic_regression",
    )?;
    if training_dimension != predict_dimension {
        return Err(invalid(node_id, "feature_dimension_mismatch", "training and predict dimensions differ")
            .with_fact("training_dimension", training_dimension)
            .with_fact("predict_dimension", predict_dimension));
    }
    let predict_validated =
        predict_packed.validate().map_err(|error| cudf_query_error(NATIVE, "invalid predict matrix", error))?;
    let predict_matrix = DeviceMatrix::from_validated(&predict_validated)
        .map_err(|error| cuml_query_error(NATIVE, "invalid predict matrix", error))?;
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlNative,
        identity
            .clone()
            .with_strategy_label("predict")
            .with_vector_relation_role("predict")
            .with_vector_dimension(predict_dimension)
            .with_input_rows(predict.num_rows() as u64)
            .with_input_bytes(matrix_bytes_for::<f32>(
                &ML_OPERATIONS,
                predict.num_rows(),
                predict_dimension,
                node_id,
                "predict",
            )?),
    );
    let predicted = (|| {
        context.check_cancellation(NATIVE)?;
        let result = model
            .predict(resources, &predict_matrix)
            .map_err(|error| cuml_query_error(NATIVE, "failed to predict cuML logistic regression", error))?;
        context.check_cancellation(NATIVE)?;
        Ok(result)
    })();
    let predicted = match predicted {
        Ok(predicted) => {
            // Int32 prediction and Float32 probability.
            guard.finish(
                Some(predict.num_rows() as u64),
                Some(matrix_bytes_for::<i32>(&ML_OPERATIONS, predict.num_rows(), 2, node_id, "prediction")?),
            );
            predicted
        },
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlAssemble,
        identity.with_vector_relation_role("predict").with_vector_dimension(predict_dimension),
    );
    let assembled = (|| {
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        let result = assemble_ml_output(
            context,
            predict,
            &params.predict,
            vec![("prediction".to_owned(), predicted.class), ("probability".to_owned(), predicted.probability)],
            TypeId::Int32,
        )?;
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        Ok(result)
    })();
    match assembled {
        Ok(output) => match output.device_allocated_size_bytes() {
            Ok(bytes) => {
                guard.finish(Some(predict.num_rows() as u64), Some(bytes));
                Ok(output)
            },
            Err(error) => {
                guard.fail(&error);
                Err(error)
            },
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

enum ForestModel {
    Classifier(RandomForestClassifier),
    Regressor(RandomForestRegressor),
}

fn execute_random_forest(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    inputs: DenseRelationInputs<'_>,
    native: (&CumlResources, (u16, u16, u16), &cudf::ExecutionBudgetToken),
    params: &MlRandomForestParams,
    algorithm: MlAlgorithm,
) -> Result<GpuDataFrame> {
    let (resources, version, token) = native;
    let training = relation_input_for(&ML_OPERATIONS, inputs, params.training.input, node_id, "training")?;
    let predict = relation_input_for(&ML_OPERATIONS, inputs, params.predict.input, node_id, "predict")?;
    let rows = training.num_rows();
    if rows < 2 {
        return Err(invalid(node_id, "insufficient_training_rows", "training requires at least two rows")
            .with_fact("training_rows", rows));
    }
    let identity = RuntimeOperatorStageIdentity::new()
        .with_node_id(node_id.to_string())
        .with_ml_algorithm(algorithm.as_str())
        .with_execution_binding_id(resources.binding_id().get())
        .with_device_ordinal(resources.device().ordinal())
        .with_native_version(version.0, version.1, version.2)
        .with_matrix_layout("row_major");
    // The pool routes must remain installed until after the forest is destroyed.
    let streams = resources
        .training_streams(NonZeroUsize::new(1).expect("one stream"), Some(token.stream_reservation()))
        .map_err(|error| {
            cuml_query_error(EXECUTE, "failed to cover RF training stream", error).with_fact("node_id", node_id)
        })?;
    let (training_packed, training_dimension) = pack_supervised_relation(
        context,
        plan,
        node_id,
        inputs,
        (resources, &identity),
        (training, &params.training, "training"),
        algorithm.as_str(),
    )?;
    let labels = training
        .frame()
        .series(params.training.label.as_deref().expect("validated training label"))
        .map_err(|error| cudf_query_error(EXECUTE, "failed to read training label", error))?;
    let training_validated =
        training_packed.validate().map_err(|error| cudf_query_error(NATIVE, "invalid training matrix", error))?;
    let training_matrix = DeviceMatrix::from_validated(&training_validated)
        .map_err(|error| cuml_query_error(NATIVE, "invalid training matrix", error))?;
    let fit_params = RandomForestParams {
        n_trees: params.n_trees,
        bootstrap: params.bootstrap,
        max_samples: checked_f32(params.max_samples, node_id, "max_samples")?,
        seed: params.seed,
        max_depth: params.max_depth,
        max_leaves: params.max_leaves.map(|value| value as i32).unwrap_or(-1),
        max_features: checked_f32(params.max_features, node_id, "max_features")?,
        max_n_bins: params.max_n_bins,
        min_samples_leaf: params.min_samples_leaf,
        min_samples_split: params.min_samples_split,
        min_impurity_decrease: checked_f32(params.min_impurity_decrease, node_id, "min_impurity_decrease")?,
        max_batch_size: params.max_batch_size,
        split_criterion: match params.criterion {
            Some(MlSplitCriterion::Gini) => SplitCriterion::Gini,
            Some(MlSplitCriterion::Entropy) => SplitCriterion::Entropy,
            None => SplitCriterion::Mse,
        },
    };
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlNative,
        identity
            .clone()
            .with_vector_relation_role("training")
            .with_vector_dimension(training_dimension)
            .with_input_rows(rows as u64)
            .with_input_bytes(matrix_bytes_for::<f32>(&ML_OPERATIONS, rows, training_dimension, node_id, "training")?)
            .with_strategy_label(match params.criterion {
                Some(MlSplitCriterion::Gini) => "rf_classifier:gini",
                Some(MlSplitCriterion::Entropy) => "rf_classifier:entropy",
                None => "rf_regressor:mse",
            }),
    );
    let fitted = (|| {
        context.check_cancellation(NATIVE)?;
        let model = if algorithm == MlAlgorithm::RandomForestClassifier {
            ForestModel::Classifier(
                RandomForestClassifier::fit(resources, &streams, &training_matrix, &labels, fit_params)
                    .map_err(|error| cuml_query_error(NATIVE, "failed to fit cuML RF classifier", error))?,
            )
        } else {
            ForestModel::Regressor(
                RandomForestRegressor::fit(resources, &streams, &training_matrix, &labels, fit_params)
                    .map_err(|error| cuml_query_error(NATIVE, "failed to fit cuML RF regressor", error))?,
            )
        };
        context.check_cancellation(NATIVE)?;
        Ok(model)
    })();
    let mut model = match fitted {
        Ok(model) => {
            guard.finish(None, None);
            model
        },
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    let (predict_packed, predict_dimension) = pack_supervised_relation(
        context,
        plan,
        node_id,
        inputs,
        (resources, &identity),
        (predict, &params.predict, "predict"),
        algorithm.as_str(),
    )?;
    if training_dimension != predict_dimension {
        return Err(invalid(node_id, "feature_dimension_mismatch", "training and predict dimensions differ")
            .with_fact("training_dimension", training_dimension)
            .with_fact("predict_dimension", predict_dimension));
    }
    let predict_validated =
        predict_packed.validate().map_err(|error| cudf_query_error(NATIVE, "invalid predict matrix", error))?;
    let predict_matrix = DeviceMatrix::from_validated(&predict_validated)
        .map_err(|error| cuml_query_error(NATIVE, "invalid predict matrix", error))?;
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlNative,
        identity
            .clone()
            .with_strategy_label("predict_nvforest")
            .with_vector_relation_role("predict")
            .with_vector_dimension(predict_dimension)
            .with_input_rows(predict.num_rows() as u64)
            .with_input_bytes(matrix_bytes_for::<f32>(
                &ML_OPERATIONS,
                predict.num_rows(),
                predict_dimension,
                node_id,
                "predict",
            )?),
    );
    let predicted = (|| {
        context.check_cancellation(NATIVE)?;
        let mut forest = match &model {
            ForestModel::Classifier(forest) => forest.forest_model(resources),
            ForestModel::Regressor(forest) => forest.forest_model(resources),
        }
        .map_err(|error| cuml_query_error(NATIVE, "failed to import cuML RF into nvForest", error))?;
        context.check_cancellation(NATIVE)?;
        let prediction = forest
            .predict(resources, &predict_matrix)
            .map_err(|error| cuml_query_error(NATIVE, "failed to predict cuML RF", error))?;
        // Reducing the output validates it on the host, so the nvForest model
        // and predict matrix outlive the kernel that reads them.
        let columns = match &model {
            ForestModel::Classifier(_) => prediction
                .into_class_prediction()
                .map_err(|error| cuml_query_error(NATIVE, "failed to predict cuML RF", error))
                .map(|predicted| {
                    vec![("prediction".to_owned(), predicted.class), ("probability".to_owned(), predicted.probability)]
                })?,
            ForestModel::Regressor(_) => {
                let series = prediction
                    .into_series()
                    .map_err(|error| cuml_query_error(NATIVE, "failed to predict cuML RF", error))?;
                let [result]: [Series; 1] = series.try_into().map_err(|_| {
                    QueryError::new(
                        QueryErrorCode::CudfInvariantViolation,
                        NATIVE,
                        "nvForest prediction series count differs from one",
                    )
                    .with_fact("reason", "invalid_native_output_contract")
                })?;
                vec![("prediction".to_owned(), result)]
            },
        };
        forest.close().map_err(|error| cuml_query_error(NATIVE, "failed to close nvForest model", error))?;
        context.check_cancellation(NATIVE)?;
        Ok(columns)
    })();
    let predicted = match predicted {
        Ok(predicted) => {
            guard.finish(
                Some(predict.num_rows() as u64),
                Some(matrix_bytes_for::<f32>(
                    &ML_OPERATIONS,
                    predict.num_rows(),
                    predicted.len(),
                    node_id,
                    "prediction",
                )?),
            );
            predicted
        },
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    match &mut model {
        ForestModel::Classifier(forest) => forest.close(),
        ForestModel::Regressor(forest) => forest.close(),
    }
    .map_err(|error| cuml_query_error(NATIVE, "failed to close cuML RF", error))?;
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlAssemble,
        identity.with_vector_relation_role("predict").with_vector_dimension(predict_dimension),
    );
    let assembled = (|| {
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        let result = assemble_ml_output(
            context,
            predict,
            &params.predict,
            predicted,
            if algorithm == MlAlgorithm::RandomForestRegressor { TypeId::Float32 } else { TypeId::Int32 },
        )?;
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        Ok(result)
    })();
    match assembled {
        Ok(output) => match output.device_allocated_size_bytes() {
            Ok(bytes) => {
                guard.finish(Some(predict.num_rows() as u64), Some(bytes));
                Ok(output)
            },
            Err(error) => {
                guard.fail(&error);
                Err(error)
            },
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

fn pack_supervised_relation(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    inputs: DenseRelationInputs<'_>,
    native: (&CumlResources, &RuntimeOperatorStageIdentity),
    relation: (&GpuDataFrame, &MlRelationBinding, &'static str),
    algorithm: &'static str,
) -> Result<(cudf::PackedMatrix, usize)> {
    let (resources, identity) = native;
    let (input, binding, role) = relation;
    pack_dense_relation_for(
        &ML_OPERATIONS,
        context,
        resources.runtime(),
        plan,
        node_id,
        role,
        algorithm,
        inputs.node_id(binding.input),
        &binding.features,
        input,
        FEATURE_ELEMENT,
        DenseMatrixLayout::RowMajor,
        identity.clone().with_vector_relation_role(role).with_input_rows(input.num_rows() as u64),
        RuntimeOperatorStageKind::MlPack,
    )
}

#[derive(Clone, Copy)]
enum RowMajorParams<'a> {
    Hdbscan(&'a MlHdbscanParams),
    Umap(&'a MlUmapParams),
}

fn execute_row_major_ml(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    ml: &MlAlgorithmNode,
    inputs: DenseRelationInputs<'_>,
    native: (&CumlResources, (u16, u16, u16)),
    params: RowMajorParams<'_>,
) -> Result<GpuDataFrame> {
    let (resources, version) = native;
    let binding = match params {
        RowMajorParams::Hdbscan(p) => &p.input,
        RowMajorParams::Umap(p) => &p.input,
    };
    let input = relation_input_for(&ML_OPERATIONS, inputs, binding.input, node_id, "input")?;
    let rows = input.num_rows();
    if rows < 2 {
        return Err(invalid(node_id, "insufficient_input_rows", "ML input requires at least two rows")
            .with_fact("input_rows", rows));
    }
    match params {
        RowMajorParams::Hdbscan(p) => {
            for (exceeded, reason) in [
                (p.min_samples >= rows, "min_samples_exceeds_input_rows"),
                (p.min_cluster_size > rows, "min_cluster_size_exceeds_input_rows"),
                (p.max_cluster_size > rows, "max_cluster_size_exceeds_input_rows"),
            ] {
                if exceeded {
                    return Err(invalid(node_id, reason, "HDBSCAN option exceeds evaluated input rows")
                        .with_fact("input_rows", rows));
                }
            }
        },
        RowMajorParams::Umap(p) if p.n_neighbors >= rows => {
            return Err(invalid(
                node_id,
                "n_neighbors_exceeds_input_rows",
                "UMAP neighbors exceed evaluated input rows",
            )
            .with_fact("input_rows", rows));
        },
        _ => {},
    }
    let identity = RuntimeOperatorStageIdentity::new()
        .with_node_id(node_id.to_string())
        .with_ml_algorithm(ml.spec.algorithm.as_str())
        .with_execution_binding_id(resources.binding_id().get())
        .with_device_ordinal(resources.device().ordinal())
        .with_native_version(version.0, version.1, version.2)
        .with_vector_relation_role("input")
        .with_matrix_layout("row_major")
        .with_input_rows(rows as u64);
    let (packed, dimension) = pack_dense_relation_for(
        &ML_OPERATIONS,
        context,
        resources.runtime(),
        plan,
        node_id,
        "input",
        ml.spec.algorithm.as_str(),
        inputs.node_id(binding.input),
        &binding.features,
        input,
        FEATURE_ELEMENT,
        DenseMatrixLayout::RowMajor,
        identity.clone(),
        RuntimeOperatorStageKind::MlPack,
    )?;
    let identity = identity.with_vector_dimension(dimension);
    let input_bytes = matrix_bytes_for::<f32>(&ML_OPERATIONS, rows, dimension, node_id, "input")?;
    let output_bytes = match params {
        RowMajorParams::Hdbscan(_) => matrix_bytes_for::<i64>(&ML_OPERATIONS, rows, 1, node_id, "labels")?
            .saturating_add(matrix_bytes_for::<f32>(&ML_OPERATIONS, rows, 1, node_id, "probabilities")?),
        RowMajorParams::Umap(p) => matrix_bytes_for::<f32>(&ML_OPERATIONS, rows, p.n_components, node_id, "embedding")?,
    };
    let strategy = match params {
        RowMajorParams::Hdbscan(p) => p.cluster_selection_method.as_str(),
        RowMajorParams::Umap(p) => p.init.as_str(),
    };
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlNative,
        identity.clone().with_strategy_label(strategy).with_input_bytes(input_bytes),
    );
    let output = (|| {
        context.check_cancellation(NATIVE)?;
        let output: Vec<(String, Series)> = match params {
            RowMajorParams::Hdbscan(p) => {
                let validated = packed
                    .validate()
                    .map_err(|error| cudf_query_error(NATIVE, "failed to validate packed HDBSCAN matrix", error))?;
                let matrix = DeviceMatrix::from_validated(&validated)
                    .map_err(|error| cuml_query_error(NATIVE, "failed to borrow HDBSCAN matrix", error))?;
                let parameters = HdbscanParams {
                    min_samples: p.min_samples,
                    min_cluster_size: p.min_cluster_size,
                    max_cluster_size: p.max_cluster_size,
                    cluster_selection_epsilon: checked_f32(
                        p.cluster_selection_epsilon,
                        node_id,
                        "cluster_selection_epsilon",
                    )?,
                    alpha: checked_f32(p.alpha, node_id, "alpha")?,
                    allow_single_cluster: p.allow_single_cluster,
                    cluster_selection_method: match p.cluster_selection_method {
                        MlClusterSelectionMethod::Eom => ClusterSelectionMethod::Eom,
                        MlClusterSelectionMethod::Leaf => ClusterSelectionMethod::Leaf,
                    },
                };
                let fitted = HdbscanFit::fit(resources, &matrix, parameters)
                    .map_err(|error| cuml_query_error(NATIVE, "failed to fit cuML HDBSCAN", error))?;
                let output = fitted.output;
                vec![
                    (
                        "cluster_id".to_owned(),
                        output
                            .labels
                            .into_series()
                            .map_err(|error| cuml_query_error(NATIVE, "failed to transfer cluster labels", error))?,
                    ),
                    (
                        "probability".to_owned(),
                        output.probabilities.into_series().map_err(|error| {
                            cuml_query_error(NATIVE, "failed to transfer membership probabilities", error)
                        })?,
                    ),
                ]
            },
            RowMajorParams::Umap(p) => {
                let parameters = UmapParams {
                    n_components: p.n_components,
                    n_neighbors: p.n_neighbors,
                    n_epochs: p.n_epochs,
                    learning_rate: checked_f32(p.learning_rate, node_id, "learning_rate")?,
                    min_dist: checked_f32(p.min_dist, node_id, "min_dist")?,
                    spread: checked_f32(p.spread, node_id, "spread")?,
                    set_op_mix_ratio: checked_f32(p.set_op_mix_ratio, node_id, "set_op_mix_ratio")?,
                    local_connectivity: checked_f32(p.local_connectivity, node_id, "local_connectivity")?,
                    repulsion_strength: checked_f32(p.repulsion_strength, node_id, "repulsion_strength")?,
                    negative_sample_rate: p.negative_sample_rate,
                    init: match p.init {
                        MlUmapInit::Spectral => UmapInit::Spectral,
                        MlUmapInit::Random => UmapInit::Random,
                    },
                    random_state: p.random_state,
                    ..UmapParams::default()
                };
                let matrix = OwnedFloat32Matrix::from_packed(resources, packed)
                    .map_err(|error| cuml_query_error(NATIVE, "failed to adopt UMAP input", error))?;
                let embedding = UmapModel::fit(resources, matrix, parameters)
                    .map_err(|error| cuml_query_error(NATIVE, "failed to fit cuML UMAP", error))?
                    .into_embedding()
                    .into_column()
                    .map_err(|error| cuml_query_error(NATIVE, "failed to transfer UMAP embedding", error))?;
                resources
                    .runtime()
                    .unpack_dense_matrix(embedding, rows, p.n_components, DenseMatrixLayout::RowMajor)
                    .map_err(|error| cudf_query_error(NATIVE, "failed to unpack UMAP embedding", error))?
                    .into_iter()
                    .enumerate()
                    .map(|(index, series)| (format!("dim_{index}"), series))
                    .collect()
            },
        };
        context.check_cancellation(NATIVE)?;
        Ok(output)
    })();
    let output = match output {
        Ok(output) => {
            guard.finish(Some(rows as u64), Some(output_bytes));
            output
        },
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        execution_id,
        parent,
        RuntimeOperatorStageKind::MlAssemble,
        identity,
    );
    let result = (|| {
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        let output = assemble_ml_output(context, input, binding, output, TypeId::Int32)?;
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        Ok(output)
    })();
    match result {
        Ok(output) => {
            let bytes = match output.device_allocated_size_bytes() {
                Ok(bytes) => bytes,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            guard.finish(Some(rows as u64), Some(bytes));
            Ok(output)
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

/// cuDF columns and packed matrices index elements with `i32`, which bounds
/// both a forest batch's feature matrix and its prediction buffer.
const FOREST_BATCH_ELEMENTS: usize = i32::MAX as usize;

#[cfg(feature = "test-hooks")]
static FOREST_BATCH_ELEMENT_LIMIT: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "test-hooks")]
static FOREST_AFTER_IMPORT_HOOK: Mutex<Option<fn(&ExecutionContext)>> = Mutex::new(None);

/// Overrides the per-batch element bound of forest prediction; `None` restores it.
#[cfg(feature = "test-hooks")]
pub fn set_forest_batch_element_limit_for_tests(limit: Option<usize>) {
    FOREST_BATCH_ELEMENT_LIMIT.store(limit.unwrap_or(0), Ordering::SeqCst);
}

/// Runs `hook` after forest import and before the first prediction batch.
#[cfg(feature = "test-hooks")]
pub fn set_forest_after_import_hook_for_tests(hook: Option<fn(&ExecutionContext)>) {
    *FOREST_AFTER_IMPORT_HOOK.lock().unwrap_or_else(PoisonError::into_inner) = hook;
}

fn forest_batch_element_limit() -> usize {
    #[cfg(feature = "test-hooks")]
    {
        let limit = FOREST_BATCH_ELEMENT_LIMIT.load(Ordering::SeqCst);
        if limit != 0 {
            return limit;
        }
    }
    FOREST_BATCH_ELEMENTS
}

/// Rows per prediction batch so that neither the feature matrix nor the
/// prediction buffer exceeds `max_elements`; at least one row.
fn forest_rows_per_batch(rows: usize, features: usize, stride: usize, max_elements: usize) -> usize {
    (max_elements / features.max(stride).max(1)).min(rows).max(1)
}

fn ml_stage<T>(
    context: &ExecutionContext,
    kind: RuntimeOperatorStageKind,
    identity: RuntimeOperatorStageIdentity,
    run: impl FnOnce() -> Result<(T, Option<u64>, Option<u64>)>,
) -> Result<T> {
    let (observation, execution_id, parent) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(observation, execution_id, parent, kind, identity);
    match run() {
        Ok((value, rows, bytes)) => {
            guard.finish(rows, bytes);
            Ok(value)
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

fn forest_output_contract(kind: ForestOutputKind) -> MlForestOutputContract {
    match kind {
        ForestOutputKind::Score => MlForestOutputContract::Score,
        ForestOutputKind::ClassIndex { classes } => MlForestOutputContract::ClassIndex { classes },
        ForestOutputKind::Vector(width) => MlForestOutputContract::Vector(width),
    }
}

/// Planning proved the snapshot with the same importer; a difference here
/// means the checkpoint or the native library changed underneath the plan.
fn verify_forest_snapshot(model: &ForestInferenceModel, snapshot: &MlForestModelSnapshot) -> Result<()> {
    let info = model.info();
    let imported = MlForestModelFacts {
        num_features: info.num_features,
        num_trees: info.num_trees,
        native_output_stride: info.native_output_stride,
        precision: match info.precision {
            ForestPrecision::Single => MlForestPrecision::Single,
            ForestPrecision::Double => MlForestPrecision::Double,
        },
    };
    let output = forest_output_contract(model.output_kind());
    if imported == snapshot.facts() && output == snapshot.output() {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InternalInvariantViolation,
        NATIVE,
        "imported forest facts differ from the planned snapshot",
    )
    .with_fact("reason", "forest_snapshot_mismatch")
    .with_fact("model_digest", snapshot.digest())
    .with_fact("planned_facts", format!("{:?} output={}", snapshot.facts(), snapshot.output()))
    .with_fact("imported_facts", format!("{imported:?} output={output}")))
}

/// Consumes the Float32 matrix so its device copy is released before the
/// native call reads the Float64 copy.
fn widen_forest_batch(runtime: &Runtime, packed: PackedMatrix) -> Result<PackedMatrix> {
    runtime
        .widen_packed_matrix_to_float64(&packed)
        .map_err(|error| cudf_query_error(NATIVE, "failed to widen forest features to Float64", error))
}

fn execute_forest_predict(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    inputs: DenseRelationInputs<'_>,
    resources: &CumlResources,
    params: &MlForestPredictParams,
) -> Result<GpuDataFrame> {
    let algorithm = MlAlgorithm::ForestPredict.as_str();
    let binding = &params.input;
    let snapshot = &params.model;
    let input = relation_input_for(&ML_OPERATIONS, inputs, binding.input, node_id, "input")?;
    let rows = input.num_rows();
    let element = relation_element_type_for(
        &ML_OPERATIONS,
        plan,
        inputs,
        binding.input,
        &binding.features,
        node_id,
        algorithm,
        "input",
    )?;
    if element != FEATURE_ELEMENT {
        return Err(QueryError::new(QueryErrorCode::UnsupportedType, EXECUTE, "cuML requires Float32 features")
            .with_fact("node_id", node_id)
            .with_fact("reason", "unsupported_feature_type")
            .with_fact("element", element.name())
            .with_fact("supported_types", "Float32"));
    }
    let facts = snapshot.facts();
    let features = facts.num_features;
    let output = snapshot.output();
    let width = output.width();
    let digest = snapshot.digest().short_hex();
    // Identity stages report the nvForest version the cuML shim compiled against.
    let version = cuml::forest_inference_version()
        .map_err(|error| cuml_query_error(EXECUTE, "failed to read the nvForest version", error))?;
    let identity = RuntimeOperatorStageIdentity::new()
        .with_node_id(node_id.to_string())
        .with_ml_algorithm(algorithm)
        .with_execution_binding_id(resources.binding_id().get())
        .with_device_ordinal(resources.device().ordinal())
        .with_native_version(version.0, version.1, version.2)
        .with_matrix_layout("row_major")
        .with_vector_relation_role("input")
        .with_vector_dimension(features);
    let request = match output {
        MlForestOutputContract::ClassIndex { .. } => ForestOutput::ClassIndex,
        MlForestOutputContract::Score | MlForestOutputContract::Vector(_) => ForestOutput::Scores,
    };
    let mut model = ml_stage(
        context,
        RuntimeOperatorStageKind::MlNative,
        identity
            .clone()
            .with_strategy_label(format!("forest_import:{digest}:{}", facts.precision))
            .with_input_bytes(snapshot.checkpoint().len() as u64),
        || {
            context.check_cancellation(NATIVE)?;
            let model = ForestInferenceModel::import(resources, snapshot.checkpoint(), request)
                .map_err(|error| cuml_query_error(NATIVE, "failed to import forest model", error))?;
            verify_forest_snapshot(&model, snapshot)?;
            Ok((model, None, None))
        },
    )?;
    #[cfg(feature = "test-hooks")]
    if let Some(hook) = *FOREST_AFTER_IMPORT_HOOK.lock().unwrap_or_else(PoisonError::into_inner) {
        hook(context);
    }
    context.check_cancellation(NATIVE)?;
    let batch_rows = forest_rows_per_batch(rows, features, facts.native_output_stride, forest_batch_element_limit());
    let mut predictions: Vec<Vec<Series>> = (0..width).map(|_| Vec::new()).collect();
    for (batch_index, offset) in (0..rows).step_by(batch_rows).enumerate() {
        let len = batch_rows.min(rows - offset);
        let sliced = if len == rows {
            None
        } else {
            Some(GpuDataFrame::new(input.frame().on(Exec::RETAINED).slice(offset, len).map_err(|error| {
                cudf_query_error(EXECUTE, "failed to slice the forest input batch", error)
                    .with_fact("offset", offset)
                    .with_fact("rows", len)
            })?))
        };
        let batch = sliced.as_ref().unwrap_or(input);
        let (packed, dimension) = pack_dense_relation_for(
            &ML_OPERATIONS,
            context,
            resources.runtime(),
            plan,
            node_id,
            "input",
            algorithm,
            inputs.node_id(binding.input),
            &binding.features,
            batch,
            FEATURE_ELEMENT,
            DenseMatrixLayout::RowMajor,
            identity.clone().with_chunk_index(batch_index).with_input_rows(len as u64),
            RuntimeOperatorStageKind::MlPack,
        )?;
        if dimension != features {
            return Err(invalid(
                node_id,
                "forest_feature_count_mismatch",
                "forest feature count differs from the model",
            )
            .with_fact("expected", features)
            .with_fact("actual", dimension));
        }
        let packed = match facts.precision {
            MlForestPrecision::Single => packed,
            MlForestPrecision::Double => widen_forest_batch(resources.runtime(), packed)?,
        };
        let validated =
            packed.validate().map_err(|error| cudf_query_error(NATIVE, "invalid forest input matrix", error))?;
        let matrix = match facts.precision {
            MlForestPrecision::Single => DeviceMatrix::from_validated(&validated),
            MlForestPrecision::Double => DeviceMatrix::from_validated_float64(&validated),
        }
        .map_err(|error| cuml_query_error(NATIVE, "invalid forest input matrix", error))?;
        let input_bytes = match facts.precision {
            MlForestPrecision::Single => matrix_bytes_for::<f32>(&ML_OPERATIONS, len, features, node_id, "input")?,
            MlForestPrecision::Double => matrix_bytes_for::<f64>(&ML_OPERATIONS, len, features, node_id, "input")?,
        };
        // nvForest writes `native_output_stride` values per row in the model's
        // precision before the output contract selects `width` columns.
        let stride = facts.native_output_stride;
        let output_bytes = match facts.precision {
            MlForestPrecision::Single => matrix_bytes_for::<f32>(&ML_OPERATIONS, len, stride, node_id, "prediction")?,
            MlForestPrecision::Double => matrix_bytes_for::<f64>(&ML_OPERATIONS, len, stride, node_id, "prediction")?,
        };
        let identity = identity
            .clone()
            .with_chunk_index(batch_index)
            .with_strategy_label(format!("forest_predict:{digest}"))
            .with_input_rows(len as u64)
            .with_input_bytes(input_bytes);
        let series = ml_stage(context, RuntimeOperatorStageKind::MlNative, identity, || {
            context.check_cancellation(NATIVE)?;
            // into_series validates the output on the host, so the model and
            // input matrix outlive the kernel that reads them.
            let series = model
                .predict(resources, &matrix)
                .and_then(|prediction| prediction.into_series())
                .map_err(|error| cuml_query_error(NATIVE, "failed to predict with the forest model", error))?;
            context.check_cancellation(NATIVE)?;
            Ok((series, Some(len as u64), Some(output_bytes)))
        })?;
        if series.len() != width {
            return Err(QueryError::new(
                QueryErrorCode::CudfInvariantViolation,
                NATIVE,
                "forest prediction series count differs from the planned output",
            )
            .with_fact("reason", "invalid_native_output_contract")
            .with_fact("expected", width)
            .with_fact("actual", series.len()));
        }
        for (column, values) in predictions.iter_mut().zip(series) {
            column.push(values);
        }
    }
    // Closing here, before assembly, releases the device forest while the
    // attempt's allocation domain is still open.
    model.close().map_err(|error| cuml_query_error(NATIVE, "failed to close forest model", error))?;
    let prediction_type = match output {
        MlForestOutputContract::ClassIndex { .. } => TypeId::Int32,
        MlForestOutputContract::Score | MlForestOutputContract::Vector(_) => TypeId::Float32,
    };
    ml_stage(context, RuntimeOperatorStageKind::MlAssemble, identity.with_input_rows(rows as u64), || {
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        let columns = predictions
            .into_iter()
            .enumerate()
            .map(|(index, batches)| {
                let name = match output {
                    MlForestOutputContract::Vector(_) => format!("output_{index}"),
                    MlForestOutputContract::Score | MlForestOutputContract::ClassIndex { .. } => {
                        "prediction".to_owned()
                    },
                };
                let series = match <[Series; 1]>::try_from(batches) {
                    Ok([series]) => series,
                    Err(batches) => {
                        Series::concat(&batches.iter().collect::<Vec<_>>(), Exec::RETAINED).map_err(|error| {
                            cudf_query_error(ML_OPERATIONS.assemble, "failed to concatenate forest batches", error)
                        })?
                    },
                };
                Ok((name, series))
            })
            .collect::<Result<Vec<_>>>()?;
        let output = assemble_ml_output(context, input, binding, columns, prediction_type)?;
        context.check_cancellation(ML_OPERATIONS.assemble)?;
        let bytes = output.device_allocated_size_bytes()?;
        Ok((output, Some(rows as u64), Some(bytes)))
    })
}

#[cfg(test)]
mod tests {
    use super::forest_rows_per_batch;

    #[test]
    fn forest_batches_bound_both_the_feature_matrix_and_the_prediction_buffer() {
        let max = i32::MAX as usize;
        assert_eq!(forest_rows_per_batch(10, 4, 1, max), 10);
        assert_eq!(forest_rows_per_batch(1 << 30, 4, 1, max), max / 4);
        assert_eq!(forest_rows_per_batch(1 << 30, 4, 2538, max), max / 2538);
        assert_eq!(forest_rows_per_batch(10, 4, 3, 12), 3);
        assert_eq!(forest_rows_per_batch(10, 5, 1, 4), 1);
        assert_eq!(forest_rows_per_batch(0, 4, 1, max), 1);
    }
}
