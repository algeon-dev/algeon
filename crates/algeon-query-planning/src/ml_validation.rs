//! cuML node contract validation shared by preflight and capability analysis.

use std::collections::BTreeSet;

use arrow::datatypes::DataType;

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{
        CumlOperation, GpuOptionKind, MlAlgorithmNode, MlAlgorithmParams, NodeId, QueryPlan, VectorElementType,
        ml_embedding_output_schema, ml_hdbscan_output_schema, ml_predictions_output_schema, vector_pca_output_schema,
    },
    vector_validation::validate_dense_relation_binding,
};

fn invalid(message: &'static str, reason: &'static str, node_id: NodeId) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "query_engine.ml_algorithm.validate", message)
        .with_fact("reason", reason)
        .with_fact("node_id", node_id)
}

pub fn validate_ml_algorithm_node_contract(ml: &MlAlgorithmNode, plan: &QueryPlan, node_id: NodeId) -> Result<()> {
    let columns = plan
        .nodes()
        .iter()
        .take(node_id.index())
        .map(|node| node.output_schema().fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();
    validate_ml_algorithm_node(ml, plan, &columns, node_id)
}

pub fn validate_ml_algorithm_node(
    ml: &MlAlgorithmNode,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    if !ml.spec.params.supports_algorithm(ml.spec.algorithm) {
        return Err(invalid("ML algorithm and parameters differ", "algorithm_parameter_mismatch", node_id));
    }
    let expected_inputs = ml.spec.params.relation_bindings().map(|binding| binding.input).collect::<Vec<_>>();
    if ml.inputs != expected_inputs {
        return Err(invalid("ML node inputs must match relation bindings", "relation_input_mismatch", node_id));
    }
    let input = ml.spec.params.relation_bindings().next().expect("ML input");
    let (operation, integers, numbers, n_components) = match &ml.spec.params {
        MlAlgorithmParams::Pca(p) => (
            CumlOperation::Pca,
            vec![("n_components", p.n_components as u64), ("n_iterations", p.n_iterations as u64)],
            vec![("tol", p.tolerance)],
            Some(p.n_components),
        ),
        MlAlgorithmParams::Tsvd(p) => (
            CumlOperation::Tsvd,
            vec![("n_components", p.n_components as u64), ("n_iterations", p.n_iterations as u64)],
            vec![("tol", p.tolerance)],
            Some(p.n_components),
        ),
        MlAlgorithmParams::Hdbscan(p) => (
            CumlOperation::Hdbscan,
            vec![
                ("min_samples", p.min_samples as u64),
                ("min_cluster_size", p.min_cluster_size as u64),
                ("max_cluster_size", p.max_cluster_size as u64),
            ],
            vec![("cluster_selection_epsilon", p.cluster_selection_epsilon), ("alpha", p.alpha)],
            None,
        ),
        MlAlgorithmParams::RandomForestClassifier(p) | MlAlgorithmParams::RandomForestRegressor(p) => (
            if matches!(&ml.spec.params, MlAlgorithmParams::RandomForestClassifier(_)) {
                CumlOperation::RandomForestClassifier
            } else {
                CumlOperation::RandomForestRegressor
            },
            vec![
                ("n_trees", p.n_trees as u64),
                ("max_depth", p.max_depth as u64),
                ("max_n_bins", p.max_n_bins as u64),
                ("min_samples_leaf", p.min_samples_leaf as u64),
                ("min_samples_split", p.min_samples_split as u64),
                ("max_batch_size", p.max_batch_size as u64),
                ("seed", p.seed),
            ]
            .into_iter()
            .chain(p.max_leaves.map(|value| ("max_leaves", value as u64)))
            .collect(),
            vec![
                ("max_features", p.max_features),
                ("max_samples", p.max_samples),
                ("min_impurity_decrease", p.min_impurity_decrease),
            ],
            None,
        ),
        MlAlgorithmParams::LogisticRegression(p) => (
            CumlOperation::LogisticRegression,
            vec![
                ("max_iter", p.max_iter as u64),
                ("linesearch_max_iter", p.linesearch_max_iter as u64),
                ("lbfgs_memory", p.lbfgs_memory as u64),
            ],
            vec![
                ("penalty_l1", p.penalty_l1),
                ("penalty_l2", p.penalty_l2),
                ("grad_tol", p.grad_tol),
                ("change_tol", p.change_tol),
            ],
            None,
        ),
        MlAlgorithmParams::Umap(p) => (
            CumlOperation::Umap,
            vec![
                ("n_components", p.n_components as u64),
                ("n_neighbors", p.n_neighbors as u64),
                ("n_epochs", p.n_epochs as u64),
                ("negative_sample_rate", p.negative_sample_rate as u64),
                ("random_state", p.random_state),
            ],
            vec![
                ("learning_rate", p.learning_rate),
                ("min_dist", p.min_dist),
                ("spread", p.spread),
                ("set_op_mix_ratio", p.set_op_mix_ratio),
                ("local_connectivity", p.local_connectivity),
                ("repulsion_strength", p.repulsion_strength),
            ],
            Some(p.n_components),
        ),
    };
    for (option, value) in integers {
        let Some(GpuOptionKind::Integer { minimum, maximum, .. }) =
            operation.descriptor().option(option).map(|entry| entry.kind)
        else {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.ml_algorithm.validate",
                "missing cuML integer registry contract",
            ));
        };
        if !(minimum..=maximum).contains(&value) {
            return Err(invalid("ML parameter outside registry range", "parameter_exceeds_registry_maximum", node_id)
                .with_fact("option", option)
                .with_fact("value", value)
                .with_fact("maximum", maximum));
        }
    }
    for (option, value) in numbers {
        let valid = match operation.descriptor().option(option).map(|entry| entry.kind) {
            Some(GpuOptionKind::NonnegativeNumber { .. }) => value.is_finite() && value >= 0.0,
            Some(GpuOptionKind::PositiveNumber { .. }) => value.is_finite() && value > 0.0,
            Some(GpuOptionKind::UnitInterval { .. }) => value.is_finite() && (0.0..=1.0).contains(&value),
            Some(GpuOptionKind::Fraction { .. }) => value.is_finite() && value > 0.0 && value <= 1.0,
            _ => false,
        };
        if !valid {
            return Err(invalid("ML numeric option is outside its registry range", "invalid_numeric_option", node_id)
                .with_fact("option", option));
        }
    }
    if let MlAlgorithmParams::Umap(p) = &ml.spec.params
        && p.min_dist > p.spread
    {
        return Err(invalid("min_dist must not exceed spread", "invalid_numeric_option", node_id)
            .with_fact("option", "min_dist,spread"));
    }
    let supervised = matches!(
        operation,
        CumlOperation::LogisticRegression
            | CumlOperation::RandomForestClassifier
            | CumlOperation::RandomForestRegressor
    );
    if !supervised && input.label.is_some() {
        return Err(invalid("unsupervised ML input must not bind a label", "unexpected_label_binding", node_id));
    }
    let relation = validate_dense_relation_binding(
        (&input.features, &input.row_id),
        input.input,
        node_id,
        if supervised { "training" } else { "input" },
        plan,
        available_columns_by_node,
        Some("Float32"),
        |relation| {
            if relation.element_type != VectorElementType::Float32 {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "query_engine.ml_algorithm.validate",
                    "cuML requires Float32 features",
                )
                .with_fact("reason", "unsupported_feature_type")
                .with_fact("node_id", node_id)
                .with_fact("data_type", relation.element_type.as_str())
                .with_fact("supported_types", "Float32"));
            }
            Ok(())
        },
    )?;
    if let Some(dimension) = relation.dimension {
        if dimension < 2 && matches!(operation, CumlOperation::Pca | CumlOperation::Tsvd) {
            return Err(invalid(
                "ML input needs at least two feature dimensions",
                "insufficient_feature_dimension",
                node_id,
            )
            .with_fact("feature_count", dimension));
        }
        if n_components.is_some_and(|count| count > dimension)
            && matches!(operation, CumlOperation::Pca | CumlOperation::Tsvd)
        {
            return Err(invalid("ML components exceed input dimension", "invalid_component_count", node_id)
                .with_fact("n_components", n_components.expect("decomposition components"))
                .with_fact("feature_count", dimension));
        }
    }
    let expected_schema = if supervised {
        let (training, predict_binding) = match &ml.spec.params {
            MlAlgorithmParams::LogisticRegression(p) => (&p.training, &p.predict),
            MlAlgorithmParams::RandomForestClassifier(p) | MlAlgorithmParams::RandomForestRegressor(p) => {
                (&p.training, &p.predict)
            },
            _ => unreachable!("supervised operation"),
        };
        if let MlAlgorithmParams::RandomForestClassifier(p) | MlAlgorithmParams::RandomForestRegressor(p) =
            &ml.spec.params
            && (operation == CumlOperation::RandomForestClassifier) != p.criterion.is_some()
        {
            return Err(invalid("RF criterion does not match forest family", "invalid_split_criterion", node_id));
        }
        let label =
            training.label.as_ref().ok_or_else(|| invalid("training label is required", "missing_label", node_id))?;
        if label != "label" {
            return Err(invalid("training label must use the canonical name", "missing_label", node_id));
        }
        let field = plan
            .node(training.input)
            .expect("validated input")
            .output_schema()
            .field_with_name(label)
            .map_err(|_| invalid("training label column is missing", "missing_label", node_id))?;
        let label_type =
            if operation == CumlOperation::RandomForestRegressor { DataType::Float32 } else { DataType::Int32 };
        if field.data_type() != &label_type {
            return Err(invalid("training label has unsupported type", "unsupported_type", node_id)
                .with_fact("field", label)
                .with_fact("actual_type", format!("{:?}", field.data_type())));
        }
        if field.is_nullable() {
            return Err(invalid("training label must not be nullable", "nullable_label", node_id));
        }
        if predict_binding.label.is_some()
            || plan
                .node(predict_binding.input)
                .expect("validated input")
                .output_schema()
                .field_with_name("label")
                .is_ok()
        {
            return Err(invalid("predict relation must not contain label", "predict_relation_has_label", node_id));
        }
        let predict = validate_dense_relation_binding(
            (&predict_binding.features, &predict_binding.row_id),
            predict_binding.input,
            node_id,
            "predict",
            plan,
            available_columns_by_node,
            Some("Float32"),
            |relation| {
                if relation.element_type != VectorElementType::Float32 {
                    return Err(invalid("cuML requires Float32 features", "unsupported_feature_type", node_id));
                }
                Ok(())
            },
        )?;
        if let (Some(training), Some(predict_dimension)) = (relation.dimension, predict.dimension)
            && training != predict_dimension
        {
            return Err(invalid("training and predict dimensions differ", "feature_dimension_mismatch", node_id)
                .with_fact("training_dimension", training)
                .with_fact("predict_dimension", predict_dimension));
        }
        ml_predictions_output_schema(
            predict.id_type,
            if operation == CumlOperation::RandomForestRegressor { DataType::Float32 } else { DataType::Int32 },
        )
    } else {
        match n_components {
            Some(count) if operation == CumlOperation::Umap => ml_embedding_output_schema(relation.id_type, count),
            Some(count) => vector_pca_output_schema(relation.id_type, count),
            None => ml_hdbscan_output_schema(relation.id_type),
        }
    };
    if ml.output_schema().as_ref() != expected_schema.as_ref() {
        return Err(invalid(
            "ML output schema must match the components contract",
            "unexpected_output_schema",
            node_id,
        ));
    }
    Ok(())
}
