use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions, coop::CooperativeExec, empty::EmptyExec,
};
use query_planning::{
    expr::Expr,
    plan::{ProjectionOutput, QueryPlanBuilder},
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

use super::SharedResidentResults;
use crate::{
    cugraph_sql::{
        CugraphAlgoExec, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphGraphOptions,
        CugraphLeidenParams, CugraphSimilarityParams, GpuIslandHostBoundaryConfig, result_schema_for_params,
    },
    native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig, NativeOutput, admit_for_test, gpu_resident_producer},
};

fn native_input(columns: &[&str], name: &str) -> Arc<dyn ExecutionPlan> {
    let schema =
        Arc::new(Schema::new(columns.iter().map(|name| Field::new(*name, DataType::Int64, false)).collect::<Vec<_>>()));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(format!("{name}.parquet")).with_row_group_rows(vec![3])],
        schema,
        columns.iter().map(|name| (*name).to_string()).collect(),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source).unwrap().clone_with_gpu_resident_sink();
    let config = AlgeonNativeExecutionConfig::default();
    Arc::new(
        AlgeonNativeExec::try_new(
            admit_for_test(&plan, &config).unwrap(),
            Vec::new(),
            NativeOutput::GpuResident { input_name: name.to_string() },
            config,
            None,
        )
        .unwrap(),
    )
}

fn algorithm(
    name: &str,
    params: CugraphAlgorithmParams,
    limit: Option<usize>,
    host_input: bool,
) -> Arc<dyn ExecutionPlan> {
    let kind = match &params {
        CugraphAlgorithmParams::Similarity(_) => CugraphAlgorithm::Jaccard,
        CugraphAlgorithmParams::Leiden(_) => CugraphAlgorithm::Leiden,
        _ => unreachable!(),
    };
    let edges = native_input(&["src", "dst"], "edges");
    let edges = if host_input { Arc::new(EmptyExec::new(edges.schema())) as Arc<dyn ExecutionPlan> } else { edges };
    let sides = if kind.is_similarity() { vec![native_input(&["first", "second"], "pairs")] } else { Vec::new() };
    let spec = CugraphAlgorithmSpec {
        algorithm: kind,
        has_weight: false,
        has_edge_id: false,
        graph_options: CugraphGraphOptions { directed: false, ..CugraphGraphOptions::default() },
        output_schema: result_schema_for_params(kind, &params),
        params,
    };
    Arc::new(
        CugraphAlgoExec::try_new_with_gpu_result_input_name(
            spec,
            edges,
            sides,
            None,
            limit,
            AlgeonNativeExecutionConfig::default(),
            GpuIslandHostBoundaryConfig::default(),
            name.to_string(),
        )
        .unwrap(),
    )
}

#[test]
fn repeated_invocations_preserve_output_pushdown_and_independent_calls() {
    let full =
        algorithm("similar", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
    let limited =
        algorithm("similar", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), Some(1), false);
    let independent =
        algorithm("other", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
    let wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(Arc::clone(&full)));
    let results = SharedResidentResults::for_gpu_inputs(&[
        wrapped,
        Arc::clone(&full),
        Arc::clone(&limited),
        Arc::clone(&limited),
        Arc::clone(&independent),
    ])
    .unwrap();
    let full_identity = results.reusable_identity(full.as_ref()).unwrap().unwrap();
    let limited_identity = results.reusable_identity(limited.as_ref()).unwrap().unwrap();
    assert!(full_identity != limited_identity, "LIMIT variants must retain separate outputs");
    assert_eq!(results.reusable_consumers[&full_identity], 2, "cooperative wrappers do not add a reading occurrence");
    assert_eq!(results.reusable_consumers[&limited_identity], 2);
    assert!(results.reusable_identity(independent.as_ref()).unwrap().is_none());
}

#[test]
fn replaced_relation_inputs_do_not_share_a_result() {
    let original =
        algorithm("similar", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
    let changed = Arc::clone(&original)
        .replace_children(
            vec![native_input(&["src", "dst"], "changed_edges"), native_input(&["first", "second"], "pairs")],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let results = SharedResidentResults::for_gpu_inputs(&[
        Arc::clone(&original),
        Arc::clone(&original),
        Arc::clone(&changed),
        Arc::clone(&changed),
    ])
    .unwrap();
    let original = results.reusable_identity(original.as_ref()).unwrap().unwrap();
    let changed = results.reusable_identity(changed.as_ref()).unwrap().unwrap();
    assert!(original != changed, "same invocation and schema with different input contents must remain separate");
}

fn similarity_edges(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let input_name = gpu_resident_producer(input.as_ref()).unwrap().gpu_result_input_name().to_string();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, input.schema())));
    let projection = builder
        .add_projection(
            source,
            vec![
                ProjectionOutput::new(Expr::column("first"), Field::new("src", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("second"), Field::new("dst", DataType::Int64, false)),
            ],
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap().clone_with_gpu_resident_sink();
    let config = AlgeonNativeExecutionConfig::default();
    Arc::new(
        AlgeonNativeExec::try_new(
            admit_for_test(&plan, &config).unwrap(),
            vec![input],
            NativeOutput::GpuResident { input_name: "nested_edges".to_string() },
            config,
            None,
        )
        .unwrap(),
    )
}

#[test]
fn nested_function_input_contents_are_part_of_reuse_identity() {
    let inner = algorithm("inner", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
    let changed_inner = Arc::clone(&inner)
        .replace_children(
            vec![native_input(&["src", "dst"], "changed_edges"), native_input(&["first", "second"], "pairs")],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let outer = algorithm("outer", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
    let original = Arc::clone(&outer)
        .replace_children(
            vec![similarity_edges(inner), native_input(&["first", "second"], "pairs")],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let changed = outer
        .replace_children(
            vec![similarity_edges(changed_inner), native_input(&["first", "second"], "pairs")],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let results = SharedResidentResults::for_gpu_inputs(&[
        Arc::clone(&original),
        Arc::clone(&original),
        Arc::clone(&changed),
        Arc::clone(&changed),
    ])
    .unwrap();
    let original = results.reusable_identity(original.as_ref()).unwrap().unwrap();
    let changed = results.reusable_identity(changed.as_ref()).unwrap().unwrap();
    assert!(original != changed, "a nested function's changed input must distinguish the outer result");
}

#[test]
fn reused_functions_register_only_executed_reads_of_shared_inputs() {
    let original =
        algorithm("similar", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
    let edge = native_input(&["src", "dst"], "edges");
    let mut shared = edge.downcast_ref::<AlgeonNativeExec>().unwrap().clone();
    shared.shared_consumer_count = 3;
    let graph = Arc::clone(&original)
        .replace_children(
            vec![Arc::new(shared), native_input(&["first", "second"], "pairs")],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let results =
        SharedResidentResults::for_gpu_inputs(&[Arc::clone(&graph), Arc::clone(&graph), Arc::clone(&graph)]).unwrap();
    assert_eq!(results.shared_consumers["edges"], 1, "reused Jaccard occurrences skip their shared edge input");
}

#[test]
fn nondeterministic_functions_and_host_staging_exclude_reuse() {
    let leiden = algorithm("worlds", CugraphAlgorithmParams::Leiden(CugraphLeidenParams::default()), None, false);
    let host = algorithm("host", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, true);
    let results = SharedResidentResults::for_gpu_inputs(&[
        Arc::clone(&leiden),
        Arc::clone(&leiden),
        Arc::clone(&host),
        Arc::clone(&host),
    ])
    .unwrap();
    for plan in [leiden, host] {
        let producer = gpu_resident_producer(plan.as_ref()).unwrap();
        assert!(!producer.gpu_result_is_repeatable());
        assert!(results.reusable_identity(plan.as_ref()).unwrap().is_none());
    }
}
