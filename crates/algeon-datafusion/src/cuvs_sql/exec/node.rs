use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::{DataFusionError, Result},
    physical_expr::LexOrdering,
    physical_plan::ExecutionPlan,
};
use query_engine::plan::QueryPlan;

use super::{
    super::{
        CuvsOperation, CuvsValidatedOptions, CuvsVectorRelationBinding, DenseVectorBinding,
        error::{feature_disabled, invalid_call},
    },
    lowering::lower_cuvs_query_plan,
};
use crate::native_exec::{
    AlgeonNativeExecutionConfig,
    child_inputs::NativeChildBinding,
    relation_function::{
        RelationFunctionExec, RelationFunctionMessages, RelationFunctionOperations, RelationFunctionSpec,
        RelationFunctionText,
    },
};

static NEXT_CUVS_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct CuvsRelationFunctionSpec {
    operation: CuvsOperation,
    options: CuvsValidatedOptions,
}

pub type CuvsAlgoExec = RelationFunctionExec<CuvsRelationFunctionSpec>;

impl RelationFunctionExec<CuvsRelationFunctionSpec> {
    pub(crate) fn try_new(
        operation: CuvsOperation,
        options: CuvsValidatedOptions,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        input_orderings: Vec<Option<LexOrdering>>,
        output_schema: SchemaRef,
        native_execution_config: AlgeonNativeExecutionConfig,
    ) -> Result<Self> {
        let id = NEXT_CUVS_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        let input_names =
            (0..inputs.len()).map(|index| format!("__algeon_datafusion_cuvs_input_{id}_{index}")).collect();
        Self::try_new_with_names(
            CuvsRelationFunctionSpec { operation, options },
            inputs,
            output_schema,
            native_execution_config,
            input_names,
            input_orderings,
            format!("__algeon_datafusion_cuvs_result_{id}"),
        )
    }

    pub fn operation(&self) -> CuvsOperation {
        self.spec().operation
    }
}

impl RelationFunctionSpec for CuvsRelationFunctionSpec {
    const EXEC_NAME: &'static str = "CuvsAlgoExec";
    const HOST_OUTPUT_MEMORY_CONSUMER: &'static str = "CuvsAlgoExec[host-output-staging]";
    fn operation_name(&self) -> &'static str {
        self.operation.descriptor().sql_function_name
    }
    fn binding_count(&self) -> usize {
        self.options.bindings().len()
    }
    fn roles(&self) -> Vec<&'static str> {
        self.operation.descriptor().relation_roles.iter().map(|role| role.as_str()).collect()
    }
    fn required_projection(&self, index: usize, schema: SchemaRef) -> Result<Vec<usize>> {
        required_projection(self.operation, schema, &self.options.bindings()[index])
    }
    fn lower(&self, bindings: &[NativeChildBinding], schema: SchemaRef) -> Result<QueryPlan> {
        lower_cuvs_query_plan(self.operation, &self.options, bindings, schema)
    }
    fn feature_available(&self) -> bool {
        cfg!(feature = "cuvs")
    }
    fn feature_disabled(&self) -> DataFusionError {
        feature_disabled(self.operation)
    }
    fn invalid_child_count(&self) -> DataFusionError {
        invalid_call(
            self.operation,
            "physical_child_count_mismatch",
            "cuVS physical inputs do not match validated relation bindings",
        )
    }
    const TEXT: RelationFunctionText = RelationFunctionText {
        operations: RelationFunctionOperations {
            execute: "algeon_datafusion.cuvs_sql.execute",
            execute_gpu: "algeon_datafusion.cuvs_sql.execute_gpu",
            gpu_execution_pool: "algeon_datafusion.cuvs_sql.gpu_execution_pool",
            host_batch_cursor: "algeon_datafusion.cuvs_sql.host_batch_cursor",
            preflight: "algeon_datafusion.cuvs_sql.preflight",
            with_new_children: "algeon_datafusion.cuvs_sql.with_new_children",
        },
        messages: RelationFunctionMessages {
            preflight_failed: "native cuVS query-plan preflight failed",
            staged_inputs_missing: "cuVS GPU-resident producer inputs were not prepared before native context acquisition",
            staged_inputs_poisoned: "cuVS staged-input ownership was poisoned",
            staged_inputs_consumed: "cuVS staged inputs were already consumed",
            execution_failed: "native cuVS vector query execution failed",
            gpu_partition: "cuVS GPU-resident producer exposes exactly one output partition",
            unexpected_host_result: "native cuVS producer materialized an unexpected host result",
            child_count: "cuVS execution received an unexpected child count",
            host_partition: "cuVS execution exposes exactly one output partition",
            gpu_pool_closed: "query service closed before cuVS acquired its GPU execution pool",
            deferred_result: "deferred cuVS host execution did not produce a GPU-resident cursor input",
            cursor_failed: "failed to bind the deferred cuVS host-output cursor",
        },
    };
}

fn required_projection(
    operation: CuvsOperation,
    schema: SchemaRef,
    binding: &CuvsVectorRelationBinding,
) -> Result<Vec<usize>> {
    let mut names = Vec::new();
    names.push(binding.id());
    match binding.vector() {
        DenseVectorBinding::Columns(columns) => names.extend(columns.iter().map(String::as_str)),
        DenseVectorBinding::VectorColumn(column) => names.push(column),
    }
    let mut field_indexes = HashMap::with_capacity(schema.fields().len());
    for (index, field) in schema.fields().iter().enumerate() {
        field_indexes.entry(field.name().as_str()).or_insert(index);
    }
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert(*name))
        .map(|name| {
            field_indexes.get(name).copied().ok_or_else(|| {
                invalid_call(
                    operation,
                    "physical_projection_field_missing",
                    format!("cuVS physical input has no field named '{name}'"),
                )
            })
        })
        .collect()
}
