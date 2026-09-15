use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::{DataFusionError, Result},
    physical_plan::ExecutionPlan,
};
use query_engine::plan::QueryPlan;

use super::{
    super::{
        CumlOperation, CumlRelationBinding, CumlValidatedOptions, DenseVectorBinding,
        error::{feature_disabled, invalid_call},
    },
    lowering::lower_cuml_query_plan,
};
use crate::native_exec::{
    AlgeonNativeExecutionConfig, GpuResidentProducer,
    child_inputs::NativeChildBinding,
    register_gpu_resident_producer_discovery,
    relation_function::{
        RelationFunctionExec, RelationFunctionMessages, RelationFunctionOperations, RelationFunctionSink,
        RelationFunctionSpec, RelationFunctionText,
    },
};

static NEXT_CUML_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct CumlRelationFunctionSpec {
    operation: CumlOperation,
    options: CumlValidatedOptions,
}

pub type CumlAlgoExec = RelationFunctionExec<CumlRelationFunctionSpec>;

impl RelationFunctionExec<CumlRelationFunctionSpec> {
    pub(crate) fn try_new(
        operation: CumlOperation,
        options: CumlValidatedOptions,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        output_schema: SchemaRef,
        native_execution_config: AlgeonNativeExecutionConfig,
    ) -> Result<Self> {
        register_cuml_execution_spi();
        let id = NEXT_CUML_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        let input_names =
            (0..inputs.len()).map(|index| format!("__algeon_datafusion_cuml_input_{id}_{index}")).collect();
        Self::try_new_with_names(
            CumlRelationFunctionSpec { operation, options },
            inputs,
            output_schema,
            native_execution_config,
            input_names,
            format!("__algeon_datafusion_cuml_result_{id}"),
        )
    }

    pub fn operation(&self) -> CumlOperation {
        self.spec().operation
    }
}

fn cuml_gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    plan.downcast_ref::<CumlAlgoExec>().map(|producer| producer as &dyn GpuResidentProducer)
}

fn register_cuml_execution_spi() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| register_gpu_resident_producer_discovery(cuml_gpu_resident_producer));
}

impl RelationFunctionSpec for CumlRelationFunctionSpec {
    const EXEC_NAME: &'static str = "CumlAlgoExec";
    const HOST_OUTPUT_MEMORY_CONSUMER: &'static str = "CumlAlgoExec[host-output-staging]";
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
    fn lower(
        &self,
        bindings: &[NativeChildBinding],
        schema: SchemaRef,
        sink: RelationFunctionSink,
    ) -> Result<QueryPlan> {
        lower_cuml_query_plan(self.operation, &self.options, bindings, schema, sink)
    }
    fn feature_available(&self) -> bool {
        cfg!(feature = "cuml")
    }
    fn feature_disabled(&self) -> DataFusionError {
        feature_disabled(self.operation)
    }
    fn invalid_child_count(&self) -> DataFusionError {
        invalid_call(
            self.operation,
            "physical_child_count_mismatch",
            "cuML physical inputs do not match validated relation bindings",
        )
    }
    const TEXT: RelationFunctionText = RelationFunctionText {
        operations: RelationFunctionOperations {
            execute: "algeon_datafusion.cuml_sql.execute",
            execute_gpu: "algeon_datafusion.cuml_sql.execute_gpu",
            gpu_execution_pool: "algeon_datafusion.cuml_sql.gpu_execution_pool",
            host_batch_cursor: "algeon_datafusion.cuml_sql.host_batch_cursor",
            preflight: "algeon_datafusion.cuml_sql.preflight",
            with_new_children: "algeon_datafusion.cuml_sql.with_new_children",
        },
        messages: RelationFunctionMessages {
            preflight_failed: "native cuML query-plan preflight failed",
            staged_inputs_missing: "cuML GPU-resident producer inputs were not prepared before native context acquisition",
            staged_inputs_poisoned: "cuML staged-input ownership was poisoned",
            staged_inputs_consumed: "cuML staged inputs were already consumed",
            execution_failed: "native cuML query execution failed",
            gpu_partition: "cuML GPU-resident producer exposes exactly one output partition",
            unexpected_host_result: "native cuML producer materialized an unexpected host result",
            child_count: "cuML execution received an unexpected child count",
            host_partition: "cuML execution exposes exactly one output partition",
            gpu_pool_closed: "query service closed before cuML acquired its GPU execution pool",
            deferred_result: "deferred cuML host execution did not produce a GPU-resident cursor input",
            cursor_failed: "failed to bind the deferred cuML host-output cursor",
        },
    };
}

fn required_projection(
    operation: CumlOperation,
    schema: SchemaRef,
    binding: &CumlRelationBinding,
) -> Result<Vec<usize>> {
    let mut names = vec![binding.id()];
    match binding.features() {
        DenseVectorBinding::Columns(columns) => names.extend(columns.iter().map(String::as_str)),
        DenseVectorBinding::VectorColumn(column) => names.push(column),
    }
    if let Some(label) = binding.label() {
        names.push(label);
    }
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert((*name).to_string()))
        .map(|name| {
            schema.index_of(name).map_err(|_| {
                invalid_call(
                    operation,
                    "physical_projection_field_missing",
                    format!("cuML physical input has no field named '{name}'"),
                )
            })
        })
        .collect()
}
