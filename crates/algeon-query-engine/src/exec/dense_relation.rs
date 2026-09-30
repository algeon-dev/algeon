//! Shared dense-relation device packing for GPU UDTF executors.

use std::{collections::BTreeMap, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef};
use cudf::{DenseMatrixLayout, PackedMatrix, Runtime, Series, TypeId, frame::dataframe::GpuRecordBatch};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    exec::{cudf_query_error, invalid_node_reference},
    observability::{
        RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
    },
    plan::{DenseVectorBinding, NodeId, QueryPlan, VectorRowIdBinding},
    runtime::{ExecutionContext, resident::GpuDataFrame},
};

pub(super) struct DenseRelationOperations {
    pub(super) pack: &'static str,
    pub(super) before_pack: &'static str,
    pub(super) before_uniform_list_pack: &'static str,
    pub(super) execute: &'static str,
    pub(super) assemble: &'static str,
}

pub(super) const VECTOR_OPERATIONS: DenseRelationOperations = DenseRelationOperations {
    pack: "query_engine.vector_algorithm.pack",
    before_pack: "query_engine.vector_algorithm.before_pack",
    before_uniform_list_pack: "query_engine.vector_algorithm.before_uniform_list_pack",
    execute: "query_engine.vector_algorithm.execute",
    assemble: "query_engine.vector_algorithm.assemble",
};
#[cfg(feature = "cuml")]
pub(super) const ML_OPERATIONS: DenseRelationOperations = DenseRelationOperations {
    pack: "query_engine.ml_algorithm.pack",
    before_pack: "query_engine.ml_algorithm.before_pack",
    before_uniform_list_pack: "query_engine.ml_algorithm.before_uniform_list_pack",
    execute: "query_engine.ml_algorithm.execute",
    assemble: "query_engine.ml_algorithm.assemble",
};
pub(super) fn relation_row_ordinals_for(
    ops: &DenseRelationOperations,
    runtime: &Runtime,
    rows: usize,
    message: &'static str,
) -> Result<Series> {
    runtime.sequence(0_u64, 1_u64, rows).map_err(|error| cudf_query_error(ops.assemble, message, error))
}

pub(super) fn relation_id_series_for(
    ops: &DenseRelationOperations,
    binding: &VectorRowIdBinding,
    input: &GpuDataFrame,
    relation_role: &'static str,
) -> Result<Series> {
    let VectorRowIdBinding::Column(column) = binding else {
        return Err(QueryError::new(
            QueryErrorCode::Internal,
            ops.assemble,
            "validated vector result relation is missing its logical row-id column",
        )
        .with_fact("relation_role", relation_role)
        .with_fact("reason", "missing_logical_row_id_column"));
    };
    input.frame().series(column).map_err(|error| {
        cudf_query_error(ops.assemble, "failed to access vector relation logical ids", error)
            .with_fact("relation_role", relation_role)
            .with_fact("column", column)
    })
}

pub(super) fn require_series_contract_for(
    ops: &DenseRelationOperations,
    series: &Series,
    expected_type: TypeId,
    expected_rows: usize,
    output: &str,
) -> Result<()> {
    let actual_type = series.dtype().map_err(|error| {
        cudf_query_error(ops.assemble, "failed to inspect vector native output type", error).with_fact("output", output)
    })?;
    if actual_type.id() != expected_type || series.len() != expected_rows || series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvariantViolation,
            ops.assemble,
            "vector native output does not match its device assembly contract",
        )
        .with_fact("output", output)
        .with_fact("expected_type", expected_type.name())
        .with_fact("actual_type", actual_type.id().name())
        .with_fact("expected_rows", expected_rows)
        .with_fact("actual_rows", series.len())
        .with_fact("null_count", series.null_count())
        .with_fact("reason", "invalid_native_output_contract"));
    }
    Ok(())
}

#[cfg(feature = "cuvs")]
pub(super) fn require_libcudf_row_count(rows: usize, algorithm: &'static str) -> Result<()> {
    if rows <= i32::MAX as usize {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::CudfInvalidInput,
        VECTOR_OPERATIONS.assemble,
        "vector result row count exceeds libcudf size_type range",
    )
    .with_fact("algorithm", algorithm)
    .with_fact("rows", rows)
    .with_fact("maximum_rows", i32::MAX)
    .with_fact("reason", "row_count_exceeds_libcudf_size_type"))
}

#[expect(clippy::too_many_arguments, reason = "the relation binding and family identity identify the failure")]
pub(super) fn relation_element_type_for(
    ops: &DenseRelationOperations,
    plan: &QueryPlan,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    input_id: NodeId,
    binding: &DenseVectorBinding,
    node_id: NodeId,
    algorithm: &str,
    role: &'static str,
) -> Result<TypeId> {
    let schema = plan
        .node(input_id)
        .ok_or_else(|| invalid_node_reference(input_id, "vector relation references a missing input"))?
        .output_schema();
    let column = match binding {
        DenseVectorBinding::Columns(columns) => {
            let input = relation_input_for(ops, inputs, input_id, node_id, role)?;
            return input
                .frame()
                .series(&columns[0])
                .and_then(|series| series.dtype())
                .map(|dtype| dtype.id())
                .map_err(|error| {
                    cudf_query_error(ops.pack, "failed to inspect evaluated vector element type", error)
                        .with_fact("relation_role", role)
                });
        },
        DenseVectorBinding::VectorColumn(column) => column,
    };
    let field = schema.field_with_name(column).map_err(|_| {
        QueryError::new(QueryErrorCode::CudfInvalidInput, ops.execute, "vector column is absent from input schema")
            .with_fact("node_id", node_id)
            .with_fact("algorithm", algorithm)
            .with_fact("reason", "missing_feature_column")
    })?;
    let element = match field.data_type() {
        DataType::FixedSizeList(element, _) | DataType::List(element) | DataType::LargeList(element) => {
            element.data_type()
        },
        other => other,
    };
    match element {
        DataType::Float32 => Ok(TypeId::Float32),
        DataType::Float64 => Ok(TypeId::Float64),
        DataType::Int8 => Ok(TypeId::Int8),
        DataType::UInt8 => Ok(TypeId::UInt8),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            ops.pack,
            "vector element type is not supported by this operation",
        )
        .with_fact("node_id", node_id)
        .with_fact("algorithm", algorithm)
        .with_fact("relation_role", role)
        .with_fact("reason", "unsupported_vector_element_type")
        .with_fact("arrow_type", format!("{element:?}"))),
    }
}

pub(super) fn matrix_layout_name(layout: DenseMatrixLayout) -> &'static str {
    match layout {
        DenseMatrixLayout::RowMajor => "row_major",
        DenseMatrixLayout::ColumnMajor => "column_major",
    }
}

#[expect(clippy::too_many_arguments, reason = "the caller supplies its relation binding and observation identity")]
pub(super) fn pack_dense_relation_for(
    ops: &DenseRelationOperations,
    context: &ExecutionContext,
    runtime: &Runtime,
    plan: &QueryPlan,
    node_id: NodeId,
    role: &'static str,
    algorithm: &'static str,
    input_id: NodeId,
    binding: &DenseVectorBinding,
    input: &GpuDataFrame,
    element: TypeId,
    layout: DenseMatrixLayout,
    identity: RuntimeOperatorStageIdentity,
    stage: RuntimeOperatorStageKind,
) -> Result<(PackedMatrix, usize)> {
    let schema = plan
        .node(input_id)
        .ok_or_else(|| invalid_node_reference(input_id, "vector relation references a missing input"))?
        .output_schema();
    let pack_operation = ops.pack;
    let column = match binding {
        DenseVectorBinding::Columns(_) => None,
        DenseVectorBinding::VectorColumn(column) => Some(column.as_str()),
    };
    let uniform = column.is_some_and(|name| {
        matches!(
            schema.field_with_name(name).map(|field| field.data_type()),
            Ok(DataType::List(_) | DataType::LargeList(_))
        )
    });
    if uniform {
        context.check_cancellation(ops.before_pack)?;
        let column = column.expect("uniform binding is a vector column");
        let batch = GpuRecordBatch::new(input.frame().clone(), Arc::clone(schema)).map_err(|error| {
            cudf_query_error(
                pack_operation,
                "failed to bind the vector relation to its variable-width List schema",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
        })?;
        let field_index = field_index(schema, column, node_id, role, pack_operation)?;
        let validated = batch.try_prepare_uniform_list_export(field_index).map_err(|error| {
            cudf_query_error(pack_operation, "failed to prove uniform List vector offsets and finite values", error)
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
        })?;
        require_element(validated.element(), element, node_id, algorithm, role, pack_operation)?;
        let dimension = validated.dimension();
        let rows = validated.rows();
        let bytes = packed_bytes(rows, dimension, element, node_id, role, ops)?;
        let guard = pack_guard(context, stage, identity, role, dimension, layout, rows, input)?;
        if let Err(error) = context.check_cancellation(ops.before_uniform_list_pack) {
            guard.fail(&error);
            return Err(error);
        }
        let result = validated.pack_active_child_into_runtime(runtime, layout).map_err(|error| {
            cudf_query_error(
                pack_operation,
                "failed to pack the validated uniform List vector child into an owned matrix",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
        });
        return finish_pack(guard, result, rows, dimension, bytes);
    }
    let dimension = relation_dimension_for(binding, schema, node_id, role, pack_operation)?;
    let rows = input.num_rows();
    let bytes = packed_bytes(rows, dimension, element, node_id, role, ops)?;
    let guard = pack_guard(context, stage, identity, role, dimension, layout, rows, input)?;
    if let Err(error) = context.check_cancellation(ops.before_pack) {
        guard.fail(&error);
        return Err(error);
    }
    let result = match binding {
        DenseVectorBinding::Columns(columns) => {
            let selected = input.frame().select(columns.iter().map(String::as_str)).map_err(|error| {
                cudf_query_error(pack_operation, "failed to select vector feature columns", error)
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
            });
            selected.and_then(|selected| {
                let series =
                    columns.iter().map(|column| selected.series(column)).collect::<cudf::Result<Vec<_>>>().map_err(
                        |error| {
                            cudf_query_error(pack_operation, "failed to access vector feature columns", error)
                                .with_fact("node_id", node_id)
                                .with_fact("relation_role", role)
                        },
                    )?;
                let actual = series[0].dtype().map_err(|error| {
                    cudf_query_error(pack_operation, "failed to inspect vector feature element type", error)
                })?;
                require_element(actual.id(), element, node_id, algorithm, role, pack_operation)?;
                runtime.pack_columns(&series.iter().map(Series::as_column).collect::<Vec<_>>(), layout).map_err(
                    |error| {
                        cudf_query_error(
                            pack_operation,
                            "failed to pack the wide feature columns into an owned matrix",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                    },
                )
            })
        },
        DenseVectorBinding::VectorColumn(column) => (|| {
            let batch = GpuRecordBatch::new(input.frame().clone(), Arc::clone(schema)).map_err(|error| {
                cudf_query_error(
                    pack_operation,
                    "failed to bind the vector relation to its FixedSizeList schema",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            })?;
            let field_index = field_index(schema, column, node_id, role, pack_operation)?;
            let validated = batch.try_prepare_fixed_size_list_export(field_index).map_err(|error| {
                cudf_query_error(
                    pack_operation,
                    "failed to prove the FixedSizeList vector child range and finite values",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            })?;
            require_element(validated.element(), element, node_id, algorithm, role, pack_operation)?;
            validated.pack_active_child_into_runtime(runtime, layout).map_err(|error| {
                cudf_query_error(
                    pack_operation,
                    "failed to pack the validated FixedSizeList vector child into an owned matrix",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            })
        })(),
    };
    finish_pack(guard, result, rows, dimension, bytes)
}

fn require_element(
    actual: TypeId,
    expected: TypeId,
    node_id: NodeId,
    algorithm: &str,
    role: &'static str,
    pack_operation: &'static str,
) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        pack_operation,
        "vector element type is not supported by this operation",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", algorithm)
    .with_fact("relation_role", role)
    .with_fact("reason", "unsupported_vector_element_type")
    .with_fact("element", actual.name()))
}

fn field_index(
    schema: &SchemaRef,
    column: &str,
    node_id: NodeId,
    role: &'static str,
    pack_operation: &'static str,
) -> Result<usize> {
    schema.index_of(column).map_err(|_| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            pack_operation,
            "vector column is absent from the evaluated relation schema",
        )
        .with_fact("node_id", node_id)
        .with_fact("relation_role", role)
        .with_fact("column", column)
        .with_fact("reason", "missing_feature_column")
    })
}

#[cfg(feature = "cuvs")]
pub(super) fn relation_dimension(
    binding: &DenseVectorBinding,
    schema: &SchemaRef,
    node_id: NodeId,
    role: &'static str,
) -> Result<usize> {
    relation_dimension_for(binding, schema, node_id, role, VECTOR_OPERATIONS.pack)
}

fn relation_dimension_for(
    binding: &DenseVectorBinding,
    schema: &SchemaRef,
    node_id: NodeId,
    role: &'static str,
    pack_operation: &'static str,
) -> Result<usize> {
    match binding {
        DenseVectorBinding::Columns(columns) => Ok(columns.len()),
        DenseVectorBinding::VectorColumn(column) => match schema
            .field_with_name(column)
            .map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    pack_operation,
                    "vector column is absent from its input schema",
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
                .with_fact("reason", "missing_feature_column")
            })?
            .data_type()
        {
            DataType::FixedSizeList(_, dimension) if *dimension > 0 => Ok(*dimension as usize),
            DataType::List(_) | DataType::LargeList(_) => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                pack_operation,
                "variable List<Float32> vector columns require a uniform-dimension device witness",
            )
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
            .with_fact("reason", "uniform_list_dimension_witness_unavailable")
            .with_fact("host_fallback", "disabled")),
            actual => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                pack_operation,
                "vector column is not a positive-width FixedSizeList",
            )
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
            .with_fact("data_type", format!("{actual:?}"))
            .with_fact("reason", "unsupported_feature_type")),
        },
    }
}

#[expect(clippy::too_many_arguments, reason = "the stage identity and relation facts are supplied by the caller")]
fn pack_guard(
    context: &ExecutionContext,
    stage: RuntimeOperatorStageKind,
    identity: RuntimeOperatorStageIdentity,
    role: &'static str,
    dimension: usize,
    layout: DenseMatrixLayout,
    rows: usize,
    input: &GpuDataFrame,
) -> Result<RuntimeOperatorStageGuard> {
    let input_bytes = input.device_allocated_size_bytes()?;
    let identity = identity
        .with_vector_relation_role(role)
        .with_vector_dimension(dimension)
        .with_matrix_layout(matrix_layout_name(layout))
        .with_input_rows(rows as u64)
        .with_input_bytes(input_bytes);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    Ok(RuntimeOperatorStageGuard::start(observation, plan_execution_id, parent_span_id, stage, identity))
}

fn finish_pack(
    guard: RuntimeOperatorStageGuard,
    result: Result<PackedMatrix>,
    rows: usize,
    dimension: usize,
    bytes: u64,
) -> Result<(PackedMatrix, usize)> {
    match result {
        Ok(packed) => {
            guard.finish(Some(rows as u64), Some(bytes));
            Ok((packed, dimension))
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

fn packed_bytes(
    rows: usize,
    dimension: usize,
    element: TypeId,
    node_id: NodeId,
    role: &'static str,
    ops: &DenseRelationOperations,
) -> Result<u64> {
    let width = match element {
        TypeId::Float32 => 4,
        TypeId::Float64 => 8,
        TypeId::Int8 | TypeId::UInt8 => 1,
        _ => return Err(size_overflow_for(ops, node_id, role)),
    };
    rows.checked_mul(dimension)
        .and_then(|values| values.checked_mul(width))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| size_overflow_for(ops, node_id, role))
}

pub(super) fn relation_input_for<'a>(
    ops: &DenseRelationOperations,
    inputs: &'a BTreeMap<NodeId, GpuDataFrame>,
    input_id: NodeId,
    node_id: NodeId,
    role: &'static str,
) -> Result<&'a GpuDataFrame> {
    inputs.get(&input_id).ok_or_else(|| {
        QueryError::new(QueryErrorCode::InvalidPlan, ops.execute, "evaluated vector relation input is missing")
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("input_node_id", input_id)
            .with_fact("reason", "missing_evaluated_relation")
    })
}

#[cfg(feature = "cuvs")]
pub(super) fn checked_output_rows(rows: usize, factor: usize, node_id: NodeId, algorithm: &str) -> Result<usize> {
    let output_rows = rows.checked_mul(factor).ok_or_else(|| {
        QueryError::new(QueryErrorCode::SizeOverflow, VECTOR_OPERATIONS.execute, "vector output row count overflowed")
            .with_fact("node_id", node_id)
            .with_fact("algorithm", algorithm)
            .with_fact("reason", "output_row_count_overflow")
    })?;
    if output_rows > i32::MAX as usize {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvalidInput,
            VECTOR_OPERATIONS.execute,
            "vector output row count exceeds libcudf size_type range",
        )
        .with_fact("node_id", node_id)
        .with_fact("algorithm", algorithm)
        .with_fact("input_rows", rows)
        .with_fact("factor", factor)
        .with_fact("output_rows", output_rows)
        .with_fact("maximum_rows", i32::MAX)
        .with_fact("reason", "row_count_exceeds_libcudf_size_type"));
    }
    Ok(output_rows)
}

pub(super) fn matrix_bytes_for<T>(
    ops: &DenseRelationOperations,
    rows: usize,
    dimension: usize,
    node_id: NodeId,
    role: &'static str,
) -> Result<u64> {
    rows.checked_mul(dimension)
        .and_then(|values| values.checked_mul(std::mem::size_of::<T>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| size_overflow_for(ops, node_id, role))
}

pub(super) fn size_overflow_for(ops: &DenseRelationOperations, node_id: NodeId, resource: &'static str) -> QueryError {
    let error =
        QueryError::new(QueryErrorCode::SizeOverflow, ops.execute, "vector execution size calculation overflowed")
            .with_fact("node_id", node_id)
            .with_fact("resource", resource)
            .with_fact("reason", "size_calculation_overflow");
    if ops.execute == VECTOR_OPERATIONS.execute { error.with_fact("algorithm", "brute_force_knn") } else { error }
}
