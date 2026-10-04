use cudf::{Series, TypeId};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    plan::MlRelationBinding,
};
use query_runtime::{native_error::cudf_query_error, runtime::ExecutionContext};

use crate::exec::{
    GpuDataFrame,
    dense_relation::{ML_OPERATIONS, relation_id_series_for, relation_row_ordinals_for, require_series_contract_for},
};

const ASSEMBLE: &str = "query_engine.ml_algorithm.assemble";

pub(super) fn assemble_components(
    context: &ExecutionContext,
    input: &GpuDataFrame,
    binding: &MlRelationBinding,
    components: Vec<Series>,
) -> Result<GpuDataFrame> {
    assemble_ml_output(
        context,
        input,
        binding,
        components.into_iter().enumerate().map(|(index, series)| (format!("pc_{index}"), series)).collect(),
        TypeId::Int32,
    )
}

pub(super) fn assemble_ml_output(
    context: &ExecutionContext,
    input: &GpuDataFrame,
    binding: &MlRelationBinding,
    columns: Vec<(String, Series)>,
    prediction_type: TypeId,
) -> Result<GpuDataFrame> {
    let rows = input.num_rows();
    for (name, series) in &columns {
        let dtype = match name.as_str() {
            "cluster_id" => TypeId::Int64,
            "prediction" => prediction_type,
            _ => TypeId::Float32,
        };
        require_series_contract_for(&ML_OPERATIONS, series, dtype, rows, name)?;
    }
    let runtime = context.runtime()?;
    let ordinal = relation_row_ordinals_for(&ML_OPERATIONS, &runtime, rows, "failed to construct ML row ordinals")?;
    require_series_contract_for(&ML_OPERATIONS, &ordinal, TypeId::UInt64, rows, "row_ordinal")?;
    let role = if columns.iter().any(|(name, _)| name == "prediction") { "predict" } else { "input" };
    let id = relation_id_series_for(&ML_OPERATIONS, &binding.row_id, input, role)?
        .copy_into_runtime(&runtime)
        .map_err(|error| cudf_query_error(ASSEMBLE, "failed to bind ML output IDs to runtime", error))?;
    let id_type =
        id.dtype().map_err(|error| cudf_query_error(ASSEMBLE, "failed to inspect ML output IDs", error))?.id();
    require_series_contract_for(&ML_OPERATIONS, &id, id_type, rows, "id")?;
    let output = runtime
        .empty_dataframe()
        .assign([("row_ordinal", ordinal), ("id", id)])
        .map_err(|error| cudf_query_error(ASSEMBLE, "failed to assemble ML relation", error))?;
    let output = columns.into_iter().try_fold(output, |frame, (name, series)| {
        frame
            .with_column(&name, series)
            .map_err(|error| cudf_query_error(ASSEMBLE, "failed to attach ML component", error))
    })?;
    if output.num_rows() != rows {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvariantViolation,
            ASSEMBLE,
            "ML output row count differs from input",
        )
        .with_fact("reason", "invalid_native_output_contract"));
    }
    Ok(GpuDataFrame::new(output))
}
