use super::{DataFrame, QueryError, QueryErrorCode, Result, cudf_query_error};

#[derive(Debug, Clone, Copy)]
pub(super) enum EdgeColumnRole {
    Vertex,
    Weight,
    EdgeId,
}

pub(super) fn validate_edge_column(frame: &DataFrame, column: &str, role: EdgeColumnRole) -> Result<cudf::DataType> {
    let series = frame.series(column).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "failed to access cuGraph edge-list column",
            error,
        )
        .with_fact("column", column)
    })?;
    if series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::GraphInputInvalid,
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "cuGraph materialized input columns must be non-null",
        )
        .with_fact("column", column)
        .with_fact("null_count", series.null_count()));
    }
    let dtype = series.dtype().map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "failed to inspect cuGraph edge-list column type",
            error,
        )
        .with_fact("column", column)
    })?;
    let supported = match role {
        EdgeColumnRole::Vertex | EdgeColumnRole::EdgeId => {
            matches!(dtype.id(), cudf::TypeId::Int32 | cudf::TypeId::Int64)
        },
        EdgeColumnRole::Weight => matches!(dtype.id(), cudf::TypeId::Float32 | cudf::TypeId::Float64),
    };
    if supported {
        return Ok(dtype);
    }
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "nexus_query_engine.graph_algorithm.validate_edge_column",
        "cuGraph edge-list column has unsupported dtype",
    )
    .with_fact("column", column)
    .with_fact(
        "role",
        match role {
            EdgeColumnRole::Vertex => "vertex",
            EdgeColumnRole::Weight => "weight",
            EdgeColumnRole::EdgeId => "edge_id",
        },
    )
    .with_fact("cudf_dtype", format!("{dtype:?}")))
}
