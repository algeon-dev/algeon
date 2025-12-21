use super::*;

pub(super) fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}

pub(super) fn native_query_error(error: cudf_query_engine::QueryError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}
