use super::*;

pub(super) fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}

pub(super) fn tpch_error(error: TpchWorkloadError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

pub(super) fn native_error(error: nexus_query_engine::error::QueryError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}
