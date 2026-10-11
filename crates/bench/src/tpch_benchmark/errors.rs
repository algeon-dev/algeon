use super::{DataFusionError, WorkloadError};

pub(super) fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}

pub(super) fn tpch_error(error: WorkloadError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}
