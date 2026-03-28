use datafusion::common::DataFusionError;

use crate::error::Error;

pub(super) fn find_adapter_error(error: &DataFusionError) -> Option<&Error> {
    match error {
        DataFusionError::External(source) => source.downcast_ref::<Error>(),
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => find_adapter_error(source),
        DataFusionError::Collection(errors) => errors.iter().find_map(find_adapter_error),
        DataFusionError::Shared(source) => find_adapter_error(source),
        _ => None,
    }
}
