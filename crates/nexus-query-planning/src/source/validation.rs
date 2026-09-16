//! Structural validation for frontend-neutral source contracts.

use super::LiteralBatchSource;
use crate::{
    backend::AnalysisContext,
    error::{QueryError, QueryErrorCode, Result},
};

mod local_parquet;

pub use local_parquet::validate_local_parquet_source;

/// Validate the structural contract of an in-memory literal-batch source.
#[doc(hidden)]
pub fn validate_literal_batch_source(source: &LiteralBatchSource, ctx: AnalysisContext<'_>) -> Result<()> {
    if source.batches.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.literal_batch_source.validate",
            "literal batch source requires at least one batch",
        )
        .with_fact("reason", "empty_batches"));
    }
    for field in source.schema.fields() {
        if !ctx.backend().supports_type(field.data_type()) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.literal_batch_source.validate_schema",
                "schema field type is not supported",
            )
            .with_fact("field", field.name())
            .with_fact("data_type", format!("{:?}", field.data_type())));
        }
    }
    for (index, batch) in source.batches.iter().enumerate() {
        if batch.schema().as_ref() != source.schema.as_ref() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.literal_batch_source.validate",
                "literal batch schema must match declared source schema",
            )
            .with_fact("batch_index", index)
            .with_fact("declared_fields", source.schema.fields().len())
            .with_fact("batch_fields", batch.schema().fields().len()));
        }
    }
    Ok(())
}
