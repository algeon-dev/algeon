//! The Arrow schema an Iceberg table exposes to DataFusion.

use std::sync::Arc;

use algeon_datafusion::error::{Error, ErrorCode, Result};
use datafusion::{arrow::datatypes::SchemaRef, datasource::physical_plan::parquet::transform_schema_to_view};
use iceberg::{
    Error as IcebergError,
    arrow::{schema_to_arrow_schema, strip_metadata_from_schema},
    spec::Schema as IcebergSchema,
};

use crate::error::SCAN_OPERATION;

/// The table schema DataFusion plans against.
///
/// Field-id metadata is removed: DataFusion projections and aliases drop
/// field metadata, and its physical-versus-logical schema check treats the
/// difference as a mismatch. Top-level strings and binaries use the view
/// types, as a DataFusion `ListingTable` over the same files does under the
/// default `schema_force_view_types`, so native lowering and the CPU
/// operators above the scan see the types they see for a listing table.
pub(crate) fn table_arrow_schema(schema: &IcebergSchema) -> Result<SchemaRef> {
    let conversion_error = |error: IcebergError| {
        Error::new(ErrorCode::UnsupportedType, SCAN_OPERATION, "the Iceberg schema has no Arrow equivalent")
            .with_source(error)
    };
    let arrow = schema_to_arrow_schema(schema).map_err(conversion_error)?;
    let arrow = strip_metadata_from_schema(&arrow).map_err(conversion_error)?;
    Ok(Arc::new(transform_schema_to_view(&arrow)))
}
