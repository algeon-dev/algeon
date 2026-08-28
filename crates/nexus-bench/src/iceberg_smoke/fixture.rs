use super::*;

/// Resolves one table exactly once, carrying the same frozen source policy to
/// the CPU delegate, leaf scan planner, and native execution profile.
pub(crate) async fn load_iceberg_table(fixture: &IcebergNativeSmokeFixture) -> SmokeResult<ResolvedTable> {
    match fixture {
        IcebergNativeSmokeFixture::LocalMetadata { metadata_location, table_ident } => {
            let ident = table_ident_from_parts(table_ident)?;
            let table = StaticTable::from_metadata_file(metadata_location, ident, FileIO::new_with_fs()).await?;
            Ok(ResolvedTable::from_table(table.into_table(), SourceAccessPolicy::s3_default_chain(None))?)
        },
        IcebergNativeSmokeFixture::GlueS3 { database, table, .. } => {
            let backend =
                GlueCatalogBackend::connect(fixture.glue_catalog_config().expect("Glue fixture has a catalog config"))
                    .await?
                    .into_backend();
            Ok(backend.load_table(TableIdentifier { namespace: vec![database.clone()], name: table.clone() }).await?)
        },
        IcebergNativeSmokeFixture::RestS3 { namespace, table, .. } => {
            let backend =
                RestCatalogBackend::connect(fixture.rest_config().expect("REST fixture has a catalog config"))
                    .await?
                    .into_backend();
            Ok(backend.load_table(TableIdentifier { namespace: vec![namespace.clone()], name: table.clone() }).await?)
        },
    }
}
