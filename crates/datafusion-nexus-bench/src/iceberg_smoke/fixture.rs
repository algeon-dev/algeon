use super::*;

pub(crate) async fn load_iceberg_table(fixture: &IcebergNativeSmokeFixture) -> SmokeResult<Table> {
    match fixture {
        IcebergNativeSmokeFixture::LocalMetadata { metadata_location, table_ident } => {
            let ident = table_ident_from_parts(table_ident)?;
            let table = StaticTable::from_metadata_file(metadata_location, ident, FileIO::new_with_fs()).await?;
            Ok(table.into_table())
        },
        IcebergNativeSmokeFixture::GlueS3 {
            catalog_name,
            database,
            table,
            warehouse,
            catalog_id,
            region,
            profile,
            uri,
            s3_endpoint,
            s3_region,
            s3_path_style_access,
            ..
        } => {
            let mut config = GlueIcebergTableConfig::new(
                catalog_name.clone(),
                database.clone(),
                table.clone(),
                warehouse.clone(),
                region.clone(),
            );
            if let Some(catalog_id) = catalog_id.clone() {
                config = config.with_catalog_id(catalog_id);
            }
            if let Some(profile) = profile.clone() {
                config = config.with_profile(profile);
            }
            if let Some(uri) = uri.clone() {
                config = config.with_uri(uri);
            }
            if let Some(endpoint) = s3_endpoint.clone() {
                config = config.with_s3_endpoint(endpoint);
            }
            if let Some(region) = s3_region.clone() {
                config = config.with_s3_region(region);
            }
            if let Some(path_style) = s3_path_style_access {
                config = config.with_s3_path_style_access(*path_style);
            }
            Ok(load_glue_table(config).await?.table)
        },
        IcebergNativeSmokeFixture::RestS3 { namespace, table, .. } => {
            let backend =
                RestIcebergCatalogBackend::from_config_env(fixture.rest_config().expect("rest config")).await?;
            let ident = TableIdent::new(NamespaceIdent::new(namespace.clone()), table.clone());
            Ok(backend.load_table(&ident).await?)
        },
    }
}

pub(crate) fn iceberg_table_arrow_schema(table: &Table) -> SmokeResult<arrow::datatypes::SchemaRef> {
    Ok(iceberg_schema_to_datafusion_arrow_schema(table.current_schema_ref().as_ref())?)
}
