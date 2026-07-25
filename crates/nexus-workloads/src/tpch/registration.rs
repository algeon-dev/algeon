use std::{path::Path, sync::Arc};

use datafusion::{
    catalog::TableProvider,
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    prelude::{ParquetReadOptions, SessionContext},
};

use super::{TPCH_TABLES, TpchTableLayout, TpchWorkloadError, TpchWorkloadResult, discover_tpch_table_layout};

pub async fn register_tpch_tables(ctx: &SessionContext, root: &Path) -> TpchWorkloadResult<()> {
    for table in TPCH_TABLES {
        let layout = discover_tpch_table_layout(root, table)?;
        register_tpch_table_layout(ctx, table, &layout).await?;
    }
    Ok(())
}

pub async fn register_tpch_table_layout(
    ctx: &SessionContext,
    table: &str,
    layout: &TpchTableLayout,
) -> TpchWorkloadResult<()> {
    if layout.files.len() == 1 || layout.anchor.is_dir() {
        let path = layout.anchor.to_str().ok_or_else(|| TpchWorkloadError::NonUtf8Path(layout.anchor.clone()))?;
        ctx.register_parquet(table, path, ParquetReadOptions::default()).await?;
        return Ok(());
    }

    let urls = layout
        .files
        .iter()
        .map(|path| -> TpchWorkloadResult<ListingTableUrl> {
            let path_str = path.to_str().ok_or_else(|| TpchWorkloadError::NonUtf8Path(path.clone()))?;
            Ok(ListingTableUrl::parse(format!("file://{path_str}"))?)
        })
        .collect::<TpchWorkloadResult<Vec<_>>>()?;
    let listing_options = ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
    let inferred_schema = listing_options.infer_schema(&ctx.state(), &urls[0]).await?;
    let config = ListingTableConfig::new_with_multi_paths(urls)
        .with_listing_options(listing_options)
        .with_schema(inferred_schema);
    let listing_table: Arc<dyn TableProvider> = Arc::new(ListingTable::try_new(config)?);
    ctx.register_table(table, listing_table)?;
    Ok(())
}
