use std::{path::PathBuf, sync::Arc};

use datafusion::{
    catalog::TableProvider,
    datasource::{
        file_format::options::ReadOptions,
        listing::{ListingTable, ListingTableConfig, ListingTableUrl},
    },
    prelude::{ParquetReadOptions, SessionContext},
};

use super::{CLICKBENCH_RAW_TABLE, CLICKBENCH_TABLE, ClickbenchLayout, WorkloadError, WorkloadResult};

pub async fn register_clickbench_table(ctx: &SessionContext, layout: &ClickbenchLayout) -> WorkloadResult<()> {
    match layout {
        ClickbenchLayout::Single { file } => {
            let path_str = file.to_str().ok_or_else(|| WorkloadError::NonUtf8Path(file.clone()))?;
            ctx.register_parquet(CLICKBENCH_RAW_TABLE, path_str, ParquetReadOptions::default()).await?;
        },
        ClickbenchLayout::Partitioned { files, .. } => register_clickbench_partitioned_shards(ctx, files).await?,
    }
    let view_sql = format!(
        r#"CREATE VIEW {CLICKBENCH_TABLE} AS SELECT * EXCEPT ("EventDate"), CAST(CAST("EventDate" AS INTEGER) AS DATE) AS "EventDate" FROM {CLICKBENCH_RAW_TABLE}"#
    );
    ctx.sql(&view_sql).await?;
    Ok(())
}

async fn register_clickbench_partitioned_shards(ctx: &SessionContext, files: &[PathBuf]) -> WorkloadResult<()> {
    let urls = files
        .iter()
        .map(|path| -> WorkloadResult<ListingTableUrl> {
            let absolute = path.canonicalize().map_err(|error| {
                WorkloadError::InvalidDataset(format!(
                    "failed to canonicalize ClickBench shard {}: {error}",
                    path.display()
                ))
            })?;
            let absolute_str = absolute.to_str().ok_or_else(|| WorkloadError::NonUtf8Path(absolute.clone()))?;
            Ok(ListingTableUrl::parse(format!("file://{absolute_str}"))?)
        })
        .collect::<WorkloadResult<Vec<_>>>()?;
    if urls.is_empty() {
        return Err(WorkloadError::InvalidDataset(
            "register_clickbench_partitioned_shards called with zero shards".to_string(),
        ));
    }
    let session_state = ctx.state();
    let listing_options =
        ParquetReadOptions::default().to_listing_options(session_state.config(), session_state.default_table_options());
    let inferred_schema = listing_options.infer_schema(&session_state, &urls[0]).await?;
    let config = ListingTableConfig::new_with_multi_paths(urls)
        .with_listing_options(listing_options)
        .with_schema(inferred_schema);
    let listing_table: Arc<dyn TableProvider> = Arc::new(ListingTable::try_new(config)?);
    ctx.register_table(CLICKBENCH_RAW_TABLE, listing_table)?;
    Ok(())
}
