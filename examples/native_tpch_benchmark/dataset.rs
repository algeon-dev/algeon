use super::*;

pub(super) async fn prepare_benchmark_dataset(tiny_root: &Path) -> DataFusionResult<(PathBuf, TpchDatasetMetadata)> {
    prepare_tpch_report_dataset(tiny_root).map_err(tpch_error)
}

pub(super) async fn register_tpch_tables(ctx: &SessionContext, metadata: &TpchDatasetMetadata) -> DataFusionResult<()> {
    for table in &metadata.tables {
        let path = table
            .path
            .to_str()
            .ok_or_else(|| plan_error(format!("TPC-H table path is not valid UTF-8: {}", table.path.display())))?;
        ctx.register_parquet(&table.table, path, ParquetReadOptions::default()).await?;
    }
    Ok(())
}
