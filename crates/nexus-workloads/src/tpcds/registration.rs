use std::{fs::File, path::Path};

use arrow::datatypes::Schema;
use datafusion::{
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    prelude::{ParquetReadOptions, SessionContext},
};

use super::{TPCDS_TABLES, TpcdsTableLayout, WorkloadError, WorkloadResult, discover_tpcds_table_layout};

pub async fn register_tpcds_tables(ctx: &SessionContext, root: &Path) -> WorkloadResult<()> {
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table)?;
        let path = layout.anchor.to_str().ok_or_else(|| WorkloadError::NonUtf8Path(layout.anchor.clone()))?;
        let schema = tpcds_table_schema(&layout)?;
        ctx.register_parquet(table, path, ParquetReadOptions::default().schema(&schema)).await?;
    }
    Ok(())
}

fn tpcds_table_schema(layout: &TpcdsTableLayout) -> WorkloadResult<Schema> {
    let first_file = layout.files.first().ok_or_else(|| {
        WorkloadError::InvalidDataset("expected at least one Parquet file for TPC-DS table".to_string())
    })?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(File::open(first_file)?)?.schema().as_ref().clone())
}
