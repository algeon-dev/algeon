use std::path::Path;

use super::*;

pub(super) async fn register_tpch_tables(ctx: &SessionContext, root: &Path) -> DataFusionResult<()> {
    for table in TPCH_TABLES {
        let layout = discover_tpch_table_layout(root, table).map_err(|error| plan_error(error.to_string()))?;
        register_tpch_table_layout(ctx, table, &layout).await.map_err(|error| plan_error(error.to_string()))?;
    }
    Ok(())
}

pub(super) fn select_planning_row(
    query_id: &str,
    rows: Vec<NativeDataFusionPlanningRow>,
) -> DataFusionResult<NativeDataFusionPlanningRow> {
    match rows.len() {
        1 => {
            let row = rows.into_iter().next().expect("single planning row exists");
            if row.query_id != query_id {
                return Err(plan_error(format!(
                    "planning row query_id mismatch; requested {query_id}, observed {}",
                    row.query_id
                )));
            }
            Ok(row)
        },
        count => Err(plan_error(format!("expected one planning row for {query_id}, observed {count}"))),
    }
}

pub(super) fn tpch_query(query_id: &str) -> DataFusionResult<&'static TpchQuery> {
    TPCH_QUERIES
        .iter()
        .find(|query| query.id == query_id)
        .ok_or_else(|| plan_error(format!("unknown TPC-H query id: {query_id}")))
}

pub(super) fn measure_output(schema: &ArrowSchemaRef, batches: &[RecordBatch]) -> DataFusionResult<CpuBaseline> {
    Ok(CpuBaseline {
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_fingerprint: fingerprint_batches(schema, batches)?,
    })
}

pub(super) fn fingerprint_batches(schema: &ArrowSchemaRef, batches: &[RecordBatch]) -> DataFusionResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-nexus-query-service-tpch-output-v1");
    for batch in batches {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(plan_error("TPC-H output fingerprint requires collected batches to match the planned schema"));
        }
    }
    hasher.update_field("schema_metadata");
    hasher.update_metadata(schema.metadata());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(field.name());
        hasher.update_field(&field.data_type().to_string());
        hasher.update_field(if field.is_nullable() { "nullable" } else { "non_nullable" });
        hasher.update_metadata(field.metadata());
    }
    hasher.update_field(&batches.iter().map(RecordBatch::num_rows).sum::<usize>().to_string());
    hasher.update_field(&schema.fields().len().to_string());
    for batch in batches {
        for row in 0..batch.num_rows() {
            for column in 0..batch.num_columns() {
                let array = batch.column(column);
                if array.is_null(row) {
                    hasher.update_field("null");
                } else {
                    hasher.update_field("value");
                    hasher.update_field(&array_value_to_string(array.as_ref(), row)?);
                }
            }
        }
    }
    Ok(format!("output-fnv1a64:{:016x}", hasher.finish()))
}
