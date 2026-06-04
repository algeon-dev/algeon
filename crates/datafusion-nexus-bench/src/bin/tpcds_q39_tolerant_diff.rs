use std::{error::Error, path::PathBuf};

use arrow::{
    array::{Array, Float32Array, Float64Array},
    datatypes::DataType,
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use datafusion_nexus::{
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityNativeRuntimeOutcome, StatisticalAggregateExecutionMode,
};
use datafusion_nexus_bench::{
    CUDF_TPCDS_PARQUET_DIR_ENV, tpcds_native_optimizer_config, tpcds_parity_workload_items, tpcds_report_session_config,
};

const ABS_TOLERANCE: f64 = 1e-8;
const REL_TOLERANCE: f64 = 1e-8;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(std::env::var(CUDF_TPCDS_PARQUET_DIR_ENV)?);
    let query_filter = vec!["q39".to_string()];
    let optimizer_config = tpcds_native_optimizer_config()
        .with_statistical_aggregate_execution_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let mut items =
        tpcds_parity_workload_items(&root, tpcds_report_session_config(), optimizer_config, Some(&query_filter))
            .await?;
    let item = items.pop().ok_or("q39 workload item missing")?;
    let NativeTpcdsParityCpuOutcome::Succeeded { schema: cpu_schema, batches: cpu_batches } = item.cpu_outcome else {
        return Err("q39 CPU baseline did not succeed".into());
    };
    let NativeTpcdsParityNativeRuntimeOutcome::Succeeded { schema: native_schema, batches: native_batches, .. } =
        item.native_runtime
    else {
        return Err("q39 native tolerant runtime did not succeed".into());
    };
    if cpu_schema.fields().len() != native_schema.fields().len() {
        return Err(format!(
            "schema field count differs: cpu={} native={}",
            cpu_schema.fields().len(),
            native_schema.fields().len()
        )
        .into());
    }

    let summary = diff_batches(&cpu_batches, &native_batches)?;
    println!("metric\tvalue");
    println!("query_id\tq39");
    println!("cpu_rows\t{}", summary.cpu_rows);
    println!("native_rows\t{}", summary.native_rows);
    println!("row_count_equal\t{}", summary.cpu_rows == summary.native_rows);
    println!("ordered_key_nonfloat_mismatch_count\t{}", summary.non_float_mismatch_count);
    println!("float_cell_count\t{}", summary.float_cell_count);
    println!("max_absolute_error\t{:.17e}", summary.max_absolute_error);
    println!("max_relative_error\t{:.17e}", summary.max_relative_error);
    println!("outside_tolerance_count\t{}", summary.outside_tolerance_count);
    println!("abs_tolerance\t{ABS_TOLERANCE:.17e}");
    println!("rel_tolerance\t{REL_TOLERANCE:.17e}");
    Ok(())
}

#[derive(Debug, Default)]
struct DiffSummary {
    cpu_rows: usize,
    native_rows: usize,
    non_float_mismatch_count: usize,
    float_cell_count: usize,
    max_absolute_error: f64,
    max_relative_error: f64,
    outside_tolerance_count: usize,
}

fn diff_batches(cpu_batches: &[RecordBatch], native_batches: &[RecordBatch]) -> Result<DiffSummary, Box<dyn Error>> {
    let cpu_rows = flatten_rows(cpu_batches)?;
    let native_rows = flatten_rows(native_batches)?;
    let mut summary = DiffSummary { cpu_rows: cpu_rows.len(), native_rows: native_rows.len(), ..Default::default() };
    let row_count = cpu_rows.len().min(native_rows.len());
    for row_index in 0..row_count {
        let cpu = &cpu_rows[row_index];
        let native = &native_rows[row_index];
        for column_index in 0..cpu.values.len().min(native.values.len()) {
            match (&cpu.values[column_index], &native.values[column_index]) {
                (CellValue::Float(cpu), CellValue::Float(native)) => {
                    summary.float_cell_count += 1;
                    if let (Some(cpu), Some(native)) = (cpu, native) {
                        let abs = (cpu - native).abs();
                        let rel = relative_error(*cpu, *native);
                        summary.max_absolute_error = summary.max_absolute_error.max(abs);
                        summary.max_relative_error = summary.max_relative_error.max(rel);
                        let limit = ABS_TOLERANCE.max(REL_TOLERANCE * cpu.abs().max(native.abs()));
                        if abs > limit {
                            summary.outside_tolerance_count += 1;
                        }
                    } else if cpu != native {
                        summary.outside_tolerance_count += 1;
                    }
                },
                _ if cpu.values[column_index] != native.values[column_index] => {
                    summary.non_float_mismatch_count += 1;
                },
                _ => {},
            }
        }
        if cpu.values.len() != native.values.len() {
            summary.non_float_mismatch_count += cpu.values.len().abs_diff(native.values.len());
        }
    }
    if cpu_rows.len() != native_rows.len() {
        summary.non_float_mismatch_count += cpu_rows.len().abs_diff(native_rows.len());
    }
    Ok(summary)
}

fn relative_error(cpu: f64, native: f64) -> f64 {
    let denom = cpu.abs().max(native.abs());
    if denom == 0.0 { 0.0 } else { (cpu - native).abs() / denom }
}

#[derive(Debug, Clone, PartialEq)]
struct RowValue {
    values: Vec<CellValue>,
}

#[derive(Debug, Clone, PartialEq)]
enum CellValue {
    Float(Option<f64>),
    Other(Option<String>),
}

fn flatten_rows(batches: &[RecordBatch]) -> Result<Vec<RowValue>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(batch.num_columns());
            for column in 0..batch.num_columns() {
                let array = batch.column(column).as_ref();
                values.push(cell_value(array, row)?);
            }
            rows.push(RowValue { values });
        }
    }
    Ok(rows)
}

fn cell_value(array: &dyn Array, row: usize) -> Result<CellValue, Box<dyn Error>> {
    if array.is_null(row) {
        return Ok(match array.data_type() {
            DataType::Float32 | DataType::Float64 => CellValue::Float(None),
            _ => CellValue::Other(None),
        });
    }
    match array.data_type() {
        DataType::Float32 => {
            let values = array.as_any().downcast_ref::<Float32Array>().ok_or("expected Float32Array")?;
            Ok(CellValue::Float(Some(f64::from(values.value(row)))))
        },
        DataType::Float64 => {
            let values = array.as_any().downcast_ref::<Float64Array>().ok_or("expected Float64Array")?;
            Ok(CellValue::Float(Some(values.value(row))))
        },
        _ => Ok(CellValue::Other(Some(array_value_to_string(array, row)?))),
    }
}
