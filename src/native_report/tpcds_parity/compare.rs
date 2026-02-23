use std::collections::BTreeSet;

use super::*;

/// Result of comparing CPU and native Arrow outputs.
#[derive(Debug, Clone)]
pub struct ParityComparisonResult {
    /// Match, mismatch, or skip status.
    pub status: ParityStatus,
    /// Row-order and float-tolerance mode used for comparison.
    pub mode: ParityComparisonMode,
    /// Mismatch or skip detail when available.
    pub detail: Option<String>,
    /// Stable diagnostic hash of canonical CPU output.
    pub cpu_fingerprint: String,
    /// Stable diagnostic hash of canonical native output.
    pub native_fingerprint: String,
}

/// Compares CPU and native Arrow output using TPC-DS parity rules.
///
/// Schema differences are reported before row comparison. When `ordered` is
/// false, canonical rows are sorted before comparison. Float tolerance applies
/// only to float columns and is recorded in the returned comparison mode.
pub fn compare_record_batches(
    cpu_schema: &Schema,
    cpu_batches: &[RecordBatch],
    native_schema: &Schema,
    native_batches: &[RecordBatch],
    ordered: bool,
    tolerance: ParityFloatTolerance,
) -> ParityComparisonResult {
    compare_record_batches_with_options(
        cpu_schema,
        cpu_batches,
        native_schema,
        native_batches,
        ordered,
        tolerance,
        ParityCompareOptions::strict_native(),
    )
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ParityCompareOptions {
    pub(super) allow_decimal_precision_mismatch: bool,
}

impl ParityCompareOptions {
    pub(super) const fn strict_native() -> Self {
        Self { allow_decimal_precision_mismatch: false }
    }

    pub(super) const fn cross_engine_oracle() -> Self {
        Self { allow_decimal_precision_mismatch: true }
    }
}

pub(super) fn compare_record_batches_with_options(
    cpu_schema: &Schema,
    cpu_batches: &[RecordBatch],
    native_schema: &Schema,
    native_batches: &[RecordBatch],
    ordered: bool,
    tolerance: ParityFloatTolerance,
    options: ParityCompareOptions,
) -> ParityComparisonResult {
    let schema_diff = compare_schemas(cpu_schema, native_schema, options);
    let float_tolerant = tolerance.is_enabled() && schema_has_float_column(cpu_schema);
    let mode = ParityComparisonMode::select(ordered, float_tolerant);

    if let Some(detail) = schema_diff {
        return ParityComparisonResult {
            status: ParityStatus::Mismatched,
            mode,
            detail: Some(detail),
            cpu_fingerprint: fingerprint_schema_only(cpu_schema),
            native_fingerprint: fingerprint_schema_only(native_schema),
        };
    }

    let cpu_rows = match collect_canonical_rows(cpu_schema, cpu_batches) {
        Ok(rows) => rows,
        Err(detail) => {
            return ParityComparisonResult {
                status: ParityStatus::NotCompared,
                mode,
                detail: Some(format!("cpu canonicalization failed: {detail}")),
                cpu_fingerprint: fingerprint_schema_only(cpu_schema),
                native_fingerprint: fingerprint_schema_only(native_schema),
            };
        },
    };
    let native_rows = match collect_canonical_rows(native_schema, native_batches) {
        Ok(rows) => rows,
        Err(detail) => {
            return ParityComparisonResult {
                status: ParityStatus::NotCompared,
                mode,
                detail: Some(format!("native canonicalization failed: {detail}")),
                cpu_fingerprint: fingerprint_rows(cpu_schema, &cpu_rows),
                native_fingerprint: fingerprint_schema_only(native_schema),
            };
        },
    };

    let cpu_fingerprint = fingerprint_rows(cpu_schema, &cpu_rows);
    let native_fingerprint = fingerprint_rows(native_schema, &native_rows);

    if cpu_rows.len() != native_rows.len() {
        return ParityComparisonResult {
            status: ParityStatus::Mismatched,
            mode,
            detail: Some(format!("row count differs: cpu={} native={}", cpu_rows.len(), native_rows.len())),
            cpu_fingerprint,
            native_fingerprint,
        };
    }

    let (cpu_aligned, native_aligned) = if ordered {
        (cpu_rows, native_rows)
    } else {
        let mut cpu = cpu_rows;
        let mut native = native_rows;
        cpu.sort();
        native.sort();
        (cpu, native)
    };

    for (index, (cpu_row, native_row)) in cpu_aligned.iter().zip(native_aligned.iter()).enumerate() {
        if let Some(detail) =
            compare_canonical_rows(cpu_schema, cpu_row, native_row, tolerance, float_tolerant, options)
        {
            return ParityComparisonResult {
                status: ParityStatus::Mismatched,
                mode,
                detail: Some(format!("row {index}: {detail}")),
                cpu_fingerprint,
                native_fingerprint,
            };
        }
    }

    ParityComparisonResult { status: ParityStatus::Matched, mode, detail: None, cpu_fingerprint, native_fingerprint }
}

pub(super) fn compare_schemas(cpu: &Schema, native: &Schema, options: ParityCompareOptions) -> Option<String> {
    if cpu.fields().len() != native.fields().len() {
        return Some(format!(
            "schema field count differs: cpu={} native={}",
            cpu.fields().len(),
            native.fields().len()
        ));
    }
    for (index, (cpu_field, native_field)) in cpu.fields().iter().zip(native.fields().iter()).enumerate() {
        if !logical_data_types_compatible(cpu_field.data_type(), native_field.data_type(), options) {
            return Some(format!(
                "column {} ({}/{}) type differs: cpu={:?} native={:?}",
                index,
                cpu_field.name(),
                native_field.name(),
                cpu_field.data_type(),
                native_field.data_type(),
            ));
        }
    }
    None
}

pub(super) fn logical_data_types_compatible(cpu: &DataType, native: &DataType, options: ParityCompareOptions) -> bool {
    if cpu == native {
        return true;
    }
    if options.allow_decimal_precision_mismatch {
        match (cpu, native) {
            (DataType::Decimal128(_, cpu_scale), DataType::Decimal128(_, native_scale))
            | (DataType::Decimal256(_, cpu_scale), DataType::Decimal256(_, native_scale))
                if cpu_scale == native_scale =>
            {
                return true;
            },
            _ => {},
        }
    }
    matches!(
        (cpu, native),
        (DataType::Utf8, DataType::Utf8View)
            | (DataType::Utf8View, DataType::Utf8)
            | (DataType::LargeUtf8, DataType::Utf8)
            | (DataType::Utf8, DataType::LargeUtf8)
            | (DataType::LargeUtf8, DataType::Utf8View)
            | (DataType::Utf8View, DataType::LargeUtf8)
            | (DataType::Binary, DataType::BinaryView)
            | (DataType::BinaryView, DataType::Binary)
            | (DataType::LargeBinary, DataType::Binary)
            | (DataType::Binary, DataType::LargeBinary)
            | (DataType::LargeBinary, DataType::BinaryView)
            | (DataType::BinaryView, DataType::LargeBinary)
    )
}

pub(super) fn schema_has_float_column(schema: &Schema) -> bool {
    schema.fields().iter().any(|field| matches!(field.data_type(), DataType::Float32 | DataType::Float64))
}

/// Detect whether the given SQL has a top-level `ORDER BY` clause that
/// determines the row order of the result. Strings, comments, and
/// parenthesized subqueries are ignored.
pub fn sql_has_top_level_order_by(sql: &str) -> bool {
    let depth_zero = strip_to_depth_zero(sql);
    let lowered = depth_zero.to_ascii_lowercase();
    top_level_order_by_start(&lowered).is_some()
}

/// Returns true when the top-level `ORDER BY` names every output column.
///
/// This is a conservative total-order signal for parity comparison. A query may
/// still be deterministic through uniqueness constraints, but those constraints
/// are not visible here, so partial `ORDER BY` lists are treated as order-tie
/// candidates.
pub fn sql_top_level_order_by_covers_output_columns(sql: &str, schema: &Schema) -> bool {
    let depth_zero = strip_to_depth_zero(sql);
    let lowered = depth_zero.to_ascii_lowercase();
    let Some(start) = top_level_order_by_start(&lowered) else {
        return false;
    };
    let mut clause = &lowered[start..];
    for keyword in ["limit", "fetch", "offset"] {
        if let Some(index) = find_word(clause.as_bytes(), keyword.as_bytes()) {
            clause = &clause[..index];
        }
    }
    let order_columns = clause.split(',').filter_map(normalized_order_by_column).collect::<BTreeSet<_>>();
    schema.fields().iter().all(|field| order_columns.contains(&field.name().to_ascii_lowercase()))
}

fn top_level_order_by_start(lowered_depth_zero_sql: &str) -> Option<usize> {
    let bytes = lowered_depth_zero_sql.as_bytes();
    let mut i = 0;
    while i + 8 <= bytes.len() {
        if matches_word_at(bytes, i, b"order") && {
            let after = skip_whitespace(bytes, i + 5);
            matches_word_at(bytes, after, b"by")
        } {
            return Some(skip_whitespace(bytes, skip_whitespace(bytes, i + 5) + 2));
        }
        i += 1;
    }
    None
}

fn find_word(bytes: &[u8], word: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + word.len() <= bytes.len() {
        if matches_word_at(bytes, i, word) {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn normalized_order_by_column(term: &str) -> Option<String> {
    let tokens = term.split_whitespace().collect::<Vec<_>>();
    let first = tokens.first()?.trim_matches(|c: char| c == '"' || c == '`');
    if first.is_empty() {
        return None;
    }
    if let Ok(ordinal) = first.parse::<usize>() {
        return Some(format!("#{ordinal}"));
    }
    let column = first.rsplit('.').next().unwrap_or(first).trim_matches(|c: char| c == '"' || c == '`');
    (!column.is_empty()).then(|| column.to_ascii_lowercase())
}

fn strip_to_depth_zero(sql: &str) -> String {
    let mut output = String::with_capacity(sql.len());
    let chars: Vec<char> = sql.chars().collect();
    let mut i = 0;
    let mut depth: i32 = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '-' && i + 1 < chars.len() && chars[i + 1] == '-' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            output.push(' ');
            continue;
        }
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            output.push(' ');
            continue;
        }
        if c == '\'' {
            i += 1;
            while i < chars.len() {
                if chars[i] == '\'' && i + 1 < chars.len() && chars[i + 1] == '\'' {
                    i += 2;
                    continue;
                }
                if chars[i] == '\'' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            output.push(' ');
            continue;
        }
        if c == '"' {
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }
            output.push(' ');
            continue;
        }
        if c == '(' {
            depth += 1;
            output.push(' ');
            i += 1;
            continue;
        }
        if c == ')' {
            depth = (depth - 1).max(0);
            output.push(' ');
            i += 1;
            continue;
        }
        if depth == 0 {
            output.push(c);
        } else {
            output.push(' ');
        }
        i += 1;
    }
    output
}

fn matches_word_at(bytes: &[u8], pos: usize, word: &[u8]) -> bool {
    if pos + word.len() > bytes.len() {
        return false;
    }
    if !word.iter().zip(bytes[pos..pos + word.len()].iter()).all(|(a, b)| a == b) {
        return false;
    }
    if pos > 0 && is_word_byte(bytes[pos - 1]) {
        return false;
    }
    if pos + word.len() < bytes.len() && is_word_byte(bytes[pos + word.len()]) {
        return false;
    }
    true
}

fn skip_whitespace(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
