use super::*;

pub(in crate::native) fn metadata_row_group_selections(
    files: &[&PartitionedFile],
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: &SchemaRef,
) -> Option<Vec<Vec<usize>>> {
    let coalesced_files = coalesced_local_parquet_scan_files(files).ok()?;
    let files = coalesced_files.iter().map(|file| file.file).collect::<Vec<_>>();
    if files.is_empty()
        || !file_scan.object_store_url.as_str().starts_with("file://")
        || file_scan.expr_adapter_factory.is_some()
        || file_scan.limit.is_some()
        || file_scan.preserve_order
        || !file_scan.output_ordering.is_empty()
        || files.iter().any(|file| file.extensions.is_some() || file.ordering.is_some())
    {
        return None;
    }
    if parquet_source.filter().is_some()
        && validate_pruning_only_parquet_predicate(parquet_source, filter, source_schema).is_err()
    {
        return None;
    }
    let predicate = row_group_pruning_predicate(filter?.predicate(), file_scan.file_schema())?;
    let selections =
        files.iter().map(|file| metadata_row_group_selection(file, predicate)).collect::<Option<Vec<_>>>()?;
    let selected_row_groups = selections.iter().map(|selection| selection.row_groups.len()).sum::<usize>();
    let total_row_groups = selections.iter().map(|selection| selection.total_row_groups).sum::<usize>();
    if selected_row_groups > 0 && selected_row_groups < total_row_groups {
        Some(selections.into_iter().map(|selection| selection.row_groups).collect())
    } else {
        None
    }
}

#[derive(Debug, Clone)]
struct MetadataRowGroupSelection {
    row_groups: Vec<usize>,
    total_row_groups: usize,
}

fn metadata_row_group_selection(
    file: &PartitionedFile,
    predicate: RowGroupPruningPredicate,
) -> Option<MetadataRowGroupSelection> {
    let path = local_object_path_to_filesystem(file.object_meta.location.as_ref());
    let reader = SerializedFileReader::try_from(path.as_path()).ok()?;
    let metadata = reader.metadata();
    let total_row_groups = metadata.num_row_groups();
    if total_row_groups == 0 {
        return None;
    }
    let row_groups = metadata
        .row_groups()
        .iter()
        .enumerate()
        .map(|(index, row_group)| {
            let statistics = row_group.columns().get(predicate.column_index)?.statistics()?;
            row_group_may_satisfy_predicate(statistics, predicate).map(|may_satisfy| may_satisfy.then_some(index))
        })
        .collect::<Option<Vec<_>>>()
        .map(|row_groups| row_groups.into_iter().flatten().collect())?;
    Some(MetadataRowGroupSelection { row_groups, total_row_groups })
}

#[derive(Debug, Clone, Copy)]
struct RowGroupPruningPredicate {
    column_index: usize,
    op: Operator,
    literal: NumericLiteral,
}

#[derive(Debug, Clone, Copy)]
enum NumericLiteral {
    Int64(i64),
    Float64(f64),
}

fn row_group_pruning_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Option<RowGroupPruningPredicate> {
    let binary = expr.as_any().downcast_ref::<BinaryExpr>()?;
    if let Some((column, literal, op)) = column_literal_predicate(binary.left(), binary.op(), binary.right()) {
        return Some(RowGroupPruningPredicate {
            column_index: file_schema.index_of(column.name()).ok()?,
            op,
            literal: numeric_literal(literal.value())?,
        });
    }
    let inverted_op = invert_comparison_operator(binary.op())?;
    let (column, literal, op) = column_literal_predicate(binary.right(), &inverted_op, binary.left())?;
    Some(RowGroupPruningPredicate {
        column_index: file_schema.index_of(column.name()).ok()?,
        op,
        literal: numeric_literal(literal.value())?,
    })
}

fn column_literal_predicate<'a>(
    left: &'a Arc<dyn PhysicalExpr>,
    op: &Operator,
    right: &'a Arc<dyn PhysicalExpr>,
) -> Option<(&'a Column, &'a Literal, Operator)> {
    let column = left.as_any().downcast_ref::<Column>()?;
    let literal = right.as_any().downcast_ref::<Literal>()?;
    if matches!(op, Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq) {
        Some((column, literal, *op))
    } else {
        None
    }
}

fn invert_comparison_operator(op: &Operator) -> Option<Operator> {
    match op {
        Operator::Eq => Some(Operator::Eq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        _ => None,
    }
}

fn numeric_literal(value: &ScalarValue) -> Option<NumericLiteral> {
    match value {
        ScalarValue::Int8(Some(value)) => Some(NumericLiteral::Int64(i64::from(*value))),
        ScalarValue::Int16(Some(value)) => Some(NumericLiteral::Int64(i64::from(*value))),
        ScalarValue::Int32(Some(value)) => Some(NumericLiteral::Int64(i64::from(*value))),
        ScalarValue::Int64(Some(value)) => Some(NumericLiteral::Int64(*value)),
        ScalarValue::Float32(Some(value)) => Some(NumericLiteral::Float64(f64::from(*value))),
        ScalarValue::Float64(Some(value)) => Some(NumericLiteral::Float64(*value)),
        _ => None,
    }
}

fn row_group_may_satisfy_predicate(statistics: &Statistics, predicate: RowGroupPruningPredicate) -> Option<bool> {
    match statistics {
        Statistics::Int32(stats) => {
            let min = i64::from(*stats.min_opt()?);
            let max = i64::from(*stats.max_opt()?);
            let NumericLiteral::Int64(literal) = predicate.literal else {
                return None;
            };
            Some(int_range_may_satisfy(min, max, predicate.op, literal))
        },
        Statistics::Int64(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let NumericLiteral::Int64(literal) = predicate.literal else {
                return None;
            };
            Some(int_range_may_satisfy(min, max, predicate.op, literal))
        },
        Statistics::Float(stats) => {
            let min = f64::from(*stats.min_opt()?);
            let max = f64::from(*stats.max_opt()?);
            let NumericLiteral::Float64(literal) = predicate.literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, predicate.op, literal))
        },
        Statistics::Double(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let NumericLiteral::Float64(literal) = predicate.literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, predicate.op, literal))
        },
        _ => None,
    }
}

fn int_range_may_satisfy(min: i64, max: i64, op: Operator, literal: i64) -> bool {
    match op {
        Operator::Eq => min <= literal && literal <= max,
        Operator::Lt => min < literal,
        Operator::LtEq => min <= literal,
        Operator::Gt => max > literal,
        Operator::GtEq => max >= literal,
        _ => true,
    }
}

fn float_range_may_satisfy(min: f64, max: f64, op: Operator, literal: f64) -> bool {
    if min.is_nan() || max.is_nan() || literal.is_nan() {
        return true;
    }
    match op {
        Operator::Eq => min <= literal && literal <= max,
        Operator::Lt => min < literal,
        Operator::LtEq => min <= literal,
        Operator::Gt => max > literal,
        Operator::GtEq => max >= literal,
        _ => true,
    }
}
