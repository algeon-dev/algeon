use arrow::datatypes::DataType;
use datafusion::parquet::schema::types::SchemaDescriptor;
use nexus_query_engine::parquet_stats::leaf_column_statistics;

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
        || files.iter().any(|file| !file.extensions.is_empty() || file.ordering.is_some())
    {
        return None;
    }
    if parquet_source.filter().is_some()
        && validate_pruning_only_parquet_predicate(parquet_source, filter, source_schema).is_err()
    {
        return None;
    }
    let predicate = row_group_pruning_expr(filter?.predicate(), file_scan.file_schema())?;
    let selections =
        files.iter().map(|file| metadata_row_group_selection(file, predicate.clone())).collect::<Option<Vec<_>>>()?;
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
    predicate: RowGroupPruningExpr,
) -> Option<MetadataRowGroupSelection> {
    let path = local_object_path_to_filesystem(file.object_meta.location.as_ref());
    let reader = SerializedFileReader::try_from(path.as_path()).ok()?;
    let metadata = reader.metadata();
    let schema = metadata.file_metadata().schema_descr();
    let total_row_groups = metadata.num_row_groups();
    if total_row_groups == 0 {
        return None;
    }
    let row_groups = metadata
        .row_groups()
        .iter()
        .enumerate()
        .map(|(index, row_group)| {
            row_group_may_satisfy_expr(row_group, schema, &predicate).map(|may_satisfy| may_satisfy.then_some(index))
        })
        .collect::<Option<Vec<_>>>()
        .map(|row_groups| row_groups.into_iter().flatten().collect())?;
    Some(MetadataRowGroupSelection { row_groups, total_row_groups })
}

#[derive(Debug, Clone)]
enum RowGroupPruningExpr {
    Atom(RowGroupPruningAtom),
    And(Box<RowGroupPruningExpr>, Box<RowGroupPruningExpr>),
    Or(Box<RowGroupPruningExpr>, Box<RowGroupPruningExpr>),
}

#[derive(Debug, Clone)]
enum RowGroupPruningAtom {
    Comparison { column_index: usize, op: Operator, literal: PruningLiteral },
    InList { column_index: usize, literals: Vec<PruningLiteral> },
    IsNull { column_index: usize },
    IsNotNull { column_index: usize },
}

#[derive(Debug, Clone)]
enum PruningLiteral {
    Int64(i64),
    Float64(f64),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
struct PruningColumn {
    column_index: usize,
    data_type: DataType,
}

fn row_group_pruning_expr(expr: &Arc<dyn PhysicalExpr>, file_schema: &SchemaRef) -> Option<RowGroupPruningExpr> {
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        let column = pruning_column(is_null.arg(), file_schema)?;
        return Some(RowGroupPruningExpr::Atom(RowGroupPruningAtom::IsNull { column_index: column.column_index }));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        let column = pruning_column(is_not_null.arg(), file_schema)?;
        return Some(RowGroupPruningExpr::Atom(RowGroupPruningAtom::IsNotNull { column_index: column.column_index }));
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        if in_list.negated() {
            return None;
        }
        let column = pruning_column(in_list.expr(), file_schema)?;
        let mut literals = Vec::new();
        for expr in in_list.list() {
            let literal = expr.downcast_ref::<Literal>()?;
            if literal.value().is_null() {
                continue;
            }
            literals.push(pruning_literal(literal.value(), &column.data_type)?);
        }
        return Some(RowGroupPruningExpr::Atom(RowGroupPruningAtom::InList {
            column_index: column.column_index,
            literals,
        }));
    }

    let binary = expr.downcast_ref::<BinaryExpr>()?;
    match binary.op() {
        Operator::And => return pruning_conjunction(binary.left(), binary.right(), file_schema),
        Operator::Or => return pruning_disjunction(binary.left(), binary.right(), file_schema),
        _ => {},
    }
    if let Some((column, literal, op)) =
        column_literal_predicate(binary.left(), binary.op(), binary.right(), file_schema)
    {
        return Some(RowGroupPruningExpr::Atom(RowGroupPruningAtom::Comparison {
            column_index: column.column_index,
            op,
            literal: pruning_literal(literal.value(), &column.data_type)?,
        }));
    }
    let inverted_op = invert_comparison_operator(binary.op())?;
    let (column, literal, op) = column_literal_predicate(binary.right(), &inverted_op, binary.left(), file_schema)?;
    Some(RowGroupPruningExpr::Atom(RowGroupPruningAtom::Comparison {
        column_index: column.column_index,
        op,
        literal: pruning_literal(literal.value(), &column.data_type)?,
    }))
}

fn pruning_conjunction(
    left: &Arc<dyn PhysicalExpr>,
    right: &Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Option<RowGroupPruningExpr> {
    match (row_group_pruning_expr(left, file_schema), row_group_pruning_expr(right, file_schema)) {
        (Some(left), Some(right)) => Some(RowGroupPruningExpr::And(Box::new(left), Box::new(right))),
        (Some(expr), None) | (None, Some(expr)) => Some(expr),
        (None, None) => None,
    }
}

fn pruning_disjunction(
    left: &Arc<dyn PhysicalExpr>,
    right: &Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Option<RowGroupPruningExpr> {
    let left = row_group_pruning_expr(left, file_schema)?;
    let right = row_group_pruning_expr(right, file_schema)?;
    Some(RowGroupPruningExpr::Or(Box::new(left), Box::new(right)))
}

fn column_literal_predicate<'a>(
    left: &Arc<dyn PhysicalExpr>,
    op: &Operator,
    right: &'a Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Option<(PruningColumn, &'a Literal, Operator)> {
    let column = pruning_column(left, file_schema)?;
    let literal = right.downcast_ref::<Literal>()?;
    if matches!(op, Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq) {
        Some((column, literal, *op))
    } else {
        None
    }
}

fn pruning_column(expr: &Arc<dyn PhysicalExpr>, file_schema: &SchemaRef) -> Option<PruningColumn> {
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let column = pruning_column(cast.expr(), file_schema)?;
        if order_preserving_pruning_cast(&column.data_type, cast.cast_type()) {
            return Some(PruningColumn { data_type: cast.cast_type().clone(), ..column });
        }
        return None;
    }
    let column = expr.downcast_ref::<Column>()?;
    let column_index = file_schema.index_of(column.name()).ok()?;
    let data_type = file_schema.field(column_index).data_type().clone();
    Some(PruningColumn { column_index, data_type })
}

fn order_preserving_pruning_cast(source_type: &DataType, target_type: &DataType) -> bool {
    source_type == target_type
        || signed_integer_rank(source_type)
            .zip(signed_integer_rank(target_type))
            .is_some_and(|(source, target)| source <= target)
}

fn signed_integer_rank(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Int8 => Some(0),
        DataType::Int16 => Some(1),
        DataType::Int32 => Some(2),
        DataType::Int64 => Some(3),
        _ => None,
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

fn pruning_literal(value: &ScalarValue, column_type: &DataType) -> Option<PruningLiteral> {
    match value {
        ScalarValue::Int8(Some(value)) => Some(PruningLiteral::Int64(i64::from(*value))),
        ScalarValue::Int16(Some(value)) => Some(PruningLiteral::Int64(i64::from(*value))),
        ScalarValue::Int32(Some(value)) => Some(PruningLiteral::Int64(i64::from(*value))),
        ScalarValue::Int64(Some(value)) => Some(PruningLiteral::Int64(*value)),
        ScalarValue::Date32(Some(value)) => Some(PruningLiteral::Int64(i64::from(*value))),
        ScalarValue::Date64(Some(value)) => Some(PruningLiteral::Int64(*value)),
        ScalarValue::TimestampSecond(Some(value), None)
        | ScalarValue::TimestampMillisecond(Some(value), None)
        | ScalarValue::TimestampMicrosecond(Some(value), None)
        | ScalarValue::TimestampNanosecond(Some(value), None) => Some(PruningLiteral::Int64(*value)),
        ScalarValue::Float32(Some(value)) => Some(PruningLiteral::Float64(f64::from(*value))),
        ScalarValue::Float64(Some(value)) => Some(PruningLiteral::Float64(*value)),
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value))
            if matches!(column_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) =>
        {
            Some(PruningLiteral::Bytes(value.as_bytes().to_vec()))
        },
        _ => None,
    }
}

fn row_group_may_satisfy_expr(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    schema: &SchemaDescriptor,
    expr: &RowGroupPruningExpr,
) -> Option<bool> {
    match expr {
        RowGroupPruningExpr::Atom(atom) => row_group_may_satisfy_atom(row_group, schema, atom),
        RowGroupPruningExpr::And(left, right) => {
            let left = row_group_may_satisfy_expr(row_group, schema, left);
            let right = row_group_may_satisfy_expr(row_group, schema, right);
            Some(!matches!(left, Some(false)) && !matches!(right, Some(false)))
        },
        RowGroupPruningExpr::Or(left, right) => {
            let left = row_group_may_satisfy_expr(row_group, schema, left);
            let right = row_group_may_satisfy_expr(row_group, schema, right);
            Some(!matches!((left, right), (Some(false), Some(false))))
        },
    }
}

fn row_group_may_satisfy_atom(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    schema: &SchemaDescriptor,
    atom: &RowGroupPruningAtom,
) -> Option<bool> {
    match atom {
        RowGroupPruningAtom::Comparison { column_index, op, literal } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            row_group_may_satisfy_comparison(statistics, *op, literal)
        },
        RowGroupPruningAtom::InList { column_index, literals } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            row_group_may_satisfy_in_list(statistics, literals)
        },
        RowGroupPruningAtom::IsNull { column_index } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            Some(statistics.null_count_opt()? > 0)
        },
        RowGroupPruningAtom::IsNotNull { column_index } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            let row_count = u64::try_from(row_group.num_rows()).ok()?;
            Some(statistics.null_count_opt()? < row_count)
        },
    }
}

fn row_group_may_satisfy_in_list(statistics: &Statistics, literals: &[PruningLiteral]) -> Option<bool> {
    if literals.is_empty() {
        return Some(false);
    }
    let mut saw_unknown = false;
    for literal in literals {
        match row_group_may_satisfy_comparison(statistics, Operator::Eq, literal) {
            Some(true) => return Some(true),
            Some(false) => {},
            None => saw_unknown = true,
        }
    }
    Some(saw_unknown)
}

fn row_group_may_satisfy_comparison(statistics: &Statistics, op: Operator, literal: &PruningLiteral) -> Option<bool> {
    match statistics {
        Statistics::Int32(stats) => {
            let min = i64::from(*stats.min_opt()?);
            let max = i64::from(*stats.max_opt()?);
            let PruningLiteral::Int64(literal) = literal else {
                return None;
            };
            Some(int_range_may_satisfy(min, max, op, *literal))
        },
        Statistics::Int64(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let PruningLiteral::Int64(literal) = literal else {
                return None;
            };
            Some(int_range_may_satisfy(min, max, op, *literal))
        },
        Statistics::Float(stats) => {
            let min = f64::from(*stats.min_opt()?);
            let max = f64::from(*stats.max_opt()?);
            let PruningLiteral::Float64(literal) = literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, op, *literal))
        },
        Statistics::Double(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let PruningLiteral::Float64(literal) = literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, op, *literal))
        },
        Statistics::ByteArray(stats) => {
            let min = stats.min_opt()?.data();
            let max = stats.max_opt()?.data();
            let PruningLiteral::Bytes(literal) = literal else {
                return None;
            };
            Some(bytes_range_may_satisfy(min, max, op, literal))
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

fn bytes_range_may_satisfy(min: &[u8], max: &[u8], op: Operator, literal: &[u8]) -> bool {
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
