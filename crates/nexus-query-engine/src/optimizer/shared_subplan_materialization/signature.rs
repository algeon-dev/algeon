#[derive(Debug, Clone)]
struct AggregateMaterializationSignature {
    aggregate_node_id: NodeId,
    post_aggregate_filter_columns_by_source: BTreeMap<String, BTreeSet<String>>,
    family_key: String,
    group_exprs: Vec<String>,
    aggregate_exprs: Vec<String>,
    aggregate_expr_specs: Vec<AggregateExpr>,
    source_fingerprints: Vec<String>,
    source_occurrences_unique: bool,
    source_predicates: Vec<SourcePredicateSignature>,
    source_columns: BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
    exact_source_signature: String,
    source_bytes: Option<u64>,
    source_rows: Option<u64>,
}

fn aggregate_materialization_signature(
    plan: &QueryPlan,
    aggregate_node_id: NodeId,
    input: NodeId,
    spec: &AggregateSpec,
) -> Option<AggregateMaterializationSignature> {
    let mut sources = Vec::new();
    collect_local_parquet_sources(plan, input, &mut BTreeSet::new(), &mut sources);
    if sources.is_empty() {
        return None;
    }
    sources.sort_by(|left, right| left.abstract_fingerprint.cmp(&right.abstract_fingerprint));

    let scalar_filtered_family = spec.group_exprs.is_empty() && scalar_filtered_aggregate_supported(spec);
    let source_occurrences_unique = if scalar_filtered_family {
        scalar_source_occurrences_are_unique(plan, input)?
    } else {
        local_parquet_source_node_occurrences_are_unique(plan, input)?
    };
    let source_fingerprints = sources
        .iter()
        .map(|source| {
            if scalar_filtered_family { source.rowset_fingerprint.clone() } else { source.abstract_fingerprint.clone() }
        })
        .collect::<Vec<_>>();
    let exact_source_signature = sources
        .iter()
        .map(|source| {
            if scalar_filtered_family {
                source.exact_rowset_fingerprint.as_str()
            } else {
                source.exact_fingerprint.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join("|");
    let source_bytes = sum_optional_u64(sources.iter().map(|source| source.selected_source_bytes));
    let source_rows = sum_optional_u64(sources.iter().map(|source| source.selected_rows));
    let group_exprs = spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>();
    let aggregate_exprs = spec.aggr_exprs.iter().map(aggregate_expr_signature).collect::<Vec<_>>();
    let post_aggregate_filter_columns_by_source = post_aggregate_filterable_group_columns_by_source(
        plan,
        input,
        plan.node(aggregate_node_id)?.output_schema(),
        spec,
    );
    let source_predicates = sources
        .iter()
        .map(|source| SourcePredicateSignature {
            abstract_fingerprint: if scalar_filtered_family {
                source.rowset_fingerprint.clone()
            } else {
                source.abstract_fingerprint.clone()
            },
            predicate: source.predicate.clone(),
        })
        .collect::<Vec<_>>();
    let mut source_columns = BTreeMap::<String, BTreeMap<String, SourceColumnMaterialization>>::new();
    for source in &sources {
        let fingerprint = if scalar_filtered_family {
            source.rowset_fingerprint.clone()
        } else {
            source.abstract_fingerprint.clone()
        };
        source_columns.entry(fingerprint).or_default().extend(source.columns.clone());
    }
    let scalar_count_star_family = scalar_filtered_family
        && spec.aggr_exprs.iter().all(|expr| {
            expr.function == AggregateFunction::Count
                && !expr.distinct
                && scalar_count_star_aggregate_expr(plan, input, &expr.expr)
        });
    let allow_column_projection_elision =
        scalar_count_star_family && spec.aggr_exprs.iter().any(|expr| !scalar_count_star_expr(&expr.expr));
    let scalar_input_shape = if scalar_filtered_family {
        Some(scalar_materialization_input_shape(plan, input, allow_column_projection_elision)?)
    } else {
        None
    };
    let family_signature = if scalar_filtered_family {
        format!(
            "scalar_filtered;input_shape={};sources={}",
            scalar_input_shape.as_deref().unwrap_or(""),
            source_fingerprints.join("|")
        )
    } else {
        format!(
            "groups={};aggr={};sources={}",
            group_exprs.join(","),
            aggregate_exprs.join(","),
            source_fingerprints.join("|")
        )
    };
    let family_key = format!("shared-subplan-v1-fnv64-{:016x}", stable_fnv1a64(family_signature.as_bytes()));

    Some(AggregateMaterializationSignature {
        aggregate_node_id,
        post_aggregate_filter_columns_by_source,
        family_key,
        group_exprs,
        aggregate_exprs,
        aggregate_expr_specs: spec.aggr_exprs.clone(),
        source_fingerprints,
        source_occurrences_unique,
        source_predicates,
        source_columns,
        exact_source_signature,
        source_bytes,
        source_rows,
    })
}

fn post_aggregate_filterable_group_columns_by_source(
    plan: &QueryPlan,
    input: NodeId,
    output_schema: &Arc<Schema>,
    spec: &AggregateSpec,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut columns = BTreeMap::<String, BTreeSet<String>>::new();
    for (expr, field) in spec.group_exprs.iter().zip(output_schema.fields().iter()) {
        let Some(column) = simple_group_column_name(expr) else {
            continue;
        };
        if field.name() != column {
            continue;
        }
        let Some(lineage) = materialized_column_lineage(plan, input, column) else {
            continue;
        };
        if lineage.source_column == column {
            columns.entry(lineage.source_fingerprint).or_default().insert(column.to_string());
        }
    }
    columns
}

fn simple_group_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.name()),
        Expr::Alias { expr, alias } => {
            let Expr::Column(column) = expr.as_ref() else {
                return None;
            };
            (alias == column.name()).then_some(alias.as_str())
        },
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct MaterializedColumnLineage {
    source_fingerprint: String,
    source_column: String,
}

#[derive(Debug, Clone)]
struct LocalSourceMaterializationSignature {
    rowset_fingerprint: String,
    exact_rowset_fingerprint: String,
    abstract_fingerprint: String,
    exact_fingerprint: String,
    predicate: Option<Expr>,
    selected_source_bytes: Option<u64>,
    selected_rows: Option<u64>,
    columns: BTreeMap<String, SourceColumnMaterialization>,
}

#[derive(Debug, Clone)]
struct SourcePredicateSignature {
    abstract_fingerprint: String,
    predicate: Option<Expr>,
}

#[derive(Debug, Clone)]
struct SourceColumnMaterialization {
    name: String,
    file_index: usize,
    source_ref: SourceColumnRef,
    field: Field,
}

fn collect_local_parquet_sources(
    plan: &QueryPlan,
    node_id: NodeId,
    visited: &mut BTreeSet<NodeId>,
    sources: &mut Vec<LocalSourceMaterializationSignature>,
) {
    if !visited.insert(node_id) {
        return;
    }
    let Some(node) = plan.node(node_id) else {
        return;
    };
    match &node.kind {
        PlanNode::Source(source) => {
            if let SourceNode::LocalParquet(source) = source.as_ref() {
                sources.push(local_source_materialization_signature(source));
            }
        },
        _ => {
            for input in node.input_ids() {
                collect_local_parquet_sources(plan, *input, visited, sources);
            }
        },
    }
}

fn local_source_materialization_signature(source: &LocalParquetSource) -> LocalSourceMaterializationSignature {
    let mut file_signatures = source.files.iter().map(file_signature).collect::<Vec<_>>();
    file_signatures.sort();
    let files = file_signatures.join("+");
    let read_columns = source.read_columns.join(",");
    let projected_output_columns = source.projected_output_columns.join(",");
    let pure_filter_columns = source.pure_filter_columns.join(",");
    let partition_columns = source
        .partition_columns
        .iter()
        .map(|column| format!("{}:{:?}:nullable={}", column.name, column.data_type, column.nullable))
        .collect::<Vec<_>>()
        .join(",");
    let missing_columns = source
        .missing_columns
        .iter()
        .map(|column| format!("{}:{:?}:nullable={}", column.name, column.data_type, column.nullable))
        .collect::<Vec<_>>()
        .join(",");
    let rowset_common = format!(
        "files={files};partition={partition_columns};missing={missing_columns};scan_limit={};max_rows={}",
        source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        source.max_rows.map_or_else(|| "none".to_string(), |rows| rows.to_string())
    );
    let common = format!(
        "{rowset_common};read={read_columns};projected={projected_output_columns};pure_filter={pure_filter_columns}"
    );
    let predicate = source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string);
    let selected_source_bytes =
        source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_source_bytes());
    let selected_rows = source
        .diagnostics
        .selected_row_group_stats
        .as_ref()
        .and_then(|stats| stats.selected_rows)
        .or_else(|| source.max_rows.and_then(|rows| u64::try_from(rows).ok()));
    let columns = source
        .read_columns
        .iter()
        .zip(source.read_column_file_indices.iter())
        .filter_map(|(column, file_index)| {
            let field = source.schema.field_with_name(column).ok()?.as_ref().clone();
            let source_ref = source.source_column_ref(column)?;
            Some((
                column.clone(),
                SourceColumnMaterialization { name: column.clone(), file_index: *file_index, source_ref, field },
            ))
        })
        .collect();

    LocalSourceMaterializationSignature {
        rowset_fingerprint: format!("{rowset_common};predicate=*"),
        exact_rowset_fingerprint: format!("{rowset_common};predicate={predicate}"),
        abstract_fingerprint: format!("{common};predicate=*"),
        exact_fingerprint: format!("{common};predicate={predicate}"),
        predicate: source.reader_predicate.clone(),
        selected_source_bytes,
        selected_rows,
        columns,
    }
}

fn source_column_ref_signature(source_ref: SourceColumnRef) -> String {
    match source_ref {
        SourceColumnRef::File(index) => format!("file:{index}"),
        SourceColumnRef::Partition(index) => format!("partition:{index}"),
        SourceColumnRef::MissingNull(index) => format!("missing_null:{index}"),
    }
}

fn file_signature(file: &crate::source::LocalParquetFile) -> String {
    let partition_values = file.partition_values.iter().map(|value| format!("{value:?}")).collect::<Vec<_>>().join(",");
    format!(
        "stable_id={};location={};resolved_contract={};reader_decimal_contract={};row_groups={};partition_values={}",
        file.location.diagnostic_stable_id(file.stable_id.as_str()),
        location_signature(&file.location),
        resolved_source_contract_signature(&file.resolved_source),
        reader_decimal_contract_signature(&file.reader_decimal_types),
        file.row_groups
            .as_ref()
            .map(|row_groups| row_groups.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(","))
            .unwrap_or_else(|| "all".to_string()),
        partition_values
    )
}

fn location_signature(location: &ParquetSourceLocation) -> String {
    match location {
        ParquetSourceLocation::LocalPath(path) => format!("local:{}", path.display()),
        ParquetSourceLocation::ObjectStoreUri { .. } => location.diagnostic_location(),
    }
}

fn reader_decimal_contract_signature(reader_decimal_types: &[crate::source::ParquetReaderDecimalType]) -> String {
    let mut output = String::new();
    push_contract_value(&mut output, "reader_decimal_types.len", reader_decimal_types.len().to_string().as_str());
    for (index, decimal_type) in reader_decimal_types.iter().enumerate() {
        push_contract_value(
            &mut output,
            format!("reader_decimal_types.{index}.column_name").as_str(),
            decimal_type.column_name.as_str(),
        );
        push_contract_value(
            &mut output,
            format!("reader_decimal_types.{index}.type_id").as_str(),
            decimal_type.type_id.as_str(),
        );
        push_contract_value(
            &mut output,
            format!("reader_decimal_types.{index}.cudf_scale").as_str(),
            decimal_type.cudf_scale.to_string().as_str(),
        );
    }
    output
}

fn resolved_source_contract_signature(resolved_source: &ResolvedParquetSource) -> String {
    let mut output = String::new();
    push_contract_hash(&mut output, "original_location", Some(resolved_source.original_location.as_str()));
    push_contract_hash(&mut output, "reader_location", Some(resolved_source.reader_location.as_str()));
    push_contract_hash(&mut output, "storage_domain", resolved_source.storage_domain.as_deref());
    push_contract_hash(&mut output, "content_identity", resolved_source.content_identity.as_deref());
    push_contract_value(
        &mut output,
        "storage_profile_id",
        resolved_source.storage_profile_id.as_deref().unwrap_or("none"),
    );
    push_contract_value(&mut output, "read_strategy", resolved_source.read_strategy.as_str());
    push_contract_value(&mut output, "decode_input", resolved_source.decode_input.as_str());
    push_contract_value(&mut output, "remote_read_mode", resolved_source.remote_read_mode.as_str());
    push_contract_value(&mut output, "credential_mode", resolved_source.credential_mode.as_str());
    push_contract_value(&mut output, "cache_status", resolved_source.cache_status.as_str());
    push_contract_value(&mut output, "lakehouse_cache_read_mode", resolved_source.lakehouse_cache_read_mode.as_str());
    push_contract_hash(&mut output, "lakehouse_cache_root", resolved_source.lakehouse_cache_root.as_deref());
    push_contract_hash(&mut output, "lakehouse_cache_path", resolved_source.lakehouse_cache_path.as_deref());
    push_contract_value(
        &mut output,
        "lakehouse_cache_max_bytes",
        resolved_source
            .lakehouse_cache_max_bytes
            .map_or_else(|| "none".to_string(), |bytes| bytes.to_string())
            .as_str(),
    );
    push_contract_value(&mut output, "lakehouse_cache_fill_mode", resolved_source.lakehouse_cache_fill_mode.as_str());
    push_contract_value(
        &mut output,
        "remote_host_buffer_ranged_enabled",
        if resolved_source.remote_host_buffer_ranged_enabled { "true" } else { "false" },
    );
    push_contract_value(&mut output, "fallback_reason", resolved_source.fallback_reason.as_deref().unwrap_or("none"));
    output
}

fn push_contract_hash(output: &mut String, key: &str, value: Option<&str>) {
    match value {
        Some(value) => push_contract_value(
            output,
            key,
            format!("fnv64:{:016x}:len:{}", stable_fnv1a64(value.as_bytes()), value.len()).as_str(),
        ),
        None => push_contract_value(output, key, "none"),
    }
}

fn push_contract_value(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push('=');
    output.push_str(value);
    output.push(';');
}

fn aggregate_expr_signature(expr: &AggregateExpr) -> String {
    format!(
        "fn={};distinct={};arg={};filter={};alias={}",
        aggregate_function_signature(expr.function),
        expr.distinct,
        expr.expr,
        expr.filter.as_ref().map_or_else(|| "none".to_string(), ToString::to_string),
        expr.alias
    )
}

fn aggregate_function_signature(function: AggregateFunction) -> &'static str {
    function.as_str()
}

fn sum_optional_u64(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    let mut sum = 0_u64;
    for value in values {
        sum = sum.checked_add(value?)?;
    }
    Some(sum)
}

fn stable_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
