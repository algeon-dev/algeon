fn matching_source_entry<'a, T>(
    entries: &'a BTreeMap<String, T>,
    signature: &LocalSourceMaterializationSignature,
) -> Option<&'a T> {
    entries.get(&signature.abstract_fingerprint).or_else(|| entries.get(&signature.rowset_fingerprint))
}

fn merge_source_columns(
    target: &mut BTreeMap<String, SourceColumnMaterialization>,
    source: &BTreeMap<String, SourceColumnMaterialization>,
) -> Result<(), SharedSubplanMaterializationRejectedReason> {
    for (name, column) in source {
        if let Some(existing) = target.get(name) {
            if existing.file_index != column.file_index
                || existing.source_ref != column.source_ref
                || existing.field.name() != column.field.name()
                || existing.field.data_type() != column.field.data_type()
                || existing.field.is_nullable() != column.field.is_nullable()
            {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
            }
        } else {
            target.insert(name.clone(), column.clone());
        }
    }
    Ok(())
}

fn source_column_names_are_unique_across_sources(
    sources: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> bool {
    let mut owners = BTreeMap::<&str, &str>::new();
    for (source, columns) in sources {
        for name in columns.keys() {
            if let Some(existing) = owners.insert(name.as_str(), source.as_str())
                && existing != source
            {
                return false;
            }
        }
    }
    true
}

fn source_materialization_sources_are_unique(signature: &AggregateMaterializationSignature) -> bool {
    signature.source_occurrences_unique && source_fingerprints_are_unique(signature)
}

fn source_fingerprints_are_unique(signature: &AggregateMaterializationSignature) -> bool {
    let mut seen = BTreeSet::new();
    signature.source_fingerprints.iter().all(|fingerprint| seen.insert(fingerprint))
}

fn local_parquet_source_node_occurrences_are_unique(plan: &QueryPlan, node_id: NodeId) -> Option<bool> {
    let mut source_nodes = BTreeSet::new();
    let mut active = BTreeSet::new();
    local_parquet_source_node_occurrences_are_unique_inner(plan, node_id, &mut active, &mut source_nodes)
}

fn local_parquet_source_node_occurrences_are_unique_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    active: &mut BTreeSet<NodeId>,
    source_nodes: &mut BTreeSet<NodeId>,
) -> Option<bool> {
    if !active.insert(node_id) {
        return Some(false);
    }
    let node = plan.node(node_id)?;
    let unique = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(_) => source_nodes.insert(node_id),
            _ => true,
        },
        _ => {
            let mut unique = true;
            for input in node.input_ids() {
                unique &= local_parquet_source_node_occurrences_are_unique_inner(plan, *input, active, source_nodes)?;
            }
            unique
        },
    };
    active.remove(&node_id);
    Some(unique)
}

fn scalar_source_occurrences_are_unique(plan: &QueryPlan, node_id: NodeId) -> Option<bool> {
    let mut source_nodes = BTreeSet::new();
    let mut rowsets = BTreeSet::new();
    let mut active = BTreeSet::new();
    scalar_source_occurrences_are_unique_inner(plan, node_id, &mut active, &mut source_nodes, &mut rowsets)
}

fn scalar_source_occurrences_are_unique_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    active: &mut BTreeSet<NodeId>,
    source_nodes: &mut BTreeSet<NodeId>,
    rowsets: &mut BTreeSet<String>,
) -> Option<bool> {
    if !active.insert(node_id) {
        return Some(false);
    }
    let node = plan.node(node_id)?;
    let unique = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                source_nodes.insert(node_id)
                    && rowsets.insert(local_source_materialization_signature(source).rowset_fingerprint)
            },
            _ => false,
        },
        _ => {
            let mut unique = true;
            for input in node.input_ids() {
                unique &= scalar_source_occurrences_are_unique_inner(plan, *input, active, source_nodes, rowsets)?;
            }
            unique
        },
    };
    active.remove(&node_id);
    Some(unique)
}

fn widen_local_parquet_source(
    source: &mut LocalParquetSource,
    columns: &BTreeMap<String, SourceColumnMaterialization>,
) -> bool {
    let mut changed = false;
    for column in columns.values() {
        if source.read_columns.iter().any(|existing| existing == &column.name) {
            continue;
        }
        source.read_columns.push(column.name.clone());
        source.read_column_file_indices.push(column.file_index);
        changed = true;
    }
    if !changed {
        return false;
    }

    let mut fields = source.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut field_names = fields.iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    for column in columns.values() {
        if field_names.insert(column.name.clone()) {
            fields.push(column.field.clone());
        }
    }
    source.schema = Arc::new(Schema::new(fields));
    source.projected_output_columns = source.read_columns.clone();
    source.projected_output_column_source_refs =
        source.read_column_file_indices.iter().copied().map(crate::source::SourceColumnRef::File).collect();
    source.pure_filter_columns.clear();
    source.pure_filter_column_source_refs.clear();
    source.refresh_diagnostics();
    true
}
