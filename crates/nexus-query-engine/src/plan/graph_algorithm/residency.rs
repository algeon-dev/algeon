//! Resident graph identity, fingerprinting, and planning analysis.

use super::*;

/// Concrete graph orientation used for resident graph cache keys.
///
/// The cache key is intentionally device-agnostic; residency stores are scoped
/// per device and may index the same key on multiple devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphResidencyOrientation {
    /// Forward cuGraph graph layout.
    Forward,
    /// Transposed cuGraph graph layout.
    Transposed,
}

impl GraphResidencyOrientation {
    /// Stable lowercase orientation code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Transposed => "transposed",
        }
    }
}

impl fmt::Display for GraphResidencyOrientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Diagnostic-safe fingerprint for a graph edge source rowset.
///
/// The private fingerprint contains source identities and rowset-affecting scan
/// contracts. [`Self::as_str`] returns a redacted stable hash for reports and
/// tests.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GraphSourceFingerprint {
    fingerprint: String,
    diagnostic_fingerprint: String,
}

impl std::fmt::Debug for GraphSourceFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphSourceFingerprint").field("fingerprint", &self.diagnostic_fingerprint).finish()
    }
}

impl GraphSourceFingerprint {
    fn new(fingerprint: String) -> Self {
        let mut hasher = crate::harness::StableFnv1a64::new();
        hasher.update_field(&fingerprint);
        let diagnostic_fingerprint = format!("graph-source-v1-fnv64-{:016x}", hasher.finish());
        Self { fingerprint, diagnostic_fingerprint }
    }

    /// Redacted textual fingerprint for diagnostics and tests.
    pub fn as_str(&self) -> &str {
        &self.diagnostic_fingerprint
    }
}

/// Reason a graph edge source cannot produce a safe graph-residency key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphSourceFingerprintBypassReason {
    /// The graph input is not backed by a local/table-format Parquet source.
    NonLocalParquetSource,
    /// A non-column projection or row-changing operator sits between the source
    /// and the graph edge input.
    UnsupportedTransformedInput,
    /// A required graph edge column could not be resolved to a source column.
    MissingEdgeColumn,
    /// Source and destination edge columns exist but do not share a vertex type.
    EdgeVertexDataTypeMismatch,
    /// Per-file field-id remapping can change the physical rowset behind a
    /// logical edge column; v1 bypasses until that contract is fingerprinted.
    FieldIdReadBindingsPresent,
    /// Iceberg/table-format delete files are present; v1 bypasses until delete
    /// identity is modeled for graph rowsets.
    DeleteFilesPresent,
    /// The source lacks a stable content identity strong enough for a
    /// process-lifetime graph cache.
    UnstableSourceIdentity,
}

impl GraphSourceFingerprintBypassReason {
    /// Stable lowercase bypass code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonLocalParquetSource => "non_local_parquet_source",
            Self::UnsupportedTransformedInput => "unsupported_transformed_input",
            Self::MissingEdgeColumn => "missing_edge_column",
            Self::EdgeVertexDataTypeMismatch => "edge_vertex_data_type_mismatch",
            Self::FieldIdReadBindingsPresent => "field_id_read_bindings_present",
            Self::DeleteFilesPresent => "delete_files_present",
            Self::UnstableSourceIdentity => "unstable_source_identity",
        }
    }
}

impl fmt::Display for GraphSourceFingerprintBypassReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of graph edge-source fingerprint analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphResidencyPlanAnalysis {
    /// A safe graph-residency descriptor was derived.
    Keyable(GraphResidencyDescriptor),
    /// The source must bypass graph residency.
    Bypass(GraphSourceFingerprintBypassReason),
}

impl GraphResidencyPlanAnalysis {
    /// Returns the descriptor when the plan is keyable.
    pub fn descriptor(&self) -> Option<&GraphResidencyDescriptor> {
        match self {
            Self::Keyable(descriptor) => Some(descriptor),
            Self::Bypass(_) => None,
        }
    }

    /// Returns the bypass reason when the plan is not keyable.
    pub fn bypass_reason(&self) -> Option<GraphSourceFingerprintBypassReason> {
        match self {
            Self::Keyable(_) => None,
            Self::Bypass(reason) => Some(*reason),
        }
    }
}

/// Source-side graph residency facts carried with GPU-resident edge frames.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphResidencyDescriptor {
    /// Stable source rowset fingerprint.
    pub source_fingerprint: GraphSourceFingerprint,
    /// Arrow dtype of the graph vertex columns.
    pub vertex_data_type: arrow::datatypes::DataType,
    /// Arrow dtype of the semantic weight column when the algorithm consumes
    /// weights.
    pub weight_data_type: Option<arrow::datatypes::DataType>,
    /// Arrow dtype of the edge-id column when present.
    pub edge_id_data_type: Option<arrow::datatypes::DataType>,
}

impl GraphResidencyDescriptor {
    /// Creates a descriptor for a materialized edge rowset with a caller-owned,
    /// immutable content identity.
    ///
    /// The identity must change whenever the edge rows or any value consumed by
    /// graph construction changes. It is retained only in the private cache
    /// key; diagnostics expose its redacted fingerprint.
    pub fn for_content_identity(
        content_identity: impl AsRef<str>,
        vertex_data_type: arrow::datatypes::DataType,
        weight_data_type: Option<arrow::datatypes::DataType>,
        edge_id_data_type: Option<arrow::datatypes::DataType>,
    ) -> crate::error::Result<Self> {
        let content_identity = content_identity.as_ref().trim();
        if content_identity.is_empty() {
            return Err(crate::error::QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.plan.graph_residency_content_identity",
                "graph residency requires a non-empty immutable content identity",
            )
            .with_fact("reason", "missing_content_identity"));
        }
        Ok(Self {
            source_fingerprint: GraphSourceFingerprint::new(format!(
                "mode=external_content_identity_graph_edges_v1;content_identity={content_identity}"
            )),
            vertex_data_type,
            weight_data_type,
            edge_id_data_type,
        })
    }

    /// Builds a resident graph key for one graph orientation.
    pub fn resident_key(&self, spec: &GraphAlgorithmSpec, orientation: GraphResidencyOrientation) -> ResidentGraphKey {
        ResidentGraphKey {
            source_fingerprint: self.source_fingerprint.clone(),
            source_column: spec.edge_columns.source.clone(),
            destination_column: spec.edge_columns.destination.clone(),
            weight_column: spec.algorithm.consumes_weight().then(|| spec.edge_columns.weight.clone()).flatten(),
            edge_id_column: spec.edge_columns.edge_id.clone(),
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            edge_preparation: GraphEdgePreparationKey::for_spec(spec),
            orientation,
            vertex_data_type: self.vertex_data_type.clone(),
            weight_data_type: self.weight_data_type.clone(),
            edge_id_data_type: self.edge_id_data_type.clone(),
        }
    }
}

/// Device-agnostic resident graph cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidentGraphKey {
    /// Stable source rowset fingerprint.
    pub source_fingerprint: GraphSourceFingerprint,
    /// Source-vertex binding at the graph operator boundary.
    pub source_column: String,
    /// Destination-vertex binding at the graph operator boundary.
    pub destination_column: String,
    /// Semantic weight binding only when the algorithm consumes weights.
    pub weight_column: Option<String>,
    /// Stable edge-id binding, when present.
    pub edge_id_column: Option<String>,
    /// Whether graph construction treats edges as directed.
    pub directed: bool,
    /// Whether cuGraph may renumber vertices internally.
    pub renumber: bool,
    /// Edge preparation performed before raw cuGraph construction.
    pub edge_preparation: GraphEdgePreparationKey,
    /// Concrete cuGraph graph orientation.
    pub orientation: GraphResidencyOrientation,
    /// Arrow dtype of the graph vertex columns.
    pub vertex_data_type: arrow::datatypes::DataType,
    /// Arrow dtype of the semantic weight column when consumed.
    pub weight_data_type: Option<arrow::datatypes::DataType>,
    /// Arrow dtype of the edge-id column when present.
    pub edge_id_data_type: Option<arrow::datatypes::DataType>,
}

/// Edge preparation contract folded into resident graph keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphEdgePreparationKey {
    /// The graph consumes the edge rowset as-is.
    None,
    /// The builder inserted Python-compatible edge normalization.
    PythonCompatible {
        /// Whether reverse edges are inserted before deduplication.
        add_reverse: bool,
        /// Duplicate-edge retention policy.
        dedup: EdgeDedupStrategy,
        /// Duplicate-weight conflict policy.
        weight_combiner: EdgeWeightCombiner,
        /// Source-vertex column consumed by normalization.
        source_column: String,
        /// Destination-vertex column consumed by normalization.
        destination_column: String,
        /// Weight column consumed by normalization, when present.
        weight_column: Option<String>,
        /// Edge-id column carried by normalization, when present.
        edge_id_column: Option<String>,
    },
}

impl GraphEdgePreparationKey {
    fn for_spec(spec: &GraphAlgorithmSpec) -> Self {
        if !spec.edge_normalize_inserted {
            return Self::None;
        }
        let normalize = EdgeNormalizeSpec::python_cugraph(spec.edge_columns.clone(), &spec.graph_options);
        Self::PythonCompatible {
            add_reverse: normalize.add_reverse,
            dedup: normalize.dedup,
            weight_combiner: normalize.weight_combiner,
            source_column: normalize.edge_columns.source,
            destination_column: normalize.edge_columns.destination,
            weight_column: normalize.edge_columns.weight,
            edge_id_column: normalize.edge_columns.edge_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EffectiveGraphEdgeBindings {
    source: String,
    destination: String,
    weight: Option<String>,
    edge_id: Option<String>,
}

impl EffectiveGraphEdgeBindings {
    fn for_spec(spec: &GraphAlgorithmSpec) -> Self {
        let weight_is_semantic = spec.algorithm.consumes_weight() || spec.edge_normalize_inserted;
        Self {
            source: spec.edge_columns.source.clone(),
            destination: spec.edge_columns.destination.clone(),
            weight: weight_is_semantic.then(|| spec.edge_columns.weight.clone()).flatten(),
            edge_id: spec.edge_columns.edge_id.clone(),
        }
    }

    fn columns(&self) -> impl Iterator<Item = (&'static str, &str)> {
        std::iter::once(("source", self.source.as_str()))
            .chain(std::iter::once(("destination", self.destination.as_str())))
            .chain(self.weight.as_deref().map(|column| ("weight", column)))
            .chain(self.edge_id.as_deref().map(|column| ("edge_id", column)))
    }

    fn try_map_columns(
        &self,
        mut map: impl FnMut(&str) -> Option<String>,
    ) -> std::result::Result<Self, GraphSourceFingerprintBypassReason> {
        let weight = match self.weight.as_deref() {
            Some(column) => Some(map(column).ok_or(GraphSourceFingerprintBypassReason::MissingEdgeColumn)?),
            None => None,
        };
        let edge_id = match self.edge_id.as_deref() {
            Some(column) => Some(map(column).ok_or(GraphSourceFingerprintBypassReason::MissingEdgeColumn)?),
            None => None,
        };
        Ok(Self {
            source: map(&self.source).ok_or(GraphSourceFingerprintBypassReason::MissingEdgeColumn)?,
            destination: map(&self.destination).ok_or(GraphSourceFingerprintBypassReason::MissingEdgeColumn)?,
            weight,
            edge_id,
        })
    }
}

/// Derive graph residency facts from an edge-source native plan.
///
/// This helper intentionally analyzes the plan that produced the GPU-resident
/// edge frame, not the graph plan that later consumes that frame.
pub fn graph_residency_descriptor_for_edge_plan(
    plan: &QueryPlan,
    spec: &GraphAlgorithmSpec,
) -> GraphResidencyPlanAnalysis {
    let bindings = EffectiveGraphEdgeBindings::for_spec(spec);
    match graph_residency_descriptor_for_edge_node(plan, plan.root_id(), &bindings) {
        Ok(descriptor) => GraphResidencyPlanAnalysis::Keyable(descriptor),
        Err(reason) => GraphResidencyPlanAnalysis::Bypass(reason),
    }
}

fn graph_residency_descriptor_for_edge_node(
    plan: &QueryPlan,
    node_id: NodeId,
    bindings: &EffectiveGraphEdgeBindings,
) -> std::result::Result<GraphResidencyDescriptor, GraphSourceFingerprintBypassReason> {
    let node = plan.node(node_id).ok_or(GraphSourceFingerprintBypassReason::UnsupportedTransformedInput)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => {
                graph_residency_descriptor_for_local_parquet_source(source, bindings)
            },
            crate::source::SourceNode::GpuResident(_) | crate::source::SourceNode::LiteralBatch(_) => {
                Err(GraphSourceFingerprintBypassReason::NonLocalParquetSource)
            },
        },
        PlanNode::Projection(projection) => {
            let mapped = bindings.try_map_columns(|column| projection_source_column(projection, column))?;
            graph_residency_descriptor_for_edge_node(plan, projection.input, &mapped)
        },
        PlanNode::Filter(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Join(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => Err(GraphSourceFingerprintBypassReason::UnsupportedTransformedInput),
    }
}

fn projection_source_column(projection: &ProjectionNode, output_column: &str) -> Option<String> {
    let output = projection.outputs.iter().find(|output| output.field.name() == output_column)?;
    match &output.expr {
        Expr::Column(column) => Some(column.name().to_string()),
        Expr::Alias { expr, alias } if alias == output_column => {
            let Expr::Column(column) = expr.as_ref() else {
                return None;
            };
            Some(column.name().to_string())
        },
        _ => None,
    }
}

fn graph_residency_descriptor_for_local_parquet_source(
    source: &crate::source::LocalParquetSource,
    bindings: &EffectiveGraphEdgeBindings,
) -> std::result::Result<GraphResidencyDescriptor, GraphSourceFingerprintBypassReason> {
    if source.scan_limit.is_some() {
        return Err(GraphSourceFingerprintBypassReason::UnsupportedTransformedInput);
    }
    if source.files.iter().any(|file| !file.position_delete_files.is_empty() || !file.equality_delete_files.is_empty())
    {
        return Err(GraphSourceFingerprintBypassReason::DeleteFilesPresent);
    }
    if source.files.iter().any(|file| !file.field_id_read_bindings.is_empty()) {
        return Err(GraphSourceFingerprintBypassReason::FieldIdReadBindingsPresent);
    }
    if source.files.iter().any(|file| !local_parquet_file_identity_is_graph_stable(file)) {
        return Err(GraphSourceFingerprintBypassReason::UnstableSourceIdentity);
    }

    let source_field = graph_source_field(source, &bindings.source)?;
    let destination_field = graph_source_field(source, &bindings.destination)?;
    if source_field.data_type() != destination_field.data_type() {
        return Err(GraphSourceFingerprintBypassReason::EdgeVertexDataTypeMismatch);
    }
    let weight_field = bindings.weight.as_deref().map(|column| graph_source_field(source, column)).transpose()?;
    let edge_id_field = bindings.edge_id.as_deref().map(|column| graph_source_field(source, column)).transpose()?;

    let mut fingerprint = String::new();
    push_graph_fingerprint_token(&mut fingerprint, "mode", "local_parquet_graph_edges_v1");
    push_graph_fingerprint_token(
        &mut fingerprint,
        "reader_predicate",
        source
            .reader_predicate
            .as_ref()
            .map_or_else(|| "none".to_string(), |predicate| format!("display={predicate};debug={predicate:?}")),
    );
    push_graph_fingerprint_token(
        &mut fingerprint,
        "scan_limit",
        source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
    );
    push_graph_fingerprint_token(&mut fingerprint, "file_count", source.files.len().to_string());
    for (index, file) in source.files.iter().enumerate() {
        push_graph_fingerprint_token(&mut fingerprint, format!("file.{index}.stable_id"), &file.stable_id);
        push_graph_fingerprint_token(
            &mut fingerprint,
            format!("file.{index}.content_identity"),
            file.resolved_source.content_identity.as_deref().unwrap_or("none"),
        );
        push_graph_fingerprint_token(
            &mut fingerprint,
            format!("file.{index}.size"),
            file.file_size_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
        );
        push_graph_optional_usize_vec(&mut fingerprint, format!("file.{index}.row_groups"), file.row_groups.as_deref());
        let partition_values = file.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
        push_graph_string_vec(&mut fingerprint, format!("file.{index}.partition_values"), &partition_values);
    }
    for (role, column) in bindings.columns() {
        push_graph_source_column(&mut fingerprint, role, source, column)?;
    }

    Ok(GraphResidencyDescriptor {
        source_fingerprint: GraphSourceFingerprint::new(fingerprint),
        vertex_data_type: source_field.data_type().clone(),
        weight_data_type: weight_field.map(|field| field.data_type().clone()),
        edge_id_data_type: edge_id_field.map(|field| field.data_type().clone()),
    })
}

fn graph_source_field<'a>(
    source: &'a crate::source::LocalParquetSource,
    column: &str,
) -> std::result::Result<&'a Field, GraphSourceFingerprintBypassReason> {
    source.output_schema().field_with_name(column).map_err(|_| GraphSourceFingerprintBypassReason::MissingEdgeColumn)
}

fn local_parquet_file_identity_is_graph_stable(file: &crate::source::LocalParquetFile) -> bool {
    let table_format_identity = file.stable_id.starts_with("table-format:");
    let strong_content_identity =
        file.resolved_source.content_identity.as_deref().is_some_and(graph_content_identity_is_strong);
    match &file.location {
        crate::source::ParquetSourceLocation::LocalPath(_) => {
            (file.file_size_bytes.is_some()
                && file.stable_id.starts_with("local:")
                && file.stable_id.contains(":mtime="))
                || (table_format_identity && strong_content_identity)
        },
        crate::source::ParquetSourceLocation::ObjectStoreUri { .. } => strong_content_identity,
    }
}

fn graph_content_identity_is_strong(identity: &str) -> bool {
    let identity = identity.trim();
    !identity.is_empty()
        && !identity.starts_with("weak_")
        && !identity.get(..2).is_some_and(|prefix| prefix.eq_ignore_ascii_case("w/"))
}

fn push_graph_source_column(
    output: &mut String,
    role: &str,
    source: &crate::source::LocalParquetSource,
    column: &str,
) -> std::result::Result<(), GraphSourceFingerprintBypassReason> {
    let field = graph_source_field(source, column)?;
    let source_ref = source.source_column_ref(column).ok_or(GraphSourceFingerprintBypassReason::MissingEdgeColumn)?;
    push_graph_fingerprint_token(output, format!("edge.{role}.column"), column);
    push_graph_fingerprint_token(output, format!("edge.{role}.source_ref"), graph_source_ref_label(source_ref));
    push_graph_fingerprint_token(output, format!("edge.{role}.data_type"), format!("{:?}", field.data_type()));
    push_graph_fingerprint_token(output, format!("edge.{role}.nullable"), field.is_nullable().to_string());
    Ok(())
}

fn graph_source_ref_label(source_ref: crate::source::SourceColumnRef) -> String {
    match source_ref {
        crate::source::SourceColumnRef::File(index) => format!("file:{index}"),
        crate::source::SourceColumnRef::Partition(index) => format!("partition:{index}"),
        crate::source::SourceColumnRef::MissingNull(index) => format!("missing_null:{index}"),
    }
}

fn push_graph_optional_usize_vec(output: &mut String, key: impl AsRef<str>, values: Option<&[usize]>) {
    match values {
        Some(values) => {
            push_graph_fingerprint_token(output, format!("{}.len", key.as_ref()), values.len().to_string());
            for (index, value) in values.iter().enumerate() {
                push_graph_fingerprint_token(output, format!("{}.{}", key.as_ref(), index), value.to_string());
            }
        },
        None => push_graph_fingerprint_token(output, key, "all"),
    }
}

fn push_graph_string_vec(output: &mut String, key: impl AsRef<str>, values: &[String]) {
    push_graph_fingerprint_token(output, format!("{}.len", key.as_ref()), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_graph_fingerprint_token(output, format!("{}.{}", key.as_ref(), index), value);
    }
}

fn push_graph_fingerprint_token(output: &mut String, key: impl AsRef<str>, value: impl ToString) {
    output.push_str(&crate::util::escape_kv_delimited(key.as_ref()));
    output.push('=');
    output.push_str(&crate::util::escape_kv_delimited(value.to_string().as_str()));
    output.push(';');
}
