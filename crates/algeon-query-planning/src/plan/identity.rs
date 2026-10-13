//! Stable execution identity of a [`QueryPlan`].
//!
//! [`PlanFingerprint`] is a blake3 digest of the executable graph. Values
//! enter the hasher field by field in declaration order, enum variants add
//! their name as a tag, ordered collections are walked in key order, Arrow
//! schemas hash their metadata with sorted keys, and literal batches hash
//! their Arrow IPC stream encoding. Two independently built plans with equal
//! executable content therefore produce equal digests.
//!
//! Advisory statistics and display-only source diagnostics are excluded: they
//! never change what executes, and the build side they informed is already a
//! field of the join.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::Arc,
};

use arrow::{
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::writer::StreamWriter,
    record_batch::RecordBatch,
};

use super::{Annotations, PlanNode, QueryNode, QueryPlan};
use crate::source::SourceNode;

mod impls;

/// blake3 digest identifying the executable content of one plan.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanFingerprint([u8; 32]);

/// blake3 digest of an Arrow schema, with metadata hashed in key order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SchemaFingerprint([u8; 32]);

impl SchemaFingerprint {
    /// Digest of `schema`.
    pub fn of(schema: &Schema) -> Self {
        let mut hasher = PlanHasher::new();
        schema.stable_hash(&mut hasher);
        Self(hasher.finish_digest())
    }
}

/// Renders a digest as lowercase hexadecimal.
macro_rules! digest_display {
    ($($ty:ident),*) => {$(
        impl fmt::Display for $ty {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                for byte in self.0 {
                    write!(formatter, "{byte:02x}")?;
                }
                Ok(())
            }
        }

        impl fmt::Debug for $ty {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}({self})", stringify!($ty))
            }
        }
    )*};
}

digest_display!(PlanFingerprint, SchemaFingerprint);

/// blake3 state fed by [`StableHash`].
///
/// It also implements [`Hasher`] so types whose derived [`Hash`] is already
/// deterministic, such as field-less enums and Arrow schemas, feed it
/// directly.
pub(crate) struct PlanHasher(blake3::Hasher);

impl PlanHasher {
    pub(crate) fn new() -> Self {
        Self(blake3::Hasher::new())
    }

    fn finish_digest(self) -> [u8; 32] {
        *self.0.finalize().as_bytes()
    }

    /// Names the enum variant that the following fields belong to.
    pub(crate) fn tag(&mut self, tag: &'static str) {
        tag.stable_hash(self);
    }
}

impl Hasher for PlanHasher {
    fn finish(&self) -> u64 {
        let digest = self.0.finalize();
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&digest.as_bytes()[..8]);
        u64::from_le_bytes(prefix)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

/// Feeds a value into a [`PlanHasher`].
pub(crate) trait StableHash {
    fn stable_hash(&self, hasher: &mut PlanHasher);
}

/// Implements [`StableHash`] through a derived or hand-written [`Hash`] whose
/// output does not depend on process state.
macro_rules! stable_hash_via_hash {
    ($($ty:ty),* $(,)?) => {$(
        impl StableHash for $ty {
            fn stable_hash(&self, hasher: &mut PlanHasher) {
                ::std::hash::Hash::hash(self, hasher);
            }
        }
    )*};
}

/// Implements [`StableHash`] for a struct by hashing every field in
/// declaration order. The exhaustive destructuring fails to compile when a
/// field is added without deciding whether it belongs to the identity.
macro_rules! stable_hash_fields {
    () => {};
    ($ty:ident {} $($rest:tt)*) => {
        impl StableHash for $ty {
            fn stable_hash(&self, _hasher: &mut PlanHasher) {
                let Self {} = self;
            }
        }
        $crate::plan::stable_hash_fields! { $($rest)* }
    };
    ($ty:ident { $($field:ident),+ $(,)? } $($rest:tt)*) => {
        impl StableHash for $ty {
            fn stable_hash(&self, hasher: &mut PlanHasher) {
                let Self { $($field),+ } = self;
                $( $field.stable_hash(hasher); )+
            }
        }
        $crate::plan::stable_hash_fields! { $($rest)* }
    };
}

pub(crate) use stable_hash_fields;
pub(crate) use stable_hash_via_hash;

stable_hash_via_hash!(bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, usize, DataType, Field, Schema, TimeUnit);

impl StableHash for f32 {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.to_bits().stable_hash(hasher);
    }
}

impl StableHash for f64 {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.to_bits().stable_hash(hasher);
    }
}

impl StableHash for str {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.len().stable_hash(hasher);
        hasher.write(self.as_bytes());
    }
}

impl StableHash for String {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.as_str().stable_hash(hasher);
    }
}

impl StableHash for PathBuf {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        let bytes = self.as_os_str().as_encoded_bytes();
        bytes.len().stable_hash(hasher);
        hasher.write(bytes);
    }
}

impl<T: StableHash + ?Sized> StableHash for &T {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        (**self).stable_hash(hasher);
    }
}

impl<T: StableHash + ?Sized> StableHash for Box<T> {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.as_ref().stable_hash(hasher);
    }
}

impl<T: StableHash + ?Sized> StableHash for Arc<T> {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.as_ref().stable_hash(hasher);
    }
}

impl<T: StableHash> StableHash for Option<T> {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            None => hasher.tag("none"),
            Some(value) => {
                hasher.tag("some");
                value.stable_hash(hasher);
            },
        }
    }
}

impl<T: StableHash> StableHash for [T] {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.len().stable_hash(hasher);
        for item in self {
            item.stable_hash(hasher);
        }
    }
}

impl<T: StableHash, const N: usize> StableHash for [T; N] {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.as_slice().stable_hash(hasher);
    }
}

impl<T: StableHash> StableHash for Vec<T> {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.as_slice().stable_hash(hasher);
    }
}

impl<T: StableHash> StableHash for BTreeSet<T> {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.len().stable_hash(hasher);
        for item in self {
            item.stable_hash(hasher);
        }
    }
}

impl<K: StableHash, V: StableHash> StableHash for BTreeMap<K, V> {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        self.len().stable_hash(hasher);
        for (key, value) in self {
            key.stable_hash(hasher);
            value.stable_hash(hasher);
        }
    }
}

/// A literal batch is identified by its IPC stream encoding, which covers the
/// schema and every value; `Debug` prints only the first and last values of a
/// long array.
impl StableHash for RecordBatch {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        let mut encoded = Vec::new();
        let written = StreamWriter::try_new(&mut encoded, &self.schema())
            .and_then(|mut writer| writer.write(self).and_then(|()| writer.finish()));
        match written {
            Ok(()) => {
                hasher.tag("ipc");
                encoded.as_slice().stable_hash(hasher);
            },
            // An encoding failure cannot be reported from here; the
            // row-by-row display is the fallback identity.
            Err(error) => {
                hasher.tag("ipc_unencodable");
                error.to_string().stable_hash(hasher);
            },
        }
    }
}

impl QueryPlan {
    /// Digest of the executable graph.
    ///
    /// A host sink and a [`ResultSink::DeferredHostArrow`](super::ResultSink::DeferredHostArrow)
    /// with the same schema share one fingerprint: the caller receives the same
    /// host result either way. Advisory outputs and display-only diagnostics
    /// are excluded.
    pub fn execution_fingerprint(&self) -> PlanFingerprint {
        self.execution_fingerprint_masking(&[])
    }

    /// Digest with the listed GPU-resident source facts hashed as absent.
    ///
    /// The sink contributes its input id and caller-visible host schema only,
    /// so the sink kind is excluded by construction and a host sink and a
    /// deferred host sink of the same schema agree. Dynamic-filter routes are
    /// hashed because they change which source reads a runtime filter.
    fn execution_fingerprint_masking(&self, masked: &[DeferredGraphSourceFactSlots]) -> PlanFingerprint {
        let Self { nodes, sink, annotations: Annotations { dynamic_filter_routes, advisory: _ } } = self;
        let mut hasher = PlanHasher::new();
        nodes.len().stable_hash(&mut hasher);
        for (index, node) in nodes.iter().enumerate() {
            match masked.iter().find(|slots| slots.node_index == index) {
                Some(slots) => node.masked(slots).stable_hash(&mut hasher),
                None => node.stable_hash(&mut hasher),
            }
        }
        sink.input().stable_hash(&mut hasher);
        sink.caller_visible_host_schema().stable_hash(&mut hasher);
        dynamic_filter_routes.stable_hash(&mut hasher);
        PlanFingerprint(hasher.finish_digest())
    }
}

impl QueryNode {
    fn masked(&self, slots: &DeferredGraphSourceFactSlots) -> QueryNode {
        let mut node = self.clone();
        if let PlanNode::Source(source) = &mut node.kind
            && let SourceNode::GpuResident(source) = source.as_mut()
        {
            if slots.row_count_upper_bound {
                source.row_count_upper_bound = None;
            }
            if slots.graph_vertex_upper_bound {
                source.graph_vertex_upper_bound = None;
            }
        }
        node
    }
}

/// Identity of a graph fragment whose missing GPU-resident source facts may be
/// filled in after Arrow staging.
///
/// The facts the admitted plan lacked are hashed as absent on both sides, so a
/// later plan matches only when it differs in nothing else. Facts the admitted
/// plan already knew stay part of the identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredGraphExecutionScopeIdentity {
    execution_fingerprint: PlanFingerprint,
    source_fact_slots: Vec<DeferredGraphSourceFactSlots>,
}

/// Which optional facts of one GPU-resident source were absent at admission.
/// A `true` slot is masked on both sides of the comparison; a `false` slot
/// stays in the digest, so a bound the admitted plan knew must still match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeferredGraphSourceFactSlots {
    node_index: usize,
    row_count_upper_bound: bool,
    graph_vertex_upper_bound: bool,
}

impl DeferredGraphExecutionScopeIdentity {
    pub(crate) fn for_plan(plan: &QueryPlan) -> Self {
        let source_fact_slots = plan
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(node_index, node)| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::GpuResident(source) => {
                        let slots = DeferredGraphSourceFactSlots {
                            node_index,
                            row_count_upper_bound: source.row_count_upper_bound.is_none(),
                            graph_vertex_upper_bound: source.graph_vertex_upper_bound.is_none(),
                        };
                        (slots.row_count_upper_bound || slots.graph_vertex_upper_bound).then_some(slots)
                    },
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        Self { execution_fingerprint: plan.execution_fingerprint_masking(&source_fact_slots), source_fact_slots }
    }

    /// Whether `plan` is the admitted fragment once its deferred facts are
    /// ignored. The admitted plan's slots mask both sides, so a plan that
    /// gained facts matches and one that changed a known fact does not.
    pub(crate) fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_fingerprint == plan.execution_fingerprint_masking(&self.source_fact_slots)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
    };

    use super::*;
    use crate::{
        admission::{AttemptExecutionManifest, AttemptExecutionOccurrence},
        expr::Expr,
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, LiteralBatchSource},
    };

    fn resident_plan(name: &str, row_bound: Option<u64>, vertex_bound: Option<u64>) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
        let mut source = GpuResidentSource::new(name, schema);
        if let Some(row_bound) = row_bound {
            source = source.with_row_count_upper_bound(row_bound);
        }
        if let Some(vertex_bound) = vertex_bound {
            source = source.with_graph_vertex_upper_bound(vertex_bound);
        }
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::GpuResident(source));
        builder.finish_gpu_resident(root).unwrap()
    }

    #[test]
    fn deferred_graph_identity_only_fills_facts_missing_from_the_admitted_plan() {
        let admitted = resident_plan("edges", None, Some(9));
        let identity = DeferredGraphExecutionScopeIdentity::for_plan(&admitted);

        assert!(identity.matches(&resident_plan("edges", Some(12), Some(9))));
        assert!(!identity.matches(&resident_plan("edges", Some(12), Some(10))));
        assert!(!identity.matches(&resident_plan("other_edges", Some(12), Some(9))));
    }

    #[test]
    fn deferred_graph_identity_does_not_weaken_known_source_facts() {
        let admitted = resident_plan("edges", Some(12), Some(9));
        let identity = DeferredGraphExecutionScopeIdentity::for_plan(&admitted);

        assert!(identity.matches(&admitted));
        assert!(!identity.matches(&resident_plan("edges", Some(13), Some(9))));
    }

    /// Builds the plan from scratch each time, so every call allocates fresh
    /// metadata `HashMap`s with their own iteration order.
    fn metadata_plan() -> QueryPlan {
        let metadata =
            (0..8).map(|index| (format!("key_{index}"), format!("value_{index}"))).collect::<HashMap<_, _>>();
        let field = Field::new("vertex", DataType::Int64, false).with_metadata(metadata.clone());
        let schema = Arc::new(Schema::new_with_metadata(vec![field], metadata));
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", schema)));
        builder.finish_gpu_resident(root).unwrap()
    }

    #[test]
    fn independently_built_plans_with_schema_metadata_share_one_identity() {
        let admitted = metadata_plan();
        let manifest = AttemptExecutionManifest::new(vec![AttemptExecutionOccurrence::host_arrow(
            "gpu-0",
            &admitted,
            vec![],
            false,
        )])
        .unwrap();
        for _ in 0..16 {
            let rebuilt = metadata_plan();
            assert_eq!(rebuilt.execution_fingerprint(), admitted.execution_fingerprint());
            assert!(manifest.occurrence_for_plan(&rebuilt).is_some(), "an equal plan matches its admitted occurrence");
        }
    }

    fn literal_plan(values: Vec<i64>) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(values))]).unwrap();
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        builder.finish_gpu_resident(root).unwrap()
    }

    #[test]
    fn literal_batch_values_outside_the_debug_preview_change_the_identity() {
        let values = (0..40).collect::<Vec<i64>>();
        let mut changed = values.clone();
        // `Debug` prints only the first and last ten values of a long array.
        changed[20] = -1;
        assert_ne!(literal_plan(values).execution_fingerprint(), literal_plan(changed).execution_fingerprint());
    }

    #[test]
    fn character_and_octet_length_have_distinct_identity_and_display() {
        let digest = |expr: &Expr| {
            let mut hasher = PlanHasher::new();
            expr.stable_hash(&mut hasher);
            hasher.finish_digest()
        };
        let characters = Expr::string_length(Expr::column("s"), DataType::Int32);
        let bytes = Expr::octet_length(Expr::column("s"), DataType::Int32);

        assert_ne!(digest(&characters), digest(&bytes));
        assert_ne!(characters.to_string(), bytes.to_string());
    }
}
