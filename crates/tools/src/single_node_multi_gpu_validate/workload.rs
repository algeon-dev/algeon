//! The measured query catalog and profile-label resolution.
//!
//! Relational SQL comes from `workloads`. The cuGraph SQL lives here
//! because it names `algeon-datafusion` table functions, which the neutral
//! workload crate cannot reference.
//!
//! Only the measured run executes a selected query, so the CPU-only feature
//! reaches these values from tests alone.
#![cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use workloads::{clickbench::CLICKBENCH_QUERIES, tpcds::TPCDS_QUERIES, tpch::TPCH_QUERIES};

use super::profile::{FixtureKind, FixtureWorkloadProfile};
use crate::ToolResult;

/// A resolved query the runner will submit through an attempt.
#[derive(Debug, Clone)]
pub(super) struct SelectedQuery {
    /// Position of the owning fixture workload in the recorded profile. The
    /// same fixture kind may name multiple independent DBLP graph slices, so
    /// kind alone is not a session-routing key.
    pub(super) fixture_index: usize,
    pub(super) kind: FixtureKind,
    pub(super) label: String,
    pub(super) sql: String,
    /// True when the query drives a cuGraph construction/algorithm path rather
    /// than a cuDF relational path.
    pub(super) graph: bool,
    /// Whether the query is expected to acquire a device placement. The
    /// runner-owned feature-disabled capability probe is deliberately
    /// non-executing, and its separate operability clause owns that contract.
    pub(super) placement_required: bool,
    /// The deliberately serialized public completion receiver. This is
    /// attached only by the dedicated slow-report-sink probe, so ordinary
    /// workload selection remains free of collector backpressure.
    completion_sink: Option<CompletionSink>,
}

impl SelectedQuery {
    pub(super) fn attach_completion_sink(&mut self, sink: CompletionSink) {
        self.completion_sink = Some(sink);
    }

    pub(super) fn completion_sink(&self) -> Option<&CompletionSink> {
        self.completion_sink.as_ref()
    }

    pub(super) fn descriptor(&self) -> SelectedQueryDescriptor {
        SelectedQueryDescriptor {
            fixture_index: self.fixture_index,
            kind: self.kind,
            label: self.label.clone(),
            workload_class: if self.graph { "cugraph" } else { "cudf_relational" },
        }
    }
}

/// One intentionally serialized completion consumer for the handoff-pressure probe. The semaphore
/// permit is acquired before the public completion receiver is awaited and is
/// held until that receiver has drained, creating the real handoff pressure
/// the probe is meant to test rather than merely sleeping each task.
#[derive(Debug, Clone)]
pub(super) struct CompletionSink {
    delay: Duration,
    permit: Arc<Semaphore>,
    drains_started: Arc<AtomicU64>,
    drains_completed: Arc<AtomicU64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CompletionSinkCounters {
    pub(super) delay: Duration,
    pub(super) drains_started: u64,
    pub(super) drains_completed: u64,
}

impl CompletionSink {
    pub(super) fn new(delay: Duration) -> Self {
        Self {
            delay,
            permit: Arc::new(Semaphore::new(1)),
            drains_started: Arc::new(AtomicU64::new(0)),
            drains_completed: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(super) async fn acquire(&self) -> OwnedSemaphorePermit {
        let permit = Arc::clone(&self.permit).acquire_owned().await.expect("completion sink semaphore is never closed");
        self.drains_started.fetch_add(1, Ordering::AcqRel);
        permit
    }

    pub(super) const fn delay(&self) -> Duration {
        self.delay
    }

    pub(super) fn mark_drained(&self) {
        self.drains_completed.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn counters(&self) -> CompletionSinkCounters {
        CompletionSinkCounters {
            delay: self.delay,
            drains_started: self.drains_started.load(Ordering::Acquire),
            drains_completed: self.drains_completed.load(Ordering::Acquire),
        }
    }
}

/// The artifact projection of a selected query. The SQL itself is never
/// persisted: the artifact records identity, not raw statements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct SelectedQueryDescriptor {
    pub(super) fixture_index: usize,
    pub(super) kind: FixtureKind,
    pub(super) label: String,
    pub(super) workload_class: &'static str,
}

/// The DBLP edge table the cuGraph table functions read. `edges_by_src` is the
/// source-sorted layout, which is what the algorithm paths want.
const DBLP_EDGE_TABLE: &str = "edges_by_src";
/// A traversal source that exists in the canonical V12 vertex space. DBLP paper
/// ids do not start at zero, so a synthetic source is rejected as invalid
/// input rather than traversing nothing.
const DBLP_TRAVERSAL_SOURCE: i64 = 1091;

/// cuGraph SQL table-function names.
///
/// The base runner feature is CPU-only and cannot link the adapter, so these
/// are literals here and pinned against the adapter's public constants by
/// `cugraph_function_names_match_the_adapter_constants` under the run feature.
const CUGRAPH_PAGERANK_FUNCTION: &str = "cugraph_pagerank";
const CUGRAPH_BFS_FUNCTION: &str = "cugraph_bfs";
const CUGRAPH_SSSP_FUNCTION: &str = "cugraph_sssp";
const CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION: &str = "cugraph_weakly_connected_components";
const CUGRAPH_FORCE_ATLAS2_FUNCTION: &str = "cugraph_force_atlas2";
const CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION: &str = "cugraph_betweenness_centrality";

/// cuGraph construction, algorithm, and result-path queries plus the two cuDF
/// relational queries over the same fixture.
fn dblp_queries() -> Vec<(&'static str, String, bool)> {
    vec![
        (
            "pagerank",
            format!(
                "SELECT COUNT(*) AS vertices, SUM(value) AS total \
                 FROM {CUGRAPH_PAGERANK_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', 'weight')"
            ),
            true,
        ),
        (
            "pagerank_unweighted",
            format!(
                "SELECT COUNT(*) AS vertices, SUM(value) AS total \
                 FROM {CUGRAPH_PAGERANK_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', NULL)"
            ),
            true,
        ),
        (
            "pagerank_topk",
            format!(
                "SELECT vertex, value FROM {CUGRAPH_PAGERANK_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', 'weight') \
                 ORDER BY value DESC LIMIT 100"
            ),
            true,
        ),
        (
            "bfs",
            format!(
                "SELECT COUNT(*) AS reached FROM {CUGRAPH_BFS_FUNCTION}('{DBLP_EDGE_TABLE}', {DBLP_TRAVERSAL_SOURCE}, 'src', 'dst', NULL, \
                 '{{\"output_mode\": \"normalized\"}}')"
            ),
            true,
        ),
        (
            "bfs_topk",
            format!(
                "SELECT vertex, distance, predecessor FROM {CUGRAPH_BFS_FUNCTION}('{DBLP_EDGE_TABLE}', {DBLP_TRAVERSAL_SOURCE}, 'src', 'dst', NULL, \
                 '{{\"construction_policy\": \"raw_libcugraph\"}}') ORDER BY distance LIMIT 100"
            ),
            true,
        ),
        (
            "sssp",
            format!(
                "SELECT COUNT(*) AS reached FROM {CUGRAPH_SSSP_FUNCTION}('{DBLP_EDGE_TABLE}', {DBLP_TRAVERSAL_SOURCE}, 'src', 'dst', \
                 'weight')"
            ),
            true,
        ),
        (
            "wcc",
            format!(
                "SELECT COUNT(*) AS vertices FROM \
                 {CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst')"
            ),
            true,
        ),
        ("edge_scan", format!("SELECT COUNT(*) AS edges FROM {DBLP_EDGE_TABLE}"), false),
        (
            "profile_year_histogram",
            "SELECT year, COUNT(*) AS papers FROM profiles_by_paper_id WHERE year IS NOT NULL \
             GROUP BY year ORDER BY papers DESC LIMIT 16"
                .to_string(),
            false,
        ),
        // The backend-service stress priority set.
        (
            "force_atlas2",
            format!(
                "SELECT COUNT(*) AS vertices, SUM(x) AS sum_x, SUM(y) AS sum_y \
                 FROM {CUGRAPH_FORCE_ATLAS2_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', 'weight', \
                 '{{\"max_iter\": 200, \"seed\": 20260804, \"barnes_hut_optimize\": true}}')"
            ),
            true,
        ),
        (
            "pagerank_forceatlas2_join",
            format!(
                "WITH pagerank AS ( \
                 SELECT vertex, value AS pagerank_value \
                 FROM {CUGRAPH_PAGERANK_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', 'weight') \
                 ), layout AS ( \
                 SELECT vertex, x, y \
                 FROM {CUGRAPH_FORCE_ATLAS2_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', 'weight', \
                 '{{\"max_iter\": 200, \"seed\": 20260804, \"barnes_hut_optimize\": true}}') \
                 ) \
                 SELECT p.vertex, p.pagerank_value, l.x, l.y \
                 FROM layout l JOIN pagerank p ON p.vertex = l.vertex \
                 WHERE p.pagerank_value > 0"
            ),
            true,
        ),
        (
            "bfs_deep",
            format!(
                "SELECT vertex, distance, reachable \
                 FROM {CUGRAPH_BFS_FUNCTION}('{DBLP_EDGE_TABLE}', {DBLP_TRAVERSAL_SOURCE}, 'src', 'dst', NULL, \
                 '{{\"depth_limit\": 8, \"output_mode\": \"normalized\"}}')"
            ),
            true,
        ),
        (
            "betweenness_centrality",
            format!(
                "SELECT COUNT(*) AS vertices, SUM(value) AS total \
                 FROM {CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', NULL, \
                 '{{\"k\": 64}}')"
            ),
            true,
        ),
        (
            "betweenness_centrality_wide",
            format!(
                "SELECT COUNT(*) AS vertices, SUM(value) AS total \
                 FROM {CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION}('{DBLP_EDGE_TABLE}', 'src', 'dst', NULL, \
                 '{{\"k\": 512}}')"
            ),
            true,
        ),
    ]
}

/// Resolves the profile's query labels into executable statements.
///
/// Selection order is profile order, so a recorded plan replays identically.
pub(super) fn select_queries(workloads: &[FixtureWorkloadProfile]) -> ToolResult<Vec<SelectedQuery>> {
    let mut selected = Vec::new();
    for (fixture_index, workload) in workloads.iter().enumerate() {
        for label in &workload.queries {
            selected.push(resolve(fixture_index, workload.kind, label)?);
        }
    }
    Ok(selected)
}

fn resolve(fixture_index: usize, kind: FixtureKind, label: &str) -> ToolResult<SelectedQuery> {
    let selected = |label: &str, sql: &str, graph: bool| SelectedQuery {
        fixture_index,
        kind,
        label: label.to_string(),
        sql: sql.to_string(),
        graph,
        placement_required: true,
        completion_sink: None,
    };
    match kind {
        FixtureKind::Tpch => TPCH_QUERIES
            .iter()
            .find(|query| query.id == label)
            .map(|query| selected(label, query.sql, false))
            .ok_or_else(|| unknown_label(kind, label).into()),
        FixtureKind::Tpcds => TPCDS_QUERIES
            .iter()
            .find(|query| query.id == label)
            .map(|query| selected(label, query.sql, false))
            .ok_or_else(|| unknown_label(kind, label).into()),
        FixtureKind::Clickbench => CLICKBENCH_QUERIES
            .iter()
            .find(|query| query.id == label)
            .map(|query| selected(label, query.sql, false))
            .ok_or_else(|| unknown_label(kind, label).into()),
        FixtureKind::Dblp => dblp_queries()
            .into_iter()
            .find(|(id, _, _)| *id == label)
            .map(|(_, sql, graph)| selected(label, &sql, graph))
            .ok_or_else(|| unknown_label(kind, label).into()),
    }
}

fn unknown_label(kind: FixtureKind, label: &str) -> String {
    format!("unknown {} query label {label}; known labels: {}", kind.label(), known_labels(kind).join(", "))
}

/// Query labels a profile may select, for the usage text and error messages.
pub(super) fn known_labels(kind: FixtureKind) -> Vec<&'static str> {
    match kind {
        FixtureKind::Tpch => TPCH_QUERIES.iter().map(|query| query.id).collect(),
        FixtureKind::Tpcds => TPCDS_QUERIES.iter().map(|query| query.id).collect(),
        FixtureKind::Clickbench => CLICKBENCH_QUERIES.iter().map(|query| query.id).collect(),
        FixtureKind::Dblp => dblp_queries().into_iter().map(|(label, _, _)| label).collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    #[cfg(feature = "single-node-multi-gpu-validate-run")]
    use algeon_datafusion::cugraph_sql::{
        CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION as AdapterCugraphBetweennessCentralityFunction,
        CUGRAPH_BFS_FUNCTION as AdapterCugraphBfsFunction,
        CUGRAPH_FORCE_ATLAS2_FUNCTION as AdapterCugraphForceAtlas2Function,
        CUGRAPH_PAGERANK_FUNCTION as AdapterCugraphPagerankFunction,
        CUGRAPH_SSSP_FUNCTION as AdapterCugraphSsspFunction,
        CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION as AdapterCugraphWeaklyConnectedComponentsFunction,
    };
    use tokio::task::yield_now;
    use workloads::{clickbench::CLICKBENCH_QUERY_COUNT, tpcds::TPCDS_QUERY_COUNT, tpch::TPCH_QUERY_COUNT};

    use super::*;

    fn workload(kind: FixtureKind, queries: &[&str]) -> FixtureWorkloadProfile {
        FixtureWorkloadProfile {
            kind,
            path: PathBuf::from("/fixture"),
            scale: "sf1".to_string(),
            preparation_command: "fixture/fixture.sh tpch generate --scale 1 --out-root /fixture".to_string(),
            queries: queries.iter().map(|query| (*query).to_string()).collect(),
        }
    }

    #[test]
    fn selection_preserves_profile_order_and_classifies_both_workload_classes() {
        let selected = select_queries(&[
            workload(FixtureKind::Tpch, &["q01", "q06"]),
            workload(FixtureKind::Dblp, &["pagerank", "edge_scan"]),
        ])
        .unwrap();
        assert_eq!(
            selected.iter().map(|query| query.label.as_str()).collect::<Vec<_>>(),
            ["q01", "q06", "pagerank", "edge_scan"]
        );
        assert!(selected.iter().all(|query| !query.sql.is_empty()));
        assert_eq!(selected.iter().filter(|query| query.graph).count(), 1);
        assert_eq!(selected[2].descriptor().workload_class, "cugraph");
        assert_eq!(selected[0].descriptor().workload_class, "cudf_relational");
    }

    #[test]
    fn distinct_dblp_workloads_keep_their_own_session_routing_indices() {
        let selected =
            select_queries(&[workload(FixtureKind::Dblp, &["pagerank"]), workload(FixtureKind::Dblp, &["pagerank"])])
                .unwrap();

        assert_eq!(selected.iter().map(|query| query.fixture_index).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(selected.iter().map(|query| query.kind).collect::<Vec<_>>(), [FixtureKind::Dblp, FixtureKind::Dblp]);
        assert_eq!(selected[1].descriptor().fixture_index, 1);
    }

    #[test]
    fn an_unknown_label_is_rejected_rather_than_skipped() {
        let error = select_queries(&[workload(FixtureKind::Tpch, &["q99"])]).unwrap_err();
        assert!(error.to_string().contains("unknown tpch query label q99"), "{error}");
        let error = select_queries(&[workload(FixtureKind::Dblp, &["not_a_query"])]).unwrap_err();
        assert!(error.to_string().contains("unknown dblp query label"), "{error}");
    }

    #[test]
    fn every_catalog_label_resolves_and_the_graph_catalog_covers_construction_and_algorithms() {
        for kind in [FixtureKind::Tpch, FixtureKind::Tpcds, FixtureKind::Clickbench, FixtureKind::Dblp] {
            for label in known_labels(kind) {
                resolve(0, kind, label).unwrap();
            }
        }
        assert_eq!(known_labels(FixtureKind::Tpch).len(), TPCH_QUERY_COUNT);
        assert_eq!(known_labels(FixtureKind::Tpcds).len(), TPCDS_QUERY_COUNT);
        assert_eq!(known_labels(FixtureKind::Clickbench).len(), CLICKBENCH_QUERY_COUNT);
        let graph = known_labels(FixtureKind::Dblp);
        for required in [
            "pagerank",
            "bfs",
            "wcc",
            "edge_scan",
            "force_atlas2",
            "pagerank_forceatlas2_join",
            "bfs_deep",
            "betweenness_centrality",
            "betweenness_centrality_wide",
        ] {
            assert!(graph.contains(&required), "{required}");
        }
    }

    #[test]
    fn basic_bfs_catalog_entry_keeps_the_unbounded_singleflight_shape() {
        let bfs = resolve(0, FixtureKind::Dblp, "bfs").expect("bfs catalog entry");
        assert!(bfs.graph);
        assert!(bfs.sql.contains("\"output_mode\": \"normalized\""));
        assert!(!bfs.sql.contains("depth_limit"));
    }

    #[test]
    fn composed_catalog_entry_keeps_two_graph_producers_and_native_join_consumer() {
        let composed = resolve(0, FixtureKind::Dblp, "pagerank_forceatlas2_join").expect("composed catalog entry");
        assert!(composed.graph);
        assert!(composed.sql.contains(CUGRAPH_PAGERANK_FUNCTION));
        assert!(composed.sql.contains(CUGRAPH_FORCE_ATLAS2_FUNCTION));
        assert!(composed.sql.contains("JOIN pagerank"));
        assert!(composed.sql.contains("pagerank_value > 0"));
    }

    /// The literals above must equal the adapter's exported names, or the graph
    /// workload would silently plan as an unknown table function.
    #[cfg(feature = "single-node-multi-gpu-validate-run")]
    #[test]
    fn cugraph_function_names_match_the_adapter_constants() {
        assert_eq!(CUGRAPH_PAGERANK_FUNCTION, AdapterCugraphPagerankFunction);
        assert_eq!(CUGRAPH_BFS_FUNCTION, AdapterCugraphBfsFunction);
        assert_eq!(CUGRAPH_SSSP_FUNCTION, AdapterCugraphSsspFunction);
        assert_eq!(CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION, AdapterCugraphWeaklyConnectedComponentsFunction);
        assert_eq!(CUGRAPH_FORCE_ATLAS2_FUNCTION, AdapterCugraphForceAtlas2Function);
        assert_eq!(CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION, AdapterCugraphBetweennessCentralityFunction);
    }

    #[test]
    fn descriptors_do_not_carry_sql() {
        let selected = select_queries(&[workload(FixtureKind::Tpch, &["q01"])]).unwrap();
        let descriptor = serde_json::to_value(selected[0].descriptor()).unwrap();
        assert_eq!(descriptor["label"], "q01");
        assert!(descriptor.get("sql").is_none());
    }

    #[tokio::test]
    async fn completion_sink_allows_only_one_completion_receiver_to_enter() {
        let sink = CompletionSink::new(std::time::Duration::from_millis(1));
        let first = sink.acquire().await;
        let waiting_sink = sink.clone();
        let waiting = tokio::spawn(async move {
            let permit = waiting_sink.acquire().await;
            waiting_sink.mark_drained();
            permit
        });
        yield_now().await;
        assert_eq!(sink.counters().drains_started, 1);
        drop(first);
        drop(waiting.await.unwrap());
        let counters = sink.counters();
        assert_eq!(counters.drains_started, 2);
        assert_eq!(counters.drains_completed, 1);
    }
}
