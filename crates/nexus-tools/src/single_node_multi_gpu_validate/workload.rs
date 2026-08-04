//! The measured query catalog and profile-label resolution.
//!
//! Relational SQL comes from `nexus-workloads`. The cuGraph SQL lives here
//! because it names `datafusion-nexus` table functions, which the neutral
//! workload crate cannot reference.
//!
//! Only the measured run executes a selected query, so the CPU-only feature
//! reaches these values from tests alone.
#![cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]

use serde::Serialize;

use super::profile::{FixtureKind, FixtureWorkloadProfile};
use crate::ToolResult;

/// A resolved query the runner will submit through an attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SelectedQuery {
    pub(super) kind: FixtureKind,
    pub(super) label: String,
    pub(super) sql: String,
    /// True when the query drives a cuGraph construction/algorithm path rather
    /// than a cuDF relational path.
    pub(super) graph: bool,
}

impl SelectedQuery {
    pub(super) fn descriptor(&self) -> SelectedQueryDescriptor {
        SelectedQueryDescriptor {
            kind: self.kind,
            label: self.label.clone(),
            workload_class: if self.graph { "cugraph" } else { "cudf_relational" },
        }
    }
}

/// The artifact projection of a selected query. The SQL itself is never
/// persisted: the artifact records identity, not raw statements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct SelectedQueryDescriptor {
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
            "bfs",
            format!(
                "SELECT COUNT(*) AS reached FROM {CUGRAPH_BFS_FUNCTION}('{DBLP_EDGE_TABLE}', {DBLP_TRAVERSAL_SOURCE}, 'src', 'dst', NULL, \
                 '{{\"depth_limit\": 3, \"output_mode\": \"normalized\"}}')"
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
    ]
}

/// Resolves the profile's query labels into executable statements.
///
/// Selection order is profile order, so a recorded plan replays identically.
pub(super) fn select_queries(workloads: &[FixtureWorkloadProfile]) -> ToolResult<Vec<SelectedQuery>> {
    let mut selected = Vec::new();
    for workload in workloads {
        for label in &workload.queries {
            selected.push(resolve(workload.kind, label)?);
        }
    }
    Ok(selected)
}

fn resolve(kind: FixtureKind, label: &str) -> ToolResult<SelectedQuery> {
    match kind {
        FixtureKind::Tpch => nexus_workloads::tpch::TPCH_QUERIES
            .iter()
            .find(|query| query.id == label)
            .map(|query| SelectedQuery { kind, label: label.to_string(), sql: query.sql.to_string(), graph: false })
            .ok_or_else(|| unknown_label(kind, label).into()),
        FixtureKind::Dblp => dblp_queries()
            .into_iter()
            .find(|(id, _, _)| *id == label)
            .map(|(_, sql, graph)| SelectedQuery { kind, label: label.to_string(), sql, graph })
            .ok_or_else(|| unknown_label(kind, label).into()),
    }
}

fn unknown_label(kind: FixtureKind, label: &str) -> String {
    format!("unknown {} query label {label}; known labels: {}", kind.label(), known_labels(kind).join(", "))
}

/// Query labels a profile may select, for the usage text and error messages.
pub(super) fn known_labels(kind: FixtureKind) -> Vec<&'static str> {
    match kind {
        FixtureKind::Tpch => nexus_workloads::tpch::TPCH_QUERIES.iter().map(|query| query.id).collect(),
        FixtureKind::Dblp => dblp_queries().into_iter().map(|(label, _, _)| label).collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

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
    fn an_unknown_label_is_rejected_rather_than_skipped() {
        let error = select_queries(&[workload(FixtureKind::Tpch, &["q99"])]).unwrap_err();
        assert!(error.to_string().contains("unknown tpch query label q99"), "{error}");
        let error = select_queries(&[workload(FixtureKind::Dblp, &["not_a_query"])]).unwrap_err();
        assert!(error.to_string().contains("unknown dblp query label"), "{error}");
    }

    #[test]
    fn every_catalog_label_resolves_and_the_graph_catalog_covers_construction_and_algorithms() {
        for kind in [FixtureKind::Tpch, FixtureKind::Dblp] {
            for label in known_labels(kind) {
                resolve(kind, label).unwrap();
            }
        }
        assert_eq!(known_labels(FixtureKind::Tpch).len(), nexus_workloads::tpch::TPCH_QUERY_COUNT);
        let graph = known_labels(FixtureKind::Dblp);
        for required in ["pagerank", "bfs", "wcc", "edge_scan"] {
            assert!(graph.contains(&required), "{required}");
        }
    }

    /// The literals above must equal the adapter's exported names, or the graph
    /// workload would silently plan as an unknown table function.
    #[cfg(feature = "single-node-multi-gpu-validate-run")]
    #[test]
    fn cugraph_function_names_match_the_adapter_constants() {
        assert_eq!(CUGRAPH_PAGERANK_FUNCTION, datafusion_nexus::cugraph_sql::CUGRAPH_PAGERANK_FUNCTION);
        assert_eq!(CUGRAPH_BFS_FUNCTION, datafusion_nexus::cugraph_sql::CUGRAPH_BFS_FUNCTION);
        assert_eq!(CUGRAPH_SSSP_FUNCTION, datafusion_nexus::cugraph_sql::CUGRAPH_SSSP_FUNCTION);
        assert_eq!(
            CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION,
            datafusion_nexus::cugraph_sql::CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION
        );
    }

    #[test]
    fn descriptors_do_not_carry_sql() {
        let selected = select_queries(&[workload(FixtureKind::Tpch, &["q01"])]).unwrap();
        let descriptor = serde_json::to_value(selected[0].descriptor()).unwrap();
        assert_eq!(descriptor["label"], "q01");
        assert!(descriptor.get("sql").is_none());
    }
}
