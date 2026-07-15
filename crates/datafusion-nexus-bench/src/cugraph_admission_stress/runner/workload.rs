use super::*;

pub(super) fn catalog(seed: i64) -> Vec<QuerySpec> {
    let spec = |label: &str, sql: String, needs_papers: bool, needs_fos: bool| QuerySpec {
        label: label.to_string(),
        sql,
        needs_papers,
        needs_fos,
    };
    vec![
        spec(
            "pagerank",
            "SELECT COUNT(*) AS n, SUM(value) AS total FROM cugraph_pagerank('citation_edges', 'src', 'dst', 'weight')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "pagerank_nw",
            "SELECT COUNT(*) AS n, SUM(value) AS total FROM cugraph_pagerank('citation_edges', 'src', 'dst', NULL)"
                .to_string(),
            false,
            false,
        ),
        spec(
            "bfs",
            format!(
                "SELECT COUNT(*) AS reached FROM cugraph_bfs('citation_edges', {seed}, 'src', 'dst', NULL, \
                 '{{\"depth_limit\": 3, \"output_mode\": \"normalized\"}}')"
            ),
            false,
            false,
        ),
        spec("edge_scan", "SELECT src, dst FROM citation_edges".to_string(), false, false),
        spec(
            "bfs_deep",
            format!(
                "SELECT COUNT(*) AS reached FROM cugraph_bfs('citation_edges', {seed}, 'src', 'dst', NULL, \
                 '{{\"output_mode\": \"normalized\"}}')"
            ),
            false,
            false,
        ),
        spec(
            "bfs_deep_full",
            format!(
                "SELECT vertex, distance, predecessor, reachable FROM \
                 cugraph_bfs('citation_edges', {seed}, 'src', 'dst', NULL, \
                 '{{\"output_mode\": \"normalized\"}}')"
            ),
            false,
            false,
        ),
        spec(
            "bfs_deep_full_sorted",
            format!(
                "SELECT vertex, distance, predecessor, reachable FROM \
                 cugraph_bfs('citation_edges', {seed}, 'src', 'dst', NULL, \
                 '{{\"output_mode\": \"normalized\"}}') ORDER BY vertex"
            ),
            false,
            false,
        ),
        spec(
            "sssp",
            format!("SELECT COUNT(*) AS n FROM cugraph_sssp('citation_edges', {seed}, 'src', 'dst', 'weight')"),
            false,
            false,
        ),
        spec(
            "wcc",
            "SELECT COUNT(*) AS n FROM cugraph_weakly_connected_components('citation_edges', 'src', 'dst')".to_string(),
            false,
            false,
        ),
        spec(
            "louvain",
            "SELECT COUNT(DISTINCT partition) AS communities FROM cugraph_louvain('citation_edges', 'src', 'dst', \
             NULL, '{\"directed\": false}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "core_number",
            "SELECT COUNT(*) AS n FROM cugraph_core_number('citation_edges', 'src', 'dst', NULL, '{\"directed\": \
             false}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "triangles",
            "SELECT COUNT(*) AS n FROM cugraph_triangle_count_all('citation_edges', 'src', 'dst', NULL, \
             '{\"directed\": false}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "degrees",
            "SELECT COUNT(*) AS n FROM cugraph_degrees_all('citation_edges', 'src', 'dst')".to_string(),
            false,
            false,
        ),
        spec(
            "betweenness",
            "SELECT COUNT(*) AS n FROM cugraph_betweenness_centrality('citation_edges', 'src', 'dst', NULL, \
             '{\"k\": 64}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "rel_join",
            "SELECT p.year, COUNT(*) AS papers, AVG(p.n_citation) AS avg_citation FROM paper_fos f JOIN papers p ON \
             p.paper_id = f.paper_id WHERE f.fos_name = 'Artificial intelligence' GROUP BY p.year ORDER BY p.year"
                .to_string(),
            true,
            true,
        ),
        spec(
            "rel_scan_agg",
            "SELECT year, COUNT(*) AS papers, SUM(n_citation) AS citations FROM papers GROUP BY year ORDER BY year"
                .to_string(),
            true,
            false,
        ),
        spec(
            "graph_enrich",
            "WITH pr AS (SELECT vertex, value FROM cugraph_pagerank('citation_edges', 'src', 'dst', 'weight')) \
             SELECT p.year, COUNT(*) AS papers, MAX(pr.value) AS top_pagerank FROM pr JOIN papers p ON p.paper_id = \
             pr.vertex GROUP BY p.year ORDER BY p.year"
                .to_string(),
            true,
            false,
        ),
    ]
}

pub(super) fn expand_workload(config: &HarnessConfig, specs: &[QuerySpec]) -> Result<Vec<Request>, Box<dyn Error>> {
    let mut requests = Vec::new();
    let mut task_index = 0;
    for (label, count) in &config.workload {
        if label == "catalog_smoke" {
            for spec in specs {
                requests.push(Request { task_index, spec: spec.clone() });
                task_index += 1;
            }
            continue;
        }
        let spec = specs
            .iter()
            .find(|spec| &spec.label == label)
            .ok_or_else(|| format!("unknown workload label `{label}`"))?;
        for _ in 0..*count {
            requests.push(Request { task_index, spec: spec.clone() });
            task_index += 1;
        }
    }
    Ok(requests)
}
