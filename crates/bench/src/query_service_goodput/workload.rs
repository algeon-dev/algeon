use std::{collections::BTreeMap, error::Error, sync::Arc};

use algeon_datafusion::backend::AlgeonGpuBackend;
use arrow::array::Array as _;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use workloads::{
    tpcds::{TPCDS_QUERIES, register_tpcds_tables},
    tpch::{TPCH_QUERIES, register_tpch_tables},
};

use super::config::{ENV_GRAPH_EDGES, ENV_TPCDS_DIR, ENV_TPCH_DIR, GoodputConfig};

/// Table name the graph items read. The harness registers the configured
/// edge Parquet under this name so graph SQL is dataset independent.
pub(super) const GRAPH_EDGE_TABLE: &str = "edges";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum WorkloadFamily {
    Tpch,
    Tpcds,
    Adhoc,
    Graph,
}

impl WorkloadFamily {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Tpch => "tpch",
            Self::Tpcds => "tpcds",
            Self::Adhoc => "adhoc",
            Self::Graph => "graph",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct WorkloadItem {
    pub(super) id: String,
    pub(super) family: WorkloadFamily,
    pub(super) sql: String,
    /// Units of user-visible work in one execution of this item.
    ///
    /// One for SQL items. For graph items whose parameters set how much work
    /// a single query performs (betweenness seed count), this is that count,
    /// so a split configuration and a batched configuration are comparable on
    /// completed work rather than on completed query count.
    pub(super) work_units: u64,
}

#[derive(Debug, Clone)]
pub(super) struct MixEntry {
    pub(super) item_id: String,
    pub(super) weight: u32,
}

/// Parses `item[@weight],...` into a mix, rejecting empty and zero-weight input.
pub(super) fn parse_mix(raw: &str) -> Result<Vec<MixEntry>, Box<dyn Error>> {
    let mut mix = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|part| !part.is_empty()) {
        let (item_id, weight) = match part.split_once('@') {
            Some((item_id, weight)) => (
                item_id.trim(),
                weight.trim().parse::<u32>().map_err(|error| format!("failed to parse mix weight: {error}"))?,
            ),
            None => (part, 1),
        };
        if item_id.is_empty() {
            return Err("mix entries require an item id".into());
        }
        if weight == 0 {
            return Err(format!("mix weight for {item_id} must be greater than zero").into());
        }
        if mix.iter().any(|entry: &MixEntry| entry.item_id == item_id) {
            return Err(format!("duplicate mix entry {item_id}").into());
        }
        mix.push(MixEntry { item_id: item_id.to_string(), weight });
    }
    if mix.is_empty() {
        return Err("the mix must contain at least one item".into());
    }
    Ok(mix)
}

/// Ad-hoc skew and string items over the TPC-H tables.
///
/// These cover shapes the standard queries do not: unanchored string matching,
/// high-cardinality distinct state, a deliberately skewed join key filter, and
/// a global sort. Each one stresses a different memory profile under the same
/// query cap, so the mix is not dominated by one execution shape.
const ADHOC_ITEMS: &[(&str, &str)] = &[
    (
        "string_filter",
        "SELECT COUNT(*) AS rows_matched, SUM(l_extendedprice) AS revenue FROM lineitem WHERE l_comment LIKE \
         '%special%' AND l_shipinstruct = 'DELIVER IN PERSON'",
    ),
    (
        "string_group",
        "SELECT l_shipmode, COUNT(*) AS shipments, AVG(l_extendedprice) AS avg_price FROM lineitem GROUP BY \
         l_shipmode ORDER BY l_shipmode",
    ),
    (
        "wide_distinct",
        "SELECT COUNT(DISTINCT l_partkey) AS parts, COUNT(DISTINCT l_suppkey) AS suppliers, COUNT(DISTINCT \
         l_orderkey) AS orders FROM lineitem",
    ),
    (
        "skew_join",
        "SELECT o_orderpriority, COUNT(*) AS line_items, SUM(l_quantity) AS quantity FROM lineitem JOIN orders ON \
         l_orderkey = o_orderkey WHERE o_orderdate < DATE '1993-01-01' GROUP BY o_orderpriority ORDER BY \
         o_orderpriority",
    ),
    (
        "global_sort",
        "SELECT l_orderkey, l_partkey, l_extendedprice FROM lineitem ORDER BY l_extendedprice DESC, l_orderkey ASC \
         LIMIT 100",
    ),
];

/// Graph items and the seed-count parameter that sets their work units.
#[cfg(feature = "cugraph")]
const GRAPH_ITEMS: &[(&str, u64)] = &[
    ("pagerank", 1),
    ("bfs", 1),
    ("sssp", 1),
    ("wcc", 1),
    ("triangles", 1),
    ("betweenness_k64", 64),
    ("betweenness_k256", 256),
];

/// Builds one graph item's SQL.
///
/// Each item aggregates the algorithm's output rather than counting them, so a
/// wrong result cannot pass the harness fingerprint: a plain row count is the
/// vertex count for every vertex-keyed algorithm and would be identical whether
/// the scores were computed or not.
///
/// Which aggregate is usable depends on the algorithm's reproducibility.
/// Component labels are canonicalized to the minimum vertex id of each
/// component and betweenness scores accumulate in fixed source order, so both
/// are run-stable and summed directly like the other integer and score
/// outputs. A wrong result still cannot pass the fingerprint: a plain row
/// count is the vertex count for every vertex-keyed algorithm and would be
/// identical whether the scores were computed or not.
#[cfg(feature = "cugraph")]
fn graph_sql(name: &str, seed: i64) -> Option<String> {
    let sql = match name {
        "pagerank" => format!(
            "SELECT COUNT(*) AS n, SUM(value) AS total FROM cugraph_pagerank('{GRAPH_EDGE_TABLE}', 'src', 'dst', \
             'weight')"
        ),
        "bfs" => format!(
            "SELECT COUNT(*) AS reached, SUM(distance) AS total FROM cugraph_bfs('{GRAPH_EDGE_TABLE}', {seed}, \
             'src', 'dst', NULL, '{{\"output_mode\": \"normalized\"}}')"
        ),
        "sssp" => format!(
            "SELECT COUNT(*) AS n, SUM(distance) AS total FROM cugraph_sssp('{GRAPH_EDGE_TABLE}', {seed}, 'src', \
             'dst', 'weight')"
        ),
        // Labels are canonical minimum-vertex ids, so their sum is a stable
        // fingerprint of both the partition and its canonical assignment.
        "wcc" => format!(
            "SELECT COUNT(*) AS n, SUM(label) AS total FROM \
             cugraph_weakly_connected_components('{GRAPH_EDGE_TABLE}', 'src', 'dst')"
        ),
        "triangles" => format!(
            "SELECT COUNT(*) AS n, SUM(triangle_count) AS total FROM \
             cugraph_triangle_count_all('{GRAPH_EDGE_TABLE}', 'src', 'dst', NULL, '{{\"directed\": false}}')"
        ),
        "betweenness_k64" => format!(
            "SELECT COUNT(*) AS n, SUM(value) AS total FROM \
             cugraph_betweenness_centrality('{GRAPH_EDGE_TABLE}', 'src', 'dst', NULL, '{{\"k\": 64}}')"
        ),
        "betweenness_k256" => format!(
            "SELECT COUNT(*) AS n, SUM(value) AS total FROM \
             cugraph_betweenness_centrality('{GRAPH_EDGE_TABLE}', 'src', 'dst', NULL, '{{\"k\": 256}}')"
        ),
        _ => return None,
    };
    Some(sql)
}

/// Resolves one mix item id into its executable item.
///
/// `graph_seed` is the traversal source discovered from the registered edge
/// table; it is only consulted by graph items.
pub(super) fn resolve_item(item_id: &str, graph_seed: i64) -> Result<WorkloadItem, Box<dyn Error>> {
    let (family_name, name) =
        item_id.split_once('.').ok_or_else(|| format!("mix item {item_id} must be <family>.<name>"))?;
    match family_name {
        "tpch" => TPCH_QUERIES
            .iter()
            .find(|query| query.id == name)
            .map(|query| WorkloadItem {
                id: item_id.to_string(),
                family: WorkloadFamily::Tpch,
                sql: query.sql.to_string(),
                work_units: 1,
            })
            .ok_or_else(|| format!("unknown TPC-H query {name}").into()),
        "tpcds" => TPCDS_QUERIES
            .iter()
            .find(|query| query.id == name)
            .map(|query| WorkloadItem {
                id: item_id.to_string(),
                family: WorkloadFamily::Tpcds,
                sql: query.sql.to_string(),
                work_units: 1,
            })
            .ok_or_else(|| format!("unknown TPC-DS query {name}").into()),
        "adhoc" => ADHOC_ITEMS
            .iter()
            .find(|(id, _)| *id == name)
            .map(|(_, sql)| WorkloadItem {
                id: item_id.to_string(),
                family: WorkloadFamily::Adhoc,
                sql: (*sql).to_string(),
                work_units: 1,
            })
            .ok_or_else(|| format!("unknown ad-hoc item {name}").into()),
        "graph" => resolve_graph_item(item_id, name, graph_seed),
        other => Err(format!("unknown workload family {other} in mix item {item_id}").into()),
    }
}

#[cfg(feature = "cugraph")]
fn resolve_graph_item(item_id: &str, name: &str, graph_seed: i64) -> Result<WorkloadItem, Box<dyn Error>> {
    let work_units = GRAPH_ITEMS
        .iter()
        .find(|(id, _)| *id == name)
        .map(|(_, work_units)| *work_units)
        .ok_or_else(|| format!("unknown graph item {name}"))?;
    let sql = graph_sql(name, graph_seed).ok_or_else(|| format!("unknown graph item {name}"))?;
    Ok(WorkloadItem { id: item_id.to_string(), family: WorkloadFamily::Graph, sql, work_units })
}

#[cfg(not(feature = "cugraph"))]
fn resolve_graph_item(item_id: &str, _name: &str, _graph_seed: i64) -> Result<WorkloadItem, Box<dyn Error>> {
    Err(format!("mix item {item_id} requires the bench `cugraph` feature").into())
}

pub(super) fn mix_families(mix: &[MixEntry]) -> Result<Vec<WorkloadFamily>, Box<dyn Error>> {
    let mut families = Vec::new();
    for entry in mix {
        let family_name = entry
            .item_id
            .split('.')
            .next()
            .ok_or_else(|| format!("mix item {} must be <family>.<name>", entry.item_id))?;
        let family = match family_name {
            "tpch" => WorkloadFamily::Tpch,
            "tpcds" => WorkloadFamily::Tpcds,
            "adhoc" => WorkloadFamily::Adhoc,
            "graph" => WorkloadFamily::Graph,
            other => return Err(format!("unknown workload family {other}").into()),
        };
        if !families.contains(&family) {
            families.push(family);
        }
    }
    families.sort_unstable();
    Ok(families)
}

/// One session context per workload family, all backed by the same backend.
///
/// TPC-H and TPC-DS both define a `customer` table, so a single context cannot
/// hold both datasets.
pub(super) type FamilySessions = BTreeMap<WorkloadFamily, Arc<SessionContext>>;

pub(super) async fn build_sessions(
    backend: &AlgeonGpuBackend,
    config: &GoodputConfig,
) -> Result<FamilySessions, Box<dyn Error>> {
    let mut sessions = FamilySessions::new();
    for family in mix_families(&config.mix)? {
        // Ad-hoc items read the TPC-H tables, so both families answer from one
        // context instead of registering a second copy of the same dataset.
        let owner = if family == WorkloadFamily::Adhoc { WorkloadFamily::Tpch } else { family };
        if let Some(existing) = sessions.get(&owner).cloned() {
            sessions.insert(family, existing);
            continue;
        }
        let ctx = new_session(backend)?;
        match owner {
            WorkloadFamily::Tpch | WorkloadFamily::Adhoc => {
                let root =
                    config.tpch_root.as_deref().ok_or_else(|| format!("{ENV_TPCH_DIR} is required by this mix"))?;
                register_tpch_tables(&ctx, root).await?;
            },
            WorkloadFamily::Tpcds => {
                let root =
                    config.tpcds_root.as_deref().ok_or_else(|| format!("{ENV_TPCDS_DIR} is required by this mix"))?;
                register_tpcds_tables(&ctx, root).await?;
            },
            WorkloadFamily::Graph => {
                let edges = config
                    .graph_edges
                    .as_deref()
                    .ok_or_else(|| format!("{ENV_GRAPH_EDGES} is required by this mix"))?;
                ctx.register_parquet(GRAPH_EDGE_TABLE, edges.to_string_lossy().as_ref(), ParquetReadOptions::default())
                    .await?;
            },
        }
        let ctx = Arc::new(ctx);
        sessions.insert(owner, Arc::clone(&ctx));
        sessions.insert(family, ctx);
    }
    Ok(sessions)
}

fn new_session(backend: &AlgeonGpuBackend) -> Result<SessionContext, Box<dyn Error>> {
    let session_config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    let builder = datafusion::execution::SessionStateBuilder::new().with_config(session_config).with_default_features();
    let builder = install_workload_sql(backend.install_on(builder)?)?;
    Ok(SessionContext::new_with_state(builder.build()))
}

#[cfg(feature = "cugraph")]
fn install_workload_sql(
    builder: datafusion::execution::SessionStateBuilder,
) -> Result<datafusion::execution::SessionStateBuilder, Box<dyn Error>> {
    use algeon_datafusion::{cugraph_sql::CugraphSqlConfig, session::AlgeonSessionStateBuilderExt};

    Ok(builder.try_with_cugraph_sql(CugraphSqlConfig::default())?)
}

#[cfg(not(feature = "cugraph"))]
fn install_workload_sql(
    builder: datafusion::execution::SessionStateBuilder,
) -> Result<datafusion::execution::SessionStateBuilder, Box<dyn Error>> {
    Ok(builder)
}

/// Discovers the traversal source used by the seeded graph items.
pub(super) async fn discover_graph_seed(ctx: &SessionContext) -> Result<i64, Box<dyn Error>> {
    let batches = ctx.sql(&format!("SELECT MIN(src) AS seed FROM {GRAPH_EDGE_TABLE}")).await?.collect().await?;
    let seed = batches
        .first()
        .and_then(|batch| batch.column(0).as_any().downcast_ref::<arrow::array::Int64Array>())
        .filter(|column| !column.is_empty() && !column.is_null(0))
        .map(|column| column.value(0))
        .ok_or("graph seed discovery returned no vertex")?;
    Ok(seed)
}
