use super::*;

const LARGE_KEYED_JOIN_GATE_ENV: &str = "CUDF_TPCH_BENCH_LARGE_KEYED_JOIN_GATE";
const LARGE_KEYED_JOIN_GATE_QUERY_IDS_ENV: &str = "CUDF_TPCH_BENCH_LARGE_KEYED_JOIN_GATE_QUERY_IDS";
const LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD: u64 = 64_000_000;
const LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LargeKeyedJoinStrategyGateSummary {
    pub(super) query_ids: Vec<String>,
    pub(super) large_one_shot_sites: usize,
    pub(super) large_reusable_sites: usize,
    pub(super) reservation_denied_count: u64,
}

impl LargeKeyedJoinStrategyGateSummary {
    pub(super) fn progress_line(&self) -> String {
        format!(
            "[native_tpch_benchmark] phase=large_keyed_join_strategy_gate outcome=passed query_ids={} large_one_shot_sites={} large_reusable_sites={} reservation_denied_count={}",
            self.query_ids.join(","),
            self.large_one_shot_sites,
            self.large_reusable_sites,
            self.reservation_denied_count,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LargeKeyedJoinStrategyGateConfig {
    pub(super) enabled: bool,
    pub(super) query_ids: Option<BTreeSet<String>>,
}

impl LargeKeyedJoinStrategyGateConfig {
    fn from_env() -> DataFusionResult<Self> {
        let enabled = match std::env::var_os(LARGE_KEYED_JOIN_GATE_ENV) {
            None => false,
            Some(value) => match value.to_string_lossy().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                other => {
                    return Err(plan_error(format!(
                        "invalid {LARGE_KEYED_JOIN_GATE_ENV}: {other}; expected true/false"
                    )));
                },
            },
        };
        Ok(Self { enabled, query_ids: optional_query_ids_from_env(LARGE_KEYED_JOIN_GATE_QUERY_IDS_ENV)? })
    }
}

pub(super) fn validate_large_keyed_join_strategy_gate_from_env(
    tsv: &str,
) -> DataFusionResult<Option<LargeKeyedJoinStrategyGateSummary>> {
    let config = LargeKeyedJoinStrategyGateConfig::from_env()?;
    if !config.enabled {
        return Ok(None);
    }
    let summary = validate_large_keyed_join_strategy_gate_tsv(tsv, &config)?;
    Ok(Some(summary))
}

pub(super) fn validate_large_keyed_join_strategy_gate_tsv(
    tsv: &str,
    config: &LargeKeyedJoinStrategyGateConfig,
) -> DataFusionResult<LargeKeyedJoinStrategyGateSummary> {
    let report = ParsedBenchmarkTsv::parse(tsv)?;
    let rows = report.selected_rows(config)?;
    let mut large_one_shot_sites = Vec::new();
    let mut large_reusable_sites = Vec::new();
    let mut reservation_denied_count = 0_u64;
    let mut runtime_failures = Vec::new();

    for row in &rows {
        reservation_denied_count = reservation_denied_count.saturating_add(row.reservation_denied_count);
        if row.runtime_outcome != "succeeded" {
            runtime_failures.push(format!("{}:{}", row.query_id, row.runtime_outcome));
        }
        for join in &row.join_strategy_rows {
            if !join.large_build_site() {
                continue;
            }
            match join.strategy() {
                "one_shot_equi" => large_one_shot_sites.push(join.site_label(&row.query_id)),
                "reusable_hash_equi" => large_reusable_sites.push(join.site_label(&row.query_id)),
                _ => {},
            }
        }
    }

    if !runtime_failures.is_empty() {
        return Err(gate_error(format!(
            "large keyed join strategy gate requires native runtime success; failures={}",
            runtime_failures.join(",")
        )));
    }
    if reservation_denied_count > 0 {
        return Err(gate_error(format!(
            "large keyed join strategy gate requires reservation_denied_count=0; observed {reservation_denied_count}"
        )));
    }
    if !large_reusable_sites.is_empty() {
        return Err(gate_error(format!(
            "large keyed join strategy gate found large reusable_hash_equi sites: {}",
            large_reusable_sites.join("|")
        )));
    }
    if large_one_shot_sites.is_empty() {
        return Err(gate_error("large keyed join strategy gate found no large one_shot_equi sites in selected rows"));
    }

    Ok(LargeKeyedJoinStrategyGateSummary {
        query_ids: rows.iter().map(|row| row.query_id.clone()).collect(),
        large_one_shot_sites: large_one_shot_sites.len(),
        large_reusable_sites: large_reusable_sites.len(),
        reservation_denied_count,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedBenchmarkTsv {
    rows: Vec<ParsedBenchmarkRow>,
}

impl ParsedBenchmarkTsv {
    fn parse(tsv: &str) -> DataFusionResult<Self> {
        let mut lines = tsv.lines().filter(|line| !line.starts_with('#') && !line.trim().is_empty());
        let header = lines.next().ok_or_else(|| plan_error("native TPC-H benchmark TSV is missing a header"))?;
        let header_fields = header.split('\t').collect::<Vec<_>>();
        let query_id_index = header_index(&header_fields, "query_id")?;
        let runtime_outcome_index = header_index(&header_fields, "runtime_outcome")?;
        let reservation_denied_count_index = header_index(&header_fields, "reservation_denied_count")?;
        let join_strategy_rows_index = header_index(&header_fields, "join_strategy_rows")?;

        let mut rows = Vec::new();
        for line in lines {
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != header_fields.len() {
                return Err(plan_error(format!(
                    "native TPC-H benchmark TSV row has {} fields but header has {}: {line}",
                    fields.len(),
                    header_fields.len()
                )));
            }
            rows.push(ParsedBenchmarkRow {
                query_id: fields[query_id_index].to_string(),
                runtime_outcome: fields[runtime_outcome_index].to_string(),
                reservation_denied_count: parse_u64_field(
                    "reservation_denied_count",
                    fields[reservation_denied_count_index],
                )?,
                join_strategy_rows: parse_join_strategy_rows(fields[join_strategy_rows_index])?,
            });
        }
        Ok(Self { rows })
    }

    fn selected_rows(&self, config: &LargeKeyedJoinStrategyGateConfig) -> DataFusionResult<Vec<ParsedBenchmarkRow>> {
        let rows = self
            .rows
            .iter()
            .filter(|row| config.query_ids.as_ref().is_none_or(|query_ids| query_ids.contains(row.query_id.as_str())))
            .cloned()
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return Err(gate_error("large keyed join strategy gate selected no benchmark rows"));
        }
        if let Some(query_ids) = &config.query_ids {
            let observed = rows.iter().map(|row| row.query_id.as_str()).collect::<BTreeSet<_>>();
            let missing = query_ids.iter().filter(|query_id| !observed.contains(query_id.as_str())).collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(gate_error(format!(
                    "large keyed join strategy gate missing selected query rows: {}",
                    missing.into_iter().map(String::as_str).collect::<Vec<_>>().join(",")
                )));
            }
        }
        Ok(rows)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedBenchmarkRow {
    query_id: String,
    runtime_outcome: String,
    reservation_denied_count: u64,
    join_strategy_rows: Vec<ParsedJoinStrategyRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedJoinStrategyRow {
    fields: BTreeMap<String, String>,
}

impl ParsedJoinStrategyRow {
    fn strategy(&self) -> &str {
        self.fields.get("join_strategy").map_or("unknown", String::as_str)
    }

    fn large_build_site(&self) -> bool {
        ["estimated_build_rows", "observed_build_rows"]
            .iter()
            .filter_map(|field| self.u64_field(field))
            .any(|rows| rows >= LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD)
            || self.u64_field("build_side_bytes").is_some_and(|bytes| bytes >= LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD)
    }

    fn site_label(&self, query_id: &str) -> String {
        format!(
            "{}:{}:{}:estimated_build_rows={}:observed_build_rows={}:build_side_bytes={}",
            query_id,
            self.fields.get("node_id").map_or("unknown", String::as_str),
            self.strategy(),
            self.fields.get("estimated_build_rows").map_or("none", String::as_str),
            self.fields.get("observed_build_rows").map_or("none", String::as_str),
            self.fields.get("build_side_bytes").map_or("none", String::as_str),
        )
    }

    fn u64_field(&self, field: &str) -> Option<u64> {
        let value = self.fields.get(field)?;
        if value == "none" {
            return None;
        }
        value.parse().ok()
    }
}

fn parse_join_strategy_rows(value: &str) -> DataFusionResult<Vec<ParsedJoinStrategyRow>> {
    if value == "none" {
        return Ok(Vec::new());
    }
    value
        .split('|')
        .map(|row| {
            let mut fields = BTreeMap::new();
            for field in row.split(',') {
                let Some((key, value)) = field.split_once('=') else {
                    return Err(plan_error(format!("invalid join_strategy_rows field segment: {field}")));
                };
                fields.insert(key.to_string(), value.to_string());
            }
            Ok(ParsedJoinStrategyRow { fields })
        })
        .collect()
}

fn optional_query_ids_from_env(name: &'static str) -> DataFusionResult<Option<BTreeSet<String>>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    let query_ids = value
        .to_string_lossy()
        .split(',')
        .map(str::trim)
        .filter(|query_id| !query_id.is_empty())
        .map(ToString::to_string)
        .collect::<BTreeSet<_>>();
    if query_ids.is_empty() {
        return Err(plan_error(format!("{name} must name at least one query id when set")));
    }
    Ok(Some(query_ids))
}

fn header_index(header: &[&str], name: &str) -> DataFusionResult<usize> {
    header.iter().position(|field| *field == name).ok_or_else(|| {
        plan_error(format!("native TPC-H benchmark TSV is missing required large keyed join gate field {name}"))
    })
}

fn parse_u64_field(name: &str, value: &str) -> DataFusionResult<u64> {
    value.parse::<u64>().map_err(|error| plan_error(format!("invalid {name} value {value}: {error}")))
}

fn gate_error(message: impl Into<String>) -> DataFusionError {
    plan_error(format!("native TPC-H benchmark large keyed join strategy gate failed: {}", message.into()))
}
