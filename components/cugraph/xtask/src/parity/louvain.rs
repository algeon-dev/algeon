use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    io::ErrorKind as IoErrorKind,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, anyhow, bail};
use clap::{Parser, Subcommand};
use cudf::{CudfType, DataFrame, Runtime, Series, TypeId};
use cugraph::{
    community::LouvainParams,
    cudf_interop::{
        DataFrameGraphOptions, GraphConstructionPolicy, device_array_view_from_slice, graph_from_dataframe,
    },
    graph::{Forward, Graph, GraphBuilder},
    handle::ResourceHandle,
    types::{CugraphElement, GraphProperties},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CASE_NAME: &str = "louvain";
pub const MODULARITY_TOLERANCE_ABS: f64 = 1e-6;
pub const LARGEST_COMMUNITIES_TOP_K: usize = 8;
pub const MEMBERSHIP_SAMPLE_SIZE: usize = 8;
const FIXTURE_ENV: &str = "DTGGRAPH_PARITY_FIXTURE";
const PYTHON_HELPER_PATH: &str = "../tools/parity/python/louvain_baseline.py";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    pub exit_code: i32,
    pub json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HarnessFailure {
    kind: String,
    message: String,
}

#[derive(Debug, Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: TopLevelCommand,
}

#[derive(Debug, Subcommand)]
enum TopLevelCommand {
    Parity(ParityCommand),
}

#[derive(Debug, Parser)]
struct ParityCommand {
    #[command(subcommand)]
    case: ParityCase,
}

#[derive(Debug, Subcommand)]
enum ParityCase {
    Louvain(LouvainArgs),
}

#[derive(Debug, Parser)]
pub struct LouvainArgs {
    #[arg(long)]
    fixture: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EdgeColumns {
    src: &'static str,
    dst: &'static str,
    weight: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeSchemaKind {
    CanonicalNumeric,
    AliasNumeric,
    AliasStringNeedsRenumbering,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EdgeSchema {
    columns: EdgeColumns,
    kind: EdgeSchemaKind,
}

struct PreparedFixture {
    rust_df: DataFrame,
    python_fixture_path: PathBuf,
    _temp_fixture: Option<TempFixtureFile>,
}

struct TempFixtureFile {
    path: PathBuf,
}

impl Drop for TempFixtureFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Mismatch,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffSeverity {
    Hard,
    Soft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathKind {
    PythonGraphFromCudfEdgelist,
    RustGraphBuilder,
    RustGraphFromDataframe,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LouvainFlags {
    pub undirected: bool,
    pub renumber: bool,
    pub weighted: bool,
    pub drop_multi_edges: bool,
    pub symmetrize: bool,
    pub do_expensive_check: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MembershipSampleEntry {
    pub vertex: i64,
    pub community_size: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LouvainSummary {
    pub path_kind: PathKind,
    pub flags: LouvainFlags,
    pub input_edge_rows: usize,
    pub graph_vertex_count: usize,
    pub graph_edge_count: usize,
    pub community_count: usize,
    pub modularity: f64,
    pub community_size_histogram_sorted: Vec<usize>,
    pub largest_communities_top_k: Vec<usize>,
    pub membership_sample: Vec<MembershipSampleEntry>,
    pub normalized_partition_shape_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LouvainDiff {
    pub path_kind: PathKind,
    pub flags: LouvainFlags,
    pub field: String,
    pub severity: DiffSeverity,
    pub baseline: Value,
    pub candidate: Value,
    pub delta_abs: Option<f64>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportError {
    pub kind: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LouvainParityReport {
    pub case: String,
    pub fixture: String,
    pub python_summary: Option<LouvainSummary>,
    pub rust_summaries: Vec<LouvainSummary>,
    pub diffs: Vec<LouvainDiff>,
    pub verdict: Verdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ReportError>,
}

pub fn run(args: impl IntoIterator<Item = OsString>) -> RunOutcome {
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            let report =
                error_report("<unresolved-fixture>", "usage_error", error.render().to_string().trim().to_string());
            return RunOutcome { exit_code: 2, json: serde_json::to_string(&report).expect("serialize usage error") };
        },
    };

    match cli.command {
        TopLevelCommand::Parity(ParityCommand { case: ParityCase::Louvain(args) }) => run_louvain(args),
    }
}

fn run_louvain(args: LouvainArgs) -> RunOutcome {
    let fixture =
        match resolve_fixture_path(args.fixture.as_deref(), std::env::var_os(FIXTURE_ENV), std::env::current_dir()) {
            Ok(path) => path,
            Err(error) => {
                let report = error_report("<unresolved-fixture>", "usage_error", error);
                return RunOutcome {
                    exit_code: 2,
                    json: serde_json::to_string(&report).expect("serialize usage error"),
                };
            },
        };

    match execute_louvain_harness(&fixture) {
        Ok(report) => {
            let exit_code = match report.verdict {
                Verdict::Pass => 0,
                Verdict::Mismatch => 1,
                Verdict::Error => 2,
            };
            RunOutcome { exit_code, json: serde_json::to_string(&report).expect("serialize louvain parity report") }
        },
        Err(error) => {
            let report = error_report(fixture.display().to_string(), error.kind, error.message);
            RunOutcome { exit_code: 2, json: serde_json::to_string(&report).expect("serialize environment error") }
        },
    }
}

fn error_report(
    fixture: impl Into<String>,
    kind: impl Into<String>,
    message: impl Into<String>,
) -> LouvainParityReport {
    LouvainParityReport {
        case: CASE_NAME.to_string(),
        fixture: fixture.into(),
        python_summary: None,
        rust_summaries: Vec::new(),
        diffs: Vec::new(),
        verdict: Verdict::Error,
        error: Some(ReportError { kind: kind.into(), message: message.into() }),
    }
}

fn environment_failure(message: impl Into<String>) -> HarnessFailure {
    HarnessFailure { kind: "environment_error".to_string(), message: message.into() }
}

fn environment_failure_from_error(error: anyhow::Error) -> HarnessFailure {
    environment_failure(format!("{error:#}"))
}

fn execute_louvain_harness(fixture: &Path) -> Result<LouvainParityReport, HarnessFailure> {
    let fixture = prepare_fixture_path(fixture).map_err(environment_failure_from_error)?;
    let runtime =
        Runtime::new().context("build cudf runtime for parity harness").map_err(environment_failure_from_error)?;
    let handle = ResourceHandle::for_runtime_device(&runtime)
        .context("build cuGraph resource handle for parity harness")
        .map_err(environment_failure_from_error)?;
    let prepared = prepare_parity_fixture(&runtime, &fixture).map_err(environment_failure_from_error)?;

    let python_summary = run_python_baseline(&prepared.python_fixture_path)?;
    let rust_summaries = run_rust_cases(&handle, &prepared.rust_df).map_err(environment_failure_from_error)?;

    let mut diffs = Vec::new();
    for summary in &rust_summaries {
        diffs.extend(compare_summaries(&python_summary, summary));
    }

    let verdict =
        if diffs.iter().any(|diff| diff.severity == DiffSeverity::Hard) { Verdict::Mismatch } else { Verdict::Pass };

    Ok(LouvainParityReport {
        case: CASE_NAME.to_string(),
        fixture: fixture.display().to_string(),
        python_summary: Some(python_summary),
        rust_summaries,
        diffs,
        verdict,
        error: None,
    })
}

fn prepare_fixture_path(path: &Path) -> anyhow::Result<PathBuf> {
    if !path.exists() {
        bail!("fixture not found at {}", path.display());
    }
    if !path.is_file() {
        bail!("fixture path is not a regular file: {}", path.display());
    }

    path.canonicalize().with_context(|| format!("canonicalize fixture path {}", path.display()))
}

fn load_fixture_dataframe(runtime: &Runtime, fixture: &Path) -> anyhow::Result<DataFrame> {
    let extension = fixture.extension().and_then(|ext| ext.to_str()).unwrap_or_default();
    if !extension.eq_ignore_ascii_case("parquet") {
        bail!("phase 1 parity harness only supports parquet fixtures, got {}", fixture.display());
    }

    DataFrame::read_parquet(runtime, fixture).with_context(|| format!("read parity fixture {}", fixture.display()))
}

fn prepare_parity_fixture(runtime: &Runtime, fixture: &Path) -> anyhow::Result<PreparedFixture> {
    let raw_df = load_fixture_dataframe(runtime, fixture)?;
    let schema = detect_edge_schema(&raw_df)?;
    let normalized_df = normalize_fixture_dataframe(&raw_df, &schema)?;
    validate_normalized_fixture_dataframe(&normalized_df)?;

    match schema.kind {
        EdgeSchemaKind::CanonicalNumeric => Ok(PreparedFixture {
            rust_df: normalized_df,
            python_fixture_path: fixture.to_path_buf(),
            _temp_fixture: None,
        }),
        EdgeSchemaKind::AliasNumeric | EdgeSchemaKind::AliasStringNeedsRenumbering => {
            let temp_fixture = TempFixtureFile { path: unique_temp_parquet_path("louvain_parity_normalized") };
            normalized_df
                .write_parquet(&temp_fixture.path)
                .with_context(|| format!("write normalized parity fixture {}", temp_fixture.path.display()))?;
            Ok(PreparedFixture {
                rust_df: normalized_df,
                python_fixture_path: temp_fixture.path.clone(),
                _temp_fixture: Some(temp_fixture),
            })
        },
    }
}

fn detect_edge_schema(df: &DataFrame) -> anyhow::Result<EdgeSchema> {
    let columns = df.column_names();
    let has = |name: &str| columns.contains(&name);
    let available = columns.join(", ");

    if has("src") && has("dst") && has("weight") {
        let src_type = df.series("src").context("load src column")?.dtype().context("inspect src dtype")?.id();
        let dst_type = df.series("dst").context("load dst column")?.dtype().context("inspect dst dtype")?.id();
        let weight_type =
            df.series("weight").context("load weight column")?.dtype().context("inspect weight dtype")?.id();
        let kind = classify_schema_kind(src_type, dst_type, weight_type, "src", "dst")?;
        return Ok(EdgeSchema { columns: EdgeColumns { src: "src", dst: "dst", weight: "weight" }, kind });
    }

    if has("sourceId") && has("targetId") && has("weight") {
        let src_type =
            df.series("sourceId").context("load sourceId column")?.dtype().context("inspect sourceId dtype")?.id();
        let dst_type =
            df.series("targetId").context("load targetId column")?.dtype().context("inspect targetId dtype")?.id();
        let weight_type =
            df.series("weight").context("load weight column")?.dtype().context("inspect weight dtype")?.id();
        let kind = classify_schema_kind(src_type, dst_type, weight_type, "sourceId", "targetId")?;
        return Ok(EdgeSchema { columns: EdgeColumns { src: "sourceId", dst: "targetId", weight: "weight" }, kind });
    }

    bail!(
        "fixture must contain either [src, dst, weight] or [sourceId, targetId, weight]; available columns: [{available}]"
    );
}

fn classify_schema_kind(
    src_type: TypeId,
    dst_type: TypeId,
    weight_type: TypeId,
    src_name: &str,
    dst_name: &str,
) -> anyhow::Result<EdgeSchemaKind> {
    if !matches!(weight_type, TypeId::Float32 | TypeId::Float64) {
        bail!("fixture weight column must use Float32 or Float64 values, got {:?}", weight_type);
    }
    if dst_type != src_type {
        bail!("fixture {dst_name} column dtype {:?} does not match {src_name} column dtype {:?}", dst_type, src_type);
    }

    match src_type {
        TypeId::Int32 | TypeId::Int64 => {
            if src_name == "src" && dst_name == "dst" {
                Ok(EdgeSchemaKind::CanonicalNumeric)
            } else {
                Ok(EdgeSchemaKind::AliasNumeric)
            }
        },
        TypeId::String => Ok(EdgeSchemaKind::AliasStringNeedsRenumbering),
        other => bail!("fixture vertex columns must use Int32, Int64, or String values, got {:?}", other),
    }
}

fn normalize_fixture_dataframe(df: &DataFrame, schema: &EdgeSchema) -> anyhow::Result<DataFrame> {
    let weight = df.series(schema.columns.weight).with_context(|| format!("load {} column", schema.columns.weight))?;
    let weight = normalize_weight_series(&weight)?;

    match schema.kind {
        EdgeSchemaKind::CanonicalNumeric => {
            let src = df.series("src").context("load src column")?;
            let dst = df.series("dst").context("load dst column")?;
            Ok(DataFrame::from_columns([("src", src), ("dst", dst), ("weight", weight)])
                .context("rebuild canonical numeric parity DataFrame")?)
        },
        EdgeSchemaKind::AliasNumeric => {
            let src = df.series(schema.columns.src).with_context(|| format!("load {} column", schema.columns.src))?;
            let dst = df.series(schema.columns.dst).with_context(|| format!("load {} column", schema.columns.dst))?;
            Ok(DataFrame::from_columns([("src", src), ("dst", dst), ("weight", weight)])
                .context("rename aliased numeric edge columns")?)
        },
        EdgeSchemaKind::AliasStringNeedsRenumbering => {
            let src = df.series(schema.columns.src).with_context(|| format!("load {} column", schema.columns.src))?;
            let dst = df.series(schema.columns.dst).with_context(|| format!("load {} column", schema.columns.dst))?;
            let all_vertices = Series::concat(&[&src, &dst]).context("concatenate string vertex columns")?;
            let codebook = all_vertices.codebook().context("build vertex codebook for string edge list")?;
            let src_codes = codebook.encode(&src).context("encode source string vertex ids")?.into_series();
            let dst_codes = codebook.encode(&dst).context("encode destination string vertex ids")?.into_series();
            Ok(DataFrame::from_columns([("src", src_codes), ("dst", dst_codes), ("weight", weight)])
                .context("build normalized parity edge DataFrame")?)
        },
    }
}

fn normalize_weight_series(weight: &Series) -> anyhow::Result<Series> {
    if weight.null_count() == 0 {
        return Ok(weight.clone());
    }

    match weight.dtype().context("inspect weight dtype for fillna")?.id() {
        TypeId::Float32 => weight.fillna(1.0f32).context("fill null edge weights with 1.0f32"),
        TypeId::Float64 => weight.fillna(1.0f64).context("fill null edge weights with 1.0f64"),
        other => bail!("fixture weight column must use Float32 or Float64 values, got {:?}", other),
    }
}

fn validate_normalized_fixture_dataframe(df: &DataFrame) -> anyhow::Result<()> {
    let src = df.series("src").context("load normalized src column")?;
    let dst = df.series("dst").context("load normalized dst column")?;
    let weight = df.series("weight").context("load normalized weight column")?;

    for (name, series) in [("src", &src), ("dst", &dst), ("weight", &weight)] {
        if series.null_count() > 0 {
            bail!("fixture column '{name}' contains {} null values", series.null_count());
        }
    }

    let src_type = src.dtype().context("inspect normalized src dtype")?.id();
    let dst_type = dst.dtype().context("inspect normalized dst dtype")?.id();
    let weight_type = weight.dtype().context("inspect normalized weight dtype")?.id();

    if !matches!(src_type, TypeId::Int32 | TypeId::Int64) {
        bail!("fixture src column must use Int32 or Int64 vertex ids, got {:?}", src_type);
    }
    if dst_type != src_type {
        bail!("fixture dst column dtype {:?} does not match src column dtype {:?}", dst_type, src_type);
    }
    if !matches!(weight_type, TypeId::Float32 | TypeId::Float64) {
        bail!("fixture weight column must use Float32 or Float64 values, got {:?}", weight_type);
    }

    Ok(())
}

fn unique_temp_parquet_path(prefix: &str) -> PathBuf {
    let nanos =
        SystemTime::now().duration_since(UNIX_EPOCH).expect("system time before unix epoch is unsupported").as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{}_{}.parquet", std::process::id(), nanos))
}

fn run_rust_cases(handle: &ResourceHandle, fixture_df: &DataFrame) -> anyhow::Result<Vec<LouvainSummary>> {
    let builder_python_like = LouvainFlags {
        undirected: true,
        renumber: true,
        weighted: true,
        drop_multi_edges: true,
        symmetrize: true,
        do_expensive_check: false,
    };
    let builder_common = LouvainFlags {
        undirected: true,
        renumber: true,
        weighted: true,
        drop_multi_edges: false,
        symmetrize: false,
        do_expensive_check: false,
    };
    let dataframe_flags = LouvainFlags {
        undirected: true,
        renumber: true,
        weighted: true,
        drop_multi_edges: true,
        symmetrize: true,
        do_expensive_check: false,
    };

    Ok(vec![
        run_graph_builder_case(handle, fixture_df, builder_python_like)?,
        run_graph_builder_case(handle, fixture_df, builder_common)?,
        run_graph_from_dataframe_case(handle, fixture_df, dataframe_flags)?,
    ])
}

fn run_graph_builder_case(
    handle: &ResourceHandle,
    fixture_df: &DataFrame,
    flags: LouvainFlags,
) -> anyhow::Result<LouvainSummary> {
    let src = fixture_df.series("src").context("load fixture src series for GraphBuilder case")?;
    let dst = fixture_df.series("dst").context("load fixture dst series for GraphBuilder case")?;
    let weight = fixture_df.series("weight").context("load fixture weight series for GraphBuilder case")?;

    match (src.dtype().context("inspect src dtype")?.id(), weight.dtype().context("inspect weight dtype")?.id()) {
        (TypeId::Int32, TypeId::Float32) => {
            run_graph_builder_case_typed::<i32, f32>(handle, &src, &dst, &weight, flags)
        },
        (TypeId::Int32, TypeId::Float64) => {
            run_graph_builder_case_typed::<i32, f64>(handle, &src, &dst, &weight, flags)
        },
        (TypeId::Int64, TypeId::Float32) => {
            run_graph_builder_case_typed::<i64, f32>(handle, &src, &dst, &weight, flags)
        },
        (TypeId::Int64, TypeId::Float64) => {
            run_graph_builder_case_typed::<i64, f64>(handle, &src, &dst, &weight, flags)
        },
        (vertex_type, weight_type) => bail!(
            "unsupported fixture dtype combination for GraphBuilder case: vertex {:?}, weight {:?}",
            vertex_type,
            weight_type
        ),
    }
}

fn run_graph_builder_case_typed<V, W>(
    handle: &ResourceHandle,
    src: &Series,
    dst: &Series,
    weight: &Series,
    flags: LouvainFlags,
) -> anyhow::Result<LouvainSummary>
where
    V: CudfType + CugraphElement,
    W: CudfType + CugraphElement,
{
    let src_column = src.as_column();
    let dst_column = dst.as_column();
    let weight_column = weight.as_column();

    let src_slice = src_column.device_slice::<V>().context("borrow src device slice")?;
    let dst_slice = dst_column.device_slice::<V>().context("borrow dst device slice")?;
    let weight_slice = weight_column.device_slice::<W>().context("borrow weight device slice")?;

    let src_view = device_array_view_from_slice::<V>(&src_slice).context("bridge src cudf slice into cugraph view")?;
    let dst_view = device_array_view_from_slice::<V>(&dst_slice).context("bridge dst cudf slice into cugraph view")?;
    let weight_view =
        device_array_view_from_slice::<W>(&weight_slice).context("bridge weight cudf slice into cugraph view")?;

    let mut builder = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weight_view)
        .properties(GraphProperties::undirected())
        .renumber();
    if flags.drop_multi_edges {
        builder = builder.drop_multi_edges();
    }
    if flags.symmetrize {
        builder = builder.symmetrize();
    }
    if flags.do_expensive_check {
        builder = builder.do_expensive_check();
    }

    let mut graph = builder.build::<Forward>(handle).context("build parity GraphBuilder graph")?;
    run_louvain_case(handle, &mut graph, PathKind::RustGraphBuilder, flags, src.len())
}

fn run_graph_from_dataframe_case(
    handle: &ResourceHandle,
    fixture_df: &DataFrame,
    flags: LouvainFlags,
) -> anyhow::Result<LouvainSummary> {
    let options = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber()
        .construction_policy(GraphConstructionPolicy::PythonCugraph);
    let mut graph = graph_from_dataframe::<Forward>(handle, fixture_df, &options)
        .context("build parity graph_from_dataframe graph")?;
    run_louvain_case(handle, &mut graph, PathKind::RustGraphFromDataframe, flags, fixture_df.num_rows())
}

fn run_louvain_case(
    handle: &ResourceHandle,
    graph: &mut Graph<Forward>,
    path_kind: PathKind,
    flags: LouvainFlags,
    input_edge_rows: usize,
) -> anyhow::Result<LouvainSummary> {
    let graph_edge_count = graph_edge_count(handle, graph)?;
    let params = LouvainParams { do_expensive_check: flags.do_expensive_check, ..LouvainParams::default() };
    let result = graph.louvain(handle, &params).context("run louvain parity case")?;
    let vertices =
        result.vertex_ids().context("access Louvain vertices")?.to_vec_i64(handle).context("copy Louvain vertices")?;
    let communities =
        result.clusters().context("access Louvain clusters")?.to_vec_i64(handle).context("copy Louvain clusters")?;

    if vertices.len() != communities.len() {
        bail!("Louvain result returned {} vertices but {} cluster assignments", vertices.len(), communities.len());
    }

    let memberships = vertices.into_iter().zip(communities).collect::<Vec<_>>();
    Ok(build_summary(
        path_kind,
        flags,
        input_edge_rows,
        memberships.len(),
        graph_edge_count,
        result.modularity(),
        &memberships,
    ))
}

fn graph_edge_count(handle: &ResourceHandle, graph: &mut Graph<Forward>) -> anyhow::Result<usize> {
    graph.edge_count(handle).context("query graph edge count")
}

fn run_python_baseline(fixture: &Path) -> Result<LouvainSummary, HarnessFailure> {
    #[derive(Debug, Deserialize)]
    struct PythonHelperEnvelope {
        python_summary: Option<LouvainSummary>,
        error: Option<ReportError>,
    }

    let script = python_helper_path().map_err(environment_failure_from_error)?;
    let output = run_python_helper(&script, fixture).map_err(environment_failure_from_error)?;

    if output.stdout.is_empty() {
        return Err(environment_failure(format!(
            "python baseline helper produced empty stdout (status {})",
            output.status
        )));
    }

    let envelope: PythonHelperEnvelope = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parse python baseline helper JSON from {}", script.display()))
        .map_err(environment_failure_from_error)?;

    if output.status.success() {
        return envelope.python_summary.ok_or_else(|| {
            environment_failure("python baseline helper succeeded but did not return python_summary".to_string())
        });
    }

    if let Some(error) = envelope.error {
        return Err(HarnessFailure { kind: error.kind, message: error.message });
    }

    Err(environment_failure(format!(
        "python baseline helper failed with status {} without structured error",
        output.status
    )))
}

fn python_helper_path() -> anyhow::Result<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(PYTHON_HELPER_PATH);
    if !path.exists() {
        bail!("python baseline helper is missing at {}", path.display());
    }

    match path.canonicalize() {
        Ok(canonical) => Ok(canonical),
        Err(_) => Ok(path),
    }
}

fn python_interpreter_candidates(virtual_env: Option<OsString>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(virtual_env) = virtual_env {
        let mut path = PathBuf::from(virtual_env);
        path.push("bin");
        path.push("python");
        candidates.push(path);
    }

    candidates.push(PathBuf::from("python3"));
    candidates.push(PathBuf::from("python"));
    candidates
}

fn run_python_helper(script: &Path, fixture: &Path) -> anyhow::Result<Output> {
    let mut last_not_found = None;

    for interpreter in python_interpreter_candidates(std::env::var_os("VIRTUAL_ENV")) {
        let mut command = Command::new(&interpreter);
        command
            .arg(script)
            .arg("--fixture")
            .arg(fixture)
            .arg("--modularity-tolerance-abs")
            .arg(MODULARITY_TOLERANCE_ABS.to_string())
            .arg("--largest-communities-top-k")
            .arg(LARGEST_COMMUNITIES_TOP_K.to_string())
            .arg("--membership-sample-size")
            .arg(MEMBERSHIP_SAMPLE_SIZE.to_string());

        match command.output() {
            Ok(output) => return Ok(output),
            Err(error) if error.kind() == IoErrorKind::NotFound => {
                last_not_found = Some(format!("{}: {error}", interpreter.display()));
            },
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("spawn python baseline helper with interpreter {}", interpreter.display())
                });
            },
        }
    }

    Err(anyhow!(
        "failed to spawn python baseline helper; interpreters not found: {}",
        last_not_found.unwrap_or_else(|| "python3, python".to_string())
    ))
}

pub fn resolve_fixture_path(
    explicit: Option<&Path>,
    env_value: Option<OsString>,
    current_dir: std::io::Result<PathBuf>,
) -> Result<PathBuf, String> {
    let raw = if let Some(explicit) = explicit {
        explicit.to_path_buf()
    } else if let Some(env_value) = env_value {
        PathBuf::from(env_value)
    } else {
        return Err(format!("missing fixture path: pass --fixture <path> or set {}", FIXTURE_ENV));
    };

    let current_dir = current_dir.map_err(|error| format!("failed to resolve current directory: {error}"))?;
    let absolute = if raw.is_absolute() { raw } else { current_dir.join(raw) };
    Ok(absolute)
}

pub fn compare_summaries(baseline: &LouvainSummary, candidate: &LouvainSummary) -> Vec<LouvainDiff> {
    let mut diffs = Vec::new();
    push_exact_diff(
        &mut diffs,
        candidate,
        "graph_vertex_count",
        baseline.graph_vertex_count,
        candidate.graph_vertex_count,
        "graph vertex count differs",
    );
    push_exact_diff(
        &mut diffs,
        candidate,
        "graph_edge_count",
        baseline.graph_edge_count,
        candidate.graph_edge_count,
        "graph edge count differs",
    );
    push_exact_diff(
        &mut diffs,
        candidate,
        "community_count",
        baseline.community_count,
        candidate.community_count,
        "community count differs",
    );
    push_exact_diff(
        &mut diffs,
        candidate,
        "largest_communities_top_k",
        &baseline.largest_communities_top_k,
        &candidate.largest_communities_top_k,
        "largest community profile differs",
    );
    push_exact_diff(
        &mut diffs,
        candidate,
        "community_size_histogram_sorted",
        &baseline.community_size_histogram_sorted,
        &candidate.community_size_histogram_sorted,
        "community size histogram differs",
    );

    let modularity_delta = (baseline.modularity - candidate.modularity).abs();
    if modularity_delta > MODULARITY_TOLERANCE_ABS {
        diffs.push(LouvainDiff {
            path_kind: candidate.path_kind,
            flags: candidate.flags.clone(),
            field: "modularity".to_string(),
            severity: DiffSeverity::Hard,
            baseline: serde_json::json!(baseline.modularity),
            candidate: serde_json::json!(candidate.modularity),
            delta_abs: Some(modularity_delta),
            message: format!("modularity delta {modularity_delta} exceeds tolerance {MODULARITY_TOLERANCE_ABS}"),
        });
    }

    let overlap = membership_sample_overlap(baseline, candidate);
    if overlap < baseline.membership_sample.len().min(candidate.membership_sample.len()) {
        diffs.push(LouvainDiff {
            path_kind: candidate.path_kind,
            flags: candidate.flags.clone(),
            field: "membership_sample_overlap".to_string(),
            severity: DiffSeverity::Soft,
            baseline: serde_json::json!(baseline.membership_sample),
            candidate: serde_json::json!(candidate.membership_sample),
            delta_abs: None,
            message: format!("membership sample overlap is {overlap}"),
        });
    }

    if baseline.normalized_partition_shape_fingerprint != candidate.normalized_partition_shape_fingerprint {
        diffs.push(LouvainDiff {
            path_kind: candidate.path_kind,
            flags: candidate.flags.clone(),
            field: "normalized_partition_shape_fingerprint".to_string(),
            severity: DiffSeverity::Soft,
            baseline: serde_json::json!(baseline.normalized_partition_shape_fingerprint),
            candidate: serde_json::json!(candidate.normalized_partition_shape_fingerprint),
            delta_abs: None,
            message: "normalized partition-shape fingerprint differs".to_string(),
        });
    }

    diffs
}

fn membership_sample_overlap(baseline: &LouvainSummary, candidate: &LouvainSummary) -> usize {
    baseline.membership_sample.iter().zip(candidate.membership_sample.iter()).filter(|(lhs, rhs)| lhs == rhs).count()
}

fn push_exact_diff<T: Serialize + PartialEq>(
    diffs: &mut Vec<LouvainDiff>,
    candidate_summary: &LouvainSummary,
    field: &str,
    baseline: T,
    candidate: T,
    message: &str,
) {
    if baseline != candidate {
        diffs.push(LouvainDiff {
            path_kind: candidate_summary.path_kind,
            flags: candidate_summary.flags.clone(),
            field: field.to_string(),
            severity: DiffSeverity::Hard,
            baseline: serde_json::to_value(baseline).expect("serialize baseline"),
            candidate: serde_json::to_value(candidate).expect("serialize candidate"),
            delta_abs: None,
            message: message.to_string(),
        });
    }
}

pub fn build_summary(
    path_kind: PathKind,
    flags: LouvainFlags,
    input_edge_rows: usize,
    graph_vertex_count: usize,
    graph_edge_count: usize,
    modularity: f64,
    memberships: &[(i64, i64)],
) -> LouvainSummary {
    let mut memberships = memberships.to_vec();
    memberships.sort_by_key(|(vertex, _)| *vertex);

    let mut community_sizes = BTreeMap::<i64, usize>::new();
    for (_, community) in &memberships {
        *community_sizes.entry(*community).or_default() += 1;
    }

    let mut histogram: Vec<usize> = community_sizes.values().copied().collect();
    histogram.sort_unstable();

    let mut largest = histogram.clone();
    largest.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
    largest.truncate(LARGEST_COMMUNITIES_TOP_K);

    let membership_sample = memberships
        .iter()
        .take(MEMBERSHIP_SAMPLE_SIZE)
        .map(|(vertex, community)| MembershipSampleEntry {
            vertex: *vertex,
            community_size: *community_sizes.get(community).expect("community size"),
        })
        .collect::<Vec<_>>();

    let normalized_partition_shape_fingerprint = if graph_vertex_count == 0 {
        "empty".to_string()
    } else {
        histogram
            .iter()
            .map(|size| format!("{:.6}", (*size as f64) / (graph_vertex_count as f64)))
            .collect::<Vec<_>>()
            .join(",")
    };

    LouvainSummary {
        path_kind,
        flags,
        input_edge_rows,
        graph_vertex_count,
        graph_edge_count,
        community_count: histogram.len(),
        modularity,
        community_size_histogram_sorted: histogram,
        largest_communities_top_k: largest,
        membership_sample,
        normalized_partition_shape_fingerprint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(path_kind: PathKind) -> LouvainFlags {
        match path_kind {
            PathKind::PythonGraphFromCudfEdgelist => LouvainFlags {
                undirected: true,
                renumber: true,
                weighted: true,
                drop_multi_edges: false,
                symmetrize: false,
                do_expensive_check: false,
            },
            PathKind::RustGraphBuilder => LouvainFlags {
                undirected: true,
                renumber: true,
                weighted: true,
                drop_multi_edges: true,
                symmetrize: true,
                do_expensive_check: false,
            },
            PathKind::RustGraphFromDataframe => LouvainFlags {
                undirected: true,
                renumber: true,
                weighted: true,
                drop_multi_edges: true,
                symmetrize: true,
                do_expensive_check: false,
            },
        }
    }

    fn baseline_summary() -> LouvainSummary {
        build_summary(
            PathKind::PythonGraphFromCudfEdgelist,
            flags(PathKind::PythonGraphFromCudfEdgelist),
            12,
            6,
            7,
            0.812345,
            &[(50, 1), (10, 1), (12, 1), (77, 9), (90, 9), (91, 4)],
        )
    }

    #[test]
    fn resolve_fixture_prefers_explicit_path_over_env() {
        let resolved = resolve_fixture_path(
            Some(Path::new("cli.parquet")),
            Some(OsString::from("env.parquet")),
            Ok(PathBuf::from("/tmp/work")),
        )
        .expect("fixture");

        assert_eq!(resolved, PathBuf::from("/tmp/work/cli.parquet"));
    }

    #[test]
    fn resolve_fixture_uses_env_when_cli_is_absent() {
        let resolved =
            resolve_fixture_path(None, Some(OsString::from("fixtures/louvain.parquet")), Ok(PathBuf::from("/repo")))
                .expect("fixture");

        assert_eq!(resolved, PathBuf::from("/repo/fixtures/louvain.parquet"));
    }

    #[test]
    fn resolve_fixture_rejects_missing_sources() {
        let error =
            resolve_fixture_path(None, None, Ok(PathBuf::from("/repo"))).expect_err("missing fixture must fail");
        assert!(error.contains("--fixture"));
        assert!(error.contains(FIXTURE_ENV));
    }

    #[test]
    fn python_interpreter_candidates_prefer_virtualenv_python() {
        let candidates = python_interpreter_candidates(Some(OsString::from("/tmp/venv")));

        assert_eq!(candidates[0], PathBuf::from("/tmp/venv/bin/python"));
        assert_eq!(candidates[1], PathBuf::from("python3"));
        assert_eq!(candidates[2], PathBuf::from("python"));
    }

    #[test]
    fn classify_schema_kind_accepts_canonical_numeric_edges() {
        let kind = classify_schema_kind(TypeId::Int32, TypeId::Int32, TypeId::Float32, "src", "dst").expect("kind");
        assert_eq!(kind, EdgeSchemaKind::CanonicalNumeric);
    }

    #[test]
    fn classify_schema_kind_accepts_alias_string_edges() {
        let kind = classify_schema_kind(TypeId::String, TypeId::String, TypeId::Float64, "sourceId", "targetId")
            .expect("kind");
        assert_eq!(kind, EdgeSchemaKind::AliasStringNeedsRenumbering);
    }

    #[test]
    fn classify_schema_kind_rejects_mismatched_vertex_dtypes() {
        let error = classify_schema_kind(TypeId::Int32, TypeId::Int64, TypeId::Float32, "src", "dst")
            .expect_err("mismatched vertex dtypes must fail");
        assert!(format!("{error:#}").contains("does not match"));
    }

    #[test]
    fn build_summary_uses_community_sizes_for_membership_sample() {
        let summary = baseline_summary();

        assert_eq!(
            summary.membership_sample,
            vec![
                MembershipSampleEntry { vertex: 10, community_size: 3 },
                MembershipSampleEntry { vertex: 12, community_size: 3 },
                MembershipSampleEntry { vertex: 50, community_size: 3 },
                MembershipSampleEntry { vertex: 77, community_size: 2 },
                MembershipSampleEntry { vertex: 90, community_size: 2 },
                MembershipSampleEntry { vertex: 91, community_size: 1 },
            ]
        );
        assert_eq!(summary.community_size_histogram_sorted, vec![1, 2, 3]);
        assert_eq!(summary.largest_communities_top_k, vec![3, 2, 1]);
    }

    #[test]
    fn compare_summaries_accepts_modularity_within_tolerance() {
        let baseline = baseline_summary();
        let mut candidate = baseline.clone();
        candidate.path_kind = PathKind::RustGraphFromDataframe;
        candidate.modularity = baseline.modularity + (MODULARITY_TOLERANCE_ABS * 0.5);

        let diffs = compare_summaries(&baseline, &candidate);
        assert!(diffs.is_empty(), "expected no diffs, got {diffs:?}");
    }

    #[test]
    fn compare_summaries_reports_hard_and_soft_differences() {
        let baseline = baseline_summary();
        let candidate = build_summary(
            PathKind::RustGraphBuilder,
            flags(PathKind::RustGraphBuilder),
            12,
            8,
            9,
            baseline.modularity + (MODULARITY_TOLERANCE_ABS * 10.0),
            &[(50, 3), (10, 3), (12, 3), (77, 8), (90, 8), (91, 4), (111, 4), (112, 6)],
        );

        let diffs = compare_summaries(&baseline, &candidate);
        assert!(diffs.iter().any(|diff| diff.field == "graph_vertex_count" && diff.severity == DiffSeverity::Hard));
        assert!(diffs.iter().any(|diff| diff.field == "graph_edge_count" && diff.severity == DiffSeverity::Hard));
        assert!(diffs.iter().any(|diff| diff.field == "community_count" && diff.severity == DiffSeverity::Hard));
        assert!(diffs.iter().any(|diff| diff.field == "modularity" && diff.severity == DiffSeverity::Hard));
        assert!(
            diffs.iter().any(|diff| diff.field == "membership_sample_overlap" && diff.severity == DiffSeverity::Soft)
        );
        assert!(diffs.iter().any(|diff| {
            diff.field == "normalized_partition_shape_fingerprint" && diff.severity == DiffSeverity::Soft
        }));
    }

    #[test]
    fn error_report_serializes_required_top_level_fields() {
        let report = error_report("/tmp/fixture.parquet", "environment_error", "python3 not found");
        let json = serde_json::to_value(report).expect("report json");

        assert_eq!(json["case"], CASE_NAME);
        assert_eq!(json["fixture"], "/tmp/fixture.parquet");
        assert!(json.get("python_summary").is_some());
        assert!(json.get("rust_summaries").is_some());
        assert!(json.get("diffs").is_some());
        assert_eq!(json["verdict"], "error");
        assert_eq!(json["error"]["kind"], "environment_error");
    }
}
