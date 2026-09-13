//! Registry-driven cuGraph parity harness.
//!
//! The CPU tests validate local dataset SHA-256 values and oracle headers.
//! The registry-driven GPU test executes every advertised algorithm through
//! its public SQL table function and compares Arrow results with pinned Python
//! cuGraph oracle data, curated Katz CSVs, or a small hand-verified contract.
//! ECG is the exception: its randomized ensemble is checked for replay
//! stability and partition quality against the oracle, not an exact
//! cross-build partition. Spectral modularity maximization is the same kind
//! of exception: its seed only fixes the random initialization, so replays
//! are checked for partition quality against the initial run and the oracle,
//! not exact equality.

#[path = "cugraph_parity/compare.rs"]
mod compare;
#[path = "cugraph_parity/datasets.rs"]
mod datasets;
#[path = "cugraph_parity/oracle.rs"]
mod oracle;

use std::{collections::BTreeSet, sync::Arc};

use algeon_datafusion::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    cugraph_sql::{
        CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, CugraphSqlConfig,
        parity::{
            CUGRAPH_PARITY_SPECS, OracleLookup, ParityComparator, ParitySpec, ParityTruthSource, data_root,
            load_ref_csv_vertex_list,
        },
    },
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{ArrayRef, Float32Array, Int32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    datasource::MemTable,
    execution::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect, displayable},
    prelude::{CsvReadOptions, SessionConfig, SessionContext},
};
use serde_json::{Map, Value};

#[test]
fn parity_manifest_loads_and_dataset_checksums_match() {
    let loaded = datasets::load_and_verify().expect("the committed cuGraph parity fixture must be available");
    assert!(
        loaded.manifest.datasets.iter().any(|d| d.id == "karate"),
        "local manifest must include the karate dataset",
    );
}

#[test]
fn parity_specs_reference_existing_datasets() {
    let loaded = datasets::load_and_verify().expect("the committed cuGraph parity fixture must be available");
    for spec in CUGRAPH_PARITY_SPECS {
        assert!(
            loaded.manifest.dataset(spec.dataset).is_some(),
            "parity spec '{}' references unknown dataset '{}'",
            spec.label,
            spec.dataset,
        );
    }
}

#[test]
fn parity_specs_cover_all_registered_algorithms() {
    let mut algorithms: Vec<_> = CUGRAPH_PARITY_SPECS.iter().map(|spec| spec.algorithm).collect();
    algorithms.sort();
    algorithms.dedup();
    let registered: Vec<_> = CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.algorithm).collect();
    let mut sorted_registered = registered;
    sorted_registered.sort();
    assert_eq!(
        algorithms, sorted_registered,
        "parity registry must cover every currently registered cuGraph algorithm exactly once",
    );
    assert!(
        CUGRAPH_PARITY_SPECS
            .iter()
            .any(|spec| spec.algorithm == CugraphAlgorithm::PageRank && spec.dataset == "karate"),
        "the registry must keep at least the karate PageRank entry as the canonical baseline",
    );
}

#[test]
fn parity_oracle_inventory_exactly_matches_specs() {
    fn collect_json_files(root: &std::path::Path, current: &std::path::Path, files: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(current)
            .unwrap_or_else(|error| panic!("failed to read oracle directory {}: {error}", current.display()))
        {
            let path = entry.expect("oracle directory entry must be readable").path();
            if path.is_dir() {
                collect_json_files(root, &path, files);
            } else if path.extension().is_some_and(|extension| extension == "json") {
                let relative = path.strip_prefix(root).expect("oracle file must be below oracle root");
                files.insert(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }

    let loaded = datasets::load_and_verify().expect("the committed cuGraph parity fixture must be available");
    let oracle_root = loaded.data_root.join("oracle");
    let expected =
        CUGRAPH_PARITY_SPECS.iter().filter_map(|spec| spec.oracle_relpath.map(str::to_string)).collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    collect_json_files(&oracle_root, &oracle_root, &mut actual);
    assert_eq!(actual, expected, "oracle files must exactly match the registry-driven parity specs");
}

#[test]
fn parity_specs_declare_a_comparator_family() {
    for spec in CUGRAPH_PARITY_SPECS {
        let label = compare::family_label(spec.comparator);
        assert!(!label.is_empty(), "spec '{}' has an empty comparator family label", spec.label,);
        let sql = parity_sql(spec);
        let unordered = spec.algorithm == CugraphAlgorithm::EdgeBetweennessCentrality
            || matches!(spec.comparator, ParityComparator::EdgeSubgraph | ParityComparator::SimilarityMultiset);
        if unordered {
            assert!(
                !sql.contains(" ORDER BY "),
                "spec '{}' must not request an unsupported multi-key sort",
                spec.label
            );
        } else {
            assert!(sql.ends_with(" ORDER BY vertex"), "spec '{}' must retain its supported vertex sort", spec.label);
        }
    }
}

#[test]
fn parity_oracle_lookup_reports_missing_with_refresh_hint() {
    let loaded = datasets::load_and_verify().expect("the committed cuGraph parity fixture must be available");
    let mut missing = Vec::new();
    let mut mismatches = Vec::new();
    let mut present = 0usize;
    for spec in CUGRAPH_PARITY_SPECS {
        if matches!(spec.truth_source, ParityTruthSource::HandVerified { .. }) {
            continue;
        }
        // Every oracle-backed spec carries an `oracle_relpath`; for
        // `OracleJson` truth it is the primary ground truth; for `RefCsv`
        // truth it is a secondary cross-check generated by Python cuGraph.
        // Look it up unconditionally so a missing secondary oracle for a
        // RefCsv spec still surfaces as "needs refresh".
        match oracle::lookup(&loaded.data_root, &loaded.manifest, spec) {
            OracleLookup::Present { .. } => present += 1,
            OracleLookup::Missing { path } => {
                missing.push(format!(
                    "  - {label}: oracle missing at {path}\n    {hint}",
                    label = spec.label,
                    path = path.display(),
                    hint = oracle::refresh_hint(&loaded.data_root, spec),
                ));
            },
            OracleLookup::Invalid { path, reason } | OracleLookup::Mismatch { path, reason } => {
                mismatches.push(format!(
                    "  - {label}: oracle at {path} rejected: {reason}",
                    label = spec.label,
                    path = path.display(),
                ));
            },
            OracleLookup::NotApplicable => unreachable!("hand-verified rows were filtered above"),
        }
        // RefCsv-truth specs additionally require their primary
        // reference file to be present and parseable on disk. Surface
        // a missing/invalid ref CSV as a hard test failure: the GPU
        // comparator path relies on this file being loadable, and silently
        // degrading would hide that gap.
        if let ParityTruthSource::RefCsv { path } = spec.truth_source
            && let Err(error) = load_ref_csv_vertex_list(&loaded.data_root, path)
        {
            mismatches.push(format!(
                "  - {label}: ref CSV at {path} rejected: {error}",
                label = spec.label,
                path = path,
            ));
        }
    }
    assert!(mismatches.is_empty(), "local oracle JSON disagrees with parity spec:\n{}", mismatches.join("\n"),);
    if !missing.is_empty() {
        // The CPU lookup test reports a structured refresh command rather
        // than failing so fixture preparation remains diagnosable. The GPU
        // acceptance test above requires every oracle to be present.
        eprintln!(
            "cuGraph parity harness: {} oracle JSON file(s) need refresh; {} present.\n{}",
            missing.len(),
            present,
            missing.join("\n"),
        );
    }
}

#[test]
fn parity_data_root_lives_under_workspace_fixture_dir() {
    // `data_root` honors `ALGEON_CUGRAPH_DATA_ROOT` so refresh tooling can
    // point at a staging fixture path. The default-case assertion only
    // runs when the override is absent.
    if std::env::var_os("ALGEON_CUGRAPH_DATA_ROOT").is_some() {
        return;
    }
    let root = data_root();
    let suffix = std::path::Path::new("fixture").join("graph").join("cugraph");
    assert!(root.ends_with(suffix), "parity data root unexpected: {}", root.display());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn all_registered_algorithms_execute_on_gpu_and_match_oracles() {
    let loaded = datasets::load_and_verify().expect("the committed cuGraph parity fixture must be available");
    let (_backend, ctx) = parity_session();
    register_datasets(&ctx, &loaded).await;
    register_personalization_table(&ctx);
    register_similarity_pairs(&ctx);

    let mut executed = BTreeSet::new();
    for spec in CUGRAPH_PARITY_SPECS {
        let document = match spec.truth_source {
            ParityTruthSource::HandVerified { document_json } => serde_json::from_str(document_json)
                .unwrap_or_else(|error| panic!("{}: hand-verified document is invalid JSON: {error}", spec.label)),
            ParityTruthSource::OracleJson | ParityTruthSource::RefCsv { .. } => {
                match oracle::lookup(&loaded.data_root, &loaded.manifest, spec) {
                    OracleLookup::Present { document, .. } => document,
                    OracleLookup::Missing { path } => panic!(
                        "{}: oracle missing at {}\n{}",
                        spec.label,
                        path.display(),
                        oracle::refresh_hint(&loaded.data_root, spec)
                    ),
                    OracleLookup::Invalid { path, reason } | OracleLookup::Mismatch { path, reason } => {
                        panic!("{}: oracle at {} rejected: {reason}", spec.label, path.display())
                    },
                    OracleLookup::NotApplicable => unreachable!("oracle-backed rows always have an oracle path"),
                }
            },
        };
        let sql = parity_sql(spec);
        let df = ctx
            .sql(&sql)
            .await
            .unwrap_or_else(|error| panic!("{}: SQL planning failed for {sql}: {error}", spec.label));
        let plan = df
            .create_physical_plan()
            .await
            .unwrap_or_else(|error| panic!("{}: physical planning failed for {sql}: {error}", spec.label));
        assert!(
            plan_contains_cugraph_exec(plan.as_ref()),
            "{}: parity query did not plan a CugraphAlgoExec:\n{}",
            spec.label,
            displayable(plan.as_ref()).indent(false)
        );
        let batches = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap_or_else(|error| panic!("{}: GPU execution failed for {sql}: {error}", spec.label));
        let dataset = loaded.manifest.dataset(spec.dataset).expect("parity spec dataset was validated");
        let edges = datasets::load_edges(&loaded.data_root, dataset)
            .unwrap_or_else(|error| panic!("{}: dataset edge load failed: {error}", spec.label));
        if matches!(spec.algorithm, CugraphAlgorithm::Ecg | CugraphAlgorithm::SpectralModularityMaximization) {
            // A DataFusion physical plan owns consumable file-scan state, so
            // collecting the same `plan` twice only replays an exhausted
            // source. Rebuild it from the same logical DataFrame to exercise
            // the same SQL, binary, and seed with a fresh execution graph.
            let replay_plan = df
                .create_physical_plan()
                .await
                .unwrap_or_else(|error| panic!("{}: replay physical planning failed for {sql}: {error}", spec.label));
            assert!(
                plan_contains_cugraph_exec(replay_plan.as_ref()),
                "{}: replay query did not plan a CugraphAlgoExec:\n{}",
                spec.label,
                displayable(replay_plan.as_ref()).indent(false)
            );
            let replay = collect(replay_plan, ctx.task_ctx())
                .await
                .unwrap_or_else(|error| panic!("{}: GPU replay failed for {sql}: {error}", spec.label));
            if spec.algorithm == CugraphAlgorithm::Ecg {
                compare::compare_repeated_partitions(&batches, &replay).unwrap_or_else(|reason| {
                    panic!("{}: same-binary, same-seed ECG replay was unstable: {reason}\nSQL: {sql}", spec.label)
                });
            } else {
                // Spectral promises a stable quality, not a bit-identical
                // partition: a fixed seed does not survive GPU scheduling
                // order in k-means centroid accumulation.
                compare::compare_repeated_partition_quality(&batches, &replay, &edges, spec.tolerance).unwrap_or_else(
                    |reason| {
                        panic!(
                            "{}: same-binary, same-seed spectral replay was unstable: {reason}\nSQL: {sql}",
                            spec.label
                        )
                    },
                );
            }
        }
        let reference_vertices = match spec.truth_source {
            ParityTruthSource::RefCsv { path } => Some(
                load_ref_csv_vertex_list(&loaded.data_root, path)
                    .unwrap_or_else(|error| panic!("{}: reference CSV failed: {error}", spec.label)),
            ),
            ParityTruthSource::OracleJson | ParityTruthSource::HandVerified { .. } => None,
        };
        compare::compare(spec, &batches, &document, reference_vertices.as_deref(), &edges)
            .unwrap_or_else(|reason| panic!("{}: GPU/oracle parity failed: {reason}\nSQL: {sql}", spec.label));
        assert!(
            executed.insert(spec.algorithm),
            "parity registry contains more than one row for algorithm {:?}",
            spec.algorithm
        );
    }

    let registered = CUGRAPH_OPERATION_REGISTRY.iter().map(|definition| definition.algorithm).collect();
    assert_eq!(executed, registered, "the GPU parity run must cover every registered algorithm");
}

fn parity_session() -> (AlgeonGpuBackend, SessionContext) {
    let config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().expect("parity backend must build");
    let state = backend
        .install_on(SessionStateBuilder::new().with_config(config).with_default_features())
        .expect("backend must install")
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("cuGraph SQL must install")
        .build();
    (backend, SessionContext::new_with_state(state))
}

async fn register_datasets(ctx: &SessionContext, loaded: &datasets::LoadedDatasets) {
    let schema = Schema::new(vec![
        Field::new("dst", DataType::Int32, false),
        Field::new("src", DataType::Int32, false),
        Field::new("weight", DataType::Float32, false),
    ]);
    for dataset in &loaded.manifest.datasets {
        assert_eq!(dataset.format, "space_separated_edges", "unsupported parity dataset format");
        assert_eq!(dataset.edge_columns, ["dst", "src", "weight"], "unexpected parity dataset columns");
        let table_name = dataset_table_name(&dataset.id);
        ctx.register_csv(
            &table_name,
            loaded.data_root.join(&dataset.path).to_str().expect("fixture path must be UTF-8"),
            CsvReadOptions::new().schema(&schema).has_header(false).delimiter(b' '),
        )
        .await
        .unwrap_or_else(|error| panic!("failed to register parity dataset '{}': {error}", dataset.id));
    }
}

fn register_personalization_table(ctx: &SessionContext) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int32, false),
        Field::new("value", DataType::Float32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![0])) as ArrayRef, Arc::new(Float32Array::from(vec![1.0])) as ArrayRef],
    )
    .expect("personalization batch must build");
    let table = MemTable::try_new(schema, vec![vec![batch]]).expect("personalization table must build");
    ctx.register_table("ppr_seeds", Arc::new(table)).expect("personalization table must register");
}

fn register_similarity_pairs(ctx: &SessionContext) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Int32, false),
        Field::new("second", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0, 3, 0, 1])) as ArrayRef,
            Arc::new(Int32Array::from(vec![3, 3, 0, 0, 4])) as ArrayRef,
        ],
    )
    .expect("similarity pair batch must build");
    let table = MemTable::try_new(schema, vec![vec![batch]]).expect("similarity pair table must build");
    ctx.register_table("similarity_pairs", Arc::new(table)).expect("similarity pair table must register");
}

fn parity_sql(spec: &ParitySpec) -> String {
    let mut options = serde_json::from_str::<Map<String, Value>>(spec.params_json)
        .unwrap_or_else(|error| panic!("{} has invalid params JSON: {error}", spec.label));
    let source = options.remove("source_vertex").and_then(|value| value.as_i64());
    options.remove("personalization_vertices");
    options.remove("personalization_values");
    let is_similarity = matches!(
        spec.algorithm,
        CugraphAlgorithm::Jaccard | CugraphAlgorithm::Sorensen | CugraphAlgorithm::Overlap | CugraphAlgorithm::Cosine
    );
    if is_similarity {
        options.insert("vertex_pairs_table".to_string(), Value::String("similarity_pairs".to_string()));
        options.insert("first_vertex_col".to_string(), Value::String("first".to_string()));
        options.insert("second_vertex_col".to_string(), Value::String("second".to_string()));
    }
    let options = serde_json::to_string(&options).expect("parity options must serialize");
    let table = dataset_table_name(spec.dataset);
    let args = if is_similarity {
        format!("'{table}', 'src', 'dst', NULL, '{}'", options.replace('\'', "''"))
    } else if matches!(spec.algorithm, CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp) {
        format!(
            "'{table}', {}, 'src', 'dst', 'weight', '{}'",
            source.unwrap_or_else(|| panic!("{} is missing source_vertex", spec.label)),
            options.replace('\'', "''")
        )
    } else {
        format!("'{table}', 'src', 'dst', 'weight', '{}'", options.replace('\'', "''"))
    };
    let sql = format!("SELECT * FROM {}({args})", spec.algorithm.function_name());
    if spec.algorithm == CugraphAlgorithm::EdgeBetweennessCentrality
        || matches!(spec.comparator, ParityComparator::EdgeSubgraph | ParityComparator::SimilarityMultiset)
    {
        sql
    } else {
        format!("{sql} ORDER BY vertex")
    }
}

fn dataset_table_name(dataset: &str) -> String {
    format!("parity_{dataset}")
}

fn plan_contains_cugraph_exec(plan: &dyn ExecutionPlan) -> bool {
    plan.name() == "CugraphAlgoExec"
        || plan.children().into_iter().any(|child| plan_contains_cugraph_exec(child.as_ref()))
}
