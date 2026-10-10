use std::{collections::BTreeMap, fs, path::Path, sync::Arc};

use algeon_datafusion::{
    cugraph_sql::{
        CUGRAPH_ECG_FUNCTION, CUGRAPH_LEIDEN_FUNCTION, CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_PAGERANK_FUNCTION,
        CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION, CUGRAPH_SSSP_FUNCTION, CugraphSqlConfig,
    },
    error::{DataFusionErrorExt, ErrorCode},
    gpu_functions::{GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION},
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{Array, BooleanArray, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{datasource::MemTable, execution::SessionStateBuilder, prelude::SessionContext};
use serde_json::{Map, Value, json};

const INVALID_VALUE: &str = "missing_or_invalid_option";
const UNKNOWN_OPTION: &str = "unknown_option";

fn register_table(ctx: &SessionContext, name: &str, columns: &[(&str, DataType)]) {
    let schema = Arc::new(Schema::new(
        columns.iter().map(|(column, data_type)| Field::new(*column, data_type.clone(), false)).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    let provider = MemTable::try_new(schema, vec![vec![batch]]).expect("fixture MemTable");
    ctx.register_table(name, Arc::new(provider)).expect("register fixture table");
}

/// A session with the `Int64`-vertex tables every function's first descriptor
/// example names.
fn cugraph_context() -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL contracts")
        .build();
    let ctx = SessionContext::new_with_state(state);
    let vertex = DataType::Int64;
    register_table(&ctx, "edges", &[("src", vertex.clone()), ("dst", vertex.clone())]);
    register_table(
        &ctx,
        "target_edges",
        &[("src", vertex.clone()), ("dst", vertex.clone()), ("weight", DataType::Float64)],
    );
    register_table(&ctx, "ppr_seeds", &[("vertex", vertex.clone()), ("value", DataType::Float64)]);
    register_table(&ctx, "candidate_pairs", &[("first", vertex.clone()), ("second", vertex)]);
    ctx
}

async fn collect(ctx: &SessionContext, sql: &str) -> RecordBatch {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(batches.len(), 1, "{sql}");
    batches.into_iter().next().unwrap()
}

fn text(batch: &RecordBatch, column: &str, row: usize) -> Option<String> {
    let array = batch.column_by_name(column).unwrap_or_else(|| panic!("missing column {column}"));
    let array = array.as_any().downcast_ref::<StringArray>().unwrap_or_else(|| panic!("{column} is not Utf8"));
    (!array.is_null(row)).then(|| array.value(row).to_string())
}

/// One cuGraph function as `gpu_describe_function` advertises it, with the
/// first example's relations and options as a call that is valid on its own.
struct Described {
    function: String,
    options_schema: Value,
    relations: BTreeMap<String, String>,
    base_options: Map<String, Value>,
}

async fn cugraph_functions(ctx: &SessionContext) -> Vec<String> {
    let batch = collect(
        ctx,
        &format!("SELECT function_name FROM {GPU_LIST_FUNCTIONS_FUNCTION}() WHERE provider = 'cugraph' ORDER BY 1"),
    )
    .await;
    (0..batch.num_rows()).map(|row| text(&batch, "function_name", row).unwrap()).collect()
}

async fn describe(ctx: &SessionContext, function: &str) -> Described {
    let batch = collect(
        ctx,
        &format!(
            "SELECT options_schema_json, examples_json \
             FROM {GPU_DESCRIBE_FUNCTION_FUNCTION}('{function}')"
        ),
    )
    .await;
    let options_schema = serde_json::from_str(&text(&batch, "options_schema_json", 0).unwrap()).unwrap();
    let examples: Value = serde_json::from_str(&text(&batch, "examples_json", 0).unwrap()).unwrap();
    let call = &examples[0]["call_json"];
    let relations = call["relations"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(role, relation)| (role.clone(), relation["table"].as_str().unwrap().to_string()))
        .collect();
    Described {
        function: function.to_string(),
        options_schema,
        relations,
        base_options: call["options"].as_object().unwrap().clone(),
    }
}

fn sql_literal(value: &Value) -> String {
    match value {
        Value::String(value) => format!("'{}'", value.replace('\'', "''")),
        other => other.to_string(),
    }
}

impl Described {
    fn sql(&self, overrides: &Map<String, Value>) -> String {
        let mut options = self.base_options.clone();
        options.extend(overrides.clone());
        let relations = self.relations.iter().map(|(role, table)| format!("{role} => (SELECT * FROM {table})"));
        let values = options.iter().map(|(name, value)| format!("{name} => {}", sql_literal(value)));
        format!("SELECT * FROM {}({})", self.function, relations.chain(values).collect::<Vec<_>>().join(", "))
    }

    fn call_json(&self, overrides: &Map<String, Value>) -> String {
        let mut options = self.base_options.clone();
        options.extend(overrides.clone());
        let relations = self.relations.iter().map(|(role, table)| (role.clone(), json!({"table": table}))).collect();
        json!({"relations": Value::Object(relations), "options": Value::Object(options)}).to_string()
    }
}

/// The typed adapter rejection of a SQL call.
struct Rejection {
    code: ErrorCode,
    reason: Option<String>,
    argument: Option<String>,
    message: String,
}

/// Plans one SQL call and returns its typed rejection, if any.
async fn plan_rejection(
    ctx: &SessionContext,
    described: &Described,
    overrides: &Map<String, Value>,
) -> Option<Rejection> {
    let sql = described.sql(overrides);
    let error = ctx.state().create_logical_plan(&sql).await.err()?;
    let typed = error.find_algeon_error().unwrap_or_else(|| panic!("untyped planning error for {sql}: {error:?}"));
    Some(Rejection {
        code: typed.code(),
        reason: typed.fact_value("reason").map(str::to_string),
        argument: typed.fact_value("argument").map(str::to_string),
        message: typed.message().to_string(),
    })
}

/// `gpu_validate_call` outcome: `(valid, error_code, message, details)`.
async fn validate(
    ctx: &SessionContext,
    described: &Described,
    overrides: &Map<String, Value>,
) -> (bool, Option<String>, String, Value) {
    let call = described.call_json(overrides).replace('\'', "''");
    let batch = collect(
        ctx,
        &format!(
            "SELECT valid, error_code, message, details_json FROM {GPU_VALIDATE_CALL_FUNCTION}('{}', '{call}')",
            described.function
        ),
    )
    .await;
    let valid = batch.column_by_name("valid").unwrap().as_any().downcast_ref::<BooleanArray>().unwrap().value(0);
    let details = serde_json::from_str(&text(&batch, "details_json", 0).unwrap()).unwrap();
    (valid, text(&batch, "error_code", 0), text(&batch, "message", 0).unwrap(), details)
}

/// One probe value: the SQL/JSON value and whether the descriptor admits it.
struct Case {
    value: Value,
    admitted: bool,
}

/// The numeric type of an advertised option schema, looking through the
/// `oneOf [type, null]` wrapper of a nullable option.
fn numeric_type(schema: &Value) -> Option<&str> {
    let declared = schema.get("type").or_else(|| schema["oneOf"].get(0)?.get("type"))?.as_str()?;
    matches!(declared, "integer" | "number").then_some(declared)
}

/// An advertised bound. Top-level bounds are the declared range; the value
/// type's own bounds sit inside the nullable wrapper.
fn bound(schema: &Value, key: &str) -> Option<serde_json::Number> {
    schema.get(key).or_else(|| schema["oneOf"].get(0)?.get(key))?.as_number().cloned()
}

fn integer(number: &serde_json::Number) -> i128 {
    number.as_i64().map(i128::from).or_else(|| number.as_u64().map(i128::from)).expect("integer bound")
}

fn integer_value(value: i128) -> Option<Value> {
    i64::try_from(value).map(Value::from).ok().or_else(|| u64::try_from(value).map(Value::from).ok())
}

/// Values on and just past every bound the schema advertises. Integer bounds
/// step by one; float bounds step by a margin far above `f64` resolution and
/// below the spacing of any bound the descriptors declare.
fn cases(schema: &Value, numeric: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    for (is_lower, inclusive, exclusive) in
        [(true, "minimum", "exclusiveMinimum"), (false, "maximum", "exclusiveMaximum")]
    {
        let (number, is_exclusive) = match (bound(schema, inclusive), bound(schema, exclusive)) {
            (Some(number), _) => (number, false),
            (None, Some(number)) => (number, true),
            (None, None) => continue,
        };
        // An exclusive bound is itself the nearest value outside, and the
        // nearest value inside is one step away from it.
        let (inside, outside) = if numeric == "integer" {
            let at = integer(&number);
            let step = if is_lower { 1 } else { -1 };
            let (inside, outside) = if is_exclusive { (at + step, at) } else { (at, at - step) };
            // A value past the 64-bit literal range cannot be written as SQL.
            (integer_value(inside), integer_value(outside))
        } else {
            const MARGIN: f64 = 1e-6;
            let at = number.as_f64().unwrap();
            let step = if is_lower { MARGIN } else { -MARGIN };
            let (inside, outside) = if is_exclusive { (at + step, at) } else { (at, at - step) };
            (Some(json!(inside)), Some(json!(outside)))
        };
        cases.extend(inside.map(|value| Case { value, admitted: true }));
        cases.extend(outside.map(|value| Case { value, admitted: false }));
    }
    cases
}

/// Options whose acceptance depends on another option. libcugraph requires
/// `n_eigenvectors <= n_clusters`, so probing one raises the other with it.
fn companion_options(function: &str, option: &str, value: &Value) -> Map<String, Value> {
    let mut companions = Map::new();
    if function == CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION && option == "n_eigenvectors" {
        let clusters = value.as_i64().unwrap_or(2).clamp(2, i64::from(i32::MAX));
        companions.insert("n_clusters".to_string(), json!(clusters));
    }
    companions
}

/// The numeric ranges `gpu_describe_function` advertises are the ranges both
/// SQL planning and `gpu_validate_call` enforce: the boundary value inside is
/// accepted and the value just outside is rejected as an invalid option value
/// of that argument. The walk covers every function and every numeric option,
/// so a new argument is held to its own descriptor without a new test.
#[tokio::test]
async fn cugraph_numeric_options_enforce_exactly_the_advertised_bounds() {
    let ctx = cugraph_context();
    let mut probed = Vec::new();
    for function in cugraph_functions(&ctx).await {
        let described = describe(&ctx, &function).await;
        let properties = described.options_schema["properties"].as_object().unwrap().clone();
        for (option, schema) in &properties {
            let Some(numeric) = numeric_type(schema) else { continue };
            probed.push(format!("{function}.{option}"));
            for case in cases(schema, numeric) {
                let mut overrides = companion_options(&function, option, &case.value);
                overrides.insert(option.clone(), case.value.clone());
                let context = format!("{function} {option} => {}", case.value);

                let planned = plan_rejection(&ctx, &described, &overrides).await;
                let (valid, code, message, details) = validate(&ctx, &described, &overrides).await;
                if case.admitted {
                    assert!(
                        planned.is_none(),
                        "{context} must plan: {}",
                        planned.map_or_else(String::new, |r| r.message)
                    );
                    assert!(valid, "{context} must validate: {message}");
                    continue;
                }
                let planned = planned.unwrap_or_else(|| panic!("{context} must not plan"));
                assert_eq!(planned.code, ErrorCode::InvalidRequest, "{context}");
                assert_eq!(planned.reason.as_deref(), Some(INVALID_VALUE), "{context}: {}", planned.message);
                assert_eq!(planned.argument.as_deref(), Some(option.as_str()), "{context}");
                assert!(!valid, "{context} must not validate");
                assert_eq!(code.as_deref(), Some(INVALID_VALUE), "{context}: {message}");
                assert_eq!(details["path"], format!("options.{option}"), "{context}");
                assert_eq!(details["argument"], option.as_str(), "{context}");
            }
        }
    }
    for expected in [
        "cugraph_pagerank.alpha",
        "cugraph_katz_centrality.beta",
        "cugraph_louvain.threshold",
        "cugraph_ecg.threshold",
        "cugraph_ecg.resolution",
        "cugraph_spectral_modularity_maximization.n_clusters",
        "cugraph_spectral_modularity_maximization.evs_tolerance",
        "cugraph_force_atlas2.barnes_hut_theta",
        "cugraph_bfs.depth_limit",
        "cugraph_betweenness_centrality.k",
        "cugraph_k_core.k",
    ] {
        assert!(probed.iter().any(|probe| probe == expected), "the descriptor walk never reached {expected}");
    }
}

fn advertised(schema: &Value, option: &str, bound: &str) -> Option<f64> {
    schema["properties"][option][bound].as_f64()
}

/// Native cuGraph requires ECG's `threshold` and `resolution` in (0.0, 1.0]
/// (`ecg_impl.cuh`), while Louvain and Leiden accept any positive resolution
/// and any non-negative Louvain threshold. The descriptors must say so.
#[tokio::test]
async fn cugraph_community_descriptors_state_the_native_ranges() {
    let ctx = cugraph_context();
    let ecg = describe(&ctx, CUGRAPH_ECG_FUNCTION).await.options_schema;
    for option in ["threshold", "resolution"] {
        assert_eq!(advertised(&ecg, option, "exclusiveMinimum"), Some(0.0), "ecg {option}");
        assert_eq!(advertised(&ecg, option, "maximum"), Some(1.0), "ecg {option}");
        assert_eq!(advertised(&ecg, option, "minimum"), None, "ecg {option}");
    }
    for function in [CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_LEIDEN_FUNCTION] {
        let schema = describe(&ctx, function).await.options_schema;
        assert_eq!(advertised(&schema, "resolution", "exclusiveMinimum"), Some(0.0), "{function}");
        assert_eq!(advertised(&schema, "resolution", "maximum"), None, "{function}");
    }
    let louvain = describe(&ctx, CUGRAPH_LOUVAIN_FUNCTION).await.options_schema;
    assert_eq!(advertised(&louvain, "threshold", "minimum"), Some(0.0));
    assert_eq!(advertised(&louvain, "threshold", "maximum"), None);
}

/// libcugraph ignores Leiden's `theta`, so the SQL surface does not offer it.
#[tokio::test]
async fn cugraph_leiden_does_not_offer_the_ignored_theta_argument() {
    let ctx = cugraph_context();
    let leiden = describe(&ctx, CUGRAPH_LEIDEN_FUNCTION).await;
    assert!(leiden.options_schema["properties"].get("theta").is_none());
    let overrides = Map::from_iter([("theta".to_string(), json!(1.0))]);
    let rejection = plan_rejection(&ctx, &leiden, &overrides).await.expect("theta must not plan");
    assert_eq!(rejection.reason.as_deref(), Some("unknown_argument"));
    let (valid, code, ..) = validate(&ctx, &leiden, &overrides).await;
    assert!(!valid);
    assert_eq!(code.as_deref(), Some(UNKNOWN_OPTION));
}

/// libcugraph rejects more spectral eigenvectors than clusters; no per-option
/// range expresses that, so it is a cross-option check.
#[tokio::test]
async fn cugraph_spectral_rejects_more_eigenvectors_than_clusters() {
    let ctx = cugraph_context();
    let spectral = describe(&ctx, CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION).await;
    let call = |clusters: i64, eigenvectors: i64| {
        Map::from_iter([
            ("n_clusters".to_string(), json!(clusters)),
            ("n_eigenvectors".to_string(), json!(eigenvectors)),
        ])
    };

    assert!(plan_rejection(&ctx, &spectral, &call(3, 3)).await.is_none());
    assert!(validate(&ctx, &spectral, &call(3, 3)).await.0);

    let rejection =
        plan_rejection(&ctx, &spectral, &call(3, 4)).await.expect("4 eigenvectors over 3 clusters must not plan");
    assert_eq!(rejection.code, ErrorCode::InvalidRequest);
    assert_eq!(rejection.reason.as_deref(), Some(INVALID_VALUE));
    assert_eq!(rejection.argument.as_deref(), Some("n_eigenvectors"));
    let (valid, code, _, details) = validate(&ctx, &spectral, &call(3, 4)).await;
    assert!(!valid);
    assert_eq!(code.as_deref(), Some(INVALID_VALUE));
    assert_eq!(details["path"], "options.n_eigenvectors");
}

/// The spectral embedding keeps the trivial Laplacian eigenvector, which is
/// constant on a connected graph, so one eigenvector would put every vertex in
/// one community.
#[tokio::test]
async fn cugraph_spectral_rejects_a_single_eigenvector() {
    let ctx = cugraph_context();
    let spectral = describe(&ctx, CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION).await;
    let overrides = Map::from_iter([("n_eigenvectors".to_string(), json!(1))]);

    let rejection = plan_rejection(&ctx, &spectral, &overrides).await.expect("one eigenvector must not plan");
    assert_eq!(rejection.reason.as_deref(), Some(INVALID_VALUE));
    assert_eq!(rejection.argument.as_deref(), Some("n_eigenvectors"));
    let (valid, code, ..) = validate(&ctx, &spectral, &overrides).await;
    assert!(!valid);
    assert_eq!(code.as_deref(), Some(INVALID_VALUE));
}

/// A mistyped argument names the function's valid arguments and the nearest
/// one, in SQL planning and in `gpu_validate_call`. Agents guessed
/// `sources => [...]` and `source => 2595` for `cugraph_sssp`, whose source
/// argument is `source_vertex`.
#[tokio::test]
async fn cugraph_unknown_arguments_suggest_the_nearest_declared_name() {
    let ctx = cugraph_context();
    let sssp = describe(&ctx, CUGRAPH_SSSP_FUNCTION).await;

    for (guess, value) in [("sources", json!([2595])), ("source", json!(2595)), ("souce_vertex", json!(2595))] {
        let overrides = Map::from_iter([(guess.to_string(), value.clone())]);

        let rejection = plan_rejection(&ctx, &sssp, &overrides).await.expect("an undeclared argument must not plan");
        assert_eq!(rejection.code, ErrorCode::InvalidRequest, "{guess}");
        assert_eq!(rejection.reason.as_deref(), Some("unknown_argument"), "{guess}");
        for expected in [format!("unknown argument '{guess}'"), "Did you mean 'source_vertex'?".into(), "cutoff".into()]
        {
            assert!(rejection.message.contains(&expected), "{guess}: {}", rejection.message);
        }

        let (valid, code, message, details) = validate(&ctx, &sssp, &overrides).await;
        assert!(!valid, "{guess}");
        assert_eq!(code.as_deref(), Some(UNKNOWN_OPTION), "{guess}");
        for expected in [format!("unknown option '{guess}'"), "Did you mean 'source_vertex'?".into(), "cutoff".into()] {
            assert!(message.contains(&expected), "{guess}: {message}");
        }
        assert_eq!(details["path"], format!("options.{guess}"), "{guess}");
    }

    // The valid names are listed even when no declared name is near enough to suggest.
    let pagerank = describe(&ctx, CUGRAPH_PAGERANK_FUNCTION).await;
    let overrides = Map::from_iter([("zzzzzzzz".to_string(), json!(1))]);
    let rejection = plan_rejection(&ctx, &pagerank, &overrides).await.expect("an undeclared argument must not plan");
    assert!(!rejection.message.contains("Did you mean"), "{}", rejection.message);
    assert!(rejection.message.contains("alpha, epsilon, max_iterations"), "{}", rejection.message);
}

/// An option another function declares is unknown here, including the BFS-only
/// `source_vertices` on a function without it and `source_vertex` on a function
/// that takes no source.
#[tokio::test]
async fn cugraph_validation_reports_undeclared_source_options_as_unknown() {
    let ctx = cugraph_context();
    for (function, option, value) in [
        (CUGRAPH_SSSP_FUNCTION, "source_vertices", json!([1, 2])),
        (CUGRAPH_PAGERANK_FUNCTION, "source_vertex", json!(1)),
    ] {
        let described = describe(&ctx, function).await;
        let overrides = Map::from_iter([(option.to_string(), value)]);
        let (valid, code, _, details) = validate(&ctx, &described, &overrides).await;
        assert!(!valid, "{function} {option}");
        assert_eq!(code.as_deref(), Some(UNKNOWN_OPTION), "{function} {option}");
        assert_eq!(details["argument"], option, "{function} {option}");
    }
}

#[tokio::test]
async fn cugraph_ecg_sql_rejects_out_of_range_threshold_and_resolution_during_planning() {
    let ctx = cugraph_context();
    let ecg = describe(&ctx, CUGRAPH_ECG_FUNCTION).await;
    for (option, rejected, accepted) in [("resolution", json!(3), json!(1.0)), ("threshold", json!(0), json!(1.0))] {
        let overrides = Map::from_iter([(option.to_string(), rejected.clone())]);
        let rejection = plan_rejection(&ctx, &ecg, &overrides)
            .await
            .unwrap_or_else(|| panic!("{option} => {rejected} must not plan"));
        assert_eq!(rejection.code, ErrorCode::InvalidRequest, "{option}");
        assert_eq!(rejection.reason.as_deref(), Some(INVALID_VALUE), "{option}");
        assert_eq!(rejection.argument.as_deref(), Some(option), "{option}");

        let overrides = Map::from_iter([(option.to_string(), accepted.clone())]);
        assert!(
            plan_rejection(&ctx, &ecg, &overrides).await.is_none(),
            "{option} => {accepted} is the native upper bound"
        );
    }
}

/// The committed specs the website renders must equal what
/// `gpu_describe_function` returns, so a descriptor change cannot ship with a
/// stale spec. Refresh them with `website/scripts/capture-cugraph-specs.sh`.
#[tokio::test]
async fn captured_cugraph_specs_match_the_descriptors() {
    let ctx = cugraph_context();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("adapter manifest must be two levels below the repository root");
    let spec_directory = repository.join("website/gpu-function-specs/cugraph");
    let functions = cugraph_functions(&ctx).await;

    let mut captured_names = fs::read_dir(&spec_directory)
        .unwrap_or_else(|error| panic!("read captured cuGraph specs at {}: {error}", spec_directory.display()))
        .map(|entry| entry.expect("read captured cuGraph spec entry").path())
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("json"))
        .map(|path| path.file_stem().and_then(|stem| stem.to_str()).expect("UTF-8 spec file name").to_string())
        .collect::<Vec<_>>();
    captured_names.sort();
    assert_eq!(captured_names, functions, "captured specs must cover exactly the registered cuGraph functions");

    for function in &functions {
        let batch = collect(&ctx, &format!("SELECT * FROM {GPU_DESCRIBE_FUNCTION_FUNCTION}('{function}')")).await;
        let captured: Value =
            serde_json::from_str(&fs::read_to_string(spec_directory.join(format!("{function}.json"))).unwrap())
                .unwrap();
        for field in batch.schema().fields() {
            let described = match field.data_type() {
                DataType::Boolean => {
                    json!(
                        batch
                            .column_by_name(field.name())
                            .unwrap()
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap()
                            .value(0)
                    )
                },
                _ => text(&batch, field.name(), 0).map_or(Value::Null, Value::String),
            };
            assert_eq!(
                captured[field.name()],
                described,
                "{function}.{} drifted from its captured spec; run website/scripts/capture-cugraph-specs.sh",
                field.name()
            );
        }
        assert_eq!(captured.as_object().unwrap().len(), batch.num_columns(), "{function} spec field set drifted");
    }
}

/// One `gpu_validate_call` row reports every option error and the relation
/// schema error together, in Algeon's own words.
#[tokio::test]
async fn cugraph_validation_reports_option_and_schema_problems_in_one_row() {
    let ctx = cugraph_context();
    let call =
        json!({"relations": {"edges": {"table": "candidate_pairs"}}, "options": {"alpha": 2.0, "epsilon": -1.0}})
            .to_string();
    let batch = collect(
        &ctx,
        &format!("SELECT valid, error_code, message FROM {GPU_VALIDATE_CALL_FUNCTION}('{CUGRAPH_PAGERANK_FUNCTION}', '{call}')"),
    )
    .await;
    assert!(!batch.column(0).as_any().downcast_ref::<BooleanArray>().unwrap().value(0));
    assert_eq!(text(&batch, "error_code", 0).as_deref(), Some(INVALID_VALUE));
    let message = text(&batch, "message", 0).unwrap();
    assert!(!message.contains("External error"), "{message}");
    for problem in ["'alpha'", "'epsilon'", "missing canonical source column 'src'"] {
        assert!(message.contains(problem), "{problem}: {message}");
    }
    assert_eq!(message.matches("; ").count(), 2, "{message}");
}

#[tokio::test]
async fn cugraph_vertex_dtype_rejection_lists_the_supported_vertex_types() {
    let ctx = cugraph_context();
    register_table(&ctx, "float_edges", &[("src", DataType::Float64), ("dst", DataType::Float64)]);
    let error = ctx
        .state()
        .create_logical_plan("SELECT * FROM cugraph_pagerank(edges => (SELECT src, dst FROM float_edges))")
        .await
        .unwrap_err();
    let typed = error.find_algeon_error().expect("typed vertex dtype rejection");
    assert_eq!(typed.code(), ErrorCode::UnsupportedType);
    assert_eq!(
        typed.message(),
        "cuGraph cugraph_pagerank source column has unsupported dtype Float64; supported vertex types: Int32, Int64, Utf8, LargeUtf8, Utf8View"
    );
}
