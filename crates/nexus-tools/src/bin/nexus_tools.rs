//! Unified developer/operator tool entrypoint.
//!
//! E2E runners are dispatched as explicit subcommands. The test harness only
//! includes pure parser/CLI unit tests; external-service runners remain behind
//! `not(test)` and are executed explicitly with `cargo run`.

#![allow(clippy::allow_attributes)] // reason = "tool binary contains extensive multi-feature conditional compilation"

use std::{env, error::Error};

type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[cfg(all(any(feature = "clickbench-stage-probe", feature = "cugraph-similarity-calibrate"), not(test)))]
#[path = "../gpu_backend.rs"]
mod gpu_backend;

#[cfg(all(feature = "clickbench-stage-probe", not(test)))]
#[path = "../clickbench_stage_probe.rs"]
mod clickbench_stage_probe;
// The timeout, exponential, and abort retry paths are only constructed by the
// feature-gated server e2e commands, so allow dead code when they are off.
#[cfg(any(not(test), feature = "server-runtime-observability-e2e"))]
#[cfg_attr(any(test, not(feature = "server-iceberg-e2e")), allow(dead_code))]
#[path = "../connection_retry.rs"]
mod connection_retry;
#[cfg(all(feature = "cugraph-oracle", not(test)))]
#[path = "../../../../tests/cugraph/cugraph_bfs_predicate_fixture.rs"]
mod cugraph_bfs_predicate_fixture;
#[cfg(all(feature = "cugraph-oracle", not(test)))]
#[path = "../../../../tests/cugraph/optional_fixture.rs"]
mod cugraph_optional_fixture;
#[cfg(all(feature = "cugraph-similarity-calibrate", not(test)))]
#[path = "../cugraph_similarity_calibrate.rs"]
mod cugraph_similarity_calibrate;
#[cfg(all(feature = "cugraph-oracle", not(test)))]
#[path = "../../../../tests/cugraph/cugraph_sql_tests.rs"]
// Path-includes the shared cuGraph SQL fixtures for the pagerank tools runner; most
// helpers are only exercised by Cargo test modules that this binary does not pull in.
#[allow(dead_code, unused_imports)]
mod cugraph_sql_tests;
#[cfg(all(feature = "datafusion-baseline-contract-validate", not(test)))]
#[path = "../datafusion_baseline_contract_validate.rs"]
mod datafusion_baseline_contract_validate;
#[cfg(all(feature = "embedded-backend-e2e", not(test)))]
#[path = "../e2e/embedded_backend_e2e.rs"]
mod embedded_backend_e2e;
#[cfg(not(test))]
#[path = "../e2e/flight_sql_cugraph_smoke.rs"]
mod flight_sql_cugraph_smoke;
#[cfg(not(test))]
#[path = "../flight_sql_query.rs"]
mod flight_sql_query;
// The stress driver carries pure parser/reduction unit tests, so it compiles
// under `test` like single_node_multi_gpu_validate; only its network runner
// stays behind `not(test)` inside the module.
#[cfg_attr(test, allow(dead_code))]
#[path = "../flight_sql_stress.rs"]
mod flight_sql_stress;
#[cfg(all(feature = "iceberg-e2e", not(test)))]
#[path = "../e2e/iceberg_glue_moto_e2e.rs"]
mod iceberg_glue_moto_e2e;
#[cfg(all(feature = "iceberg-e2e", not(test)))]
#[path = "../e2e/iceberg_rest_local_e2e.rs"]
mod iceberg_rest_local_e2e;
#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
#[path = "../e2e/server_iceberg_glue_moto_e2e.rs"]
mod server_iceberg_glue_moto_e2e;
#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
#[path = "../e2e/server_iceberg_glue_remote_direct_e2e.rs"]
mod server_iceberg_glue_remote_direct_e2e;
#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
#[path = "../e2e/server_iceberg_rest_multigpu_e2e.rs"]
mod server_iceberg_rest_multigpu_e2e;
#[cfg(all(feature = "server-runtime-observability-e2e", test))]
#[cfg_attr(test, allow(dead_code))]
#[path = "../e2e/server_iceberg_rest_multigpu_e2e.rs"]
mod server_iceberg_rest_multigpu_e2e;
#[cfg(feature = "single-node-multi-gpu-validate")]
#[cfg_attr(test, allow(dead_code))]
#[path = "../single_node_multi_gpu_validate/mod.rs"]
mod single_node_multi_gpu_validate;
#[cfg(feature = "validate-query")]
#[path = "../validate_query.rs"]
mod validate_query;
#[tokio::main(flavor = "current_thread")]
async fn main() -> ToolResult<()> {
    let command = env::args().nth(1);
    match command.as_deref() {
        Some("flight-sql-cugraph-smoke") => run_flight_sql_cugraph_smoke(env::args().nth(2)).await,
        Some("flight-sql-query") => run_flight_sql_query(env::args().skip(2).collect()).await,
        Some("flight-sql-stress") => run_flight_sql_stress(env::args().skip(2).collect()).await,
        Some("iceberg-rest-local-e2e") => run_iceberg_rest_local_e2e().await,
        Some("iceberg-glue-moto-e2e") => run_iceberg_glue_moto_e2e().await,
        Some("server-iceberg-glue-moto-e2e") => run_server_iceberg_glue_moto_e2e().await,
        Some("server-iceberg-glue-remote-direct-e2e") => run_server_iceberg_glue_remote_direct_e2e().await,
        Some("server-iceberg-rest-multigpu-e2e") => run_server_iceberg_rest_multigpu_e2e().await,
        Some("server-runtime-observability-e2e") => {
            run_server_runtime_observability_e2e(env::args().skip(2).collect()).await
        },
        Some("embedded-backend-e2e") => run_embedded_backend_e2e(env::args().skip(2).collect()).await,
        Some("cugraph-fixture-e2e") => run_cugraph_fixture_e2e(env::args().nth(2)).await,
        Some("cugraph-similarity-calibrate") => run_cugraph_similarity_calibrate(env::args().skip(2).collect()).await,
        Some("validate-query") => run_validate_query(env::args().skip(2).collect()).await,
        Some("clickbench-stage-probe") => run_clickbench_stage_probe(env::args().skip(2).collect()).await,
        Some("datafusion-baseline-contract-validate") => {
            run_datafusion_baseline_contract_validate(env::args().skip(2).collect()).await
        },
        Some("single-node-multi-gpu-validate") => {
            run_single_node_multi_gpu_validate(env::args().skip(2).collect()).await
        },
        Some("-h" | "--help") | None => {
            print_usage();
            if command.is_none() { Err("missing tool command".into()) } else { Ok(()) }
        },
        Some(other) => {
            print_usage();
            Err(format!("unknown nexus-tools command: {other}").into())
        },
    }
}

fn print_usage() {
    eprintln!(
        "nexus_tools\n\
         \n\
         USAGE:\n\
           cargo run -p nexus-tools --features validate-query -- validate-query --sql <SQL> [--format json|human] [--parquet TABLE=PATH]...\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features clickbench-stage-probe -- clickbench-stage-probe [OPTIONS]\n\
           cargo run -p nexus-tools -- flight-sql-cugraph-smoke [URL]\n\
           cargo run -p nexus-tools -- flight-sql-query [URL] [SQL]\n\
           cargo run -p nexus-tools -- flight-sql-stress --config flight-stress-profile.json --out-dir DIR\n\
           cargo run -p nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e\n\
           cargo run -p nexus-tools --features iceberg-e2e -- iceberg-glue-moto-e2e\n\
           cargo run -p nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-moto-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-remote-direct-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features server-iceberg-e2e -- server-iceberg-rest-multigpu-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features server-runtime-observability-e2e -- server-runtime-observability-e2e [OPTIONS]\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features embedded-backend-e2e -- embedded-backend-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features cugraph-oracle -- cugraph-fixture-e2e <pagerank|bfs-include-edges|bfs-utf8-include-edges|bfs-utf8-path|bfs-utf8-predicates>\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features cugraph-similarity-calibrate -- cugraph-similarity-calibrate --algorithm <jaccard|sorensen|overlap|cosine> --pair-rows <N> [--weighted] [--vertex-dtype int32|int64] [--topology sparse-uniform|hub-skewed] [--edge-parquet PATH --edge-row-groups N] [--observation-out PATH]\n\
           flock /tmp/cudf-gpu.lock cargo run -p nexus-tools --features datafusion-baseline-contract-validate -- datafusion-baseline-contract-validate [--out-dir DIR]\n\
           cargo run -p nexus-tools --features single-node-multi-gpu-validate -- single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --dry-run\n           cargo run -p nexus-tools --features single-node-multi-gpu-validate -- single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --fixture-preflight\n           flock /tmp/cudf-gpu.lock cargo run --release -p nexus-tools --features single-node-multi-gpu-validate-run -- single-node-multi-gpu-validate --config run-profile.json --out-dir run-artifacts --measured-run [--nsight-trace PATH]"
    );
}

#[cfg(not(test))]
async fn run_flight_sql_cugraph_smoke(url: Option<String>) -> ToolResult<()> {
    flight_sql_cugraph_smoke::run(url).await
}

#[cfg(test)]
async fn run_flight_sql_cugraph_smoke(_url: Option<String>) -> ToolResult<()> {
    Err("flight-sql-cugraph-smoke requires normal cargo run execution".into())
}

#[cfg(not(test))]
async fn run_flight_sql_query(args: Vec<String>) -> ToolResult<()> {
    flight_sql_query::run(args).await
}

#[cfg(test)]
async fn run_flight_sql_query(_args: Vec<String>) -> ToolResult<()> {
    Err("flight-sql-query requires normal cargo run execution".into())
}

#[cfg(not(test))]
async fn run_flight_sql_stress(args: Vec<String>) -> ToolResult<()> {
    flight_sql_stress::run(args).await
}

#[cfg(test)]
async fn run_flight_sql_stress(_args: Vec<String>) -> ToolResult<()> {
    Err("flight-sql-stress requires normal cargo run execution".into())
}

#[cfg(feature = "validate-query")]
async fn run_validate_query(args: Vec<String>) -> ToolResult<()> {
    validate_query::run(args).await
}

#[cfg(not(feature = "validate-query"))]
async fn run_validate_query(_args: Vec<String>) -> ToolResult<()> {
    Err("validate-query requires --features validate-query".into())
}

#[cfg(all(feature = "datafusion-baseline-contract-validate", not(test)))]
async fn run_datafusion_baseline_contract_validate(args: Vec<String>) -> ToolResult<()> {
    datafusion_baseline_contract_validate::run(args).await
}

#[cfg(not(all(feature = "datafusion-baseline-contract-validate", not(test))))]
async fn run_datafusion_baseline_contract_validate(_args: Vec<String>) -> ToolResult<()> {
    Err(
        "datafusion-baseline-contract-validate requires --features datafusion-baseline-contract-validate and normal cargo run execution"
            .into(),
    )
}

#[cfg(all(feature = "single-node-multi-gpu-validate", not(test)))]
async fn run_single_node_multi_gpu_validate(args: Vec<String>) -> ToolResult<()> {
    single_node_multi_gpu_validate::run(args)
}

#[cfg(not(all(feature = "single-node-multi-gpu-validate", not(test))))]
async fn run_single_node_multi_gpu_validate(_args: Vec<String>) -> ToolResult<()> {
    Err("single-node-multi-gpu-validate requires --features single-node-multi-gpu-validate and normal cargo run execution".into())
}

#[cfg(all(feature = "clickbench-stage-probe", not(test)))]
async fn run_clickbench_stage_probe(args: Vec<String>) -> ToolResult<()> {
    clickbench_stage_probe::run(args).await
}

#[cfg(not(all(feature = "clickbench-stage-probe", not(test))))]
async fn run_clickbench_stage_probe(_args: Vec<String>) -> ToolResult<()> {
    Err("clickbench-stage-probe requires --features clickbench-stage-probe and normal cargo run execution".into())
}

#[cfg(all(feature = "iceberg-e2e", not(test)))]
async fn run_iceberg_rest_local_e2e() -> ToolResult<()> {
    iceberg_rest_local_e2e::run().await
}

#[cfg(not(all(feature = "iceberg-e2e", not(test))))]
async fn run_iceberg_rest_local_e2e() -> ToolResult<()> {
    Err("iceberg-rest-local-e2e requires --features iceberg-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "iceberg-e2e", not(test)))]
async fn run_iceberg_glue_moto_e2e() -> ToolResult<()> {
    iceberg_glue_moto_e2e::run().await
}

#[cfg(not(all(feature = "iceberg-e2e", not(test))))]
async fn run_iceberg_glue_moto_e2e() -> ToolResult<()> {
    Err("iceberg-glue-moto-e2e requires --features iceberg-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
async fn run_server_iceberg_glue_moto_e2e() -> ToolResult<()> {
    server_iceberg_glue_moto_e2e::run().await
}

#[cfg(not(all(feature = "server-iceberg-e2e", not(test))))]
async fn run_server_iceberg_glue_moto_e2e() -> ToolResult<()> {
    Err("server-iceberg-glue-moto-e2e requires --features server-iceberg-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
async fn run_server_iceberg_glue_remote_direct_e2e() -> ToolResult<()> {
    server_iceberg_glue_remote_direct_e2e::run().await
}

#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
async fn run_server_iceberg_rest_multigpu_e2e() -> ToolResult<()> {
    server_iceberg_rest_multigpu_e2e::run().await
}

#[cfg(all(feature = "server-runtime-observability-e2e", not(test)))]
async fn run_server_runtime_observability_e2e(args: Vec<String>) -> ToolResult<()> {
    server_iceberg_rest_multigpu_e2e::run_runtime_observability(args).await
}

#[cfg(not(all(feature = "server-runtime-observability-e2e", not(test))))]
async fn run_server_runtime_observability_e2e(_args: Vec<String>) -> ToolResult<()> {
    Err("server-runtime-observability-e2e requires --features server-runtime-observability-e2e and normal cargo run execution".into())
}

#[cfg(not(all(feature = "server-iceberg-e2e", not(test))))]
async fn run_server_iceberg_rest_multigpu_e2e() -> ToolResult<()> {
    Err("server-iceberg-rest-multigpu-e2e requires --features server-iceberg-e2e and normal cargo run execution".into())
}

#[cfg(not(all(feature = "server-iceberg-e2e", not(test))))]
async fn run_server_iceberg_glue_remote_direct_e2e() -> ToolResult<()> {
    Err("server-iceberg-glue-remote-direct-e2e requires --features server-iceberg-e2e and normal cargo run execution"
        .into())
}

#[cfg(all(feature = "embedded-backend-e2e", not(test)))]
async fn run_embedded_backend_e2e(args: Vec<String>) -> ToolResult<()> {
    embedded_backend_e2e::run(args).await
}

#[cfg(not(all(feature = "embedded-backend-e2e", not(test))))]
async fn run_embedded_backend_e2e(_args: Vec<String>) -> ToolResult<()> {
    Err("embedded-backend-e2e requires --features embedded-backend-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "cugraph-oracle", not(test)))]
async fn run_cugraph_fixture_e2e(case: Option<String>) -> ToolResult<()> {
    match case.as_deref() {
        Some("pagerank") => cugraph_sql_tests::gpu::run_pagerank_filter_sort_limit_acceptance().await,
        Some("bfs-include-edges") => cugraph_bfs_predicate_fixture::run_numeric_include_edges_acceptance().await,
        Some("bfs-utf8-include-edges") => cugraph_bfs_predicate_fixture::run_utf8_include_edges_acceptance().await,
        Some("bfs-utf8-path") => cugraph_bfs_predicate_fixture::run_utf8_path_acceptance().await,
        Some("bfs-utf8-predicates") => cugraph_bfs_predicate_fixture::run_utf8_vertex_predicates_acceptance().await,
        Some(other) => return Err(format!("unknown cugraph fixture acceptance case: {other}").into()),
        None => return Err("cugraph-fixture-e2e requires an acceptance case".into()),
    }
    Ok(())
}

#[cfg(not(all(feature = "cugraph-oracle", not(test))))]
async fn run_cugraph_fixture_e2e(_case: Option<String>) -> ToolResult<()> {
    Err("cugraph-fixture-e2e requires --features cugraph-oracle and normal cargo run execution".into())
}

#[cfg(all(feature = "cugraph-similarity-calibrate", not(test)))]
async fn run_cugraph_similarity_calibrate(args: Vec<String>) -> ToolResult<()> {
    cugraph_similarity_calibrate::run(args)
}

#[cfg(not(all(feature = "cugraph-similarity-calibrate", not(test))))]
async fn run_cugraph_similarity_calibrate(_args: Vec<String>) -> ToolResult<()> {
    Err("cugraph-similarity-calibrate requires --features cugraph-similarity-calibrate and normal cargo run execution"
        .into())
}
