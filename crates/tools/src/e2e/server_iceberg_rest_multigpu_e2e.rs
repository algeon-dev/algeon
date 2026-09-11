//! Production-topology Flight SQL acceptance against local Iceberg REST + RustFS.
//!
//! This is deliberately an explicit tool, not a Cargo test. It launches the
//! release `algeon_server` as a child with one Tokio worker and four
//! GPUs, drives concurrent TPC-H statements through Flight SQL, and retains
//! machine-readable evidence under `target/production_acceptance`.
//!
//! Use `fixture/iceberg-local/e2e.sh iceberg-local --server-rest-multigpu-only` so
//! the REST/RustFS stack, fixture load, release build, and GPU flock are all
//! handled consistently.

#[path = "server_iceberg_rest_multigpu_e2e/artifacts.rs"]
mod artifacts;
#[path = "server_iceberg_rest_multigpu_e2e/churn.rs"]
mod churn;
#[path = "server_iceberg_rest_multigpu_e2e/flight.rs"]
mod flight;
#[path = "server_iceberg_rest_multigpu_e2e/probe.rs"]
mod probe;
#[cfg(feature = "server-runtime-observability-e2e")]
#[path = "server_iceberg_rest_multigpu_e2e/runtime_observability/mod.rs"]
mod runtime_observability;
#[path = "server_iceberg_rest_multigpu_e2e/server.rs"]
mod server;
#[path = "server_iceberg_rest_multigpu_e2e/validation.rs"]
mod validation;

use std::{
    collections::BTreeMap,
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use tonic::transport::Channel;

use self::{
    artifacts::{write_probe_artifacts, write_query_artifacts, write_run_metadata, write_summary},
    churn::{CHURN_TIMEOUT, run_churn},
    flight::{CpuReference, Q06_WAVE_SIZE, Workload, build_cpu_references, run_mixed_wave, run_wave, tpch_workloads},
    probe::{ProbeReport, REQUIRED_SATURATED_PROBES, SATURATION_FLOOR, run_probes},
    server::{CONCURRENT_CLIENTS, MAX_ACTIVE_ATTEMPTS_PER_DEVICE, Params, ServerChild, require_opt_in},
    validation::{
        validate_cold_and_warm_waves, validate_final_diagnostics, validate_observed_devices, validate_static_topology,
        wait_for_idle,
    },
};

type E2eResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub async fn run() -> E2eResult<()> {
    require_opt_in()?;
    let params = Params::from_env()?;
    fs::create_dir_all(&params.artifact_dir)?;
    write_run_metadata(&params)?;

    eprintln!("building CPU references through Iceberg REST/RustFS");
    let workloads = tpch_workloads(&params);
    let cpu_references = build_cpu_references(&params, &workloads).await?;

    let mut server = ServerChild::spawn(&params)?;
    let channel = match server.connect(&params.server_url).await {
        Ok(channel) => channel,
        Err(error) => {
            let _ = fs::write(params.artifact_dir.join("FAILURE.txt"), format!("{error:#}\n"));
            return Err(error);
        },
    };

    let acceptance = run_acceptance(&params, channel, &workloads, &cpu_references).await;
    if let Err(error) = &acceptance {
        let _ = fs::write(params.artifact_dir.join("FAILURE.txt"), format!("{error:#}\n"));
    }
    let shutdown = server.shutdown().await;
    acceptance?;
    shutdown?;

    println!("server_iceberg_rest_multigpu_e2e ok: {}", params.artifact_dir.display());
    Ok(())
}

#[cfg(feature = "server-runtime-observability-e2e")]
pub async fn run_runtime_observability(args: Vec<String>) -> E2eResult<()> {
    runtime_observability::run(args).await
}

async fn run_acceptance(
    params: &Params,
    channel: Channel,
    workloads: &BTreeMap<&'static str, Workload>,
    cpu_references: &BTreeMap<&'static str, CpuReference>,
) -> E2eResult<()> {
    let before = flight::get_admission_stats(&channel).await?;
    validate_static_topology(&before, &params.devices)?;
    fs::write(params.artifact_dir.join("admission.before.txt"), &before)?;

    let probe_stop = Arc::new(AtomicBool::new(false));
    let probe_task = tokio::spawn(run_probes(channel.clone(), Arc::clone(&probe_stop)));

    let waves = async {
        eprintln!("running synchronized cold Q06 wave ({Q06_WAVE_SIZE} statements)");
        let q06 = workloads.get("q06").expect("q06 workload").clone();
        let q06_cpu = cpu_references.get("q06").expect("q06 CPU reference").clone();
        let cold = run_wave(&channel, "cold_q06", vec![q06.clone(); Q06_WAVE_SIZE], q06_cpu.clone()).await?;

        eprintln!("running synchronized warm Q06 wave ({Q06_WAVE_SIZE} statements)");
        let warm = run_wave(&channel, "warm_q06", vec![q06; Q06_WAVE_SIZE], q06_cpu).await?;

        eprintln!("running mixed concurrent Q01/Q03/Q06/Q10 wave");
        let mut mixed_workloads = Vec::with_capacity(CONCURRENT_CLIENTS);
        for _ in 0..MAX_ACTIVE_ATTEMPTS_PER_DEVICE {
            for query_id in ["q01", "q03", "q06", "q10"] {
                mixed_workloads.push(workloads.get(query_id).expect("mixed workload").clone());
            }
        }
        let mixed = run_mixed_wave(&channel, "mixed_olap", mixed_workloads, cpu_references).await?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((cold, warm, mixed))
    }
    .await;

    probe_stop.store(true, Ordering::Release);
    let probes: E2eResult<ProbeReport> = match probe_task.await {
        Ok(result) => result,
        Err(error) => Err(format!("probe task failed: {error}").into()),
    };
    let (cold, warm, mixed) = waves?;
    let probes = probes?;
    if probes.saturated_count < REQUIRED_SATURATED_PROBES {
        return Err(format!(
            "only {} diagnostics/GetFlightInfo probe pairs completed while active GPU work was at least {SATURATION_FLOOR}; required {REQUIRED_SATURATED_PROBES}",
            probes.saturated_count
        )
        .into());
    }
    if probes.tickets_issued != probes.tickets_retired || probes.tickets_issued != probes.rows.len() {
        return Err(format!(
            "liveness probe ticket accounting mismatch: issued={} retired={} rows={}",
            probes.tickets_issued,
            probes.tickets_retired,
            probes.rows.len()
        )
        .into());
    }
    fs::write(params.artifact_dir.join("admission.peak.txt"), &probes.peak_payload)?;
    write_probe_artifacts(params, &probes)?;

    validate_cold_and_warm_waves(&cold, &warm, &params.devices)?;

    eprintln!("running cancellation and dropped-stream churn");
    let churn = tokio::time::timeout(
        CHURN_TIMEOUT,
        run_churn(
            &channel,
            workloads.get("q03").expect("q03 workload"),
            cpu_references.get("q03").expect("q03 CPU reference"),
        ),
    )
    .await
    .map_err(|_| format!("cancellation/drop churn exceeded {} seconds", CHURN_TIMEOUT.as_secs()))??;

    let after = wait_for_idle(&channel).await?;
    validate_final_diagnostics(&after, &params.devices)?;
    fs::write(params.artifact_dir.join("admission.after.txt"), &after)?;

    let mut successful = Vec::new();
    successful.extend(cold);
    successful.extend(warm);
    successful.extend(mixed);
    successful.extend(churn.completed.iter().cloned());
    validate_observed_devices(&successful, &params.devices)?;
    write_query_artifacts(params, &successful, &churn)?;
    write_summary(params, &successful, &churn, &probes, &after)?;
    Ok(())
}
