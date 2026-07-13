//! Synchronized allocator-path stress for query memory domains.
//!
//! Usage:
//! `cargo run --release --example query_memory_domain_alloc_storm -- MODE LANES WARMUPS ITERATIONS SIZES`
//!
//! `MODE` is one of `raw_private`, `arena`, `domain_routed`, or
//! `domain_member_direct`. `SIZES` is a comma-separated byte list.

use std::{
    str::FromStr,
    sync::{Arc, Barrier},
    time::Instant,
};

use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryResource, QueryMemoryDomain, QueryMemoryDomainOptions, ReservationArena,
    Runtime, StreamReservationLease,
};

const DEFAULT_LANES: usize = 16;
const DEFAULT_WARMUPS: usize = 1_000;
const DEFAULT_ITERATIONS: usize = 100_000;
const DEFAULT_SIZES: &[usize] = &[256, 1024, 4096, 16 * 1024, 64 * 1024];
const LANE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
enum Mode {
    RawPrivate,
    Arena,
    DomainRouted,
    DomainMemberDirect,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::RawPrivate => "raw_private",
            Self::Arena => "arena",
            Self::DomainRouted => "domain_routed",
            Self::DomainMemberDirect => "domain_member_direct",
        }
    }
}

impl FromStr for Mode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "raw_private" => Ok(Self::RawPrivate),
            "arena" => Ok(Self::Arena),
            "domain_routed" => Ok(Self::DomainRouted),
            "domain_member_direct" => Ok(Self::DomainMemberDirect),
            other => Err(format!(
                "unsupported mode {other:?}; expected raw_private|arena|domain_routed|domain_member_direct"
            )),
        }
    }
}

struct Lane {
    runtime: Runtime,
    lease: Option<StreamReservationLease>,
    _domain: Option<QueryMemoryDomain>,
    _arena: Option<ReservationArena>,
    _private_resource: Option<DeviceMemoryResource>,
}

#[derive(Debug)]
struct LaneResult {
    allocation_ns: Vec<u64>,
    free_ns: Vec<u64>,
    failures: usize,
    direct_total_allocations: u64,
    routed_total_allocations: u64,
}

fn build_lane(mode: Mode, lane: usize, device: DeviceId) -> Lane {
    let reservation_id = 100_000 + lane as u64;
    match mode {
        Mode::RawPrivate => {
            let resource = DeviceMemoryResource::cuda_async_on(device, None, None).expect("private async resource");
            let runtime = Runtime::builder()
                .with_device(device)
                .with_memory_resource(resource.clone())
                .build()
                .expect("private runtime");
            Lane { runtime, lease: None, _domain: None, _arena: None, _private_resource: Some(resource) }
        },
        Mode::Arena => {
            let runtime = Runtime::builder().with_device(device).with_device_memory_manager().build().expect("runtime");
            let manager = runtime.device_memory_manager().expect("manager");
            let arena = manager.reservation_arena(200_000 + lane as u64, LANE_BUDGET_BYTES).expect("arena");
            let lease =
                arena.reserve_stream(runtime.stream(), reservation_id, LANE_BUDGET_BYTES).expect("arena member");
            Lane { runtime, lease: Some(lease), _domain: None, _arena: Some(arena), _private_resource: None }
        },
        Mode::DomainRouted | Mode::DomainMemberDirect => {
            let routed_runtime =
                Runtime::builder().with_device(device).with_device_memory_manager().build().expect("runtime");
            let manager = routed_runtime.device_memory_manager().expect("manager");
            let domain = manager
                .query_memory_domain(
                    300_000 + lane as u64,
                    QueryMemoryDomainOptions::new(LANE_BUDGET_BYTES).with_ceiling_bytes(LANE_BUDGET_BYTES),
                )
                .expect("query domain");
            let lease = domain
                .reserve_stream(routed_runtime.stream(), reservation_id, LANE_BUDGET_BYTES)
                .expect("domain member");
            let runtime = if matches!(mode, Mode::DomainMemberDirect) {
                Runtime::builder()
                    .with_device(device)
                    .with_stream(routed_runtime.stream().clone())
                    .with_memory_resource(lease.memory_resource().clone())
                    .build()
                    .expect("direct member runtime")
            } else {
                routed_runtime
            };
            Lane { runtime, lease: Some(lease), _domain: Some(domain), _arena: None, _private_resource: None }
        },
    }
}

fn run_lane(
    lane: Lane,
    ready_barrier: Arc<Barrier>,
    start_barrier: Arc<Barrier>,
    warmups: usize,
    iterations: usize,
    sizes: Arc<Vec<usize>>,
) -> LaneResult {
    for index in 0..warmups {
        let buffer = lane.runtime.device_buffer(sizes[index % sizes.len()]).expect("warmup allocation");
        drop(buffer);
    }
    lane.runtime.synchronize().expect("warmup synchronize");
    ready_barrier.wait();
    start_barrier.wait();

    let mut allocation_ns = Vec::with_capacity(iterations);
    let mut free_ns = Vec::with_capacity(iterations);
    let mut failures = 0;
    for index in 0..iterations {
        let size = sizes[index % sizes.len()];
        let allocation_start = Instant::now();
        let buffer = match lane.runtime.device_buffer(size) {
            Ok(buffer) => buffer,
            Err(_) => {
                failures += 1;
                continue;
            },
        };
        allocation_ns.push(allocation_start.elapsed().as_nanos() as u64);
        let free_start = Instant::now();
        drop(buffer);
        free_ns.push(free_start.elapsed().as_nanos() as u64);
    }
    lane.runtime.synchronize().expect("measured synchronize");
    let (direct_total_allocations, routed_total_allocations) = lane
        .lease
        .as_ref()
        .map(|lease| {
            let usage = lease.usage().expect("member usage");
            (usage.direct_total_allocations, usage.routed_total_allocations)
        })
        .unwrap_or((0, 0));
    LaneResult { allocation_ns, free_ns, failures, direct_total_allocations, routed_total_allocations }
}

fn percentile(values: &mut [u64], percentile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index]
}

fn parse_sizes(value: Option<String>) -> Vec<usize> {
    value
        .map(|value| {
            value
                .split(',')
                .map(|part| part.parse::<usize>().expect("SIZES entries must be positive integers"))
                .collect::<Vec<_>>()
        })
        .filter(|sizes| !sizes.is_empty() && sizes.iter().all(|size| *size > 0))
        .unwrap_or_else(|| DEFAULT_SIZES.to_vec())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().as_deref().unwrap_or("domain_member_direct").parse::<Mode>().expect("MODE");
    let lanes = args.next().and_then(|value| value.parse().ok()).unwrap_or(DEFAULT_LANES);
    let warmups = args.next().and_then(|value| value.parse().ok()).unwrap_or(DEFAULT_WARMUPS);
    let iterations = args.next().and_then(|value| value.parse().ok()).unwrap_or(DEFAULT_ITERATIONS);
    let sizes = Arc::new(parse_sizes(args.next()));
    assert!(lanes > 0 && iterations > 0);

    let device = DeviceId::new(0).expect("device 0");
    if !matches!(mode, Mode::RawPrivate) {
        DeviceMemoryManager::install_for_device(device).expect("install device manager");
    }

    let setup_start = Instant::now();
    let lane_states = (0..lanes).map(|lane| build_lane(mode, lane, device)).collect::<Vec<_>>();
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    let ready_barrier = Arc::new(Barrier::new(lanes + 1));
    let start_barrier = Arc::new(Barrier::new(lanes + 1));
    let measured_start = Instant::now();
    let workers = lane_states
        .into_iter()
        .map(|lane| {
            let ready_barrier = Arc::clone(&ready_barrier);
            let start_barrier = Arc::clone(&start_barrier);
            let sizes = Arc::clone(&sizes);
            std::thread::spawn(move || run_lane(lane, ready_barrier, start_barrier, warmups, iterations, sizes))
        })
        .collect::<Vec<_>>();
    ready_barrier.wait();
    let measurement_start = Instant::now();
    start_barrier.wait();
    let results = workers.into_iter().map(|worker| worker.join().expect("worker")).collect::<Vec<_>>();
    let measured_ms = measurement_start.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = measured_start.elapsed().as_secs_f64() * 1_000.0;

    let mut allocation_ns = results.iter().flat_map(|result| result.allocation_ns.iter().copied()).collect::<Vec<_>>();
    let mut free_ns = results.iter().flat_map(|result| result.free_ns.iter().copied()).collect::<Vec<_>>();
    let failures = results.iter().map(|result| result.failures).sum::<usize>();
    let direct_total_allocations = results.iter().map(|result| result.direct_total_allocations).sum::<u64>();
    let routed_total_allocations = results.iter().map(|result| result.routed_total_allocations).sum::<u64>();
    let completed = allocation_ns.len();
    let throughput = completed as f64 / (measured_ms / 1_000.0);

    println!(
        "mode\tlanes\twarmups_per_lane\titerations_per_lane\tcompleted\tfailures\tsetup_ms\tmeasured_ms\ttotal_ms\tthroughput_ops_s\talloc_p50_ns\talloc_p95_ns\talloc_p99_ns\tfree_p50_ns\tfree_p95_ns\tfree_p99_ns\tdirect_total_allocations\trouted_total_allocations\tsizes"
    );
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        mode.as_str(),
        lanes,
        warmups,
        iterations,
        completed,
        failures,
        setup_ms,
        measured_ms,
        total_ms,
        throughput,
        percentile(&mut allocation_ns.clone(), 0.50),
        percentile(&mut allocation_ns, 0.95),
        percentile(&mut allocation_ns, 0.99),
        percentile(&mut free_ns.clone(), 0.50),
        percentile(&mut free_ns, 0.95),
        percentile(&mut free_ns, 0.99),
        direct_total_allocations,
        routed_total_allocations,
        sizes.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
    );
}
