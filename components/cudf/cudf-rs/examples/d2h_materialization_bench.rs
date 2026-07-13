//! Explicit device-to-host materialization benchmark.
//!
//! This tool compares reusable pageable and pinned destinations with identical
//! synchronization semantics, includes symmetric pageable-to-pageable and
//! pinned-to-pageable staging variants, and measures the existing
//! `DataFrame::to_record_batch` path end to end.
//! It prints one TSV row per trial and has no timing assertions.
//!
//! Run GPU measurements under the repository-wide GPU lock, for example:
//! `flock /tmp/cudf-gpu.lock cargo run --release -p cudf --example d2h_materialization_bench`
//!
//! Configuration environment variables:
//! - `CUDF_D2H_DEVICE_ORDINAL` (default `0`)
//! - `CUDF_D2H_SIZES_BYTES` (default `65536,4194304,33554432`)
//! - `CUDF_D2H_CONCURRENCIES` (default `1,8`)
//! - `CUDF_D2H_CASES` (default `i32,f64,nullable_i64,utf8_16,mixed_fixed`)
//! - `CUDF_D2H_TRIALS` (default `3`)
//! - `CUDF_D2H_WARMUPS` (default `2` per lane)
//! - `CUDF_D2H_TARGET_MEASURED_BYTES` (default `134217728` per trial)
//! - `CUDF_D2H_OUTPUT_PATH` (optional TSV output path; stdout when unset)

use std::{
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    str::FromStr,
    sync::{Arc, Barrier},
    time::Instant,
};

use arrow::{
    array::{ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DeviceId, DeviceMemoryManager, Runtime,
    testing::{
        BorrowedViewTestingExt,
        column::{D2hBenchmarkBuffer, D2hBenchmarkHostMemory},
    },
};

const DEFAULT_SIZES: &[usize] = &[64 * 1024, 4 * 1024 * 1024, 32 * 1024 * 1024];
const DEFAULT_CONCURRENCIES: &[usize] = &[1, 8];
const DEFAULT_TARGET_MEASURED_BYTES: usize = 128 * 1024 * 1024;
const MAX_ITERATIONS_PER_LANE: usize = 2_000;
const MIN_ITERATIONS_PER_LANE: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypeCase {
    I32,
    F64,
    NullableI64,
    Utf8_16,
    MixedFixed,
}

impl TypeCase {
    fn as_str(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::F64 => "f64",
            Self::NullableI64 => "nullable_i64",
            Self::Utf8_16 => "utf8_16",
            Self::MixedFixed => "mixed_fixed",
        }
    }

    fn estimated_d2h_bytes_per_row(self) -> usize {
        match self {
            Self::I32 => 4,
            Self::F64 => 8,
            Self::NullableI64 => 8,
            Self::Utf8_16 => 20,
            Self::MixedFixed => 21,
        }
    }

    fn direct_copy_supported(self) -> bool {
        matches!(self, Self::I32 | Self::F64)
    }

    fn rows_for_target_bytes(self, target_bytes: usize) -> usize {
        (target_bytes / self.estimated_d2h_bytes_per_row()).max(1)
    }

    fn payload_bytes(self, rows: usize) -> usize {
        match self {
            Self::I32 => rows * 4,
            Self::F64 => rows * 8,
            Self::NullableI64 => rows * 8 + rows.div_ceil(8),
            Self::Utf8_16 => (rows + 1) * 4 + rows * 16,
            Self::MixedFixed => rows * (4 + 8 + 8 + 1),
        }
    }

    fn record_batch_copy_info(self) -> &'static str {
        match self {
            Self::I32 | Self::F64 => "fixed_data:1",
            Self::NullableI64 => "fixed_data:1,null_mask:1",
            Self::Utf8_16 => "string_offsets+chars:1_ffi_call",
            Self::MixedFixed => "fixed_data:4",
        }
    }
}

impl FromStr for TypeCase {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "i32" => Ok(Self::I32),
            "f64" => Ok(Self::F64),
            "nullable_i64" => Ok(Self::NullableI64),
            "utf8_16" => Ok(Self::Utf8_16),
            "mixed_fixed" => Ok(Self::MixedFixed),
            other => Err(format!(
                "unsupported CUDF_D2H_CASES entry {other:?}; expected i32|f64|nullable_i64|utf8_16|mixed_fixed"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    RawPageable,
    RawPinned,
    PageableStagedPageable,
    PinnedStagedPageable,
    ToRecordBatch,
}

impl Path {
    fn as_str(self) -> &'static str {
        match self {
            Self::RawPageable => "raw_pageable",
            Self::RawPinned => "raw_pinned",
            Self::PageableStagedPageable => "pageable_staged_pageable",
            Self::PinnedStagedPageable => "pinned_staged_pageable",
            Self::ToRecordBatch => "to_record_batch",
        }
    }

    fn final_host_memory(self) -> &'static str {
        match self {
            Self::RawPinned => "pinned",
            Self::RawPageable | Self::PageableStagedPageable | Self::PinnedStagedPageable | Self::ToRecordBatch => {
                "pageable"
            },
        }
    }

    fn synchronization(self) -> &'static str {
        match self {
            Self::RawPageable | Self::RawPinned | Self::PageableStagedPageable | Self::PinnedStagedPageable => {
                "pre_visibility_fence+ffi_stream_sync"
            },
            Self::ToRecordBatch => "production_to_record_batch",
        }
    }
}

struct Config {
    device_ordinal: usize,
    sizes: Vec<usize>,
    concurrencies: Vec<usize>,
    cases: Vec<TypeCase>,
    trials: usize,
    warmups: usize,
    target_measured_bytes: usize,
}

struct LaneResult {
    latencies_ns: Vec<u64>,
    completed: usize,
    host_output_bytes: usize,
    checksum: u64,
}

struct TrialResult {
    wall_ms: f64,
    latencies_ns: Vec<u64>,
    completed: usize,
    host_output_bytes: usize,
    checksum: u64,
}

struct TrialSpec {
    case: TypeCase,
    target_size: usize,
    rows: usize,
    payload_bytes: usize,
    path: Path,
    lanes: usize,
    trial: usize,
    warmups: usize,
    iterations: usize,
}

fn parse_csv_usize(name: &str, defaults: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| part.parse::<usize>().unwrap_or_else(|_| panic!("{name} entries must be integers")))
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty() && values.iter().all(|value| *value > 0))
        .unwrap_or_else(|| defaults.to_vec())
}

fn parse_positive_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.parse::<usize>().unwrap_or_else(|_| panic!("{name} must be an integer")))
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn config() -> Config {
    let cases = std::env::var("CUDF_D2H_CASES")
        .unwrap_or_else(|_| "i32,f64,nullable_i64,utf8_16,mixed_fixed".to_string())
        .split(',')
        .map(|value| value.parse::<TypeCase>().expect("CUDF_D2H_CASES"))
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "CUDF_D2H_CASES must not be empty");
    Config {
        device_ordinal: std::env::var("CUDF_D2H_DEVICE_ORDINAL")
            .ok()
            .map(|value| value.parse().expect("CUDF_D2H_DEVICE_ORDINAL must be an integer"))
            .unwrap_or(0),
        sizes: parse_csv_usize("CUDF_D2H_SIZES_BYTES", DEFAULT_SIZES),
        concurrencies: parse_csv_usize("CUDF_D2H_CONCURRENCIES", DEFAULT_CONCURRENCIES),
        cases,
        trials: parse_positive_usize("CUDF_D2H_TRIALS", 3),
        warmups: parse_positive_usize("CUDF_D2H_WARMUPS", 2),
        target_measured_bytes: parse_positive_usize("CUDF_D2H_TARGET_MEASURED_BYTES", DEFAULT_TARGET_MEASURED_BYTES),
    }
}

fn build_host_batch(case: TypeCase, rows: usize) -> RecordBatch {
    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = match case {
        TypeCase::I32 => (
            vec![Field::new("value", ArrowDataType::Int32, false)],
            vec![Arc::new(Int32Array::from_iter_values((0..rows).map(|row| row as i32)))],
        ),
        TypeCase::F64 => (
            vec![Field::new("value", ArrowDataType::Float64, false)],
            vec![Arc::new(Float64Array::from_iter_values((0..rows).map(|row| row as f64 + 0.25)))],
        ),
        TypeCase::NullableI64 => (
            vec![Field::new("value", ArrowDataType::Int64, true)],
            vec![Arc::new(Int64Array::from_iter((0..rows).map(|row| (row % 8 != 0).then_some(row as i64))))],
        ),
        TypeCase::Utf8_16 => (
            vec![Field::new("value", ArrowDataType::Utf8, false)],
            vec![Arc::new(StringArray::from_iter_values(std::iter::repeat_n("0123456789abcdef", rows)))],
        ),
        TypeCase::MixedFixed => (
            vec![
                Field::new("i32", ArrowDataType::Int32, false),
                Field::new("i64", ArrowDataType::Int64, false),
                Field::new("f64", ArrowDataType::Float64, false),
                Field::new("bool", ArrowDataType::Boolean, false),
            ],
            vec![
                Arc::new(Int32Array::from_iter_values((0..rows).map(|row| row as i32))),
                Arc::new(Int64Array::from_iter_values((0..rows).map(|row| row as i64))),
                Arc::new(Float64Array::from_iter_values((0..rows).map(|row| row as f64 + 0.5))),
                Arc::new(BooleanArray::from_iter((0..rows).map(|row| Some(row % 2 == 0)))),
            ],
        ),
    };
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("benchmark host RecordBatch")
}

fn sample_checksum(bytes: &[u8]) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut hash = bytes.len() as u64;
    for index in [0, bytes.len() / 3, bytes.len() / 2, bytes.len() - 1] {
        hash = hash.rotate_left(9) ^ u64::from(bytes[index]);
    }
    hash
}

fn run_direct_lane(
    path: Path,
    frame: DataFrame,
    warmups: usize,
    iterations: usize,
    ready: Arc<Barrier>,
    start: Arc<Barrier>,
    done: Arc<Barrier>,
) -> LaneResult {
    let series = frame.series("value").expect("benchmark value series");
    let view = series.testing_view().expect("benchmark value view");
    let copy_bytes = match path {
        Path::RawPageable | Path::RawPinned | Path::PageableStagedPageable | Path::PinnedStagedPageable => {
            let dtype = series.dtype().expect("benchmark dtype");
            series.len()
                * usize::try_from(dtype.id().size_in_bytes().expect("type width").expect("fixed width"))
                    .expect("non-negative width")
        },
        Path::ToRecordBatch => unreachable!(),
    };
    let memory_kind = match path {
        Path::RawPageable | Path::PageableStagedPageable => D2hBenchmarkHostMemory::Pageable,
        Path::RawPinned | Path::PinnedStagedPageable => D2hBenchmarkHostMemory::Pinned,
        Path::ToRecordBatch => unreachable!(),
    };
    let mut destination = D2hBenchmarkBuffer::new(memory_kind, copy_bytes).expect("benchmark destination");
    let mut copy_once = || {
        let copied = destination.copy_from_fixed_width(&view).expect("D2H benchmark copy");
        if matches!(path, Path::PageableStagedPageable | Path::PinnedStagedPageable) {
            // Symmetric transfer comparison: both staging lanes reuse their
            // first D2H destination, then allocate the same final pageable
            // output and copy every byte into it. This models the transfer and
            // ownership handoff, not the rest of Arrow array construction.
            let mut pageable = vec![0u8; copy_bytes];
            pageable.copy_from_slice(destination.initialized_bytes());
            black_box(pageable.as_slice());
            black_box(sample_checksum(pageable.as_slice()))
        } else {
            black_box(sample_checksum(destination.initialized_bytes()))
        }
        .wrapping_add(copied as u64)
    };
    for _ in 0..warmups {
        black_box(copy_once());
    }
    ready.wait();
    start.wait();

    let mut latencies_ns = Vec::with_capacity(iterations);
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let started = Instant::now();
        checksum = checksum.wrapping_add(copy_once());
        latencies_ns.push(started.elapsed().as_nanos() as u64);
    }
    done.wait();
    LaneResult {
        latencies_ns,
        completed: iterations,
        host_output_bytes: copy_bytes.saturating_mul(iterations),
        checksum,
    }
}

fn run_record_batch_lane(
    frame: DataFrame,
    warmups: usize,
    iterations: usize,
    ready: Arc<Barrier>,
    start: Arc<Barrier>,
    done: Arc<Barrier>,
) -> LaneResult {
    for _ in 0..warmups {
        let batch = frame.to_record_batch().expect("warmup to_record_batch");
        black_box(batch);
    }
    ready.wait();
    start.wait();

    let mut latencies_ns = Vec::with_capacity(iterations);
    let mut host_output_bytes = 0usize;
    let mut checksum = 0u64;
    for _ in 0..iterations {
        let started = Instant::now();
        let batch = frame.to_record_batch().expect("measured to_record_batch");
        latencies_ns.push(started.elapsed().as_nanos() as u64);
        let bytes = batch.get_array_memory_size();
        host_output_bytes = host_output_bytes.saturating_add(bytes);
        checksum =
            checksum.wrapping_add((batch.num_rows() as u64).rotate_left(7) ^ bytes as u64 ^ batch.num_columns() as u64);
        black_box(batch);
    }
    done.wait();
    LaneResult { latencies_ns, completed: iterations, host_output_bytes, checksum }
}

fn run_trial(
    device: DeviceId,
    case: TypeCase,
    rows: usize,
    path: Path,
    lanes: usize,
    warmups: usize,
    iterations: usize,
) -> TrialResult {
    let ready = Arc::new(Barrier::new(lanes + 1));
    let start = Arc::new(Barrier::new(lanes + 1));
    let done = Arc::new(Barrier::new(lanes + 1));
    let workers = (0..lanes)
        .map(|_| {
            let ready = Arc::clone(&ready);
            let start = Arc::clone(&start);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let runtime = Runtime::builder()
                    .with_device(device)
                    .with_device_memory_manager()
                    .build()
                    .expect("benchmark runtime");
                let host = build_host_batch(case, rows);
                let frame = DataFrame::from_record_batch(&runtime, &host).expect("benchmark device frame");
                runtime.synchronize().expect("benchmark setup synchronize");
                if path == Path::ToRecordBatch {
                    run_record_batch_lane(frame, warmups, iterations, ready, start, done)
                } else {
                    run_direct_lane(path, frame, warmups, iterations, ready, start, done)
                }
            })
        })
        .collect::<Vec<_>>();
    ready.wait();
    let started = Instant::now();
    start.wait();
    done.wait();
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let results = workers.into_iter().map(|worker| worker.join().expect("benchmark lane")).collect::<Vec<_>>();

    TrialResult {
        wall_ms,
        latencies_ns: results.iter().flat_map(|result| result.latencies_ns.iter().copied()).collect(),
        completed: results.iter().map(|result| result.completed).sum(),
        host_output_bytes: results.iter().map(|result| result.host_output_bytes).sum(),
        checksum: results.iter().fold(0u64, |checksum, result| checksum.wrapping_add(result.checksum)),
    }
}

fn percentile_ns(values: &mut [u64], quantile: f64) -> u64 {
    values.sort_unstable();
    values[((values.len() - 1) as f64 * quantile).round() as usize]
}

fn iterations_per_lane(payload_bytes: usize, lanes: usize, target_measured_bytes: usize) -> usize {
    target_measured_bytes
        .checked_div(payload_bytes.saturating_mul(lanes).max(1))
        .unwrap_or(0)
        .clamp(MIN_ITERATIONS_PER_LANE, MAX_ITERATIONS_PER_LANE)
}

fn print_result(output: &mut dyn Write, spec: &TrialSpec, mut result: TrialResult) {
    let seconds = result.wall_ms / 1_000.0;
    let payload_total = spec.payload_bytes.saturating_mul(result.completed);
    let payload_gib_s = payload_total as f64 / seconds / (1024.0 * 1024.0 * 1024.0);
    let host_output_gib_s = result.host_output_bytes as f64 / seconds / (1024.0 * 1024.0 * 1024.0);
    let ops_s = result.completed as f64 / seconds;
    let p50_us = percentile_ns(&mut result.latencies_ns.clone(), 0.50) as f64 / 1_000.0;
    let p95_us = percentile_ns(&mut result.latencies_ns.clone(), 0.95) as f64 / 1_000.0;
    let p99_us = percentile_ns(&mut result.latencies_ns, 0.99) as f64 / 1_000.0;
    let copy_info = match spec.path {
        Path::PageableStagedPageable | Path::PinnedStagedPageable => "fixed_data:1+cpu_memcpy:1+pageable_alloc:1",
        Path::ToRecordBatch => spec.case.record_batch_copy_info(),
        Path::RawPageable | Path::RawPinned => "fixed_data:1",
    };
    writeln!(
        output,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.3}\t{:.3}\t{:.3}\t{}",
        spec.path.as_str(),
        spec.case.as_str(),
        spec.target_size,
        spec.rows,
        spec.payload_bytes,
        spec.path.final_host_memory(),
        spec.path.synchronization(),
        copy_info,
        spec.lanes,
        spec.trial,
        spec.warmups,
        spec.iterations,
        result.completed,
        result.wall_ms,
        payload_gib_s,
        host_output_gib_s,
        ops_s,
        p50_us,
        p95_us,
        p99_us,
        result.checksum,
    )
    .expect("write benchmark TSV row");
}

fn main() {
    let config = config();
    let device = DeviceId::new(config.device_ordinal).expect("CUDF_D2H_DEVICE_ORDINAL");
    DeviceMemoryManager::install_for_device(device).expect("install device memory manager");
    let mut output: Box<dyn Write> = match std::env::var("CUDF_D2H_OUTPUT_PATH") {
        Ok(path) => {
            let path = std::path::Path::new(&path);
            if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).expect("create CUDF_D2H_OUTPUT_PATH parent");
            }
            Box::new(BufWriter::new(File::create(path).expect("create CUDF_D2H_OUTPUT_PATH")))
        },
        Err(_) => Box::new(BufWriter::new(std::io::stdout())),
    };

    writeln!(output, "# cudf_d2h_materialization_benchmark=1").expect("write benchmark metadata");
    writeln!(output, "# device_ordinal={}", config.device_ordinal).expect("write benchmark metadata");
    writeln!(output, "# sizes_bytes={}", config.sizes.iter().map(usize::to_string).collect::<Vec<_>>().join(","))
        .expect("write benchmark metadata");
    writeln!(
        output,
        "# concurrencies={}",
        config.concurrencies.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
    )
    .expect("write benchmark metadata");
    writeln!(output, "# cases={}", config.cases.iter().map(|case| case.as_str()).collect::<Vec<_>>().join(","))
        .expect("write benchmark metadata");
    writeln!(output, "# trials={}", config.trials).expect("write benchmark metadata");
    writeln!(output, "# warmups_per_lane={}", config.warmups).expect("write benchmark metadata");
    writeln!(output, "# target_measured_bytes={}", config.target_measured_bytes).expect("write benchmark metadata");
    writeln!(
        output,
        "path\ttype_case\ttarget_size_bytes\trows\tpayload_bytes_per_op\tfinal_host_memory\tsynchronization\tcopy_info\tlanes\ttrial\twarmups_per_lane\titerations_per_lane\tcompleted\twall_ms\tpayload_gib_s\thost_output_gib_s\tops_s\tlatency_p50_us\tlatency_p95_us\tlatency_p99_us\tchecksum"
    )
    .expect("write benchmark TSV header");

    for case in &config.cases {
        for target_size in &config.sizes {
            let rows = case.rows_for_target_bytes(*target_size);
            let payload_bytes = case.payload_bytes(rows);
            for lanes in &config.concurrencies {
                let iterations = iterations_per_lane(payload_bytes, *lanes, config.target_measured_bytes);
                for trial in 0..config.trials {
                    let mut paths = if case.direct_copy_supported() {
                        vec![
                            Path::RawPageable,
                            Path::RawPinned,
                            Path::PageableStagedPageable,
                            Path::PinnedStagedPageable,
                            Path::ToRecordBatch,
                        ]
                    } else {
                        vec![Path::ToRecordBatch]
                    };
                    if trial % 2 == 1 {
                        paths.reverse();
                    }
                    for path in paths {
                        let spec = TrialSpec {
                            case: *case,
                            target_size: *target_size,
                            rows,
                            payload_bytes,
                            path,
                            lanes: *lanes,
                            trial,
                            warmups: config.warmups,
                            iterations,
                        };
                        let result = run_trial(
                            device,
                            spec.case,
                            spec.rows,
                            spec.path,
                            spec.lanes,
                            spec.warmups,
                            spec.iterations,
                        );
                        print_result(output.as_mut(), &spec, result);
                    }
                }
            }
        }
    }
    output.flush().expect("flush benchmark output");
}
