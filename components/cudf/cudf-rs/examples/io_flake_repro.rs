//! A/B reproducer for the concurrent-IO cudaErrorInvalidValue flake.
//!
//! Mode A (default): pool runtimes only — device current resource stays the
//! lazily-installed plain cuda resource, so libcudf reader temporaries churn
//! cudaMalloc/cudaFree concurrently with memcpy submissions.
//! Mode B (`--with-manager`): installs the device memory domain first, so
//! temporaries route through the device pool adaptor (production config).
//!
//! Usage: cargo run --example io_flake_repro [--with-manager] [threads] [rounds]

use std::sync::atomic::{AtomicUsize, Ordering};

use cudf::{DataFrame, DeviceId, DeviceMemoryManager, Runtime};

static FAILURES: AtomicUsize = AtomicUsize::new(0);

fn worker(thread_id: usize, rounds: usize) {
    let dir = tempfile::tempdir().expect("tempdir");
    for round in 0..rounds {
        // One runtime per round mirrors the contract-test suites: the pool's
        // slab is cudaFree'd (VA unmapped) while other threads submit copies.
        let rt = Runtime::builder()
            .with_device(DeviceId::new(0).expect("device 0"))
            .with_pool_memory_resource(32 * 1024 * 1024, None)
            .build()
            .expect("runtime");
        let path = dir.path().join(format!("t{thread_id}_r{round}.parquet"));
        let df = DataFrame::from_series([
            ("id", rt.series(&(0..512i32).collect::<Vec<_>>()).expect("series")),
            ("val", rt.series(&(0..512i32).map(|v| v * 3).collect::<Vec<_>>()).expect("series")),
        ])
        .expect("frame");
        if let Err(err) = df.write_parquet(&path) {
            eprintln!("[repro] write failure t{thread_id} r{round}: {err}");
            FAILURES.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        match DataFrame::read_parquet(&rt, &path) {
            Ok(read) => assert_eq!(read.num_rows(), 512),
            Err(err) => {
                eprintln!("[repro] read failure t{thread_id} r{round}: {err}");
                FAILURES.fetch_add(1, Ordering::Relaxed);
            },
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let with_manager = args.iter().any(|a| a == "--with-manager");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let threads: usize = positional.first().and_then(|v| v.parse().ok()).unwrap_or(16);
    let rounds: usize = positional.get(1).and_then(|v| v.parse().ok()).unwrap_or(30);

    if with_manager {
        DeviceMemoryManager::install_for_device(DeviceId::new(0).unwrap()).expect("install domain");
        println!("mode: device memory domain installed (production config)");
    } else {
        println!("mode: pool runtimes only (no device domain)");
    }

    let workers: Vec<_> = (0..threads).map(|t| std::thread::spawn(move || worker(t, rounds))).collect();
    for w in workers {
        w.join().expect("worker");
    }
    println!("threads={threads} rounds={rounds} failures={}", FAILURES.load(Ordering::Relaxed));
    std::process::exit(if FAILURES.load(Ordering::Relaxed) > 0 { 1 } else { 0 });
}
