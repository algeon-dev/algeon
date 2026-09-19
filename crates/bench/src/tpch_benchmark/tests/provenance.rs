use super::*;

#[test]
fn run_start_line_renders_unobservable_facts_as_unknown_not_defaults() {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q01".to_string(), "q06".to_string()],
        warmup_iterations: 1,
        iterations: 3,
        native_memory_policy: NativeTpchMemoryPolicy::bounded_custom(
            NativeTpchDeviceBudget::Explicit { bytes: 15_032_385_536 },
            536_870_912,
            1,
            DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            query_engine::runtime::SourceChunkPlanningMode::Conservative,
        ),
        dataset_mode: NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf100),
        ..NativeTpchBenchmarkConfig::default()
    };

    let line = run_start_line(&config, &RunProvenance::default());

    assert!(line.starts_with("[native_tpch_benchmark] stage=run_start "));
    for key in [
        "revision",
        "worktree_dirty",
        "cudf_source_revision",
        "cudf_source_dirty",
        "cudf_install_lib_mtime_unix",
        "driver_version",
        "cpu_model",
        "cpu_logical",
        "fd_soft_limit",
        "gpu_count",
    ] {
        assert!(line.contains(&format!(" {key}=unknown")), "{key} must render as unknown: {line}");
    }
    assert!(line.contains(" kvikio_compat_mode=unset "));
    assert!(line.contains(" dataset_mode=sf100_external query_count=2 warmups=1 iterations=3 "));
    assert!(line.contains(
        " native_profile=custom profile_source=direct_config device_budget=15032385536 query_cap_bytes=15032385536 "
    ));
    assert!(line.contains(" max_source_chunk_bytes=536870912 max_row_groups_per_chunk=1 "));
    assert!(line.contains(" source_chunk_planning_mode=conservative max_source_reads_in_flight=2"));
    assert!(!line.contains("gpu0_"));
}

#[test]
fn run_start_line_carries_device_identity_and_capacity_per_ordinal() {
    let provenance = RunProvenance {
        revision: Some("9a3dc99ab13b".to_string()),
        worktree_dirty: Some(true),
        fd_soft_limit: Some(1024),
        kvikio_compat_mode: Some("ON".to_string()),
        cpu_model: Some("Intel(R) Xeon(R) Platinum 8559C".to_string()),
        devices: vec![DeviceProvenance {
            ordinal: 0,
            model_name: Some("NVIDIA RTX PRO 6000 Blackwell Workstation Edition".to_string()),
            uuid: None,
            physical_total_bytes: Some(102_636_388_352),
            cuda_total_bytes: 101_973_819_392,
            managed_capacity_bytes: 15_032_385_536,
            default_query_limit_bytes: 15_032_385_536,
        }],
        ..RunProvenance::default()
    };

    let line = run_start_line(&NativeTpchBenchmarkConfig::default(), &provenance);

    assert!(line.contains(
        " native_profile=conservative profile_source=default device_budget=automatic query_cap_bytes=15032385536 "
    ));
    assert!(line.contains(" revision=9a3dc99ab13b worktree_dirty=true "));
    assert!(line.contains(" cpu_model=Intel(R)_Xeon(R)_Platinum_8559C "));
    assert!(line.contains(" fd_soft_limit=1024 kvikio_compat_mode=ON "));
    assert!(
        line.contains(" gpu_count=1 gpu0_model=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition gpu0_uuid=unknown ")
    );
    assert!(line.contains(" gpu0_physical_total_bytes=102636388352 gpu0_cuda_total_bytes=101973819392 "));
    assert!(line.ends_with(" gpu0_managed_capacity_bytes=15032385536"));
    assert_eq!(line.split_whitespace().filter(|token| !token.contains('=')).count(), 1, "{line}");
}

#[test]
fn run_start_line_reports_unknown_cap_for_unresolved_automatic_budget() {
    let line = run_start_line(&NativeTpchBenchmarkConfig::default(), &RunProvenance::default());

    assert!(line.contains(" device_budget=automatic query_cap_bytes=unknown "));
}

#[test]
fn proc_parsers_extract_driver_cpu_and_fd_limit() {
    let version = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.178.04  Release Build  (dvs-builder@U22)  Tue Jul  7 12:18:12 UTC 2026\nGCC version:  gcc version 13.3.0\n";
    assert_eq!(parse_driver_version(version.to_string()).as_deref(), Some("580.178.04"));
    assert_eq!(parse_driver_version("GCC version:  gcc version 13.3.0\n".to_string()), None);

    let cpuinfo =
        "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Platinum 8559C\nflags\t: fpu\n";
    assert_eq!(parse_cpu_model(cpuinfo.to_string()).as_deref(), Some("Intel(R) Xeon(R) Platinum 8559C"));
    assert_eq!(parse_cpu_model("processor\t: 0\n".to_string()), None);

    let limits = "Limit                     Soft Limit           Hard Limit           Units     \nMax cpu time              unlimited            unlimited            seconds   \nMax open files            1024                 1048576              files     \n";
    assert_eq!(parse_fd_soft_limit(limits.to_string()), Some(1024));
    assert_eq!(parse_fd_soft_limit("Max cpu time  unlimited  unlimited  seconds\n".to_string()), None);
}
