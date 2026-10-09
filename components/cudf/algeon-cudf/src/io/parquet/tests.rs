//! Parquet reader evidence and remote profile unit tests.

use super::{
    evidence::validate_required_remote_evidence,
    remote::{RemoteProfileFfi, remote_known_size_bytes_raw, remote_parquet_capability_with_status},
    *,
};
use crate::core::error::ErrorCode;

fn direct_evidence() -> ParquetReadEvidence {
    ParquetReadEvidence {
        requested_remote_read_mode: Some(RemoteReadMode::RequireDirect),
        input_kind: "remote_path",
        backend: "cudf_sys_kvikio_remote_datasource",
        fallback_reason: "none",
        remote_bytes: Some(1024),
        remote_range_count: Some(3),
        device_read_support: "supported_used",
        backend_build_status: RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled,
        pruning: None,
    }
}

#[test]
fn require_direct_accepts_kvikio_remote_datasource_evidence() {
    validate_required_remote_evidence(&direct_evidence(), Some(RemoteReadMode::RequireDirect), "test").unwrap();
}

#[test]
fn require_direct_rejects_unverified_remote_path_evidence() {
    let evidence = ParquetReadEvidence {
        requested_remote_read_mode: Some(RemoteReadMode::RequireDirect),
        input_kind: "remote_path",
        backend: "libcudf_path_reader_unverified",
        fallback_reason: "not_reported_by_libcudf",
        remote_bytes: None,
        remote_range_count: None,
        device_read_support: "not_reported_by_libcudf",
        backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
        pruning: None,
    };

    let error = validate_required_remote_evidence(&evidence, Some(RemoteReadMode::RequireDirect), "test").unwrap_err();

    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("require_direct_evidence_unavailable"));
    assert_eq!(error.detail_value("backend"), Some("libcudf_path_reader_unverified"));
}

#[test]
fn require_direct_rejects_host_only_kvikio_remote_datasource_payload_evidence() {
    let mut evidence = direct_evidence();
    evidence.device_read_support = "supported_not_used";

    let error = validate_required_remote_evidence(&evidence, Some(RemoteReadMode::RequireDirect), "test").unwrap_err();

    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("require_direct_evidence_unavailable"));
    assert_eq!(error.detail_value("device_read_capable"), Some("true"));
    assert_eq!(error.detail_value("device_read_used"), Some("false"));
    assert_eq!(error.detail_value("payload_device_read_required"), Some("true"));
}

#[test]
fn require_direct_accepts_pruned_metadata_only_kvikio_remote_datasource_evidence() {
    let mut evidence = direct_evidence();
    evidence.device_read_support = "supported_not_used";
    evidence.pruning = Some(ParquetPruningEvidence {
        input_row_groups: 4,
        row_groups_after_stats_filter: Some(0),
        row_groups_after_bloom_filter: None,
    });

    validate_required_remote_evidence(&evidence, Some(RemoteReadMode::RequireDirect), "test").unwrap();
}

#[test]
fn require_direct_rejects_device_read_unsupported_kvikio_remote_datasource_evidence() {
    let mut evidence = direct_evidence();
    evidence.device_read_support = "unsupported";

    let error = validate_required_remote_evidence(&evidence, Some(RemoteReadMode::RequireDirect), "test").unwrap_err();

    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("require_direct_evidence_unavailable"));
    assert_eq!(error.detail_value("backend"), Some("cudf_sys_kvikio_remote_datasource"));
    assert_eq!(error.detail_value("device_read_capable"), Some("false"));
    assert_eq!(error.detail_value("device_read_used"), Some("false"));
}

#[test]
fn require_kvikio_accepts_compatibility_io_from_explicit_remote_datasource() {
    for device_read_support in ["supported_not_used", "unsupported"] {
        let mut evidence = direct_evidence();
        evidence.requested_remote_read_mode = Some(RemoteReadMode::RequireKvikio);
        evidence.device_read_support = device_read_support;

        validate_required_remote_evidence(&evidence, Some(RemoteReadMode::RequireKvikio), "test")
            .expect("KvikIO compatibility IO is internal to the explicit datasource");
    }
}

#[test]
fn require_kvikio_rejects_unverified_path_reader_and_backend_fallback() {
    let mut path_reader = direct_evidence();
    path_reader.requested_remote_read_mode = Some(RemoteReadMode::RequireKvikio);
    path_reader.backend = "libcudf_path_reader_unverified";
    path_reader.fallback_reason = "not_reported_by_libcudf";

    let error =
        validate_required_remote_evidence(&path_reader, Some(RemoteReadMode::RequireKvikio), "test").unwrap_err();
    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("require_kvikio_evidence_unavailable"));
    assert_eq!(error.detail_value("backend"), Some("libcudf_path_reader_unverified"));

    let mut backend_fallback = direct_evidence();
    backend_fallback.requested_remote_read_mode = Some(RemoteReadMode::RequireKvikio);
    backend_fallback.fallback_reason = "backend_unavailable";
    let error =
        validate_required_remote_evidence(&backend_fallback, Some(RemoteReadMode::RequireKvikio), "test").unwrap_err();
    assert_eq!(error.detail_value("reason"), Some("require_kvikio_evidence_unavailable"));
    assert_eq!(error.detail_value("fallback_reason"), Some("backend_unavailable"));
}

#[test]
fn require_kvikio_is_fail_closed_when_backend_is_unavailable() {
    let source = RemoteParquetSource::s3("s3://bucket/table/part-0.parquet")
        .unwrap()
        .with_read_mode(RemoteReadMode::RequireKvikio);

    assert!(!source.can_use_path_reader_fallback());
    assert!(source.should_use_explicit_remote_datasource());

    let disabled =
        remote_parquet_capability_with_status(&source, RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled);
    assert_eq!(disabled.disposition, RemoteParquetCapabilityDisposition::BackendBuildDisabled);
    assert_eq!(disabled.read_mode, RemoteReadMode::RequireKvikio);

    let unknown = remote_parquet_capability_with_status(&source, RemoteParquetBackendBuildStatus::Unknown);
    assert_eq!(unknown.disposition, RemoteParquetCapabilityDisposition::KvikioEvidenceUnavailable);
    assert_eq!(RemoteReadMode::RequireKvikio.as_str(), "require_kvikio");
}

#[test]
fn remote_profile_raw_carries_known_size_bytes_hint() {
    let source = RemoteParquetSource::s3("s3://bucket/table/part-0.parquet").unwrap().with_known_size_bytes(4096);
    let profile_ffi = RemoteProfileFfi::new(source.profile(), "test").unwrap();

    let raw = profile_ffi.as_raw(source.profile(), source.known_size_bytes());

    assert_eq!(source.known_size_bytes(), Some(4096));
    assert_eq!(raw.known_size_bytes, 4096);
    assert_eq!(remote_known_size_bytes_raw(None), -1);
    assert_eq!(remote_known_size_bytes_raw(Some(i64::MAX as u64 + 1)), -1);
}
