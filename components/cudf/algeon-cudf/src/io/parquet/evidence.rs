//! Parquet read evidence construction and required remote evidence validation.

use super::{
    options::ParquetMultiTableInput,
    remote::{
        RemoteParquetBackendBuildStatus, RemoteParquetSource, RemoteReadMode, remote_parquet_backend_build_status,
    },
};
use crate::core::error::{Error, ErrorCode, Result};

/// Per-read Parquet source evidence returned by cudf.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParquetReadEvidence {
    pub requested_remote_read_mode: Option<RemoteReadMode>,
    pub input_kind: &'static str,
    pub backend: &'static str,
    pub fallback_reason: &'static str,
    pub remote_bytes: Option<u64>,
    pub remote_range_count: Option<u64>,
    pub device_read_support: &'static str,
    pub backend_build_status: RemoteParquetBackendBuildStatus,
    pub pruning: Option<ParquetPruningEvidence>,
}

/// Row-group pruning evidence returned by libcudf for a Parquet read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParquetPruningEvidence {
    /// Row groups participating in this read after any caller row-group selection.
    pub input_row_groups: u64,
    pub row_groups_after_stats_filter: Option<u64>,
    pub row_groups_after_bloom_filter: Option<u64>,
}

impl ParquetReadEvidence {
    pub(super) fn remote_path_reader(source: &RemoteParquetSource) -> Self {
        Self {
            requested_remote_read_mode: Some(source.read_mode()),
            input_kind: "remote_path",
            backend: "libcudf_path_reader_unverified",
            fallback_reason: "not_reported_by_libcudf",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_reported_by_libcudf",
            backend_build_status: remote_parquet_backend_build_status(),
            pruning: None,
        }
    }

    pub(super) fn local_path_reader() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "local_path",
            backend: "libcudf_path_reader",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    pub(crate) fn local_path_footer_materialized() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "local_path",
            backend: "parquet_footer_materialized",
            fallback_reason: "all_null_columns_elided",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    pub(crate) fn local_path_multi_footer_materialized() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "local_path_multi",
            backend: "parquet_footer_materialized",
            fallback_reason: "all_null_columns_elided",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    pub(super) fn host_buffer() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "host_buffer",
            backend: "host_buffer",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    pub(super) fn with_pruning(mut self, pruning: Option<ParquetPruningEvidence>) -> Self {
        self.pruning = pruning;
        self
    }
}

pub(super) fn empty_remote_read_evidence_raw() -> cudf_sys::cudf_io_remote_parquet_read_evidence_t {
    cudf_sys::cudf_io_remote_parquet_read_evidence_t {
        backend: 0,
        fallback_reason: 1,
        supports_device_read: 0,
        device_read_used: 0,
        remote_bytes: -1,
        remote_range_count: -1,
        host_read_bytes: -1,
        host_read_count: -1,
        device_read_bytes: -1,
        device_read_count: -1,
    }
}

pub(super) fn empty_pruning_evidence_raw() -> cudf_sys::cudf_io_parquet_pruning_evidence_t {
    cudf_sys::cudf_io_parquet_pruning_evidence_t {
        input_row_groups: -1,
        has_row_groups_after_stats_filter: 0,
        row_groups_after_stats_filter: -1,
        has_row_groups_after_bloom_filter: 0,
        row_groups_after_bloom_filter: -1,
    }
}

pub(super) fn pruning_evidence_from_raw(
    raw: cudf_sys::cudf_io_parquet_pruning_evidence_t,
) -> Option<ParquetPruningEvidence> {
    let input_row_groups = u64::try_from(raw.input_row_groups).ok()?;
    Some(ParquetPruningEvidence {
        input_row_groups,
        row_groups_after_stats_filter: if raw.has_row_groups_after_stats_filter != 0 {
            u64::try_from(raw.row_groups_after_stats_filter).ok()
        } else {
            None
        },
        row_groups_after_bloom_filter: if raw.has_row_groups_after_bloom_filter != 0 {
            u64::try_from(raw.row_groups_after_bloom_filter).ok()
        } else {
            None
        },
    })
}

pub(super) fn remote_read_evidence_from_raw(
    source: Option<&RemoteParquetSource>,
    input_kind: &'static str,
    raw: cudf_sys::cudf_io_remote_parquet_read_evidence_t,
) -> ParquetReadEvidence {
    let requested_remote_read_mode = source.map(RemoteParquetSource::read_mode);
    let backend = match raw.backend {
        1 => "cudf_sys_kvikio_remote_datasource",
        _ => "unavailable",
    };
    let fallback_reason = match raw.fallback_reason {
        0 => "none",
        _ => "backend_unavailable",
    };
    let device_read_support = if raw.supports_device_read == 0 {
        "unsupported"
    } else if raw.device_read_used != 0 {
        "supported_used"
    } else {
        "supported_not_used"
    };
    ParquetReadEvidence {
        requested_remote_read_mode,
        input_kind,
        backend,
        fallback_reason,
        remote_bytes: u64::try_from(raw.remote_bytes).ok(),
        remote_range_count: u64::try_from(raw.remote_range_count).ok(),
        device_read_support,
        backend_build_status: remote_parquet_backend_build_status(),
        pruning: None,
    }
}

pub(super) fn validate_required_remote_evidence(
    evidence: &ParquetReadEvidence,
    required_mode: Option<RemoteReadMode>,
    operation: &'static str,
) -> Result<()> {
    let Some(required_mode) = required_mode.filter(|mode| mode.requires_explicit_kvikio_datasource()) else {
        return Ok(());
    };
    let explicit_kvikio_backend = evidence.backend == "cudf_sys_kvikio_remote_datasource";
    let no_fallback = evidence.fallback_reason == "none";
    if required_mode == RemoteReadMode::RequireKvikio {
        if explicit_kvikio_backend && no_fallback {
            return Ok(());
        }
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "RequireKvikio remote Parquet read did not produce required explicit KvikIO datasource evidence",
        )
        .with_detail("resource", "remote_parquet_read_evidence")
        .with_detail("reason", "require_kvikio_evidence_unavailable")
        .with_detail("remote_read_mode", required_mode.as_str())
        .with_detail("input_kind", evidence.input_kind)
        .with_detail("backend", evidence.backend)
        .with_detail("fallback_reason", evidence.fallback_reason)
        .with_detail("remote_backend_build_status", evidence.backend_build_status.as_str()));
    }
    let has_remote_bytes = evidence.remote_bytes.is_some_and(|bytes| bytes > 0);
    let has_remote_ranges = evidence.remote_range_count.is_some_and(|ranges| ranges > 0);
    // Footer and metadata reads are host-facing even on the explicit KvikIO
    // datasource. `supported_not_used` is acceptable only when pruning evidence
    // proves the reader had no row-group payload left to fetch.
    let device_read_capable = matches!(evidence.device_read_support, "supported_used" | "supported_not_used");
    let used_device_read = evidence.device_read_support == "supported_used";
    let payload_device_read_required = remote_payload_device_read_required(evidence);
    let direct_device_read_proven = used_device_read || (device_read_capable && !payload_device_read_required);
    if explicit_kvikio_backend && no_fallback && has_remote_bytes && has_remote_ranges && direct_device_read_proven {
        return Ok(());
    }
    Err(Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "RequireDirect remote Parquet read did not produce required direct remote evidence",
    )
    .with_detail("resource", "remote_parquet_read_evidence")
    .with_detail("reason", "require_direct_evidence_unavailable")
    .with_detail("remote_read_mode", required_mode.as_str())
    .with_detail("input_kind", evidence.input_kind)
    .with_detail("backend", evidence.backend)
    .with_detail("fallback_reason", evidence.fallback_reason)
    .with_detail("remote_bytes_positive", has_remote_bytes.to_string())
    .with_detail("remote_range_count_positive", has_remote_ranges.to_string())
    .with_detail("device_read_support", evidence.device_read_support)
    .with_detail("device_read_capable", device_read_capable.to_string())
    .with_detail("device_read_used", used_device_read.to_string())
    .with_detail("payload_device_read_required", payload_device_read_required.to_string())
    .with_detail("remote_backend_build_status", evidence.backend_build_status.as_str()))
}

fn remote_payload_device_read_required(evidence: &ParquetReadEvidence) -> bool {
    let Some(pruning) = evidence.pruning else {
        return true;
    };
    if pruning.input_row_groups == 0 {
        return false;
    }
    let remaining_row_groups = pruning
        .row_groups_after_bloom_filter
        .or(pruning.row_groups_after_stats_filter)
        .unwrap_or(pruning.input_row_groups);
    remaining_row_groups > 0
}

pub(super) fn multi_table_read_evidence(inputs: &[ParquetMultiTableInput]) -> Option<ParquetReadEvidence> {
    let remote_sources = inputs.iter().filter_map(ParquetMultiTableInput::remote_source).collect::<Vec<_>>();
    if remote_sources.is_empty() {
        return Some(ParquetReadEvidence {
            requested_remote_read_mode: None,
            input_kind: "local_path_multi",
            backend: "libcudf_path_reader",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        });
    }
    let requested_remote_read_mode = remote_sources
        .first()
        .map(|source| source.read_mode())
        .filter(|first_mode| remote_sources.iter().all(|source| source.read_mode() == *first_mode));
    Some(ParquetReadEvidence {
        requested_remote_read_mode,
        input_kind: if remote_sources.len() == inputs.len() { "remote_path_multi" } else { "mixed_path_multi" },
        backend: "libcudf_path_reader_unverified",
        fallback_reason: "not_reported_by_libcudf",
        remote_bytes: None,
        remote_range_count: None,
        device_read_support: "not_reported_by_libcudf",
        backend_build_status: remote_parquet_backend_build_status(),
        pruning: None,
    })
}
