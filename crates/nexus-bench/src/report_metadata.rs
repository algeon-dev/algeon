use datafusion_nexus::native_report::WorkloadDatasetMetadata;
use nexus_workloads::{
    clickbench::ClickbenchDatasetMetadata, metadata::WorkloadDatasetMetadata as NeutralWorkloadDatasetMetadata,
    tpcds::TpcdsDatasetMetadata, tpch::TpchDatasetMetadata,
};

/// Bench/report-layer conversion from neutral workload metadata into the
/// adapter's stable report header shape.
pub trait ReportDatasetMetadataExt {
    fn to_workload_dataset_metadata(&self) -> WorkloadDatasetMetadata;
}

impl ReportDatasetMetadataExt for TpchDatasetMetadata {
    fn to_workload_dataset_metadata(&self) -> WorkloadDatasetMetadata {
        report_metadata(self.to_neutral_metadata())
    }
}

impl ReportDatasetMetadataExt for TpcdsDatasetMetadata {
    fn to_workload_dataset_metadata(&self) -> WorkloadDatasetMetadata {
        report_metadata(self.to_neutral_metadata())
    }
}

impl ReportDatasetMetadataExt for ClickbenchDatasetMetadata {
    fn to_workload_dataset_metadata(&self) -> WorkloadDatasetMetadata {
        report_metadata(self.to_neutral_metadata())
    }
}

fn report_metadata(metadata: NeutralWorkloadDatasetMetadata) -> WorkloadDatasetMetadata {
    WorkloadDatasetMetadata {
        workload: metadata.workload,
        root: metadata.root,
        scale_factor: metadata.scale_factor,
        provenance: metadata.provenance,
        generation_command: metadata.generation_command,
        table_count: metadata.table_count,
        file_count: metadata.file_count,
        total_rows: metadata.total_rows,
        total_bytes: metadata.total_bytes,
        metadata_fingerprint: metadata.metadata_fingerprint,
        content_fingerprint: metadata.content_fingerprint,
        table_file_counts: metadata.table_file_counts,
        table_row_counts: metadata.table_row_counts,
        table_byte_sizes: metadata.table_byte_sizes,
        table_row_group_counts: metadata.table_row_group_counts,
    }
}
