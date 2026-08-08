pub mod catalog;
pub mod dataset;
pub mod fixture;
pub mod registration;

pub use catalog::{TPCDS_QUERIES, TPCDS_QUERY_COUNT, TpcdsQuery, tpcds_query_ids};
pub use dataset::{
    DEFAULT_LOCAL_TPCDS_SF1_PARQUET_DIR, NEXUS_TPCDS_CONTENT_FINGERPRINT_ENV, NEXUS_TPCDS_GENERATION_COMMAND_ENV,
    NEXUS_TPCDS_SCALE_FACTOR_ENV, TPCDS_TABLES, TpcdsDatasetMetadata, TpcdsTableFileMetadata, TpcdsTableLayout,
    TpcdsTableMetadata, default_tpcds_sf1_parquet_root, describe_tpcds_dataset,
    describe_tpcds_dataset_with_content_fingerprint, discover_tpcds_table_layout, external_tpcds_dataset_metadata,
    prepare_tpcds_report_dataset, tiny_tpcds_dataset_metadata,
};
pub use fixture::write_tiny_tpcds_parquet_dataset;
pub use registration::register_tpcds_tables;

pub use crate::{
    NEXUS_PARQUET_DIR_ENV,
    error::{WorkloadError, WorkloadResult},
};
