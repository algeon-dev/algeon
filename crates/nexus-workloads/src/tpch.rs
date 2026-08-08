pub mod catalog;
pub mod dataset;
pub mod fixture;
pub mod registration;

pub use catalog::{TPCH_QUERIES, TPCH_QUERY_COUNT, TpchQuery, tpch_query_ids};
pub use dataset::{
    DEFAULT_LOCAL_TPCH_SF1_PARQUET_DIR, DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR,
    DEFAULT_LOCAL_TPCH_SF100_OPTIMIZED_PARQUET_DIR, NEXUS_TPCH_CONTENT_FINGERPRINT_ENV,
    NEXUS_TPCH_GENERATION_COMMAND_ENV, NEXUS_TPCH_SCALE_FACTOR_ENV, TPCH_TABLES, TpchDatasetMetadata, TpchDefaultScale,
    TpchTableFileMetadata, TpchTableLayout, TpchTableMetadata, default_tpch_dataset,
    default_tpch_dataset_with_fingerprint, default_tpch_parquet_root, describe_tpch_dataset,
    describe_tpch_dataset_with_content_fingerprint, discover_tpch_table_layout, external_tpch_dataset_metadata,
    prepare_tpch_report_dataset, tiny_tpch_dataset_metadata,
};
pub use fixture::write_tiny_tpch_parquet_dataset;
pub use registration::{register_tpch_table_layout, register_tpch_tables};

pub use crate::NEXUS_PARQUET_DIR_ENV;

pub use crate::error::{TpchWorkloadError, TpchWorkloadResult};
