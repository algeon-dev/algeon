use std::{
    collections::HashMap,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int64Array, StringArray},
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, NullEquality, ScalarValue},
    datasource::TableProvider,
    execution::context::TaskContext,
    logical_expr::{JoinType, Operator},
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{
        ExecutionPlan,
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        joins::{HashJoinExec, PartitionMode},
        limit::GlobalLimitExec,
    },
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    error::{DataFusionErrorExt, ErrorCode, ErrorStatus},
    native::lower_physical_plan_to_native,
    policy::FallbackReason,
    report::CandidateOutcome,
    source_cardinality::schema_with_source_unique_keys,
    table_format::{
        NexusTableFormatScanExec, TableFormat, TableFormatColumnMapping, TableFormatColumnRole,
        TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat, TableFormatDeleteFile,
        TableFormatDeleteFileContent, TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyConfig,
        TableFormatParquetSourceMetadata, TableFormatPartitionTransform, TableFormatPartitionValue, TableFormatPlanner,
        TableFormatReadCapability, TableFormatScalarValue, TableFormatScanPlan, TableFormatScanRequest,
        TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode, iceberg::NexusIcebergTableProvider,
        resolve_table_format_parquet_source, resolve_table_format_parquet_source_with_metadata,
    },
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    exec::{execute_query_plan, output::QueryOutput},
    expr::LiteralValue as NativeLiteralValue,
    plan::PlanNode,
    runtime::ExecutionContext,
    source::{
        CredentialMode, LakehouseCacheStatus, ParquetReadStrategy, ResolvedParquetInput, ResolvedParquetSource,
        SourceColumnRef, SourceNode,
    },
};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};

include!("table_format_provider_tests/support.rs");
include!("table_format_provider_tests/provider.rs");
include!("table_format_provider_tests/count_star.rs");
include!("table_format_provider_tests/partitions.rs");
include!("table_format_provider_tests/filters.rs");
include!("table_format_provider_tests/lowering.rs");
include!("table_format_provider_tests/resolver.rs");
