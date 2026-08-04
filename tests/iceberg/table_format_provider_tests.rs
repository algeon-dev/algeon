use std::{
    collections::HashMap,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "cuvs")]
use arrow::array::{Array, Float32Array};
use arrow::{
    array::{
        ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, ListArray, MapArray, StringArray, StructArray,
    },
    buffer::OffsetBuffer,
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::{
    catalog::Session,
    common::{DataFusionError, NullEquality, ScalarValue, project_schema},
    datasource::{MemTable, TableProvider},
    execution::{SessionStateBuilder, context::TaskContext},
    logical_expr::{Expr, JoinType, Operator, TableProviderFilterPushDown, TableType},
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        coalesce_partitions::CoalescePartitionsExec,
        collect,
        empty::EmptyExec,
        filter::FilterExec,
        joins::{HashJoinExec, PartitionMode},
        limit::GlobalLimitExec,
    },
    prelude::{SessionConfig, SessionContext},
};
#[cfg(feature = "cuvs")]
use datafusion_nexus::backend::{GpuMemoryOwnership, NexusGpuBackend};
use datafusion_nexus::{
    backend::{GpuDeviceMemoryLimit, NexusGpuDeviceProfile},
    capability::CapabilityReason,
    error::{DataFusionErrorExt, Error, ErrorCode, ErrorStatus},
    native::lower_physical_plan_to_native,
    native_exec::NexusNativeExecutionConfig,
    planner::NexusNativeOptimizerConfig,
    report::CandidateOutcome,
    session::NexusSessionStateBuilderExt,
    source_cardinality::schema_with_source_unique_keys,
    table_format::{
        NexusIcebergNativeScanOutcome, NexusIcebergScanExec, TableFormat, TableFormatColumnMapping,
        TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat,
        TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatLakehouseCachePolicy,
        TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
        TableFormatPartitionValue, TableFormatPlanner, TableFormatReadCapability, TableFormatScalarValue,
        TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
        iceberg::{NexusIcebergTableProvider, ResolvedIcebergTable, SourceAccessPolicy},
        resolve_table_format_parquet_source, resolve_table_format_parquet_source_with_metadata,
    },
};
use futures::future::BoxFuture;
use iceberg::{
    TableIdent,
    io::FileIO,
    spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestListWriter, ManifestWriterBuilder, Struct,
        TableMetadata,
    },
    table::Table,
};
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
