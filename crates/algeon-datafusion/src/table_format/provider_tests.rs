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

use algeon_datafusion::{
    backend::{AlgeonGpuDeviceProfile, GpuDeviceMemoryLimit},
    capability::CapabilityReason,
    error::{Error, ErrorCode, ErrorStatus},
    native_exec::AlgeonNativeExecutionConfig,
    planner::AlgeonNativeOptimizerConfig,
    report::CandidateOutcome,
    session::AlgeonSessionStateBuilderExt,
    source_cardinality::schema_with_source_unique_keys,
    table_format::{
        AlgeonIcebergNativeScanOutcome, AlgeonIcebergScanExec, TableFormat, TableFormatColumnMapping,
        TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat,
        TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatLakehouseCachePolicy,
        TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
        TableFormatPartitionValue, TableFormatPlanner, TableFormatReadCapability, TableFormatScalarValue,
        TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
        iceberg::{AlgeonIcebergTableProvider, ResolvedTable, SourceAccessPolicy},
        resolve_table_format_parquet_source, resolve_table_format_parquet_source_with_metadata,
    },
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
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use query_engine::{
    expr::LiteralValue as NativeLiteralValue,
    plan::PlanNode,
    source::{
        CredentialMode, LakehouseCacheStatus, ParquetReadStrategy, ResolvedParquetInput, ResolvedParquetSource,
        SourceColumnRef, SourceNode,
    },
};

use crate::native::lower_physical_plan_to_native;

include!("provider_tests/support.rs");
include!("provider_tests/provider.rs");
include!("provider_tests/count_star.rs");
include!("provider_tests/partitions.rs");
include!("provider_tests/filters.rs");
include!("provider_tests/lowering.rs");
include!("provider_tests/resolver.rs");
