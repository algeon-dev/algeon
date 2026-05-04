use cudf::{
    plan_local_parquet_source_chunks, ByteRangeDisposition, LocalParquetFile, LocalParquetSource,
    LocalParquetSourceKind, PartitionColumn, PartitionColumnDisposition, PartitionValue, PredicateDisposition,
    RejectedSourceFeature, SchemaMappingDisposition, SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy,
    SourceColumn, SourceColumnRef, SourceColumnRole, SourceDiagnostics, SourceFormat, SourceOutputColumn,
    SourceReadMode, SourceRequiredColumn, SourceSchema, ZeroColumnCountDisposition, ZERO_COLUMN_COUNT_COLUMN,
};

fn main() {
    let _ = plan_local_parquet_source_chunks;
    let _ = ZERO_COLUMN_COUNT_COLUMN;

    let _ = std::mem::size_of::<ByteRangeDisposition>();
    let _ = std::mem::size_of::<LocalParquetFile>();
    let _ = std::mem::size_of::<LocalParquetSource>();
    let _ = std::mem::size_of::<LocalParquetSourceKind>();
    let _ = std::mem::size_of::<PartitionColumn>();
    let _ = std::mem::size_of::<PartitionColumnDisposition>();
    let _ = std::mem::size_of::<PartitionValue>();
    let _ = std::mem::size_of::<PredicateDisposition>();
    let _ = std::mem::size_of::<RejectedSourceFeature>();
    let _ = std::mem::size_of::<SchemaMappingDisposition>();
    let _ = std::mem::size_of::<SourceChunk>();
    let _ = std::mem::size_of::<SourceChunkPlan>();
    let _ = std::mem::size_of::<SourceChunkPlanningPolicy>();
    let _ = std::mem::size_of::<SourceColumn>();
    let _ = std::mem::size_of::<SourceColumnRef>();
    let _ = std::mem::size_of::<SourceColumnRole>();
    let _ = std::mem::size_of::<SourceDiagnostics>();
    let _ = std::mem::size_of::<SourceFormat>();
    let _ = std::mem::size_of::<SourceOutputColumn>();
    let _ = std::mem::size_of::<SourceReadMode>();
    let _ = std::mem::size_of::<SourceRequiredColumn>();
    let _ = std::mem::size_of::<SourceSchema>();
    let _ = std::mem::size_of::<ZeroColumnCountDisposition>();
}
