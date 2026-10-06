use std::sync::Arc;

use algeon_datafusion::{
    native_exec::AlgeonNativeExecutionConfig, planner::AlgeonNativeOptimizerConfig,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    datasource::MemTable,
    execution::{SessionState, SessionStateBuilder},
    prelude::SessionContext,
};

pub const KMEANS_OPTIONS: &str = r#"{"n_clusters":1}"#;

pub fn cuvs_context() -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cuvs_sql()
        .expect("install cuVS SQL contracts")
        .build();
    context_with_state(state)
}

pub fn cuvs_context_with_native_execution_config(config: AlgeonNativeExecutionConfig) -> SessionContext {
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(AlgeonNativeOptimizerConfig::default().with_native_execution_config(config))
        .try_with_cuvs_sql()
        .expect("install cuVS SQL contracts with native execution config")
        .build();
    context_with_state(state)
}

fn context_with_state(state: SessionState) -> SessionContext {
    let ctx = SessionContext::new_with_state(state);
    register_table(&ctx, "dataset_vectors", vector_schema("item_id", DataType::Int64, false, 2));
    register_table(&ctx, "canonical_vectors", vector_schema("id", DataType::Int64, false, 2));
    register_table(&ctx, "query_vectors", vector_schema("query_id", DataType::Utf8, false, 2));
    register_table(&ctx, "one_dim_queries", vector_schema("query_id", DataType::Utf8, false, 1));
    register_table(&ctx, "nullable_vectors", vector_schema("item_id", DataType::Int64, true, 2));
    register_table(&ctx, "nullable_list_vectors", nullable_list_schema(DataType::List));
    register_table(&ctx, "nullable_fixed_vectors", nullable_list_schema(|item| DataType::FixedSizeList(item, 2)));
    register_table(&ctx, "bad_id_vectors", vector_schema("item_id", DataType::Float32, false, 2));
    ctx
}

#[cfg(feature = "cugraph")]
pub fn cuvs_cugraph_context() -> SessionContext {
    use algeon_datafusion::cugraph_sql::CugraphSqlConfig;

    let state = SessionStateBuilder::new_with_default_features()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL contracts")
        .try_with_cuvs_sql()
        .expect("install cuGraph and cuVS SQL contracts")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_table(&ctx, "dataset_vectors", vector_schema("item_id", DataType::Int64, false, 2));
    register_table(&ctx, "query_vectors", vector_schema("query_id", DataType::Utf8, false, 2));
    register_table(
        &ctx,
        "edges",
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ])),
    );
    ctx
}

#[cfg(feature = "cugraph")]
pub fn cuvs_cugraph_native_context() -> SessionContext {
    use algeon_datafusion::cugraph_sql::CugraphSqlConfig;

    let native_execution_config = AlgeonNativeExecutionConfig::default();
    let state = SessionStateBuilder::new_with_default_features()
        .with_algeon_native(
            AlgeonNativeOptimizerConfig::default().with_native_execution_config(native_execution_config),
        )
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL contracts")
        .try_with_cuvs_sql()
        .expect("install native, cuGraph, and cuVS SQL contracts")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_table(&ctx, "dataset_vectors", vector_schema("item_id", DataType::Int64, false, 2));
    register_table(&ctx, "query_vectors", vector_schema("query_id", DataType::Utf8, false, 2));
    register_table(
        &ctx,
        "edges",
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ])),
    );
    ctx
}

/// `nullable` declares the ID and every feature field nullable.
fn vector_schema(id_name: &str, id_type: DataType, nullable: bool, dimensions: usize) -> SchemaRef {
    let mut fields = vec![Field::new(id_name, id_type, nullable)];
    fields.extend((0..dimensions).map(|index| Field::new(format!("d{index}"), DataType::Float32, nullable)));
    Arc::new(Schema::new(fields))
}

/// An ID and a vector column that are nullable, with nullable elements, like
/// a list built by `ARRAY_AGG` or `MAKE_ARRAY`.
fn nullable_list_schema(vector: impl FnOnce(Arc<Field>) -> DataType) -> SchemaRef {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(Schema::new(vec![Field::new("item_id", DataType::Int64, true), Field::new("vector", vector(item), true)]))
}

fn register_table(ctx: &SessionContext, name: &str, schema: SchemaRef) {
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    let provider = MemTable::try_new(schema, vec![vec![batch]]).expect("fixture MemTable");
    ctx.register_table(name, Arc::new(provider)).expect("register fixture table");
}
