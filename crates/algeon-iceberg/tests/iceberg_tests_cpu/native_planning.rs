use std::{collections::HashMap, sync::Arc};

use algeon_datafusion::{
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig, plan_with_native_diagnostics},
    session::AlgeonSessionStateBuilderExt,
};
use datafusion::{
    execution::{SessionStateBuilder, object_store::ObjectStoreUrl},
    object_store::{ObjectStoreExt, PutPayload, memory::InMemory, path::Path as ObjectPath},
    physical_plan::displayable,
    prelude::{SessionConfig, SessionContext},
};

use crate::support::{Warehouse, data_file, people_arrow_schema, people_batch, people_schema, write_parquet_bytes};

fn native_required_session() -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(1);
    let optimizer = AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired);
    let state =
        SessionStateBuilder::new_with_default_features().with_config(config).with_algeon_native(optimizer).build();
    SessionContext::new_with_state(state)
}

/// Native lowering reads the footers of a multi-file remote scan only from
/// the session's file metadata cache, which the provider fills during
/// `scan`; without them `native_required` planning fails. Planning goes
/// through an invocation, as a server statement's attempt does, because
/// only an invocation opens the footer scope.
#[tokio::test]
async fn a_multi_file_s3_table_plans_as_one_native_exec_under_native_required() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("remote", people_schema(), None, HashMap::new()).await;
    let store = Arc::new(InMemory::new());
    let mut entries = Vec::new();
    for (index, ids) in [[1_i64, 2], [101, 102]].iter().enumerate() {
        let bytes = write_parquet_bytes(&people_batch(&people_arrow_schema(Some([1, 2, 3])), ids));
        let size = bytes.len() as u64;
        let key = format!("warehouse/remote/part-{index}.parquet");
        store.put(&ObjectPath::from(key.as_str()), PutPayload::from(bytes)).await.unwrap();
        entries.push(data_file(&format!("s3://bucket/{key}"), size, 2).build().unwrap());
    }
    warehouse.append(&table, entries).await;
    let ctx = native_required_session();
    ctx.register_object_store(ObjectStoreUrl::parse("s3://bucket").unwrap().as_ref(), store);
    warehouse.register(&ctx).await;

    let sql = "SELECT name, count(*) FROM iceberg.ns.remote WHERE id > 1 GROUP BY name";
    let (state, logical) = ctx.sql(sql).await.unwrap().into_parts();
    let (plan, _) = plan_with_native_diagnostics(&state, &logical).await.unwrap();
    let rendered = displayable(plan.as_ref()).indent(false).to_string();
    assert!(rendered.starts_with("AlgeonNativeExec"), "{rendered}");
    assert_eq!(rendered.lines().count(), 1, "{rendered}");
}
