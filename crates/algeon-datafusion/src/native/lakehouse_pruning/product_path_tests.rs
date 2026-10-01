use std::{
    fmt,
    fs::{self, File},
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use bytes::Bytes;
use datafusion::{
    execution::{TaskContext, object_store::ObjectStoreUrl, runtime_env::RuntimeEnv},
    object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, ObjectStoreExt,
        PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as ObjectStoreResult, memory::InMemory,
        path::Path as ObjectStorePath,
    },
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::ExecutionPlan,
    prelude::{SessionConfig, SessionContext},
};
use futures::{StreamExt, stream::BoxStream};
use query_engine::{
    admission::AttemptCapabilityEvidence,
    expr::{BinaryOp, Expr, LiteralValue},
    optimizer::local_parquet_source,
    plan::{NodeId, QueryPlan, QueryPlanBuilder},
    runtime::{S3SourceStorageProfile, SourceStorageProfile},
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};
use tokio::{task::yield_now, time::timeout};

use super::prune_native_query_plan;
use crate::{
    attempt::{AlgeonAttempt, attempt_ticket_for_test},
    backend::{AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit, InstalledTestGpuBackend},
    error::DataFusionErrorExt,
    native_exec::{
        AlgeonNativeComposedExec, AlgeonNativeExec, AlgeonNativeExecutionConfig, AlgeonNativeGpuProducerExec,
    },
};

#[derive(Debug)]
struct TrackingObjectStore {
    inner: InMemory,
    get_calls: Arc<AtomicUsize>,
}

impl TrackingObjectStore {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let get_calls = Arc::new(AtomicUsize::new(0));
        (Self { inner: InMemory::new(), get_calls: Arc::clone(&get_calls) }, get_calls)
    }
}

impl fmt::Display for TrackingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TrackingObjectStore")
    }
}

#[async_trait::async_trait]
impl ObjectStore for TrackingObjectStore {
    async fn put_opts(
        &self,
        location: &ObjectStorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> ObjectStoreResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectStorePath,
        opts: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &ObjectStorePath, options: GetOptions) -> ObjectStoreResult<GetResult> {
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &ObjectStorePath, ranges: &[Range<u64>]) -> ObjectStoreResult<Vec<Bytes>> {
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, ObjectStoreResult<ObjectStorePath>>,
    ) -> BoxStream<'static, ObjectStoreResult<ObjectStorePath>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&ObjectStorePath>) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&ObjectStorePath>) -> ObjectStoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectStorePath,
        to: &ObjectStorePath,
        options: CopyOptions,
    ) -> ObjectStoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

struct TestS3Server {
    endpoint: String,
    request_count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl TestS3Server {
    fn start(payload: Vec<u8>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test S3 listener should bind");
        listener.set_nonblocking(true).expect("test S3 listener should become nonblocking");
        let address = listener.local_addr().expect("test S3 listener should have an address");
        let request_count = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_requests = Arc::clone(&request_count);
        let worker_stop = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        worker_requests.fetch_add(1, Ordering::Relaxed);
                        serve_s3_request(&mut stream, &payload);
                    },
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    },
                    Err(_) => break,
                }
            }
        });
        Self { endpoint: format!("http://{address}"), request_count, stop, worker: Some(worker) }
    }
}

impl Drop for TestS3Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().expect("test S3 worker should stop cleanly");
        }
    }
}

fn serve_s3_request(stream: &mut TcpStream, payload: &[u8]) {
    stream.set_read_timeout(Some(Duration::from_secs(2))).expect("test S3 read timeout should be configurable");
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = stream.read(&mut buffer).expect("test S3 request should be readable");
        if count == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..count]);
    }
    let request = String::from_utf8_lossy(&request);
    let method = request.split_whitespace().next().unwrap_or_default();
    let range = request.lines().find_map(|line| {
        let value = line.strip_prefix("Range: bytes=").or_else(|| line.strip_prefix("range: bytes="))?;
        let (start, end) = value.trim().split_once('-')?;
        Some((start.parse::<usize>().ok()?, end.parse::<usize>().ok()?))
    });
    let (status, body, content_range) = match (method, range) {
        ("HEAD", _) => ("200 OK", &[][..], None),
        ("GET", Some((start, end))) if start < payload.len() => {
            let end = end.min(payload.len() - 1);
            ("206 Partial Content", &payload[start..=end], Some((start, end)))
        },
        ("GET", Some(_)) => ("416 Range Not Satisfiable", &[][..], None),
        ("GET", None) => ("200 OK", payload, None),
        _ => ("405 Method Not Allowed", &[][..], None),
    };
    let mut response = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nLast-Modified: Wed, 01 Jan 2025 00:00:00 GMT\r\nETag: \"test-etag\"\r\nAccept-Ranges: bytes\r\nx-amz-request-id: test-request\r\nConnection: close\r\n",
        if method == "HEAD" { payload.len() } else { body.len() }
    );
    if let Some((start, end)) = content_range {
        response.push_str(&format!("Content-Range: bytes {start}-{end}/{}\r\n", payload.len()));
    }
    response.push_str("\r\n");
    stream.write_all(response.as_bytes()).expect("test S3 response headers should be writable");
    if method != "HEAD" {
        stream.write_all(body).expect("test S3 response body should be writable");
    }
}

fn prunable_object_store_source(
    uri: &str,
    file_size_bytes: usize,
    storage_profile_id: Option<&str>,
) -> LocalParquetSource {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut file = LocalParquetFile::object_store_uri(uri)
        .expect("test object-store URI should parse")
        .with_file_size_bytes(file_size_bytes as u64);
    if let Some(profile_id) = storage_profile_id {
        let resolved_source = file.resolved_source.clone().with_storage_profile_id(profile_id.to_string());
        file = file
            .with_resolved_parquet_source(resolved_source)
            .expect("test storage profile should preserve the source location");
    }
    LocalParquetSource::from_files(vec![file], schema, vec!["value".to_string()]).with_reader_predicate(Expr::binary(
        Expr::column("value"),
        BinaryOp::Eq,
        Expr::literal(LiteralValue::Int64(9)),
    ))
}

const UNREACHABLE_ENGINE_PROFILE_ID: &str = "unreachable-engine-profile";

// Engine-side S3 access for these fixtures must stay on loopback. Without a
// profile the engine falls back to the default AWS chain and public endpoint,
// and a host that reaches S3 turns the fixture's footer fetch into a
// `NoSuchBucket` failure before the chunk read these tests assert.
fn unreachable_storage_profile(id: &str) -> SourceStorageProfile {
    SourceStorageProfile::s3(
        id,
        S3SourceStorageProfile::default_chain()
            .with_region("us-east-1")
            .with_endpoint("http://127.0.0.1:9")
            .with_path_style_access(true)
            .with_unsigned_requests(),
    )
}

fn hermetic_backend() -> InstalledTestGpuBackend {
    InstalledTestGpuBackend::with_config(
        AlgeonNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1)
            .with_source_storage_profile(unreachable_storage_profile(UNREACHABLE_ENGINE_PROFILE_ID)),
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 }),
        AlgeonGpuQueuePolicy::default(),
    )
}

fn prunable_query_plan(uri: &str, file_size_bytes: usize, gpu_resident_sink: bool) -> (QueryPlan, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(prunable_object_store_source(
        uri,
        file_size_bytes,
        Some(UNREACHABLE_ENGINE_PROFILE_ID),
    )));
    let plan =
        if gpu_resident_sink { builder.finish_gpu_resident(source) } else { builder.finish_with_host_sink(source) };
    (plan, source)
}

/// A config carrying the ticket of an attempt that the caller keeps alive
/// while the config executes.
fn admission_execution_config(
    backend: &InstalledTestGpuBackend,
    plan: &QueryPlan,
) -> (AlgeonAttempt, AlgeonNativeExecutionConfig) {
    let source = backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority");
    let (attempt, ticket) =
        attempt_ticket_for_test(source.clone(), Some(AttemptCapabilityEvidence::for_query_plan(plan)));
    let config = AlgeonNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_existing_query_admission_ticket(ticket)
        .with_lakehouse_footer_pruning(true);
    (attempt, config)
}

/// [`admission_execution_config`] for a composed root, whose evidence
/// completion extends the attempt's ticket.
fn composed_admission_execution_config(
    backend: &InstalledTestGpuBackend,
) -> (AlgeonAttempt, AlgeonNativeExecutionConfig) {
    let source = backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority");
    let (attempt, ticket) = attempt_ticket_for_test(source.clone(), None);
    let config = AlgeonNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_existing_query_admission_ticket(ticket)
        .with_lakehouse_footer_pruning(true);
    (attempt, config)
}

fn task_context_with_runtime(runtime_env: Arc<RuntimeEnv>) -> Arc<TaskContext> {
    SessionContext::new_with_config_rt(SessionConfig::new(), runtime_env).task_ctx()
}

async fn register_tracking_parquet_store(
    runtime_env: &Arc<RuntimeEnv>,
    store_url: &str,
    bytes: &[u8],
) -> Arc<AtomicUsize> {
    let (store, get_calls) = TrackingObjectStore::new();
    store
        .put(&ObjectStorePath::from("table/data.parquet"), Bytes::copy_from_slice(bytes).into())
        .await
        .expect("test object-store payload should be writable");
    let store_url = ObjectStoreUrl::parse(store_url).expect("test object-store URL should parse");
    runtime_env.register_object_store(store_url.as_ref(), Arc::new(store));
    get_calls
}

#[tokio::test]
async fn product_pruning_uses_runtime_object_store_and_noops_when_store_is_missing() {
    let (dir, bytes) = prunable_parquet_fixture("lakehouse-object-store-pruning");
    let object_path = ObjectStorePath::from("table/data.parquet");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    store.put(&object_path, bytes.clone().into()).await.expect("test object-store payload should be writable");
    let runtime_env = Arc::new(RuntimeEnv::default());
    let store_url = ObjectStoreUrl::parse("s3://lakehouse").expect("test object-store URL should parse");
    runtime_env.register_object_store(store_url.as_ref(), Arc::clone(&store));

    let source = prunable_object_store_source("s3://lakehouse/table/data.parquet", bytes.len(), None);
    let mut builder = QueryPlanBuilder::new();
    let source_node_id = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source_node_id);

    let pruned = prune_native_query_plan(plan.clone(), &runtime_env, &[]).await;
    let pruned_source = local_parquet_source(&pruned, source_node_id).expect("pruned source should remain present");
    assert_eq!(pruned_source.files[0].row_groups.as_deref(), Some(&[1][..]));

    let unchanged = prune_native_query_plan(plan, &Arc::new(RuntimeEnv::default()), &[]).await;
    let unchanged_source =
        local_parquet_source(&unchanged, source_node_id).expect("unpruned source should remain present");
    assert_eq!(unchanged_source.files[0].row_groups, None);
    remove_fixture(dir);
}

#[tokio::test]
async fn product_pruning_prefers_runtime_object_store_over_source_storage_profile() {
    let (dir, bytes) = prunable_parquet_fixture("lakehouse-runtime-store-precedence");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    store
        .put(&ObjectStorePath::from("table/data.parquet"), bytes.clone().into())
        .await
        .expect("test object-store payload should be writable");
    let runtime_env = Arc::new(RuntimeEnv::default());
    let store_url = ObjectStoreUrl::parse("s3://lakehouse").expect("test object-store URL should parse");
    runtime_env.register_object_store(store_url.as_ref(), store);

    let source =
        prunable_object_store_source("s3://lakehouse/table/data.parquet", bytes.len(), Some("unreachable-profile"));
    let mut builder = QueryPlanBuilder::new();
    let source_node_id = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source_node_id);
    let unreachable_profile = unreachable_storage_profile("unreachable-profile");

    let pruned = prune_native_query_plan(plan, &runtime_env, &[unreachable_profile]).await;
    let pruned_source = local_parquet_source(&pruned, source_node_id).expect("pruned source should remain present");
    assert_eq!(pruned_source.files[0].row_groups.as_deref(), Some(&[1][..]));
    remove_fixture(dir);
}

#[tokio::test]
async fn product_pruning_falls_back_to_source_storage_profile() {
    let (dir, bytes) = prunable_parquet_fixture("lakehouse-profile-store-fallback");
    let server = TestS3Server::start(bytes.clone());
    let profile = SourceStorageProfile::s3(
        "local-profile",
        S3SourceStorageProfile::default_chain()
            .with_region("us-east-1")
            .with_endpoint(server.endpoint.clone())
            .with_path_style_access(true)
            .with_unsigned_requests(),
    );
    let source = prunable_object_store_source("s3://lakehouse/table/data.parquet", bytes.len(), Some("local-profile"));
    let mut builder = QueryPlanBuilder::new();
    let source_node_id = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source_node_id);

    let pruned = prune_native_query_plan(plan, &Arc::new(RuntimeEnv::default()), &[profile]).await;
    let pruned_source = local_parquet_source(&pruned, source_node_id).expect("pruned source should remain present");
    assert_eq!(pruned_source.files[0].row_groups.as_deref(), Some(&[1][..]));
    assert!(server.request_count.load(Ordering::Relaxed) > 0, "storage-profile fallback should reach S3");
    remove_fixture(dir);
}

#[tokio::test]
async fn native_host_resolves_lakehouse_metadata_before_admission_context() {
    let backend = hermetic_backend();
    let (dir, bytes) = prunable_parquet_fixture("lakehouse-host-before-admission");
    let runtime_env = Arc::new(RuntimeEnv::default());
    let get_calls = register_tracking_parquet_store(&runtime_env, "s3://host-source", &bytes).await;
    let (plan, _) = prunable_query_plan("s3://host-source/table/data.parquet", bytes.len(), false);
    let (_attempt, config) = admission_execution_config(&backend, &plan);
    let exec = AlgeonNativeExec::try_new(plan.clone(), config).expect("test native host execution should build");

    let mut stream = exec
        .execute(0, task_context_with_runtime(runtime_env))
        .expect("native host execution should return a lazy stream");
    let result = stream.next().await.expect("native host stream should resolve once");

    let error = result.expect_err("the object-store fixture reaches native execution after admission");
    let native = error.find_algeon_error().expect("native execution failure remains structured");
    assert_eq!(
        native.operation(),
        "algeon_datafusion.native_exec.execute",
        "metadata resolution must complete before the real admission context reaches the intentionally unsupported read"
    );
    assert_eq!(native.fact_value("native_operation"), Some("query_engine.local_parquet_chunk.read"));
    assert!(get_calls.load(Ordering::Relaxed) > 0, "footer metadata must resolve before context acquisition");
    remove_fixture(dir);
}

#[tokio::test]
async fn native_host_refines_its_fragment_without_dropping_combined_ticket_evidence() {
    let backend = hermetic_backend();
    let (dir, bytes) = prunable_parquet_fixture("lakehouse-host-combined-ticket");
    let runtime_env = Arc::new(RuntimeEnv::default());
    let get_calls = register_tracking_parquet_store(&runtime_env, "s3://host-combined", &bytes).await;
    let (plan, _) = prunable_query_plan("s3://host-combined/table/data.parquet", bytes.len(), false);
    let (other_plan, _) = prunable_query_plan("s3://other-combined/table/data.parquet", bytes.len(), true);
    let combined_evidence = AttemptCapabilityEvidence::aggregate([
        AttemptCapabilityEvidence::for_query_plan(&plan),
        AttemptCapabilityEvidence::for_query_plan(&other_plan),
    ])
    .expect("test evidence has one manifest");
    let expected_combined_evidence = combined_evidence.clone();
    let source = backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority");
    let (_ticket_attempt, ticket) = attempt_ticket_for_test(source.clone(), Some(combined_evidence));
    let config = AlgeonNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_existing_query_admission_ticket(Arc::clone(&ticket))
        .with_lakehouse_footer_pruning(true);
    let exec = AlgeonNativeExec::try_new(plan, config).expect("test native host execution should build");

    let mut stream = exec
        .execute(0, task_context_with_runtime(Arc::clone(&runtime_env)))
        .expect("native host execution should return a lazy stream");
    let result = stream.next().await.expect("native host stream should resolve once");

    let error = result.expect_err("the fixture reaches the intentionally unsupported native read");
    let native = error.find_algeon_error().expect("native execution failure remains structured");
    assert_eq!(native.operation(), "algeon_datafusion.native_exec.execute");
    assert!(get_calls.load(Ordering::Relaxed) > 0, "footer metadata must resolve before ticket validation");
    let refined = ticket.attempt_capability_evidence_for_rebind().expect("combined evidence remains bound");
    assert!(
        expected_combined_evidence
            .execution_scopes()
            .iter()
            .all(|expected| { refined.execution_scopes().iter().any(|actual| actual == expected) })
    );
    drop(stream);
    drop(exec);
    drop(ticket);
    timeout(Duration::from_secs(5), async {
        while backend.backend().admission_snapshot().unwrap().active_queries != 0 {
            yield_now().await;
        }
    })
    .await
    .expect("combined ticket cleanup must release its grant");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
    remove_fixture(dir);
}

#[tokio::test]
async fn native_host_rejects_an_unrelated_admission_ticket_before_grant() {
    let backend = hermetic_backend();
    let (dir, bytes) = prunable_parquet_fixture("lakehouse-host-unrelated-ticket");
    let runtime_env = Arc::new(RuntimeEnv::default());
    let get_calls = register_tracking_parquet_store(&runtime_env, "s3://host-unrelated", &bytes).await;
    let (plan, _) = prunable_query_plan("s3://host-unrelated/table/data.parquet", bytes.len(), false);
    let (other_plan, _) = prunable_query_plan("s3://other-unrelated/table/data.parquet", bytes.len(), true);
    let unrelated_evidence = AttemptCapabilityEvidence::for_query_plan(&other_plan);
    let expected_unrelated_evidence = unrelated_evidence.clone();
    let source = backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority");
    let (_ticket_attempt, ticket) = attempt_ticket_for_test(source.clone(), Some(unrelated_evidence));
    let config = AlgeonNativeExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_existing_query_admission_ticket(Arc::clone(&ticket))
        .with_lakehouse_footer_pruning(true);
    let exec = AlgeonNativeExec::try_new(plan, config).expect("test native host execution should build");

    let mut stream = exec
        .execute(0, task_context_with_runtime(Arc::clone(&runtime_env)))
        .expect("native host execution should return a lazy stream");
    let result = stream.next().await.expect("native host stream should resolve once");

    let error = result.expect_err("an unrelated ticket cannot acquire this host fragment");
    let native = error.find_algeon_error().expect("ticket mismatch remains structured");
    assert_eq!(native.operation(), "algeon_datafusion.native_exec.admission_evidence");
    assert_eq!(native.fact_value("reason"), Some("native_host_admission_evidence_scope_mismatch"));
    assert_eq!(native.fact_value("ticket_evidence_scope"), Some("different_native_plan"));
    assert!(get_calls.load(Ordering::Relaxed) > 0, "footer metadata must resolve before ticket validation");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(ticket.attempt_capability_evidence_for_rebind(), Some(expected_unrelated_evidence));
    remove_fixture(dir);
}

#[tokio::test]
async fn native_composed_resolves_consumer_and_producer_metadata_before_admission_context() {
    let backend = hermetic_backend();
    let (dir, bytes) = prunable_parquet_fixture("lakehouse-composed-before-admission");
    let runtime_env = Arc::new(RuntimeEnv::default());
    let consumer_gets = register_tracking_parquet_store(&runtime_env, "s3://consumer-source", &bytes).await;
    let producer_gets = register_tracking_parquet_store(&runtime_env, "s3://producer-source", &bytes).await;

    let (producer_plan, _) = prunable_query_plan("s3://producer-source/table/data.parquet", bytes.len(), true);
    let producer = AlgeonNativeGpuProducerExec::try_new_with_metrics_capture_and_shape(
        "producer",
        producer_plan,
        None,
        None,
        true,
        Vec::new(),
    )
    .expect("test native producer should build");

    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let consumer_source = builder.add_source(SourceNode::LocalParquet(prunable_object_store_source(
        "s3://consumer-source/table/data.parquet",
        bytes.len(),
        Some(UNREACHABLE_ENGINE_PROFILE_ID),
    )));
    let producer_source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("producer", Arc::clone(&schema))));
    let union = builder.add_union(vec![consumer_source, producer_source], Arc::clone(&schema));
    let consumer_plan = builder.finish_with_host_sink(union);
    let (_attempt, config) = composed_admission_execution_config(&backend);
    let exec = AlgeonNativeComposedExec::try_new(consumer_plan, vec![Arc::new(producer)], config)
        .expect("test native composed execution should build");

    let mut stream = exec
        .execute(0, task_context_with_runtime(runtime_env))
        .expect("native composed execution should return a lazy stream");
    let result = stream.next().await.expect("native composed stream should resolve once");

    let error = result.expect_err("the object-store fixture reaches native execution after admission");
    let native = error.find_algeon_error().expect("native execution failure remains structured");
    assert_eq!(
        native.operation(),
        "algeon_datafusion.native_gpu_producer.execute_gpu",
        "consumer and producer metadata must resolve before their real admission contexts reach the intentionally unsupported read"
    );
    assert_eq!(native.fact_value("native_operation"), Some("query_engine.local_parquet_chunk.read"));
    assert!(consumer_gets.load(Ordering::Relaxed) > 0, "consumer footer metadata must resolve before admission");
    assert!(producer_gets.load(Ordering::Relaxed) > 0, "producer footer metadata must resolve before admission");
    remove_fixture(dir);
}

fn prunable_parquet_fixture(prefix: &str) -> (PathBuf, Vec<u8>) {
    let dir = unique_test_dir(prefix);
    fs::create_dir_all(&dir).expect("temporary lakehouse pruning directory should be created");
    let path = dir.join("data.parquet");
    write_prunable_i64_parquet(&path);
    let bytes = fs::read(path).expect("test parquet payload should be readable");
    (dir, bytes)
}

fn write_prunable_i64_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let values = Arc::new(Int64Array::from(vec![1, 9])) as ArrayRef;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values]).expect("test batch should be valid");
    let file = File::create(path).expect("test parquet file should be creatable");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("test parquet writer should be valid");
    writer.write(&batch).expect("test parquet batch should be writable");
    writer.close().expect("test parquet writer should close");
}

fn unique_test_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be valid").as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}

fn remove_fixture(dir: PathBuf) {
    fs::remove_dir_all(dir).expect("temporary lakehouse pruning directory should be removable");
}
