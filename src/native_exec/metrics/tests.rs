#[cfg(feature = "server")]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::{collections::BTreeMap, time::Duration};

#[cfg(feature = "server")]
use arrow::{
    array::Int32Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
#[cfg(feature = "server")]
use datafusion::common::DataFusionError;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricCategory};
#[cfg(feature = "server")]
use nexus_query_engine::{
    error::QueryErrorCode,
    exec::execute_query_plan_observed,
    observability::SealedRuntimeObservationSnapshot,
    plan::QueryPlanBuilder,
    runtime::{AdmittedQueryContext, ExecutionCancellation, ExecutionContext, QueryHandle},
    source::{GpuResidentSource, LiteralBatchSource, SourceNode},
};
use nexus_query_engine::{
    metrics::{
        JoinOperatorMetrics, OperatorMetrics, QueryMetrics, SourceReadParquetEvidenceMetrics, SourceReadRuntimeMetrics,
    },
    observability::RuntimeObservationConfig,
    runtime::LakehouseCacheFillEventSnapshot,
};

#[cfg(feature = "server")]
use super::NativeMetricsCaptureSink;
use super::{
    NATIVE_NEXUS_DATAFUSION_METRIC_NAMES, NativeMetricsCapture, NexusNativeMetrics,
    contract::{NATIVE_NEXUS_METRIC_DESCRIPTORS, NativeMetricKind},
    record::{NATIVE_NEXUS_RECORDING_DESCRIPTORS, NativeMetricRecordingKind},
};
#[cfg(feature = "server")]
use crate::backend::{GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile};
#[cfg(feature = "server")]
use crate::native_exec::{
    NexusNativeComposedExec, NexusNativeExec, NexusNativeExecutionConfig, NexusNativeGpuProducerExec,
};
use crate::{native_exec::admission::QueryAdmissionFacts, native_report::NativeDataFusionExecutionMetrics};

#[cfg(feature = "server")]
struct AdmittedTestContext {
    context: ExecutionContext,
    cancellation: ExecutionCancellation,
    _admission_context: AdmittedQueryContext,
    _handle: QueryHandle,
    _backend: NexusGpuBackend,
}

#[cfg(feature = "server")]
fn admitted_test_context(query_id: &'static str) -> AdmittedTestContext {
    let backend = NexusGpuBackend::builder()
        .device_profiles([NexusGpuDeviceProfile::new(0)])
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .build()
        .expect("test GPU backend builds");
    let handle = backend.admit_query().expect("test query admits");
    let cancellation = ExecutionCancellation::new();
    let admission_context =
        handle.acquire_context_with_cancellation(cancellation.clone()).expect("admitted test context acquires");
    let context = admission_context.context().with_query_id(query_id).expect("test query id is valid");
    AdmittedTestContext {
        context,
        cancellation,
        _admission_context: admission_context,
        _handle: handle,
        _backend: backend,
    }
}

#[test]
fn native_summary_metrics_set_datafusion_v54_categories() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let _native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let metrics = metrics_set.clone_inner();

    let category_for = |name: &str| {
        metrics.iter().find(|metric| metric.value().name() == name).and_then(|metric| metric.metric_category())
    };

    assert_eq!(category_for("reservation_live_peak_bytes"), Some(MetricCategory::Bytes));
    assert_eq!(category_for("query_grant_bytes"), Some(MetricCategory::Bytes));
    assert_eq!(category_for("resident_dataframe_peak_bytes_observed"), Some(MetricCategory::Rows));
    assert_eq!(category_for("source_chunks_committed"), Some(MetricCategory::Rows));
    assert_eq!(category_for("admission_wait_time"), Some(MetricCategory::Timing));
    assert_eq!(category_for("source_read_time"), Some(MetricCategory::Timing));
    assert_eq!(category_for("output_rows"), Some(MetricCategory::Rows));
    assert_eq!(category_for("output_bytes"), Some(MetricCategory::Bytes));
    assert_eq!(category_for("elapsed_compute"), Some(MetricCategory::Timing));
}

#[test]
fn ordered_descriptors_drive_names_registration_and_legacy_categories() {
    const TIMER_NAMES: &[&str] = &[
        "admission_wait_time",
        "source_metadata_load_time",
        "source_read_queue_wait_time",
        "parquet_read_time",
        "source_read_time",
        "operator_time",
        "chunk_operator_time",
        "final_merge_time",
        "final_sort_time",
        "materialization_time",
    ];

    let descriptor_names = NATIVE_NEXUS_METRIC_DESCRIPTORS.iter().map(|descriptor| descriptor.name).collect::<Vec<_>>();
    assert_eq!(descriptor_names, NATIVE_NEXUS_DATAFUSION_METRIC_NAMES);

    let metrics_set = ExecutionPlanMetricsSet::new();
    let _native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let registered = metrics_set.clone_inner();

    for descriptor in NATIVE_NEXUS_METRIC_DESCRIPTORS {
        let legacy_kind = if TIMER_NAMES.contains(&descriptor.name) {
            NativeMetricKind::Timer
        } else if descriptor.name.ends_with("_bytes_observed") {
            NativeMetricKind::Rows
        } else if descriptor.name.contains("bytes") {
            NativeMetricKind::Bytes
        } else {
            NativeMetricKind::Rows
        };
        assert_eq!(descriptor.kind, legacy_kind, "category drift for {}", descriptor.name);

        let matching = registered.iter().filter(|metric| metric.value().name() == descriptor.name).collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "{} must be registered exactly once", descriptor.name);
        let expected_category = match descriptor.kind {
            NativeMetricKind::Rows => MetricCategory::Rows,
            NativeMetricKind::Bytes => MetricCategory::Bytes,
            NativeMetricKind::Timer => MetricCategory::Timing,
        };
        assert_eq!(matching[0].metric_category(), Some(expected_category), "{} category", descriptor.name);
    }
}

#[test]
fn every_ordered_descriptor_has_a_compile_time_recording_mapping() {
    let recording_names = NATIVE_NEXUS_RECORDING_DESCRIPTORS.iter().map(|(name, _kind)| *name).collect::<Vec<_>>();
    assert_eq!(recording_names, NATIVE_NEXUS_DATAFUSION_METRIC_NAMES);
    assert_eq!(recording_names.len(), 110, "the stable public metric count must not change");

    for expected_kind in
        [NativeMetricRecordingKind::Query, NativeMetricRecordingKind::Admission, NativeMetricRecordingKind::CacheEvents]
    {
        assert!(
            NATIVE_NEXUS_RECORDING_DESCRIPTORS.iter().any(|(_name, kind)| *kind == expected_kind),
            "recording contract must exercise {expected_kind:?} mappings"
        );
    }
}

#[test]
fn labelled_row_recording_targets_do_not_repeat_metric_name_literals() {
    let source = include_str!("record.rs");
    assert!(
        !source.contains(".global_counter(\""),
        "labelled row targets must use descriptor-generated metric-name accessors"
    );
}

#[test]
fn native_metrics_record_query_service_admission_facts() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);

    native_metrics.record_admission_facts(QueryAdmissionFacts {
        admission_sequence: 7,
        admission_wait: Duration::from_millis(3),
        device_ordinal: 2,
        query_grant_bytes: 4096,
    });

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.admission_sequence, Some(7));
    assert_eq!(metrics.admission_wait_time_ns, Some(3_000_000));
    assert_eq!(metrics.device_ordinal, Some(2));
    assert_eq!(metrics.query_grant_bytes, Some(4096));
}

#[cfg(feature = "server")]
struct CountingCaptureSink {
    admission_facts: AtomicUsize,
    query_metrics: AtomicUsize,
    cache_events: AtomicUsize,
}

#[cfg(feature = "server")]
impl Default for CountingCaptureSink {
    fn default() -> Self {
        Self {
            admission_facts: AtomicUsize::new(0),
            query_metrics: AtomicUsize::new(0),
            cache_events: AtomicUsize::new(0),
        }
    }
}

#[cfg(feature = "server")]
impl NativeMetricsCaptureSink for CountingCaptureSink {
    fn record_admission_facts(&self, _query_id: &str, _facts: QueryAdmissionFacts) {
        self.admission_facts.fetch_add(1, Ordering::Relaxed);
    }

    fn record_query_metrics(&self, _query_id: &str, _metrics: &QueryMetrics) {
        self.query_metrics.fetch_add(1, Ordering::Relaxed);
    }

    fn record_lakehouse_cache_fill_events(&self, _query_id: &str, events: &[LakehouseCacheFillEventSnapshot]) {
        self.cache_events.fetch_add(events.len(), Ordering::Relaxed);
    }
}

#[cfg(feature = "server")]
fn test_runtime_snapshot(query_id: &str) -> Arc<SealedRuntimeObservationSnapshot> {
    Arc::new(
        nexus_query_engine::observability::RuntimeObservationSession::new(
            query_id,
            None,
            RuntimeObservationConfig::bounded(4, 4096),
        )
        .seal(nexus_query_engine::observability::RuntimeTerminalOutcome::Succeeded)
        .expect("enabled observation seals"),
    )
}

#[cfg(feature = "server")]
#[test]
fn native_metrics_capture_without_sink_retains_owned_runtime_evidence() {
    let capture = NativeMetricsCapture::without_sink("no-sink").new_execution();
    let snapshot = Arc::new(
        nexus_query_engine::observability::RuntimeObservationSession::new(
            "no-sink",
            None,
            RuntimeObservationConfig::bounded(4, 4096),
        )
        .seal(nexus_query_engine::observability::RuntimeTerminalOutcome::Succeeded)
        .expect("enabled observation seals"),
    );

    capture.record_runtime_observation(Arc::clone(&snapshot));

    let retained = capture.runtime_observations();
    assert_eq!(retained.len(), 1);
    assert!(Arc::ptr_eq(&retained[0], &snapshot));
}

#[cfg(feature = "server")]
#[test]
fn native_metrics_capture_does_not_keep_sink_alive() {
    let sink = Arc::new(CountingCaptureSink::default());
    let erased_sink: Arc<dyn NativeMetricsCaptureSink> = sink.clone();
    let weak_sink = Arc::downgrade(&erased_sink);
    let capture = NativeMetricsCapture::new("q1", erased_sink).new_execution();

    assert_eq!(Arc::strong_count(&sink), 1);
    capture.record_admission_facts(QueryAdmissionFacts {
        admission_sequence: 1,
        admission_wait: Duration::from_millis(1),
        device_ordinal: 0,
        query_grant_bytes: 1024,
    });
    capture.record_query_metrics(&QueryMetrics::default());
    assert_eq!(sink.admission_facts.load(Ordering::Relaxed), 1);
    assert_eq!(sink.query_metrics.load(Ordering::Relaxed), 1);

    drop(sink);
    assert!(weak_sink.upgrade().is_none());
    capture.record_admission_facts(QueryAdmissionFacts {
        admission_sequence: 2,
        admission_wait: Duration::from_millis(1),
        device_ordinal: 0,
        query_grant_bytes: 1024,
    });
    capture.record_query_metrics(&QueryMetrics::default());
    let session = nexus_query_engine::observability::RuntimeObservationSession::new(
        "q1",
        None,
        RuntimeObservationConfig::bounded(4, 4096),
    );
    let snapshot = Arc::new(
        session
            .seal(nexus_query_engine::observability::RuntimeTerminalOutcome::Succeeded)
            .expect("enabled observation seals"),
    );
    capture.record_runtime_observation(Arc::clone(&snapshot));
    capture.record_lakehouse_cache_fill_events(&[]);
    let retained = capture.runtime_observations();
    assert_eq!(retained.len(), 1);
    assert!(Arc::ptr_eq(&retained[0], &snapshot));
}

#[cfg(feature = "server")]
#[test]
fn local_observation_collector_retains_each_epoch_once_without_sink_publication() {
    let wiring = NativeMetricsCapture::without_sink("q1")
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(7, 777));
    let capture = wiring.new_execution();

    let first = capture.new_runtime_observation("q1", None).into_local_owner();
    let second = capture.new_runtime_observation("q1", None).into_local_owner();

    assert!(!first.handle().is_same_session(&second.handle()));
    assert_eq!(first.handle().live_snapshot().unwrap().manifest().limits.max_retained_records, 7);
    assert_eq!(first.handle().live_snapshot().unwrap().manifest().limits.max_retained_bytes, 777);
    let first_snapshot = Arc::new(
        first
            .finish_query(nexus_query_engine::observability::RuntimeTerminalOutcome::Succeeded)
            .expect("first owner seals"),
    );
    let second_snapshot = Arc::new(
        second
            .finish_query(nexus_query_engine::observability::RuntimeTerminalOutcome::Succeeded)
            .expect("second owner seals"),
    );
    capture.record_runtime_observation(Arc::clone(&first_snapshot));
    capture.record_runtime_observation(Arc::clone(&first_snapshot));
    capture.record_runtime_observation(second_snapshot);
    assert_eq!(capture.runtime_observations().len(), 2);
    assert_eq!(capture.retention_stats().recorded_total, 2);
}

#[cfg(feature = "server")]
#[test]
fn coordinated_execution_uses_external_handle_without_local_collector_or_seal_authority() {
    let root = nexus_query_engine::observability::RuntimeObservationOwner::new(
        "coordinated",
        None,
        RuntimeObservationConfig::bounded(7, 777),
    );
    let capture = NativeMetricsCapture::coordinated("coordinated", root.handle()).new_execution();
    let observation = capture.new_runtime_observation("coordinated", None);

    assert!(matches!(observation, super::capture::NativeRuntimeObservation::Coordinated(_)));
    assert_eq!(capture.retention_stats().capacity, 0);
    capture.record_runtime_observation(test_runtime_snapshot("must-not-retain"));
    assert!(capture.runtime_observations().is_empty());
    assert!(root.sealed_snapshot().is_none());
}

#[cfg(feature = "server")]
#[test]
fn standalone_execution_has_no_coordinated_activity_requirement() {
    let capture = NativeMetricsCapture::without_sink("standalone").new_execution();

    assert!(capture.new_observation_activity().is_none());
}

#[cfg(feature = "server")]
#[test]
fn cloned_config_allocates_isolated_drainable_local_collectors_and_drop_releases_evidence() {
    let wiring = NativeMetricsCapture::without_sink("reused-config")
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(7, 777));
    let config = NexusNativeExecutionConfig::bounded_default().with_metrics_capture(wiring);
    assert_eq!(config.runtime_observation_config().limits().max_retained_records, 7);
    assert_eq!(config.runtime_observation_config().limits().max_retained_bytes, 777);
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let make_plan = || {
        let mut builder = QueryPlanBuilder::new();
        let source = builder
            .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch.clone()])));
        builder.finish_with_host_sink(source)
    };

    let first = NexusNativeExec::try_new(make_plan(), config.clone()).expect("first wrapper builds");
    let second = NexusNativeExec::try_new(make_plan(), config.clone()).expect("second wrapper builds");
    let first_owner = first.metrics_capture().expect("default config creates a local collector");
    let second_owner = second.metrics_capture().expect("default config creates a local collector");
    assert_eq!(first_owner.runtime_observation_config().limits().max_retained_records, 7);
    assert_eq!(first_owner.runtime_observation_config().limits().max_retained_bytes, 777);
    assert!(!first_owner.shares_owner_with(&second_owner));

    let first_snapshot = test_runtime_snapshot("first-owner");
    let second_snapshot = test_runtime_snapshot("second-owner");
    first_owner.record_runtime_observation(Arc::clone(&first_snapshot));
    second_owner.record_runtime_observation(Arc::clone(&second_snapshot));
    assert_eq!(first_owner.runtime_observations().len(), 1);
    assert!(Arc::ptr_eq(&first_owner.runtime_observations()[0], &first_snapshot));
    assert_eq!(second_owner.runtime_observations().len(), 1);
    assert!(Arc::ptr_eq(&second_owner.runtime_observations()[0], &second_snapshot));

    let drained = first_owner.drain_runtime_observations();
    assert_eq!(drained.len(), 1);
    assert!(first_owner.runtime_observations().is_empty());
    assert_eq!(first_owner.retention_stats().drained_total, 1);
    drop(drained);

    let weak_second_snapshot = Arc::downgrade(&second_snapshot);
    drop(second_snapshot);
    drop(second_owner);
    assert!(weak_second_snapshot.upgrade().is_some(), "execution wrapper remains the durable owner");
    drop(second);
    assert!(weak_second_snapshot.upgrade().is_none(), "dropping the wrapper releases local evidence");
}

#[cfg(feature = "server")]
#[test]
fn local_collector_overflow_and_zero_capacity_have_explicit_accounting() {
    let wiring = NativeMetricsCapture::without_sink("bounded-owner");
    let owner = wiring.new_execution_with_test_capacity(2);
    let first = test_runtime_snapshot("retention-first");
    let second = test_runtime_snapshot("retention-second");
    let third = test_runtime_snapshot("retention-third");
    owner.record_runtime_observation(Arc::clone(&first));
    owner.record_runtime_observation(Arc::clone(&second));
    owner.record_runtime_observation(Arc::clone(&third));

    let retained = owner.runtime_observations();
    assert_eq!(retained.len(), 2);
    assert!(Arc::ptr_eq(&retained[0], &second), "overflow evicts the oldest snapshot first");
    assert!(Arc::ptr_eq(&retained[1], &third));
    assert_eq!(
        owner.retention_stats(),
        super::capture::NativeRuntimeObservationRetentionStats {
            capacity: 2,
            retained: 2,
            recorded_total: 3,
            dropped_total: 0,
            evicted_total: 1,
            drained_total: 0,
        }
    );
    drop(retained);
    assert_eq!(owner.drain_runtime_observations().len(), 2);
    assert_eq!(owner.retention_stats().retained, 0);
    assert_eq!(owner.retention_stats().drained_total, 2);

    let disabled_retention = wiring.new_execution_with_test_capacity(0);
    disabled_retention.record_runtime_observation(test_runtime_snapshot("retention-dropped"));
    assert_eq!(disabled_retention.retention_stats().recorded_total, 1);
    assert_eq!(disabled_retention.retention_stats().dropped_total, 1);
    assert_eq!(disabled_retention.retention_stats().retained, 0);
}

#[cfg(feature = "server")]
#[test]
fn composed_native_fragments_share_only_the_parent_local_collector_and_capacity() {
    let wiring = NativeMetricsCapture::without_sink("composed-owner")
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(9, 999));
    let config = NexusNativeExecutionConfig::bounded_default().with_metrics_capture(wiring);
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut producer_builder = QueryPlanBuilder::new();
    let producer_source = producer_builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
    let producer_plan = producer_builder.finish_with_host_sink(producer_source).clone_with_gpu_resident_sink();
    let producer = NexusNativeGpuProducerExec::try_new_with_metrics_capture(
        "fragment",
        producer_plan,
        config.metrics_capture_factory(),
    )
    .expect("producer builds");
    let original_producer_owner = producer.metrics_capture().expect("producer has a local collector");

    let mut consumer_builder = QueryPlanBuilder::new();
    let consumer_source =
        consumer_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fragment", Arc::clone(&schema))));
    let consumer_plan = consumer_builder.finish_with_host_sink(consumer_source);
    let composed = NexusNativeComposedExec::try_new(consumer_plan.clone(), vec![Arc::new(producer)], config.clone())
        .expect("composed wrapper builds");
    let composed_owner = composed.metrics_capture().expect("composed wrapper has a local collector");
    let rebound = composed.gpu_inputs()[0]
        .as_ref()
        .downcast_ref::<NexusNativeGpuProducerExec>()
        .expect("native producer remains typed")
        .metrics_capture()
        .expect("rebound producer has a local collector");
    assert!(composed_owner.shares_owner_with(&rebound));
    assert_eq!(composed_owner.runtime_observation_config().limits().max_retained_records, 9);
    assert_eq!(rebound.runtime_observation_config().limits().max_retained_bytes, 999);
    assert!(!composed_owner.shares_owner_with(&original_producer_owner));

    for index in 0..=NativeMetricsCapture::DEFAULT_RUNTIME_OBSERVATION_CAPACITY {
        let target = if index % 2 == 0 { &composed_owner } else { &rebound };
        target.record_runtime_observation(test_runtime_snapshot(&format!("composed-{index}")));
    }
    let stats = composed_owner.retention_stats();
    assert_eq!(stats.recorded_total, 257);
    assert_eq!(stats.retained, NativeMetricsCapture::DEFAULT_RUNTIME_OBSERVATION_CAPACITY);
    assert_eq!(stats.evicted_total, 1);
    assert_eq!(stats.dropped_total, 0);

    let independent = NexusNativeComposedExec::try_new(consumer_plan, composed.gpu_inputs().to_vec(), config)
        .expect("independent composed wrapper builds");
    assert!(!composed_owner.shares_owner_with(&independent.metrics_capture().expect("owner exists")));
}

#[cfg(feature = "server")]
#[test]
fn cancelled_native_outcome_retains_local_evidence_before_error_conversion() {
    let sink = Arc::new(CountingCaptureSink::default());
    let capture = NativeMetricsCapture::new("capture-cancel", sink.clone()).new_execution();
    let capture_owner = capture.clone();
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new_with_capture(&metrics_set, 0, Some(capture));
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut plan = QueryPlanBuilder::new();
    let source = plan.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let plan = plan.finish_with_host_sink(source);
    let admitted = admitted_test_context("capture-cancel");
    admitted.cancellation.cancel();
    let outcome = execute_query_plan_observed(&admitted.context, &plan, RuntimeObservationConfig::native_execution());

    let error = match native_metrics.finalize_query_outcome(outcome) {
        Err(error) => error,
        Ok(_) => panic!("cancelled native result must remain typed"),
    };
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    let _: DataFusionError = crate::native_exec::common::native_query_runtime_error(
        "datafusion_nexus.native_exec.metrics.test",
        "cancelled evidence conversion",
        error,
    )
    .into();
    assert_eq!(sink.query_metrics.load(Ordering::Relaxed), 1);
    let owned = capture_owner.runtime_observations();
    assert_eq!(owned.len(), 1);
    assert_eq!(owned[0].footer().terminal, Some(nexus_query_engine::observability::RuntimeTerminalOutcome::Cancelled));
    assert!(!owned[0].events().is_empty());
}

#[cfg(feature = "server")]
#[test]
fn successful_native_outcome_retains_one_owned_snapshot_in_local_collector() {
    let sink = Arc::new(CountingCaptureSink::default());
    let capture = NativeMetricsCapture::new("capture-success", sink.clone()).new_execution();
    let capture_owner = capture.clone();
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new_with_capture(&metrics_set, 0, Some(capture));
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut plan = QueryPlanBuilder::new();
    let source = plan.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let plan = plan.finish_with_host_sink(source);
    let admitted = admitted_test_context("capture-success");
    let outcome = execute_query_plan_observed(&admitted.context, &plan, RuntimeObservationConfig::native_execution());

    native_metrics.finalize_query_outcome(outcome).expect("native execution succeeds");

    assert_eq!(sink.query_metrics.load(Ordering::Relaxed), 1);
    let owned = capture_owner.runtime_observations();
    assert_eq!(owned.len(), 1);
    assert_eq!(owned[0].footer().terminal, Some(nexus_query_engine::observability::RuntimeTerminalOutcome::Succeeded));
}

#[test]
fn aggregate_strategy_rows_bridge_statistical_contract_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("selected_strategy", "stddev_samp_gpu_native_tolerant".to_string());
    facts.insert("execution_stage", "single".to_string());
    facts.insert("memory_mode", "bounded".to_string());
    facts.insert("group_key_count", "4".to_string());
    facts.insert("group_key_exprs", "w_warehouse_name,w_warehouse_sk,i_item_sk,d_moy".to_string());
    facts.insert("group_key_dtypes", "Utf8View,Int64,Int64,Int64".to_string());
    facts.insert("aggregate_functions", "stddev_samp,avg".to_string());
    facts.insert("floating_statistical_aggregate", "true".to_string());
    facts.insert("statistical_aggregate_execution_mode", "gpu_native_tolerant".to_string());
    facts.insert("statistical_aggregate_tolerance_contract", "float_tolerant".to_string());
    facts.insert("statistical_aggregate_functions", "stddev_samp".to_string());
    facts.insert("statistical_aggregate_ddof", "stddev_samp:1".to_string());
    facts.insert("stddev_samp_execution_mode", "gpu_native_tolerant".to_string());
    facts.insert("stddev_samp_ddof", "1".to_string());

    query_metrics.record_aggregate_strategy_decision("n12", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.node_id, "n12");
    assert_eq!(row.selected_strategy, "stddev_samp_gpu_native_tolerant");
    assert_eq!(row.floating_statistical_aggregate, "true");
    assert_eq!(row.statistical_aggregate_execution_mode, "gpu_native_tolerant");
    assert_eq!(row.statistical_aggregate_tolerance_contract, "float_tolerant");
    assert_eq!(row.statistical_aggregate_functions, "stddev_samp");
    assert_eq!(row.statistical_aggregate_ddof, "stddev_samp:1");
    assert_eq!(row.stddev_samp_execution_mode, "gpu_native_tolerant");
    assert_eq!(row.stddev_samp_ddof, "1");
}

#[test]
fn aggregate_strategy_rows_bridge_strategy_execution_option_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let facts = BTreeMap::from([
        ("selected_strategy", "libcudf_groupby".to_string()),
        ("execution_stage", "partial".to_string()),
        ("aggregate_strategy_options_version", "1".to_string()),
        ("aggregate_encoded_key_partial_low_cardinality_threshold_ppm", "175000".to_string()),
        ("aggregate_encoded_key_partial_min_input_rows", "123".to_string()),
        ("aggregate_exact_grouped_key_cardinality_probe_max_key_columns", "4".to_string()),
        ("aggregate_exact_single_string_avg_cardinality_probe_min_input_rows", "101".to_string()),
        ("aggregate_exact_single_key_cardinality_probe_min_input_rows", "202".to_string()),
        ("aggregate_exact_grouped_key_cardinality_probe_min_input_rows", "303".to_string()),
        ("aggregate_exact_single_string_key_prepared_max_groups", "404".to_string()),
        ("aggregate_exact_three_key_cardinality_probe_max_input_rows", "505".to_string()),
        ("aggregate_exact_five_key_cardinality_probe_max_input_rows", "606".to_string()),
        ("aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm", "200000".to_string()),
        ("aggregate_source_adjacent_prepared_sort_groupby_max_row_groups", "65".to_string()),
        ("aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes", "4096".to_string()),
        ("aggregate_source_adjacent_prepared_sort_groupby_max_fan_in", "7".to_string()),
    ]);

    query_metrics.record_aggregate_strategy_decision("n13", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.node_id, "n13");
    assert_eq!(row.aggregate_strategy_options_version, "1");
    assert_eq!(row.aggregate_encoded_key_partial_low_cardinality_threshold_ppm, 175_000);
    assert_eq!(row.aggregate_encoded_key_partial_min_input_rows, 123);
    assert_eq!(row.aggregate_exact_grouped_key_cardinality_probe_max_key_columns, 4);
    assert_eq!(row.aggregate_exact_single_string_avg_cardinality_probe_min_input_rows, 101);
    assert_eq!(row.aggregate_exact_single_key_cardinality_probe_min_input_rows, 202);
    assert_eq!(row.aggregate_exact_grouped_key_cardinality_probe_min_input_rows, 303);
    assert_eq!(row.aggregate_exact_single_string_key_prepared_max_groups, 404);
    assert_eq!(row.aggregate_exact_three_key_cardinality_probe_max_input_rows, 505);
    assert_eq!(row.aggregate_exact_five_key_cardinality_probe_max_input_rows, 606);
    assert_eq!(row.aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm, 200_000);
    assert_eq!(row.aggregate_source_adjacent_prepared_sort_groupby_max_row_groups, 65);
    assert_eq!(row.aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes, 4096);
    assert_eq!(row.aggregate_source_adjacent_prepared_sort_groupby_max_fan_in, 7);
}

#[test]
fn aggregate_strategy_rows_bridge_filtered_global_materialization_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("selected_strategy", "filtered_global_segmented_state".to_string());
    facts.insert("execution_stage", "single".to_string());
    facts.insert("memory_mode", "bounded".to_string());
    facts.insert("group_key_count", "0".to_string());
    facts.insert("group_key_exprs", "none".to_string());
    facts.insert("group_key_dtypes", "none".to_string());
    facts.insert("aggregate_functions", "count,sum,avg".to_string());
    facts.insert("aggregate_filter_count", "15".to_string());
    facts.insert("filtered_input_materialization_strategy", "row_chunked_segmented_state".to_string());
    facts.insert("filtered_input_materialization_estimated_bytes", "66000000000".to_string());
    facts.insert("filtered_input_materialization_budget_bytes", "85899345920".to_string());
    facts.insert("filtered_input_materialization_threshold_bytes", "42949672960".to_string());
    facts.insert(
        "filtered_input_lower_level_unresolved_contracts",
        "libcudf_dataframe_expression_temporary_workspace_and_transient_relocation_preflight_unavailable".to_string(),
    );
    facts.insert("filtered_global_rows_per_chunk", "1024".to_string());
    facts.insert("filtered_global_inputs_per_segment", "1".to_string());
    facts.insert("filtered_global_segment_target_temporary_bytes", "536870912".to_string());
    facts.insert("filtered_global_actual_chunk_count", "98".to_string());
    facts.insert("filtered_global_actual_segment_count", "196".to_string());
    facts.insert("filtered_global_max_segment_rows", "1024".to_string());
    facts.insert("filtered_global_observed_temporary_peak_bytes", "16777216".to_string());
    facts.insert("filtered_global_observed_retained_peak_bytes", "4096".to_string());
    facts.insert("filtered_global_state_rows_merged", "98".to_string());

    query_metrics.record_aggregate_strategy_decision("n5", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.selected_strategy, "filtered_global_segmented_state");
    assert_eq!(row.aggregate_filter_count, 15);
    assert_eq!(row.filtered_input_materialization_strategy, "row_chunked_segmented_state");
    assert_eq!(row.filtered_input_materialization_estimated_bytes, 66_000_000_000);
    assert_eq!(row.filtered_input_materialization_budget_bytes, 85_899_345_920);
    assert_eq!(row.filtered_input_materialization_threshold_bytes, 42_949_672_960);
    assert_eq!(
        row.filtered_input_lower_level_unresolved_contracts,
        "libcudf_dataframe_expression_temporary_workspace_and_transient_relocation_preflight_unavailable"
    );
    assert_eq!(row.filtered_global_rows_per_chunk, 1024);
    assert_eq!(row.filtered_global_inputs_per_segment, 1);
    assert_eq!(row.filtered_global_segment_target_temporary_bytes, 536_870_912);
    assert_eq!(row.filtered_global_actual_chunk_count, 98);
    assert_eq!(row.filtered_global_actual_segment_count, 196);
    assert_eq!(row.filtered_global_max_segment_rows, 1024);
    assert_eq!(row.filtered_global_observed_temporary_peak_bytes, 16_777_216);
    assert_eq!(row.filtered_global_observed_retained_peak_bytes, 4096);
    assert_eq!(row.filtered_global_state_rows_merged, 98);
}

#[test]
fn aggregate_strategy_rows_bridge_filtered_grouped_materialization_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("selected_strategy", "filtered_grouped_segmented_state".to_string());
    facts.insert("execution_stage", "partial".to_string());
    facts.insert("memory_mode", "bounded".to_string());
    facts.insert("group_key_count", "1".to_string());
    facts.insert("group_key_exprs", "key".to_string());
    facts.insert("group_key_dtypes", "Int64".to_string());
    facts.insert("aggregate_functions", "count,sum,max".to_string());
    facts.insert("aggregate_filter_count", "90".to_string());
    facts.insert("filtered_input_materialization_strategy", "row_chunked_grouped_segmented_state".to_string());
    facts.insert("filtered_input_materialization_estimated_bytes", "33000000".to_string());
    facts.insert("filtered_input_materialization_budget_bytes", "33554432".to_string());
    facts.insert("filtered_input_materialization_threshold_bytes", "16777216".to_string());
    facts.insert("filtered_input_lower_level_unresolved_contracts", "groupby_agg_temporary_workspace".to_string());
    facts.insert("filtered_grouped_rows_per_chunk", "512".to_string());
    facts.insert("filtered_grouped_inputs_per_segment", "2".to_string());
    facts.insert("filtered_grouped_segment_target_temporary_bytes", "4194304".to_string());
    facts.insert("filtered_grouped_actual_chunk_count", "40".to_string());
    facts.insert("filtered_grouped_actual_segment_count", "1200".to_string());
    facts.insert("filtered_grouped_max_segment_rows", "512".to_string());
    facts.insert("filtered_grouped_observed_temporary_peak_bytes", "1048576".to_string());
    facts.insert("filtered_grouped_observed_retained_peak_bytes", "524288".to_string());
    facts.insert("filtered_grouped_state_rows_merged", "20000".to_string());
    facts.insert("filtered_grouped_high_cardinality_admission", "chunk_local_partial_final_merge".to_string());

    query_metrics.record_aggregate_strategy_decision("n9", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.selected_strategy, "filtered_grouped_segmented_state");
    assert_eq!(row.aggregate_filter_count, 90);
    assert_eq!(row.filtered_input_materialization_strategy, "row_chunked_grouped_segmented_state");
    assert_eq!(row.filtered_input_lower_level_unresolved_contracts, "groupby_agg_temporary_workspace");
    assert_eq!(row.filtered_grouped_rows_per_chunk, 512);
    assert_eq!(row.filtered_grouped_inputs_per_segment, 2);
    assert_eq!(row.filtered_grouped_segment_target_temporary_bytes, 4_194_304);
    assert_eq!(row.filtered_grouped_actual_chunk_count, 40);
    assert_eq!(row.filtered_grouped_actual_segment_count, 1200);
    assert_eq!(row.filtered_grouped_max_segment_rows, 512);
    assert_eq!(row.filtered_grouped_observed_temporary_peak_bytes, 1_048_576);
    assert_eq!(row.filtered_grouped_observed_retained_peak_bytes, 524_288);
    assert_eq!(row.filtered_grouped_state_rows_merged, 20_000);
    assert_eq!(row.filtered_grouped_high_cardinality_admission, "chunk_local_partial_final_merge");
}

#[test]
fn join_strategy_rows_preserve_planned_strategy_while_counting_runtime_execution_strategy() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("join_strategy", "partitioned_hash_equi".to_string());
    facts.insert("estimated_build_rows", "10000".to_string());
    facts.insert("partition_count", "8".to_string());

    query_metrics.record_join_strategy_decision("n6", &facts);
    query_metrics.record_operator_metrics(OperatorMetrics::join(
        "n6",
        JoinOperatorMetrics {
            join_type: "inner".to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: 1,
            build_side: "right".to_string(),
            probe_side: "left".to_string(),
            build_rows: 10_000,
            build_bytes: 64 * 1024 * 1024,
            probe_rows: 1_000_000,
            probe_bytes: 512 * 1024 * 1024,
            gather_map_bytes: 2048,
            output_reservation_bytes: 4096,
            output_rows: 100,
            output_bytes: 8192,
            ..Default::default()
        },
    ));

    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.join_strategy_rows.len(), 1);
    let row = &metrics.join_strategy_rows[0];
    assert_eq!(row.join_strategy, "partitioned_hash_equi");
    assert_eq!(row.execution_strategy, "bounded_one_shot_equi_join");
    assert_eq!(row.partitioned_hash_join_count, 0);
    assert_eq!(row.one_shot_join_count, 1);
}

#[test]
fn source_read_runtime_rows_bridge_runtime_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();

    query_metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        query_id: None,
        source_node_id: "n7".to_string(),
        read_kind: "chunk_batch".to_string(),
        read_strategy: "remote_direct".to_string(),
        parquet_read_evidence: SourceReadParquetEvidenceMetrics {
            input_kind: "remote_path".to_string(),
            remote_read_mode: "require_kvikio".to_string(),
            backend: "cudf_sys_kvikio_remote_datasource".to_string(),
            backend_build_status: "kvikio_remote_io_enabled".to_string(),
            remote_bytes: Some(4096),
            remote_range_count: Some(3),
            device_read_support: "not_reported_by_libcudf".to_string(),
            pruning_input_row_groups: Some(9),
            row_groups_after_stats_filter: Some(6),
            row_groups_after_bloom_filter: Some(2),
            bloom_pruning_status: "bloom_pruned".to_string(),
        },
        batch_index: 2,
        chunk_count: 3,
        first_chunk_index: 4,
        last_chunk_index: 6,
        first_file_index: 10,
        last_file_index: 12,
        file_count: 3,
        row_group_count: 9,
        read_columns_per_file: 2,
        read_columns_total: 6,
        known_file_bytes: 900,
        estimated_source_read_bytes: 700,
        known_row_group_bytes: 600,
        known_projected_compressed_bytes: 500,
        known_projected_uncompressed_bytes: 800,
        known_pure_filter_compressed_bytes: 100,
        known_pure_filter_uncompressed_bytes: 200,
        output_rows: 42,
        reader_predicate: true,
        reader_dynamic_filter_count: 1,
        source_batching_reason: "batched".to_string(),
        source_chunk_planning_mode: "local_parquet_throughput".to_string(),
        source_chunk_planning_reason: "local_parquet_throughput_admitted".to_string(),
        elapsed: std::time::Duration::from_nanos(1234),
    });
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let row = &metrics.source_read_runtime_rows[0];
    assert_eq!(row.source_node_id, "n7");
    assert_eq!(row.read_kind, "chunk_batch");
    assert_eq!(row.read_strategy, "remote_direct");
    assert_eq!(row.batch_index, 2);
    assert_eq!(row.chunk_count, 3);
    assert_eq!(row.first_file_index, 10);
    assert_eq!(row.file_count, 3);
    assert_eq!(row.row_group_count, 9);
    assert_eq!(row.read_columns_total, 6);
    assert_eq!(row.known_projected_uncompressed_bytes, 800);
    assert_eq!(row.known_pure_filter_uncompressed_bytes, 200);
    assert_eq!(row.parquet_read_input_kind, "remote_path");
    assert_eq!(row.remote_read_mode, "require_kvikio");
    assert_eq!(row.parquet_read_backend, "cudf_sys_kvikio_remote_datasource");
    assert_eq!(row.remote_backend_build_status, "kvikio_remote_io_enabled");
    assert_eq!(row.remote_read_bytes, Some(4096));
    assert_eq!(row.remote_read_range_count, Some(3));
    assert_eq!(row.device_read_support, "not_reported_by_libcudf");
    assert_eq!(row.parquet_pruning_input_row_groups, Some(9));
    assert_eq!(row.parquet_row_groups_after_stats_filter, Some(6));
    assert_eq!(row.parquet_row_groups_after_bloom_filter, Some(2));
    assert_eq!(row.parquet_bloom_pruning_status, "bloom_pruned");
    assert_eq!(row.output_rows, 42);
    assert!(row.reader_predicate);
    assert_eq!(row.reader_dynamic_filter_count, 1);
    assert_eq!(row.source_batching_reason, "batched");
    assert_eq!(row.source_chunk_planning_mode, "local_parquet_throughput");
    assert_eq!(row.source_chunk_planning_reason, "local_parquet_throughput_admitted");
    assert_eq!(row.elapsed_ns, 1234);
}

#[test]
fn query_metric_row_bridge_enforces_shared_record_and_byte_caps() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let capture = NativeMetricsCapture::without_sink("bounded-metric-rows")
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(1, 4096))
        .new_execution();
    let native_metrics = NexusNativeMetrics::new_with_capture(&metrics_set, 0, Some(capture));
    let mut query_metrics = QueryMetrics::default();
    for index in 0..128 {
        query_metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
            source_node_id: format!("parquet-source-{index}"),
            read_kind: "chunk_batch".to_string(),
            read_strategy: "local_parquet".to_string(),
            ..SourceReadRuntimeMetrics::default()
        });
    }

    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let (remaining_records, remaining_bytes, truncated, overflow) = native_metrics.row_capture_state();
    assert_eq!(remaining_records, 0);
    assert!(remaining_bytes < 4096);
    assert!(truncated);
    assert!(!overflow);
    assert_eq!(
        metrics_set.clone_inner().sum_by_name(super::SERVER_CAPTURE_LOSS_METRIC_NAME).unwrap().as_usize(),
        1,
        "one bounded boolean marker reports row loss without growing per dropped row"
    );
}

#[test]
fn lakehouse_cache_fill_event_snapshot_rows_bridge_runtime_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = NexusNativeMetrics::new(&metrics_set, 0);
    native_metrics.record_lakehouse_cache_fill_events(&[LakehouseCacheFillEventSnapshot {
        outcome: "failed",
        bytes: None,
        elapsed: std::time::Duration::from_nanos(4321),
        cache_path_hash: "cacheabcd".to_string(),
        stable_id_hash: "stableabcd".to_string(),
        file_index: Some(3),
        chunk_index: Some(5),
        error_code: Some("cudf_dependency_failure".to_string()),
        error_kind: Some("dependency".to_string()),
        error_status: Some("retryable".to_string()),
        error_operation: Some("nexus_query_engine.lakehouse_cache.background_fill.fetch".to_string()),
    }]);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.lakehouse_cache_fill_event_snapshot_rows.len(), 1);
    let row = &metrics.lakehouse_cache_fill_event_snapshot_rows[0];
    assert_eq!(row.outcome, "failed");
    assert_eq!(row.bytes, None);
    assert_eq!(row.elapsed_ns, 4321);
    assert_eq!(row.cache_path_hash, "cacheabcd");
    assert_eq!(row.stable_id_hash, "stableabcd");
    assert_eq!(row.file_index, Some(3));
    assert_eq!(row.chunk_index, Some(5));
    assert_eq!(row.error_code, "cudf_dependency_failure");
    assert_eq!(row.error_operation, "nexus_query_engine.lakehouse_cache.background_fill.fetch");
}

#[test]
fn plan_task_mapping_rows_bridge_correlation_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics =
        NexusNativeMetrics::new_with_capture_and_shape(&metrics_set, 0, None, Some("AggregateExec(LocalParquetScan)"));
    let mut query_metrics = QueryMetrics::new("q-mapping");

    query_metrics.record_compiled_pipeline_task_mapping("t3", "source_read", Some("n4"));
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.plan_task_mapping_rows.len(), 1);
    let row = &metrics.plan_task_mapping_rows[0];
    assert_eq!(row.query_id, "q-mapping");
    assert_eq!(row.plan_node_id.as_deref(), Some("n4"));
    assert_eq!(row.compiled_task_id.as_deref(), Some("t3"));
    assert_eq!(row.operator_kind.as_deref(), Some("source_read"));
    assert_eq!(row.datafusion_shape.as_deref(), Some("AggregateExec(LocalParquetScan)"));
}
