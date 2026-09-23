use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::{
    array::{FixedSizeListArray, Float32Array, ListArray},
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DeviceMemoryResource, ErrorCode, Runtime, TypeId,
    core::Table,
    frame::dataframe::GpuRecordBatch,
    interop::{
        contract::{
            ActiveLeaseBackend, CompletionBinding, CompletionWitness, CompletionWitnessBackend, CompletionWitnessKind,
            IncompleteDropBehavior, LeaseActivation, LeaseBackend, LeaseCompletionResult, LeaseExportContext,
            ReadStartBinding, Result as InteropResult,
        },
        cuda_explicit_completion_required_lease_backend,
    },
    runtime::stream::CudaStream,
};
use cuvs::{
    BruteForceIndex, CuvsResources, DeviceBuffer, DistanceMetric, ErrorKind, IvfFlatIndex, IvfFlatIndexParams,
    IvfFlatSearchParams, KMeansInit, KMeansMetric, KMeansModel, KMeansParams, MatrixLayout, PcaOutputBuffers,
    PcaParams, ValidatedDeviceTensor, fit_transform_owned, pairwise_distance_into, select_k_into,
};

#[derive(Debug)]
struct TrackingWitness {
    synchronizations: Arc<AtomicUsize>,
}

impl CompletionWitnessBackend for TrackingWitness {
    fn is_ready(&self) -> InteropResult<bool> {
        Ok(false)
    }

    fn synchronize(&self) -> InteropResult<()> {
        self.synchronizations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct TrackingLeaseBackend {
    synchronizations: Arc<AtomicUsize>,
}

impl ActiveLeaseBackend for TrackingLeaseBackend {
    fn activate(
        &self,
        _export: &LeaseExportContext,
        _read_start: ReadStartBinding,
        _completion: CompletionBinding,
    ) -> InteropResult<LeaseActivation> {
        Ok(LeaseActivation::noop())
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        IncompleteDropBehavior::GuaranteedFallback
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> InteropResult<LeaseCompletionResult> {
        Ok(LeaseCompletionResult::Completed(CompletionWitness::with_backend(
            CompletionWitnessKind::ConsumerStream,
            Arc::new(TrackingWitness { synchronizations: Arc::clone(&self.synchronizations) }),
        )))
    }

    fn drop_incomplete(&self, _export: &LeaseExportContext, _completion: CompletionBinding) -> InteropResult<()> {
        self.synchronizations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

const TEST_RESERVATION_LIMIT_BYTES: usize = 1 << 30;
static NEXT_RESERVATION_ID: AtomicUsize = AtomicUsize::new(0xC0_0000);

fn resources() -> (Arc<Runtime>, CuvsResources, cudf::StreamReservationLease) {
    assert!(cudf::DeviceId::count().expect("query CUDA device count") > 0);
    let runtime = Arc::new(Runtime::new().expect("runtime"));
    let reservation_id = NEXT_RESERVATION_ID.fetch_add(1, Ordering::Relaxed) as u64;
    let _reservation =
        runtime.stream_reservation(reservation_id, TEST_RESERVATION_LIMIT_BYTES).expect("test stream reservation");
    let resources = CuvsResources::for_runtime(Arc::clone(&runtime)).expect("cuVS resources");
    (runtime, resources, _reservation)
}

fn i64_to_host(runtime: &Runtime, buffer: DeviceBuffer<i64>) -> Vec<i64> {
    let len = buffer.len();
    runtime
        .column_from_device_buffer(buffer.into_inner(), len, TypeId::Int64)
        .expect("adopt i64 output")
        .to_vec::<i64>()
        .expect("copy i64 output")
}

fn i32_to_host(runtime: &Runtime, buffer: DeviceBuffer<i32>) -> Vec<i32> {
    let len = buffer.len();
    runtime
        .column_from_device_buffer(buffer.into_inner(), len, TypeId::Int32)
        .expect("adopt i32 output")
        .to_vec::<i32>()
        .expect("copy i32 output")
}

fn f32_to_host(runtime: &Runtime, buffer: DeviceBuffer<f32>) -> Vec<f32> {
    let len = buffer.len();
    runtime
        .column_from_device_buffer(buffer.into_inner(), len, TypeId::Float32)
        .expect("adopt f32 output")
        .to_vec::<f32>()
        .expect("copy f32 output")
}

#[test]
fn resources_bind_the_runtime_stream_and_current_manager() {
    let (runtime, resources, _reservation) = resources();
    assert_eq!(resources.binding_id(), runtime.binding_id());
}

#[test]
fn resources_reject_custom_runtime_memory_resources() {
    let manager_runtime = Runtime::new().expect("device-manager runtime");
    let custom_resource = DeviceMemoryResource::cuda_on(manager_runtime.device()).expect("custom CUDA resource");
    let runtime = Arc::new(
        Runtime::builder()
            .with_device(manager_runtime.device())
            .with_memory_resource(custom_resource)
            .build()
            .expect("custom-resource runtime"),
    );

    let error = CuvsResources::for_runtime(runtime).expect_err("custom memory resource must be rejected");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("cannot bind an arbitrary runtime memory resource"));
}

#[test]
fn prepared_tensor_completion_and_incomplete_drop_are_safe() {
    let (runtime, resources, _reservation) = resources();
    let column = runtime.column_from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).expect("column");
    let export = column.try_prepare_device_column_export().expect("prepared export");

    let lease = resources.activate_prepared_tensor::<f32>(&export, &[2, 2], None).expect("activate prepared tensor");
    assert_eq!(lease.tensor().shape(), &[2, 2]);
    let completion = lease.complete().expect("complete prepared tensor");
    assert!(completion.witness().is_none());
    assert!(!completion.retains_producer_owner());
    assert!(completion.is_ready().expect("completed read readiness"));

    let abandoned = resources.activate_prepared_tensor::<f32>(&export, &[4], None).expect("activate abandoned tensor");
    drop(abandoned);
    assert!(runtime.stream().is_ready().expect("query stream readiness"));
}

#[test]
fn prepared_completion_retains_owner_until_drop_and_synchronizes_before_release() {
    let (runtime, resources, _reservation) = resources();
    let series =
        cudf::Series::from_column(runtime.column_from_slice(&[1.0_f32, 2.0]).expect("column")).expect("series");
    let marker = Arc::new(());
    let synchronizations = Arc::new(AtomicUsize::new(0));
    let export = series
        .try_prepare_device_column_export()
        .expect("prepared export")
        .with_retained_owner_arc(Arc::clone(&marker))
        .with_lease_backend(LeaseBackend::new(TrackingLeaseBackend {
            synchronizations: Arc::clone(&synchronizations),
        }));

    let completion =
        resources.activate_prepared_tensor::<f32>(&export, &[2], None).expect("activate").complete().expect("complete");
    assert!(completion.retains_producer_owner());
    assert_eq!(synchronizations.load(Ordering::SeqCst), 0);
    drop(export);
    drop(series);
    assert_eq!(Arc::strong_count(&marker), 2, "completion must retain the producer owner");

    drop(completion);
    assert_eq!(synchronizations.load(Ordering::SeqCst), 1);
    assert_eq!(Arc::strong_count(&marker), 1, "owner must release only after synchronization");
}

#[test]
fn prepared_completion_without_retained_owner_synchronizes_before_return() {
    let (runtime, resources, _reservation) = resources();
    let column = runtime.column_from_slice(&[1.0_f32, 2.0]).expect("column");
    let synchronizations = Arc::new(AtomicUsize::new(0));
    let export = column.try_prepare_device_column_export().expect("prepared export").with_lease_backend(
        LeaseBackend::new(TrackingLeaseBackend { synchronizations: Arc::clone(&synchronizations) }),
    );
    assert!(export.clone_retained_owner().is_none());

    let completion =
        resources.activate_prepared_tensor::<f32>(&export, &[2], None).expect("activate").complete().expect("complete");
    assert!(completion.witness().is_none());
    assert!(!completion.retains_producer_owner());
    assert_eq!(synchronizations.load(Ordering::SeqCst), 1);
}

#[test]
fn finite_fixed_size_list_witness_activates_validated_tensor() {
    let (runtime, resources, _reservation) = resources();
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let values = Arc::new(Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0]));
    let lists = FixedSizeListArray::new(Arc::clone(&item), 2, values, None);
    let schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 2), false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(lists)]).expect("record batch");
    let frame = cudf::DataFrame::from_record_batch(&runtime, &batch).expect("device dataframe");
    let gpu_batch = GpuRecordBatch::new(frame, schema).expect("GPU record batch");
    let validated = gpu_batch.try_prepare_fixed_size_list_float32_export(0).expect("finite vector export");

    assert_eq!(validated.rows(), 3);
    assert_eq!(validated.dimension(), 2);
    assert_eq!(validated.binding_id(), resources.binding_id());
    let lease = resources.activate_validated_fixed_size_list_tensor(&validated).expect("validated tensor lease");
    assert_eq!(lease.tensor().shape(), &[3, 2]);
    assert_eq!(lease.tensor().binding_id(), resources.binding_id());
    let index = BruteForceIndex::build(&resources, lease.tensor(), DistanceMetric::L2Expanded).expect("build index");
    index.close().expect("close index");
    lease.complete().expect("complete vector read").synchronize().expect("synchronize vector read");
}

#[test]
fn fixed_size_list_witness_rejects_non_finite_child_values() {
    let (runtime, _resources, _reservation) = resources();
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let values = Arc::new(Float32Array::from(vec![0.0_f32, 1.0, f32::NAN, 3.0]));
    let lists = FixedSizeListArray::new(Arc::clone(&item), 2, values, None);
    let schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 2), false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(lists)]).expect("record batch");
    let frame = cudf::DataFrame::from_record_batch(&runtime, &batch).expect("device dataframe");
    let gpu_batch = GpuRecordBatch::new(frame, schema).expect("GPU record batch");

    let error = gpu_batch
        .try_prepare_fixed_size_list_float32_export(0)
        .expect_err("non-finite FixedSizeList child values must not produce a cuVS witness");
    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert!(error.message().contains("not finite"));
}

#[test]
fn algorithms_reject_same_device_tensors_from_a_different_stream_binding() {
    let (producer_runtime, producer_resources, _producer_reservation) = resources();
    let values = DeviceBuffer::from_slice(&producer_resources, &[0.0_f32, 0.0, 1.0, 1.0]).expect("values");
    let tensor = values.as_validated_matrix(&producer_resources, 2, 2).expect("tensor");

    let consumer_stream = CudaStream::new().expect("consumer stream");
    let consumer_runtime = Arc::new(
        Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream)
            .build()
            .expect("consumer runtime"),
    );
    let consumer_resources = CuvsResources::for_runtime(consumer_runtime).expect("consumer resources");

    let error = BruteForceIndex::build(&consumer_resources, &tensor, DistanceMetric::L2Expanded)
        .expect_err("same-device cross-stream tensors must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.message().contains("same exact runtime binding"));
    assert!(error.context_value("tensor_binding_id").is_some());
    assert!(error.context_value("resource_binding_id").is_some());
}

#[test]
fn validated_packed_tensor_rejects_a_different_memory_resource_binding() {
    let (runtime, resources, _reservation) = resources();
    let custom_resource = DeviceMemoryResource::cuda_on(runtime.device()).expect("custom CUDA resource");
    let foreign_runtime = Runtime::builder()
        .with_device(runtime.device())
        .with_stream(runtime.stream().clone())
        .with_memory_resource(custom_resource)
        .build()
        .expect("foreign runtime");
    let first = foreign_runtime.column_from_slice(&[1.0_f32, 2.0]).expect("first");
    let second = foreign_runtime.column_from_slice(&[3.0_f32, 4.0]).expect("second");
    let packed = foreign_runtime
        .pack_float32_columns(&[&first, &second], cudf::DenseMatrixLayout::RowMajor)
        .expect("packed matrix");
    let validated = packed.validate_finite().expect("finite matrix");

    let error = ValidatedDeviceTensor::from_packed(&resources, &validated)
        .expect_err("custom-memory-resource packed input must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.message().contains("same exact runtime binding"));
}

#[test]
fn prepared_table_ingress_uses_one_root_lease_and_runtime_packing() {
    let (runtime, resources, _reservation) = resources();
    let table = Table::from_columns(vec![
        runtime.column_from_slice(&[1.0_f32, 2.0, 3.0]).expect("first"),
        runtime.column_from_slice(&[4.0_f32, 5.0, 6.0]).expect("second"),
    ])
    .expect("table");
    let synchronizations = Arc::new(AtomicUsize::new(0));
    let export = table.try_prepare_device_table_export().expect("prepared table").with_lease_backend(
        LeaseBackend::new(TrackingLeaseBackend { synchronizations: Arc::clone(&synchronizations) }),
    );

    let matrix = resources.pack_prepared_float32_table(&export, MatrixLayout::RowMajor).expect("packed prepared table");
    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.columns(), 2);
    assert_eq!(matrix.layout(), MatrixLayout::RowMajor);
    assert_eq!(matrix.binding_id(), resources.binding_id());
    assert_eq!(synchronizations.load(Ordering::SeqCst), 1, "table import must activate one root lease");
}

#[test]
fn validated_list_pack_entrypoints_preserve_public_shape_layout_and_binding() {
    let (runtime, resources, _reservation) = resources();
    let item = Arc::new(Field::new("item", ArrowDataType::Float32, false));
    let fixed =
        FixedSizeListArray::new(Arc::clone(&item), 2, Arc::new(Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0])), None);
    let fixed_schema =
        Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::FixedSizeList(item, 2), false)]));
    let fixed_host = RecordBatch::try_new(Arc::clone(&fixed_schema), vec![Arc::new(fixed)]).expect("fixed host batch");
    let fixed_gpu = GpuRecordBatch::new(
        DataFrame::from_record_batch(&runtime, &fixed_host).expect("fixed GPU frame"),
        fixed_schema,
    )
    .expect("fixed GPU batch");
    let fixed_export = fixed_gpu.try_prepare_fixed_size_list_float32_export(0).expect("fixed validated export");

    let fixed_matrix = resources
        .pack_validated_fixed_size_list_float32(&fixed_export, MatrixLayout::RowMajor)
        .expect("direct fixed-list pack");
    assert_eq!(fixed_matrix.rows(), 2);
    assert_eq!(fixed_matrix.columns(), 2);
    assert_eq!(fixed_matrix.layout(), MatrixLayout::RowMajor);
    assert_eq!(fixed_matrix.binding_id(), resources.binding_id());
    assert!(fixed_matrix.is_finite_validated());

    let element = Arc::new(Field::new("element", ArrowDataType::Float32, false));
    let lists = ListArray::new(
        Arc::clone(&element),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 4])),
        Arc::new(Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0])),
        None,
    );
    let list_schema = Arc::new(Schema::new(vec![Field::new("features", ArrowDataType::List(element), false)]));
    let list_host = RecordBatch::try_new(Arc::clone(&list_schema), vec![Arc::new(lists)]).expect("list host batch");
    let list_gpu =
        GpuRecordBatch::new(DataFrame::from_record_batch(&runtime, &list_host).expect("list GPU frame"), list_schema)
            .expect("list GPU batch");
    let list_export = list_gpu.try_prepare_uniform_list_float32_export(0).expect("uniform validated export");

    let list_matrix = resources
        .pack_validated_uniform_list_float32(&list_export, MatrixLayout::ColumnMajor)
        .expect("direct uniform-list pack");
    assert_eq!(list_matrix.rows(), 2);
    assert_eq!(list_matrix.columns(), 2);
    assert_eq!(list_matrix.layout(), MatrixLayout::ColumnMajor);
    assert_eq!(list_matrix.binding_id(), resources.binding_id());
    assert!(list_matrix.is_finite_validated());
}

#[test]
fn prepared_tensor_rejects_exports_without_incomplete_drop_fallback() {
    let (runtime, resources, _reservation) = resources();
    let column = runtime.column_from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).expect("column");
    let export = column
        .try_prepare_device_column_export()
        .expect("prepared export")
        .with_lease_backend(cuda_explicit_completion_required_lease_backend());

    let error = resources
        .activate_prepared_tensor::<f32>(&export, &[2, 2], None)
        .expect_err("explicit-completion-only export must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.message().contains("must guarantee safe fallback"));
}

#[test]
fn brute_force_search_writes_caller_owned_device_buffers() {
    let (runtime, resources, _reservation) = resources();
    let points =
        [0.0_f32, 0.0, 1.0, 0.0, 3.0, 0.0, 6.0, 0.0, 10.0, 0.0, 15.0, 0.0, 21.0, 0.0, 28.0, 0.0, 36.0, 0.0, 45.0, 0.0];
    let data = DeviceBuffer::from_slice(&resources, &points).expect("dataset");
    let matrix = data.as_validated_matrix(&resources, 10, 2).expect("matrix");
    let index = BruteForceIndex::build(&resources, &matrix, DistanceMetric::L2Expanded).expect("build");
    let mut neighbors = DeviceBuffer::<i64>::new(&resources, 30).expect("neighbors");
    let mut distances = DeviceBuffer::<f32>::new(&resources, 30).expect("distances");
    let result = index.search_into(&matrix, 3, &mut neighbors, &mut distances).expect("search");
    assert_eq!(result.n_queries, 10);
    assert_eq!(result.k, 3);
    index.close().expect("close index");
    drop(matrix);

    let neighbors = i64_to_host(&runtime, neighbors);
    let distances = f32_to_host(&runtime, distances);
    assert!(neighbors.iter().all(|neighbor| (0..10).contains(neighbor)));
    assert!(distances.iter().all(|distance| distance.is_finite()));
    assert_eq!(&neighbors[..3], &[0, 1, 2]);
}

#[test]
fn ivf_flat_all_metrics_match_brute_force_when_every_list_is_probed() {
    let (runtime, resources, _reservation) = resources();
    let data = DeviceBuffer::from_slice(
        &resources,
        &[
            1.0_f32, 0.0, 1.1, 0.2, 0.9, 0.4, 0.0, 1.0, 0.2, 0.9, 0.4, 0.8, -1.0, 0.0, -0.9, 0.2, -0.8, 0.4, 0.0, -1.0,
            0.2, -0.9, 0.4, -0.8,
        ],
    )
    .unwrap();
    let queries = DeviceBuffer::from_slice(&resources, &[1.0_f32, 0.05, 0.05, 1.0, -1.0, 0.05]).unwrap();
    let matrix = data.as_validated_matrix(&resources, 12, 2).unwrap();
    let query_matrix = queries.as_validated_matrix(&resources, 3, 2).unwrap();
    for metric in [
        DistanceMetric::L2Expanded,
        DistanceMetric::L2SqrtExpanded,
        DistanceMetric::InnerProduct,
        DistanceMetric::CosineExpanded,
    ] {
        let brute = BruteForceIndex::build(&resources, &matrix, metric).unwrap();
        let mut oracle_ids = DeviceBuffer::<i64>::new(&resources, 9).unwrap();
        let mut oracle_distances = DeviceBuffer::<f32>::new(&resources, 9).unwrap();
        brute.search_into(&query_matrix, 3, &mut oracle_ids, &mut oracle_distances).unwrap();
        let oracle_ids = i64_to_host(&runtime, oracle_ids);
        let oracle_distances = f32_to_host(&runtime, oracle_distances);
        brute.close().unwrap();

        let index = IvfFlatIndex::build(&resources, &matrix, IvfFlatIndexParams::new(metric, 3)).unwrap();
        let mut ids = DeviceBuffer::<i64>::new(&resources, 9).unwrap();
        let mut distances = DeviceBuffer::<f32>::new(&resources, 9).unwrap();
        index.search_into(&query_matrix, 3, IvfFlatSearchParams { n_probes: 3 }, &mut ids, &mut distances).unwrap();
        assert_eq!(i64_to_host(&runtime, ids), oracle_ids, "metric {metric:?} recall@3");
        for (actual, expected) in f32_to_host(&runtime, distances).iter().zip(oracle_distances.iter()) {
            assert!((actual - expected).abs() < 1e-3, "metric {metric:?}: {actual} != {expected}");
        }
        index.close().unwrap();
    }
}

#[test]
fn ivf_flat_marks_unprobed_slots_and_rejects_foreign_bindings() {
    let (runtime, resources, _reservation) = resources();
    let data = DeviceBuffer::from_slice(
        &resources,
        &[1.0_f32, 0.0, 1.1, 0.0, 1.2, 0.0, 10.0, 0.0, 10.1, 0.0, 10.2, 0.0, 20.0, 0.0, 20.1, 0.0, 20.2, 0.0],
    )
    .unwrap();
    let queries = DeviceBuffer::from_slice(&resources, &[1.0_f32, 0.0]).unwrap();
    let matrix = data.as_validated_matrix(&resources, 9, 2).unwrap();
    let query_matrix = queries.as_validated_matrix(&resources, 1, 2).unwrap();
    let index =
        IvfFlatIndex::build(&resources, &matrix, IvfFlatIndexParams::new(DistanceMetric::L2Expanded, 3)).unwrap();
    let mut ids = DeviceBuffer::<i64>::new(&resources, 8).unwrap();
    let mut distances = DeviceBuffer::<f32>::new(&resources, 8).unwrap();
    index.search_into(&query_matrix, 8, IvfFlatSearchParams { n_probes: 1 }, &mut ids, &mut distances).unwrap();
    let ids = i64_to_host(&runtime, ids);
    let distances = f32_to_host(&runtime, distances);
    assert!(ids.contains(&i64::MAX), "one probed list cannot contain eight of nine rows");
    assert!(ids.iter().zip(distances.iter()).all(|(id, distance)| *id == i64::MAX || distance.is_finite()));
    assert!(ids.iter().zip(distances.iter()).any(|(id, distance)| *id == i64::MAX && distance.is_infinite()));
    let foreign_stream = CudaStream::new().unwrap();
    let foreign_runtime =
        Arc::new(Runtime::builder().with_device(runtime.device()).with_stream(foreign_stream).build().unwrap());
    let foreign_resources = CuvsResources::for_runtime(foreign_runtime).unwrap();
    assert_eq!(
        IvfFlatIndex::build(&foreign_resources, &matrix, IvfFlatIndexParams::new(DistanceMetric::L2Expanded, 3))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    index.close().unwrap();
}

#[test]
fn pairwise_metrics_match_host_oracles_and_reject_foreign_bindings() {
    let (runtime, resources, _reservation) = resources();
    let x = DeviceBuffer::from_slice(&resources, &[1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let y = DeviceBuffer::from_slice(&resources, &[1.0_f32, 0.0, 1.0, 1.0]).unwrap();
    let x_tensor = x.as_validated_matrix(&resources, 2, 2).unwrap();
    let y_tensor = y.as_validated_matrix(&resources, 2, 2).unwrap();
    for (metric, expected) in [
        (DistanceMetric::L2Expanded, [0.0_f32, 1.0, 2.0, 1.0]),
        (DistanceMetric::L2SqrtExpanded, [0.0_f32, 1.0, 2.0_f32.sqrt(), 1.0]),
        (DistanceMetric::CosineExpanded, [0.0_f32, 1.0 - 1.0 / 2.0_f32.sqrt(), 1.0, 1.0 - 1.0 / 2.0_f32.sqrt()]),
        (DistanceMetric::InnerProduct, [1.0_f32, 1.0, 0.0, 1.0]),
    ] {
        let mut out = DeviceBuffer::<f32>::new(&resources, 4).unwrap();
        pairwise_distance_into(&resources, &x_tensor, &y_tensor, metric, &mut out).unwrap();
        for (actual, expected) in f32_to_host(&runtime, out).into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-4, "{metric:?}: {actual} != {expected}");
        }
    }
    let other_stream = CudaStream::new().unwrap();
    let other_runtime =
        Arc::new(Runtime::builder().with_device(runtime.device()).with_stream(other_stream).build().unwrap());
    let other_resources = CuvsResources::for_runtime(other_runtime).unwrap();
    let mut out = DeviceBuffer::<f32>::new(&resources, 4).unwrap();
    let error = pairwise_distance_into(&other_resources, &x_tensor, &y_tensor, DistanceMetric::L2Expanded, &mut out)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

#[test]
fn batched_top_k_selects_min_and_max_and_validates_boundaries() {
    let (runtime, resources, _reservation) = resources();
    let input = DeviceBuffer::from_slice(&resources, &[4.0_f32, 1.0, 3.0, 2.0, 0.0, 8.0]).unwrap();
    let tensor = input.as_validated_matrix(&resources, 2, 3).unwrap();
    for (select_min, expected_values, expected_positions) in
        [(true, [1.0_f32, 3.0, 0.0, 2.0], [1_i64, 2, 1, 0]), (false, [4.0_f32, 3.0, 8.0, 2.0], [0_i64, 2, 2, 0])]
    {
        let mut values = DeviceBuffer::<f32>::new(&resources, 4).unwrap();
        let mut positions = DeviceBuffer::<i64>::new(&resources, 4).unwrap();
        select_k_into(&resources, &tensor, 2, select_min, &mut values, &mut positions).unwrap();
        assert_eq!(f32_to_host(&runtime, values), expected_values);
        assert_eq!(i64_to_host(&runtime, positions), expected_positions);
    }
    let mut values = DeviceBuffer::<f32>::new(&resources, 2).unwrap();
    let mut positions = DeviceBuffer::<i64>::new(&resources, 2).unwrap();
    assert_eq!(
        select_k_into(&resources, &tensor, 4, true, &mut values, &mut positions).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let foreign_stream = CudaStream::new().unwrap();
    let foreign_runtime =
        Arc::new(Runtime::builder().with_device(runtime.device()).with_stream(foreign_stream).build().unwrap());
    let foreign = CuvsResources::for_runtime(foreign_runtime).unwrap();
    assert_eq!(
        select_k_into(&foreign, &tensor, 1, true, &mut values, &mut positions).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
}

#[test]
fn kmeans_model_retains_fitted_params_and_device_centroids() {
    let (runtime, resources, _reservation) = resources();
    let values =
        [0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 11.0, 11.0, 10.0, -10.0, 10.0, -10.0, 11.0, -9.0, 10.0];
    let data = DeviceBuffer::from_slice(&resources, &values).expect("data");
    let matrix = data.as_validated_matrix(&resources, 9, 2).expect("matrix");
    let initial =
        DeviceBuffer::from_slice(&resources, &[0.0_f32, 0.0, 10.0, 10.0, -10.0, 10.0]).expect("initial centroids");
    let mut params = KMeansParams::new(3);
    params.init = KMeansInit::Array;
    params.tol = 1e-6;
    let model = KMeansModel::fit(&resources, &matrix, params, Some(initial)).expect("fit");
    let mut labels = DeviceBuffer::<i32>::new(&resources, 9).expect("labels");
    let prediction = model.predict_into(&matrix, &mut labels).expect("predict");
    assert_eq!(prediction.rows, 9);
    assert!(prediction.inertia.is_finite());
    assert!(model.fit_result().inertia.is_finite());
    assert!(model.fit_result().n_iter > 0);
    drop(matrix);

    let labels = i32_to_host(&runtime, labels);
    for cluster in labels.as_chunks::<3>().0 {
        assert_eq!(cluster[0], cluster[1]);
        assert_eq!(cluster[1], cluster[2]);
    }
}

#[test]
fn kmeans_supports_l2_sqrt_expanded_metric() {
    let (_runtime, resources, _reservation) = resources();
    let data = DeviceBuffer::from_slice(&resources, &[0.0_f32, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0]).expect("data");
    let matrix = data.as_validated_matrix(&resources, 4, 2).expect("matrix");
    let initial = DeviceBuffer::from_slice(&resources, &[0.0_f32, 0.0, 10.0, 10.0]).expect("initial centroids");
    let mut params = KMeansParams::new(2);
    params.init = KMeansInit::Array;
    params.metric = KMeansMetric::L2SqrtExpanded;

    let model = KMeansModel::fit(&resources, &matrix, params, Some(initial)).expect("fit L2SqrtExpanded");
    assert_eq!(model.params().metric, KMeansMetric::L2SqrtExpanded);
    assert!(model.fit_result().inertia.is_finite());
}

#[test]
fn pca_packs_prepared_table_and_returns_device_dataframe() {
    let (runtime, resources, _reservation) = resources();
    let rows = 6;
    let columns = 3;
    let table = Table::from_columns(vec![
        runtime.column_from_slice(&[1.0_f32, 4.0, 2.0, 3.0, -1.0, 3.5]).expect("first"),
        runtime.column_from_slice(&[2.0_f32, 5.0, 1.0, 6.0, 0.5, 5.0]).expect("second"),
        runtime.column_from_slice(&[3.0_f32, 6.0, 4.0, 5.0, 2.0, 6.5]).expect("third"),
    ])
    .expect("table");
    let export = table.try_prepare_device_table_export().expect("prepared table");
    let mut input =
        resources.pack_prepared_float32_table(&export, MatrixLayout::ColumnMajor).expect("column-major packed input");
    let params = PcaParams::new(2);
    let mut outputs = PcaOutputBuffers::new(&resources, rows, columns, 2).expect("outputs");
    let fit = fit_transform_owned(&resources, &mut input, &params, &mut outputs).expect("fit transform");
    assert_eq!(fit.rows(), rows);
    assert_eq!(fit.columns(), columns);
    assert_eq!(fit.n_components(), 2);

    drop(export);
    let source_columns = table.into_columns().expect("source columns");
    let source_values =
        source_columns.iter().map(|column| column.to_vec::<f32>().expect("source values")).collect::<Vec<_>>();
    assert_eq!(source_values[0], [1.0_f32, 4.0, 2.0, 3.0, -1.0, 3.5]);
    assert_eq!(source_values[1], [2.0_f32, 5.0, 1.0, 6.0, 0.5, 5.0]);
    assert_eq!(source_values[2], [3.0_f32, 6.0, 4.0, 5.0, 2.0, 6.5]);

    let same_flat_len_wrong_shape =
        PcaOutputBuffers::new(&resources, 4, columns, 3).expect("same-cardinality mismatched outputs");
    let error = same_flat_len_wrong_shape
        .into_transformed_dataframe(fit)
        .expect_err("a fit token must validate every output dimension, not only the transformed flat length");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.message().contains("components"));

    let uninitialized = PcaOutputBuffers::new(&resources, rows, columns, 2).expect("uninitialized outputs");
    let error = uninitialized
        .into_transformed_dataframe(fit)
        .expect_err("unvalidated PCA buffers must not be adopted as a transformed dataframe");
    assert_eq!(error.kind(), ErrorKind::InvalidState);

    let transformed = outputs.into_transformed_dataframe(fit).expect("transformed dataframe");
    assert_eq!(transformed.num_rows(), rows);
    assert_eq!(transformed.num_columns(), 2);
    assert_eq!(transformed.column_names(), ["component_0", "component_1"]);
    for column in transformed.iter_series() {
        assert_eq!(column.expect("transformed column").placement(), runtime.placement());
    }
}

#[test]
fn float32_device_buffers_reject_non_finite_native_inputs() {
    let (_runtime, resources, _reservation) = resources();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = DeviceBuffer::from_slice(&resources, &[0.0_f32, value])
            .expect_err("non-finite Float32 input must be rejected before device allocation");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(error.message().contains("finite"));
    }
}
