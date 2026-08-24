use core::ptr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, BorrowedColumnView, BoundBytes, BufferBinding, BufferId, ColumnId,
    ColumnLayout, CompletionAuthority, CompletionBinding, CompletionWitnessBackend, CompletionWitnessKind,
    CudaEventHandle, CudaStreamHandle, DataType, DeviceMemoryResourceHandle, ExecutionBinding, ExportDiagnostics,
    ExportProfile, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation, LeaseBackend,
    LeaseCompletionResult, LeaseExportContext, NullMask, OwnedColumnTransfer, PartitionedPreparedTableExport,
    PartitionedPreparedTablePartition, PhysicalValidity, Placement, PreparedColumnExport, PreparedTableExport,
    ProducerReadiness, ReadStartAuthority, ReadStartBinding, ReleaseContract, ReleaseSynchronization,
    SchemaNullability, SemanticColumn, SemanticTable, TableId, TransferCompletionRule, TypeId,
};

fn fake_ptr(addr: usize) -> *const core::ffi::c_void {
    addr as *const core::ffi::c_void
}

fn explicit_binding() -> ExecutionBinding<'static> {
    ExecutionBinding::explicit(
        0,
        CudaStreamHandle::from_raw_on_device(fake_ptr(0x3000) as *mut core::ffi::c_void, 0).unwrap(),
        DeviceMemoryResourceHandle::from_raw(fake_ptr(0x4000) as *mut core::ffi::c_void).unwrap(),
    )
}

fn prepared_readiness() -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceStreamOrdered {
        stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, 0).unwrap(),
    }
}

fn prepared_readiness_on_device(device: i32) -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceStreamOrdered {
        stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, device).unwrap(),
    }
}

fn prepared_event_readiness() -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceEventOrdered {
        event: CudaEventHandle::from_raw_on_device(fake_ptr(0x5100) as *mut core::ffi::c_void, 0).unwrap(),
    }
}

fn prepared_event_readiness_on_device(device: i32) -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceEventOrdered {
        event: CudaEventHandle::from_raw_on_device(fake_ptr(0x5100) as *mut core::ffi::c_void, device).unwrap(),
    }
}

fn prepared_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedDeviceExport)
}

fn prepared_host_diagnostics() -> ExportDiagnostics {
    ExportDiagnostics::new(ExportProfile::PreparedHostExport)
}

fn owned_release_contract() -> ReleaseContract {
    ReleaseContract::new(
        0,
        TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
        ReleaseSynchronization::RequiresCallerSynchronization,
    )
}

fn prepared_fixed_width_table_export_on_device(
    table_id: u32,
    column_id: u32,
    buffer_id: u32,
    addr: usize,
    device: i32,
    num_rows: usize,
) -> PreparedTableExport<'static> {
    PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(table_id), num_rows, vec![ColumnId::new(column_id)], None),
        vec![fixed_width_column(column_id, buffer_id, num_rows, 0)],
        vec![device_binding(buffer_id, addr, num_rows * std::mem::size_of::<i32>(), device)],
        prepared_readiness_on_device(device),
        prepared_diagnostics(),
    )
    .unwrap()
}

#[test]
fn prepared_table_export_lookup_helpers_resolve_top_level_names_ids_and_bindings() {
    let root_src = fixed_width_column(1, 11, 3, 0);
    let root_neighbors = list_column(2, 12, 3, 3, 0);
    let child = fixed_width_column(3, 13, 5, 0);
    let export = PreparedTableExport::try_new_with_semantic_columns(
        SemanticTable::new(
            TableId::new(10),
            3,
            vec![root_src.id(), root_neighbors.id()],
            Some(vec!["src".into(), "neighbors".into()]),
        ),
        vec![root_src.clone(), root_neighbors.clone()],
        vec![root_src.clone(), root_neighbors.clone(), child.clone()],
        vec![
            device_binding(11, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x6100, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(13, 0x6200, 5 * std::mem::size_of::<i32>(), 0),
        ],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.top_level_column(root_src.id()), Some(&root_src));
    assert_eq!(export.top_level_column_by_name("neighbors"), Some(&root_neighbors));
    assert_eq!(export.top_level_column(child.id()), None);
    assert_eq!(export.semantic_column(child.id()), Some(&child));
    assert_eq!(export.top_level_column_by_name("missing"), None);
    assert_eq!(export.binding(BufferId::new(12)).map(|binding| binding.id()), Some(BufferId::new(12)));
    assert_eq!(export.binding(BufferId::new(99)), None);
}

#[test]
fn prepared_table_export_name_lookup_requires_schema_names() {
    let export = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3);

    assert_eq!(export.top_level_column_by_name("src"), None);
}

#[test]
fn prepared_column_export_retained_owner_arc_builder_preserves_owner() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 11, 3, 0),
        vec![host_binding(11, 0x6000, 3 * std::mem::size_of::<i32>())],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap()
    .with_retained_owner_arc(Arc::new([1u8, 2, 3]));

    assert!(export.clone_retained_owner().is_some());
}

#[test]
fn prepared_table_export_retained_owner_arc_builder_preserves_owner() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(10), 3, vec![ColumnId::new(1)], Some(vec!["src".into()])),
        vec![fixed_width_column(1, 11, 3, 0)],
        vec![host_binding(11, 0x6000, 3 * std::mem::size_of::<i32>())],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap()
    .with_retained_owner_arc(Arc::new(String::from("owner")));

    assert!(export.clone_retained_owner().is_some());
}

#[test]
fn execution_binding_requires_explicit_device_stream_handle() {
    let err = CudaStreamHandle::from_raw(fake_ptr(0x3000) as *mut core::ffi::c_void)
        .map(|stream| {
            ExecutionBinding::explicit(
                0,
                stream,
                DeviceMemoryResourceHandle::from_raw(fake_ptr(0x4000) as *mut core::ffi::c_void).unwrap(),
            )
        })
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert!(err.context().iter().any(|(key, value)| *key == "field" && value == "cuda_stream.device"));
}

fn same_device_completion() -> CompletionAuthority<'static> {
    CompletionAuthority::SameDeviceStream {
        stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x8000) as *mut core::ffi::c_void, 0).unwrap(),
    }
}

fn same_device_completion_on_device(device: i32) -> CompletionAuthority<'static> {
    CompletionAuthority::SameDeviceStream {
        stream: CudaStreamHandle::from_raw_on_device(fake_ptr(0x8000) as *mut core::ffi::c_void, device).unwrap(),
    }
}

fn cross_device_completion() -> CompletionAuthority<'static> {
    CompletionAuthority::CrossDeviceEvent {
        event: CudaEventHandle::from_raw_on_device(fake_ptr(0x9000) as *mut core::ffi::c_void, 1).unwrap(),
    }
}

fn cross_device_completion_on_device(device: i32) -> CompletionAuthority<'static> {
    CompletionAuthority::CrossDeviceEvent {
        event: CudaEventHandle::from_raw_on_device(fake_ptr(0x9000) as *mut core::ffi::c_void, device).unwrap(),
    }
}

fn device_binding(id: u32, addr: usize, byte_len: usize, device: i32) -> BufferBinding<'static> {
    BufferBinding::new(
        BufferId::new(id),
        BoundBytes::from_raw_parts(fake_ptr(addr) as *mut core::ffi::c_void, byte_len).unwrap(),
        Placement::CudaDevice { device },
    )
}

fn host_binding(id: u32, addr: usize, byte_len: usize) -> BufferBinding<'static> {
    BufferBinding::new(
        BufferId::new(id),
        BoundBytes::from_raw_parts(fake_ptr(addr) as *mut core::ffi::c_void, byte_len).unwrap(),
        Placement::Host,
    )
}

fn fixed_width_column(id: u32, values: u32, len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::Int32),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::FixedWidth { values: BufferId::new(values) },
    )
}

fn fixed_width_column_with_validity(
    id: u32,
    values: u32,
    len: usize,
    offset: usize,
    validity: PhysicalValidity,
) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::Int32),
        len,
        offset,
        validity,
        SchemaNullability::Unknown,
        ColumnLayout::FixedWidth { values: BufferId::new(values) },
    )
}

fn binary_like_column(id: u32, offsets: u32, data: u32, len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::String),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::BinaryLike { offsets: BufferId::new(offsets), data: BufferId::new(data) },
    )
}

fn list_column(id: u32, offsets: u32, child: u32, len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::List),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::List { offsets: BufferId::new(offsets), child: ColumnId::new(child) },
    )
}

fn struct_column(id: u32, children: &[u32], len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::Struct),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::Struct { children: children.iter().copied().map(ColumnId::new).collect() },
    )
}

fn dictionary_column(id: u32, indices: u32, keys: u32, len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::Dictionary32),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::Dictionary { indices: BufferId::new(indices), keys: ColumnId::new(keys) },
    )
}

#[derive(Default)]
struct TestWitnessState {
    ready: AtomicBool,
    sync_calls: AtomicUsize,
}

impl CompletionWitnessBackend for TestWitnessState {
    fn is_ready(&self) -> rapids_interop::Result<bool> {
        Ok(self.ready.load(Ordering::SeqCst))
    }

    fn synchronize(&self) -> rapids_interop::Result<()> {
        self.ready.store(true, Ordering::SeqCst);
        self.sync_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct TestLeaseBackend {
    behavior: IncompleteDropBehavior,
    activation_calls: Arc<AtomicUsize>,
    activation_drop_calls: Arc<AtomicUsize>,
    fail_activation: bool,
    drop_calls: Arc<AtomicUsize>,
    witness_state: Arc<TestWitnessState>,
}

impl TestLeaseBackend {
    fn guaranteed_drop_fallback() -> (LeaseBackend, Arc<AtomicUsize>, Arc<TestWitnessState>) {
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let witness_state = Arc::new(TestWitnessState::default());
        (
            LeaseBackend::new(Self {
                behavior: IncompleteDropBehavior::GuaranteedFallback,
                activation_calls: Arc::new(AtomicUsize::new(0)),
                activation_drop_calls: Arc::new(AtomicUsize::new(0)),
                fail_activation: false,
                drop_calls: Arc::clone(&drop_calls),
                witness_state: Arc::clone(&witness_state),
            }),
            drop_calls,
            witness_state,
        )
    }

    fn explicit_completion_required() -> (LeaseBackend, Arc<AtomicUsize>) {
        let drop_calls = Arc::new(AtomicUsize::new(0));
        (
            LeaseBackend::new(Self {
                behavior: IncompleteDropBehavior::ExplicitCompletionRequired,
                activation_calls: Arc::new(AtomicUsize::new(0)),
                activation_drop_calls: Arc::new(AtomicUsize::new(0)),
                fail_activation: false,
                drop_calls: Arc::clone(&drop_calls),
                witness_state: Arc::new(TestWitnessState::default()),
            }),
            drop_calls,
        )
    }

    fn guaranteed_drop_fallback_with_activation() -> (LeaseBackend, Arc<AtomicUsize>, Arc<AtomicUsize>, Arc<AtomicUsize>)
    {
        let activation_calls = Arc::new(AtomicUsize::new(0));
        let activation_drop_calls = Arc::new(AtomicUsize::new(0));
        let drop_calls = Arc::new(AtomicUsize::new(0));
        (
            LeaseBackend::new(Self {
                behavior: IncompleteDropBehavior::GuaranteedFallback,
                activation_calls: Arc::clone(&activation_calls),
                activation_drop_calls: Arc::clone(&activation_drop_calls),
                fail_activation: false,
                drop_calls: Arc::clone(&drop_calls),
                witness_state: Arc::new(TestWitnessState::default()),
            }),
            activation_calls,
            activation_drop_calls,
            drop_calls,
        )
    }

    fn activation_failure() -> (LeaseBackend, Arc<AtomicUsize>) {
        let activation_calls = Arc::new(AtomicUsize::new(0));
        (
            LeaseBackend::new(Self {
                behavior: IncompleteDropBehavior::GuaranteedFallback,
                activation_calls: Arc::clone(&activation_calls),
                activation_drop_calls: Arc::new(AtomicUsize::new(0)),
                fail_activation: true,
                drop_calls: Arc::new(AtomicUsize::new(0)),
                witness_state: Arc::new(TestWitnessState::default()),
            }),
            activation_calls,
        )
    }
}

struct TestActivationState {
    drop_calls: Arc<AtomicUsize>,
}

impl Drop for TestActivationState {
    fn drop(&mut self) {
        self.drop_calls.fetch_add(1, Ordering::SeqCst);
    }
}

impl ActiveLeaseActivation for TestActivationState {}

impl ActiveLeaseBackend for TestLeaseBackend {
    fn activate(
        &self,
        _export: &LeaseExportContext,
        _read_start: ReadStartBinding,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseActivation> {
        self.activation_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_activation {
            return Err(InteropError::new(InteropErrorKind::ReadinessUnavailable, "test activation failure"));
        }
        Ok(LeaseActivation::with_state(Arc::new(TestActivationState {
            drop_calls: Arc::clone(&self.activation_drop_calls),
        })))
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.behavior
    }

    fn complete(
        &self,
        _export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseCompletionResult> {
        Ok(match completion {
            CompletionBinding::SynchronousByReturn => LeaseCompletionResult::FallbackSynchronized,
            CompletionBinding::SameDeviceStream { .. } => {
                LeaseCompletionResult::Completed(rapids_interop::CompletionWitness::with_backend(
                    CompletionWitnessKind::ConsumerStream,
                    Arc::clone(&self.witness_state) as Arc<dyn CompletionWitnessBackend>,
                ))
            },
            CompletionBinding::CrossDeviceEvent { .. } => {
                LeaseCompletionResult::Completed(rapids_interop::CompletionWitness::with_backend(
                    CompletionWitnessKind::ConsumerEvent,
                    Arc::clone(&self.witness_state) as Arc<dyn CompletionWitnessBackend>,
                ))
            },
        })
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        _completion: CompletionBinding,
    ) -> rapids_interop::Result<()> {
        self.drop_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn borrowed_column_view_rejects_list_type() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::List),
        0,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "list"));
}

#[test]
fn borrowed_column_view_rejects_struct_type() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Struct),
        0,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "struct"));
}

#[test]
fn borrowed_column_view_rejects_dictionary32_type() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Dictionary32),
        0,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "dictionary32"));
}

#[test]
fn borrowed_column_view_accepts_binary_like_string_semantic_graph() {
    let root = binary_like_column(1, 10, 11, 3, 1);
    let borrowed = BorrowedColumnView::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone()],
        vec![device_binding(10, 0x6000, 5 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        explicit_binding(),
    )
    .unwrap();

    assert_eq!(borrowed.type_id(), TypeId::String);
    assert_eq!(borrowed.element_size_bytes(), None);
    assert_eq!(borrowed.column(), Some(&root));
    assert_eq!(borrowed.semantic_columns(), &[root]);
}

#[test]
fn borrowed_column_view_rejects_missing_root_semantic_column() {
    let root = fixed_width_column(1, 10, 3, 0);
    let err = BorrowedColumnView::try_new_with_semantic_columns(
        root,
        vec![fixed_width_column(2, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
}

#[test]
fn borrowed_column_view_accepts_dictionary32_with_string_keys_semantic_graph() {
    let root = dictionary_column(1, 10, 2, 3, 0);
    let keys = binary_like_column(2, 11, 12, 2, 0);
    let borrowed = BorrowedColumnView::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), keys.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 9, 0),
        ],
        explicit_binding(),
    )
    .unwrap();

    assert_eq!(borrowed.type_id(), TypeId::Dictionary32);
    assert_eq!(borrowed.element_size_bytes(), None);
    assert_eq!(borrowed.column(), Some(&root));
    assert_eq!(borrowed.semantic_columns(), &[root, keys]);
}

#[test]
fn borrowed_column_view_accepts_list_semantic_graphs() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = fixed_width_column(2, 11, 3, 0);
    let borrowed = BorrowedColumnView::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), child.clone()],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        explicit_binding(),
    )
    .unwrap();

    assert_eq!(borrowed.type_id(), TypeId::List);
    assert_eq!(borrowed.element_size_bytes(), None);
    assert_eq!(borrowed.data_ptr(), ptr::null());
    assert_eq!(borrowed.column(), Some(&root));
    assert_eq!(borrowed.semantic_columns(), &[root, child]);
}

#[test]
fn borrowed_column_view_accepts_struct_semantic_graphs_with_nested_list_children() {
    let root = struct_column(1, &[2, 3], 2, 0);
    let nested_list = list_column(2, 10, 4, 2, 0);
    let label = binary_like_column(3, 11, 12, 2, 0);
    let leaf = fixed_width_column(4, 13, 3, 0);
    let borrowed = BorrowedColumnView::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), nested_list.clone(), label.clone(), leaf.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x7100, 9, 0),
            device_binding(13, 0x7200, 12, 0),
        ],
        explicit_binding(),
    )
    .unwrap();

    assert_eq!(borrowed.type_id(), TypeId::Struct);
    assert_eq!(borrowed.element_size_bytes(), None);
    assert_eq!(borrowed.data_ptr(), ptr::null());
    assert_eq!(borrowed.column(), Some(&root));
    assert_eq!(borrowed.semantic_columns(), &[root, nested_list, label, leaf]);
}

#[test]
fn borrowed_column_view_rejects_null_data_pointer_for_non_empty_view() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        3,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "len" && value == "3"));
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "int32"));
}

#[test]
fn borrowed_column_view_rejects_active_range_overflow() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        1,
        usize::MAX,
        fake_ptr(0x1000),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "offset" && value == &usize::MAX.to_string()));
    assert!(err.context().iter().any(|(key, value)| *key == "elem_size" && value == "4"));
}

#[test]
fn borrowed_column_view_rejects_misaligned_active_address() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        2,
        0,
        fake_ptr(0x1001),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "alignment" && value == "4"));
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "int32"));
}

#[test]
fn borrowed_column_view_rejects_null_count_exceeding_len() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        2,
        0,
        fake_ptr(0x1000),
        NullMask::present(fake_ptr(0x2000), 2, 3).unwrap(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "null_count" && value == "3"));
    assert!(err.context().iter().any(|(key, value)| *key == "len" && value == "2"));
}

#[test]
fn borrowed_column_view_rejects_undersized_null_mask() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        3,
        2,
        fake_ptr(0x1000),
        NullMask::present(fake_ptr(0x2000), 4, 1).unwrap(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "bit_len" && value == "4"));
    assert!(err.context().iter().any(|(key, value)| *key == "required_bits" && value == "5"));
}

#[test]
fn borrowed_column_view_rejects_missing_fixed_width_value_binding() {
    let root = fixed_width_column(1, 10, 3, 0);
    let err = BorrowedColumnView::try_new_with_semantic_columns(
        root,
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![],
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "10"));
}

#[test]
fn borrowed_column_view_rejects_missing_validity_binding() {
    let root = fixed_width_column_with_validity(
        1,
        10,
        3,
        0,
        PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
    );
    let err = BorrowedColumnView::try_new_with_semantic_columns(
        root,
        vec![fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        )],
        vec![device_binding(10, 0x6000, 12, 0)],
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_accepts_fixed_width_device_binding() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column().id(), ColumnId::new(1));
    assert_eq!(export.column().layout(), &ColumnLayout::FixedWidth { values: BufferId::new(10) });
    assert_eq!(export.bindings().len(), 1);
    assert_eq!(export.readiness(), &prepared_readiness());
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedDeviceExport);
}

#[test]
fn prepared_column_export_rejects_missing_root_semantic_column() {
    let root = fixed_width_column(1, 10, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![fixed_width_column(2, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
}

#[test]
fn prepared_column_export_rejects_missing_values_binding() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "10"));
}

#[test]
fn owned_column_transfer_rejects_missing_root_semantic_column() {
    let root = fixed_width_column(1, 10, 3, 0);
    let err = OwnedColumnTransfer::try_new_with_semantic_columns(
        root,
        vec![fixed_width_column(2, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        ExportDiagnostics::new(ExportProfile::OwnedDeviceTransfer),
        owned_release_contract(),
        (),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
}

#[test]
fn prepared_column_export_rejects_duplicate_buffer_ids() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0), device_binding(10, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "10"));
}

#[test]
fn prepared_column_export_rejects_validity_mask_binding_on_wrong_device() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        ),
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 1, 1)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_missing_validity_mask_binding() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        ),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_insufficient_value_buffer_range() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 1),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "required_bytes" && value == "16"));
}

#[test]
fn prepared_column_export_rejects_validity_bit_range_smaller_than_offset_plus_len() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column_with_validity(
            1,
            10,
            7,
            2,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        ),
        vec![device_binding(10, 0x6000, 36, 0), device_binding(11, 0x7000, 1, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "required_bits" && value == "9"));
}

#[test]
fn prepared_column_export_accepts_binary_like_device_bindings() {
    let export = PreparedColumnExport::try_new(
        binary_like_column(1, 10, 11, 3, 0),
        vec![device_binding(10, 0x6000, 16, 0), device_binding(11, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(
        export.column().layout(),
        &ColumnLayout::BinaryLike { offsets: BufferId::new(10), data: BufferId::new(11) }
    );
    assert_eq!(export.bindings().len(), 2);
}

#[test]
fn prepared_column_export_rejects_missing_binary_like_data_binding() {
    let err = PreparedColumnExport::try_new(
        binary_like_column(1, 10, 11, 3, 0),
        vec![device_binding(10, 0x6000, 16, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_insufficient_binary_like_offsets_range() {
    let err = PreparedColumnExport::try_new(
        binary_like_column(1, 10, 11, 3, 1),
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 32, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "required_bytes" && value == "20"));
}

#[test]
fn prepared_column_export_rejects_binary_like_data_binding_on_wrong_device() {
    let err = PreparedColumnExport::try_new(
        binary_like_column(1, 10, 11, 3, 0),
        vec![device_binding(10, 0x6000, 16, 0), device_binding(11, 0x7000, 12, 1)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_host_column_export_accepts_fixed_width_host_binding() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![host_binding(10, 0x6000, 12)],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column().id(), ColumnId::new(1));
    assert_eq!(export.bindings().len(), 1);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
}

#[test]
fn prepared_host_column_export_rejects_missing_validity_mask_binding() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        ),
        vec![host_binding(10, 0x6000, 12)],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_host_column_export_rejects_cuda_device_binding() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "placement" && value == "cuda_device"));
}

#[test]
fn prepared_host_column_export_rejects_device_stream_readiness() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![host_binding(10, 0x6000, 12)],
        prepared_readiness(),
        prepared_host_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "readiness" && value == "device_stream_ordered"));
}

#[test]
fn prepared_host_column_export_accepts_binary_like_host_bindings() {
    let export = PreparedColumnExport::try_new(
        binary_like_column(1, 10, 11, 2, 0),
        vec![host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()), host_binding(11, 0x7000, "alicebob".len())],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column().id(), ColumnId::new(1));
    assert!(matches!(export.column().layout(), ColumnLayout::BinaryLike { .. }));
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
}

#[test]
fn prepared_host_column_export_accepts_list_layout_with_fixed_width_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = fixed_width_column(2, 11, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), child.clone()],
        vec![host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()), host_binding(11, 0x7000, 12)],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child]);
}

#[test]
fn prepared_host_column_export_accepts_list_layout_with_binary_like_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = binary_like_column(2, 11, 12, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), child.clone()],
        vec![
            host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 8),
        ],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child]);
}

#[test]
fn prepared_host_column_export_accepts_struct_layout_with_current_supported_children() {
    let root = struct_column(1, &[2, 3, 4], 3, 0);
    let fixed = fixed_width_column(2, 10, 3, 0);
    let names = binary_like_column(3, 11, 12, 3, 0);
    let lists = list_column(4, 13, 5, 3, 0);
    let list_child = fixed_width_column(5, 14, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), fixed.clone(), names.clone(), lists.clone(), list_child.clone()],
        vec![
            host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 11),
            host_binding(13, 0x9000, 4 * std::mem::size_of::<i32>()),
            host_binding(14, 0xa000, 3 * std::mem::size_of::<i32>()),
        ],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, fixed, names, lists, list_child]);
}

#[test]
fn prepared_host_column_export_accepts_dictionary_layout_with_current_supported_keys() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let keys = binary_like_column(2, 11, 12, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), keys.clone()],
        vec![
            host_binding(10, 0x6000, 4 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 14),
        ],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, keys]);
}

#[test]
fn prepared_host_column_export_accepts_broader_recursive_struct_child_layouts() {
    let root = struct_column(1, &[2], 2, 0);
    let nested = struct_column(2, &[3], 2, 0);
    let leaf = fixed_width_column(3, 10, 2, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), nested.clone(), leaf.clone()],
        vec![host_binding(10, 0x6000, 2 * std::mem::size_of::<i32>())],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, nested, leaf]);
}

#[test]
fn prepared_table_export_accepts_uniform_device_and_readiness() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(
            TableId::new(1),
            3,
            vec![ColumnId::new(1), ColumnId::new(2)],
            Some(vec!["a".into(), "b".into()]),
        ),
        vec![fixed_width_column(1, 10, 3, 0), fixed_width_column(2, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.table().id(), TableId::new(1));
    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(export.table().columns(), &[ColumnId::new(1), ColumnId::new(2)]);
    assert_eq!(export.columns().len(), 2);
}

#[test]
fn prepared_table_export_rejects_row_count_mismatch() {
    let err = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 2, 0)],
        vec![device_binding(10, 0x6000, 8, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "column_len" && value == "2"));
}

#[test]
fn prepared_table_export_rejects_mixed_binding_devices() {
    let err = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1), ColumnId::new(2)], None),
        vec![fixed_width_column(1, 10, 3, 0), fixed_width_column(2, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 12, 1)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_table_export_rejects_missing_binding_for_column_buffer() {
    let err = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "10"));
}

#[test]
fn prepared_table_export_rejects_missing_column_id_in_table_schema() {
    let err = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1), ColumnId::new(2)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "2"));
}

#[test]
fn partitioned_prepared_table_export_accepts_two_partitions_on_different_devices() {
    let left = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3);
    let right = prepared_fixed_width_table_export_on_device(11, 2, 12, 0x7000, 1, 3);

    let export = PartitionedPreparedTableExport::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedPreparedTablePartition::new(vec![0], left),
            PartitionedPreparedTablePartition::new(vec![1], right),
        ],
    )
    .unwrap();

    assert_eq!(export.num_rows(), 3);
    assert_eq!(export.schema_names().unwrap(), &["left".to_string(), "right".to_string()]);
    assert_eq!(export.partitions().len(), 2);
    assert_eq!(export.partitions()[0].top_level_column_indices(), &[0]);
    assert_eq!(export.partitions()[1].top_level_column_indices(), &[1]);
    assert_eq!(export.partitions()[0].export().diagnostics().profile(), ExportProfile::PreparedDeviceExport);
    assert_eq!(export.partitions()[1].export().diagnostics().profile(), ExportProfile::PreparedDeviceExport);
}

#[test]
fn partitioned_prepared_table_partition_activates_directly_and_preserves_top_level_indices() {
    let left = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3);
    let right = prepared_fixed_width_table_export_on_device(11, 2, 12, 0x7000, 1, 3);

    let export = PartitionedPreparedTableExport::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedPreparedTablePartition::new(vec![0], left),
            PartitionedPreparedTablePartition::new(vec![1], right),
        ],
    )
    .unwrap();

    let lease = export.partitions()[0]
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(lease.top_level_column_indices(), &[0]);
    assert_eq!(lease.export().table().id(), TableId::new(10));
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn partitioned_prepared_table_partition_activation_failure_is_local_to_that_partition() {
    let left = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3);
    let right = prepared_fixed_width_table_export_on_device(11, 2, 12, 0x7000, 1, 3);

    let export = PartitionedPreparedTableExport::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedPreparedTablePartition::new(vec![0], left),
            PartitionedPreparedTablePartition::new(vec![1], right),
        ],
    )
    .unwrap();

    let err = export.partitions()[1]
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion_on_device(0))
        .unwrap_err();
    assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);

    let lease = export.partitions()[0]
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion_on_device(0))
        .unwrap();
    assert_eq!(
        lease.complete().unwrap(),
        LeaseCompletionResult::Completed(rapids_interop::CompletionWitness::new(CompletionWitnessKind::ConsumerStream))
    );
}

#[test]
fn partitioned_prepared_table_partition_drop_fallback_stays_local_to_that_partition() {
    let (left_backend, left_drop_calls, _left_witness_state) = TestLeaseBackend::guaranteed_drop_fallback();
    let (right_backend, right_drop_calls, _right_witness_state) = TestLeaseBackend::guaranteed_drop_fallback();
    let left = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3).with_lease_backend(left_backend);
    let right = prepared_fixed_width_table_export_on_device(11, 2, 12, 0x7000, 1, 3).with_lease_backend(right_backend);

    let export = PartitionedPreparedTableExport::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedPreparedTablePartition::new(vec![0], left),
            PartitionedPreparedTablePartition::new(vec![1], right),
        ],
    )
    .unwrap();

    {
        let _lease = export.partitions()[0]
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion_on_device(0))
            .unwrap();
    }

    assert_eq!(left_drop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(right_drop_calls.load(Ordering::SeqCst), 0);

    let lease = export.partitions()[1]
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn partitioned_prepared_table_export_rejects_row_count_mismatch() {
    let left = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3);
    let right = prepared_fixed_width_table_export_on_device(11, 2, 12, 0x7000, 1, 2);

    let err = PartitionedPreparedTableExport::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedPreparedTablePartition::new(vec![0], left),
            PartitionedPreparedTablePartition::new(vec![1], right),
        ],
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "partition_num_rows" && value == "2"));
}

#[test]
fn partitioned_prepared_table_export_rejects_duplicate_top_level_coverage() {
    let left = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3);
    let right = prepared_fixed_width_table_export_on_device(11, 2, 12, 0x7000, 1, 3);

    let err = PartitionedPreparedTableExport::try_new(
        Some(vec!["left".into(), "right".into()]),
        3,
        vec![
            PartitionedPreparedTablePartition::new(vec![0], left),
            PartitionedPreparedTablePartition::new(vec![0], right),
        ],
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "top_level_column_index" && value == "0"));
}

#[test]
fn partitioned_prepared_table_export_rejects_missing_top_level_coverage() {
    let left = prepared_fixed_width_table_export_on_device(10, 1, 11, 0x6000, 0, 3);
    let right = prepared_fixed_width_table_export_on_device(11, 2, 12, 0x7000, 1, 3);

    let err = PartitionedPreparedTableExport::try_new(
        Some(vec!["left".into(), "right".into(), "right_again".into()]),
        3,
        vec![
            PartitionedPreparedTablePartition::new(vec![0], left),
            PartitionedPreparedTablePartition::new(vec![2], right),
        ],
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "missing_top_level_column_index" && value == "1"));
}

#[test]
fn prepared_host_table_export_activates_with_synchronous_completion() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![host_binding(10, 0x6000, 12)],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn prepared_host_table_export_rejects_stream_completion_authority() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![host_binding(10, 0x6000, 12)],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    let err =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "completion" && value == "same_device_stream"));
}

#[test]
fn prepared_host_table_export_accepts_binary_like_and_fixed_width_columns() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(
            TableId::new(1),
            2,
            vec![ColumnId::new(1), ColumnId::new(2)],
            Some(vec!["name".into(), "id".into()]),
        ),
        vec![binary_like_column(1, 10, 11, 2, 0), fixed_width_column(2, 12, 2, 0)],
        vec![
            host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, "alicebob".len()),
            host_binding(12, 0x8000, 2 * std::mem::size_of::<i32>()),
        ],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.table().names().unwrap(), &["name".to_string(), "id".to_string()]);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::BinaryLike { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::FixedWidth { .. }));
}

#[test]
fn prepared_host_table_export_accepts_list_column_with_supported_child_projection() {
    let list = list_column(1, 10, 2, 2, 0);
    let list_child = binary_like_column(2, 11, 12, 3, 0);
    let ids = fixed_width_column(3, 13, 2, 0);
    let export = PreparedTableExport::try_new_with_semantic_columns(
        SemanticTable::new(TableId::new(1), 2, vec![list.id(), ids.id()], Some(vec!["lists".into(), "id".into()])),
        vec![list.clone(), ids.clone()],
        vec![list.clone(), list_child.clone(), ids.clone()],
        vec![
            host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 8),
            host_binding(13, 0x9000, 8),
        ],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.table().names().unwrap(), &["lists".to_string(), "id".to_string()]);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::List { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::FixedWidth { .. }));
    assert_eq!(export.semantic_columns(), &[list, list_child, ids]);
}

#[test]
fn prepared_host_table_export_accepts_struct_and_dictionary_columns_with_current_supported_projections() {
    let point = struct_column(1, &[2, 3], 3, 0);
    let point_x = fixed_width_column(2, 10, 3, 0);
    let point_name = binary_like_column(3, 11, 12, 3, 0);
    let category = dictionary_column(4, 13, 5, 3, 0);
    let category_keys = binary_like_column(5, 14, 15, 3, 0);
    let ids = fixed_width_column(6, 16, 3, 0);
    let export = PreparedTableExport::try_new_with_semantic_columns(
        SemanticTable::new(
            TableId::new(1),
            3,
            vec![point.id(), category.id(), ids.id()],
            Some(vec!["point".into(), "category".into(), "id".into()]),
        ),
        vec![point.clone(), category.clone(), ids.clone()],
        vec![point.clone(), point_x.clone(), point_name.clone(), category.clone(), category_keys.clone(), ids.clone()],
        vec![
            host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 11),
            host_binding(13, 0x9000, 3 * std::mem::size_of::<i32>()),
            host_binding(14, 0xa000, 4 * std::mem::size_of::<i32>()),
            host_binding(15, 0xb000, 14),
            host_binding(16, 0xc000, 3 * std::mem::size_of::<i32>()),
        ],
        ProducerReadiness::HostVisible,
        prepared_host_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.table().names().unwrap(), &["point".to_string(), "category".to_string(), "id".to_string()]);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::Struct { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::Dictionary { .. }));
    assert!(matches!(export.columns()[2].layout(), ColumnLayout::FixedWidth { .. }));
}

#[test]
fn prepared_column_export_rejects_null_count_exceeding_length() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(4) },
        ),
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 1, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "null_count" && value == "4"));
    assert!(err.context().iter().any(|(key, value)| *key == "len" && value == "3"));
}

#[test]
fn prepared_readiness_stream_handle_requires_explicit_device_identity() {
    let err = CudaStreamHandle::from_raw(fake_ptr(0x5000) as *mut core::ffi::c_void)
        .map(|stream| ProducerReadiness::DeviceStreamOrdered { stream })
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert!(err.context().iter().any(|(key, value)| *key == "field" && value == "cuda_stream.device"));
}

#[test]
fn prepared_column_export_rejects_readiness_stream_on_wrong_device() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness_on_device(1),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);
    assert!(err.context().iter().any(|(key, value)| *key == "producer_device" && value == "0"));
    assert!(err.context().iter().any(|(key, value)| *key == "stream_device" && value == "1"));
}

#[test]
fn prepared_column_export_accepts_list_layout_with_fixed_width_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = fixed_width_column(2, 11, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), child.clone()],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child]);
}

#[test]
fn prepared_column_export_accepts_list_layout_with_binary_like_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = binary_like_column(2, 11, 12, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), child.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 8, 0),
        ],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child]);
}

#[test]
fn prepared_column_export_rejects_fixed_width_layout_with_non_fixed_width_logical_type() {
    let err = PreparedColumnExport::try_new(
        SemanticColumn::new(
            ColumnId::new(1),
            DataType::new(TypeId::String),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            SchemaNullability::Unknown,
            ColumnLayout::FixedWidth { values: BufferId::new(10) },
        ),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "string"));
}

#[test]
fn prepared_column_export_rejects_binary_like_layout_with_non_string_logical_type() {
    let err = PreparedColumnExport::try_new(
        SemanticColumn::new(
            ColumnId::new(1),
            DataType::new(TypeId::Int32),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            SchemaNullability::Unknown,
            ColumnLayout::BinaryLike { offsets: BufferId::new(10), data: BufferId::new(11) },
        ),
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 8, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "int32"));
}

#[test]
fn prepared_column_export_accepts_struct_layout_with_supported_children() {
    let root = struct_column(1, &[2, 3], 3, 0);
    let left = fixed_width_column(2, 10, 3, 0);
    let right = binary_like_column(3, 11, 12, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), left.clone(), right.clone()],
        vec![
            device_binding(10, 0x6000, 12, 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 8, 0),
        ],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, left, right]);
}

#[test]
fn prepared_column_export_rejects_list_layout_with_non_list_logical_type() {
    let root = SemanticColumn::new(
        ColumnId::new(1),
        DataType::new(TypeId::String),
        2,
        0,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::List { offsets: BufferId::new(10), child: ColumnId::new(2) },
    );
    let child = fixed_width_column(2, 11, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![
            SemanticColumn::new(
                ColumnId::new(1),
                DataType::new(TypeId::String),
                2,
                0,
                PhysicalValidity::NoValidityBuffer,
                SchemaNullability::Unknown,
                ColumnLayout::List { offsets: BufferId::new(10), child: ColumnId::new(2) },
            ),
            child,
        ],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "string"));
}

#[test]
fn prepared_column_export_rejects_struct_layout_with_non_struct_logical_type() {
    let root = SemanticColumn::new(
        ColumnId::new(1),
        DataType::new(TypeId::List),
        3,
        0,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::Struct { children: vec![ColumnId::new(2)] },
    );
    let child = fixed_width_column(2, 10, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![
            SemanticColumn::new(
                ColumnId::new(1),
                DataType::new(TypeId::List),
                3,
                0,
                PhysicalValidity::NoValidityBuffer,
                SchemaNullability::Unknown,
                ColumnLayout::Struct { children: vec![ColumnId::new(2)] },
            ),
            child,
        ],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "list"));
}

#[test]
fn prepared_column_export_accepts_dictionary_layout_with_fixed_width_keys() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let keys = fixed_width_column(2, 11, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), keys.clone()],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, keys]);
}

#[test]
fn prepared_column_export_rejects_dictionary_layout_with_non_dictionary_logical_type() {
    let root = SemanticColumn::new(
        ColumnId::new(1),
        DataType::new(TypeId::Struct),
        4,
        0,
        PhysicalValidity::NoValidityBuffer,
        SchemaNullability::Unknown,
        ColumnLayout::Dictionary { indices: BufferId::new(10), keys: ColumnId::new(2) },
    );
    let keys = fixed_width_column(2, 11, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![
            SemanticColumn::new(
                ColumnId::new(1),
                DataType::new(TypeId::Struct),
                4,
                0,
                PhysicalValidity::NoValidityBuffer,
                SchemaNullability::Unknown,
                ColumnLayout::Dictionary { indices: BufferId::new(10), keys: ColumnId::new(2) },
            ),
            keys,
        ],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "struct"));
}

#[test]
fn prepared_column_export_accepts_dictionary_layout_with_binary_like_keys() {
    let root = dictionary_column(1, 10, 2, 3, 0);
    let keys = binary_like_column(2, 11, 12, 2, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), keys.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 8, 0),
        ],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, keys]);
}

#[test]
fn prepared_column_export_accepts_struct_layout_with_nested_list_child() {
    let root = struct_column(1, &[2, 3], 3, 0);
    let left = fixed_width_column(2, 10, 3, 0);
    let right = list_column(3, 11, 4, 3, 0);
    let list_child = binary_like_column(4, 12, 13, 4, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), left.clone(), right.clone(), list_child.clone()],
        vec![
            device_binding(10, 0x6000, 12, 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 5 * std::mem::size_of::<i32>(), 0),
            device_binding(13, 0x9000, 10, 0),
        ],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, left, right, list_child]);
}

#[test]
fn prepared_column_export_accepts_list_layout_with_nested_struct_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = struct_column(2, &[3, 4], 3, 0);
    let left = fixed_width_column(3, 11, 3, 0);
    let right = binary_like_column(4, 12, 13, 3, 0);
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        root.clone(),
        vec![root.clone(), child.clone(), left.clone(), right.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 12, 0),
            device_binding(12, 0x8000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(13, 0x9000, 8, 0),
        ],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child, left, right]);
}

#[test]
fn prepared_column_export_rejects_missing_list_child_semantic_column() {
    let root = list_column(1, 10, 2, 2, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![list_column(1, 10, 2, 2, 0)],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "2"));
}

#[test]
fn prepared_column_export_rejects_missing_struct_child_semantic_column() {
    let root = struct_column(1, &[2, 3], 3, 0);
    let left = fixed_width_column(2, 10, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![struct_column(1, &[2, 3], 3, 0), left],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "3"));
}

#[test]
fn prepared_column_export_rejects_missing_dictionary_keys_semantic_column() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![dictionary_column(1, 10, 2, 4, 0)],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "2"));
}

#[test]
fn prepared_column_export_rejects_list_child_on_wrong_device() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = fixed_width_column(2, 11, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![list_column(1, 10, 2, 2, 0), child],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 1)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "2"));
}

#[test]
fn prepared_column_export_rejects_struct_child_on_wrong_device() {
    let root = struct_column(1, &[2, 3], 3, 0);
    let left = fixed_width_column(2, 10, 3, 0);
    let right = binary_like_column(3, 11, 12, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![struct_column(1, &[2, 3], 3, 0), left, right],
        vec![
            device_binding(10, 0x6000, 12, 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 1),
            device_binding(12, 0x8000, 8, 1),
        ],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "3"));
}

#[test]
fn prepared_column_export_rejects_dictionary_keys_on_wrong_device() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let keys = fixed_width_column(2, 11, 3, 0);
    let err = PreparedColumnExport::try_new_with_semantic_columns(
        root,
        vec![dictionary_column(1, 10, 2, 4, 0), keys],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 1)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "2"));
}

#[test]
fn prepared_column_export_accepts_event_ordered_readiness() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_event_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    assert_eq!(export.readiness(), &prepared_event_readiness());
}

#[test]
fn prepared_readiness_event_handle_requires_explicit_device_identity() {
    let err = CudaEventHandle::from_raw(fake_ptr(0x5100) as *mut core::ffi::c_void)
        .map(|event| ProducerReadiness::DeviceEventOrdered { event })
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert!(err.context().iter().any(|(key, value)| *key == "field" && value == "cuda_event.device"));
}

#[test]
fn prepared_column_export_rejects_readiness_event_on_wrong_device() {
    let err = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_event_readiness_on_device(1),
        prepared_diagnostics(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);
    assert!(err.context().iter().any(|(key, value)| *key == "producer_device" && value == "0"));
    assert!(err.context().iter().any(|(key, value)| *key == "event_device" && value == "1"));
}

#[test]
fn prepared_column_export_activates_with_synchronous_completion() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(lease.export().column().id(), ColumnId::new(1));
    assert_eq!(lease.completion(), CompletionAuthority::SynchronousByReturn);
}

#[test]
fn prepared_column_export_allows_overlapping_active_leases() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let first = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    let second = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(first.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    assert_eq!(second.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn prepared_table_export_activates_with_same_device_stream_completion() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1), ColumnId::new(2)], None),
        vec![fixed_width_column(1, 10, 3, 0), fixed_width_column(2, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    assert_eq!(lease.export().table().id(), TableId::new(1));
    assert_eq!(lease.completion(), same_device_completion());
}

#[test]
fn prepared_table_export_clone_allows_overlapping_leases_and_reacquire_after_partial_drop() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();
    let cloned = export.clone();

    let first =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    let second =
        cloned.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    drop(first);

    let third =
        cloned.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    assert_eq!(second.completion(), same_device_completion());
    assert_eq!(third.completion(), same_device_completion());
}

#[test]
fn prepared_table_export_activates_with_cross_device_event_completion() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, cross_device_completion()).unwrap();

    assert_eq!(lease.export().table().id(), TableId::new(1));
    assert_eq!(lease.completion(), cross_device_completion());
}

#[test]
fn cross_device_completion_handle_requires_explicit_device_identity() {
    let err = CudaEventHandle::from_raw(fake_ptr(0x9000) as *mut core::ffi::c_void)
        .map(|event| CompletionAuthority::CrossDeviceEvent { event })
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert!(err.context().iter().any(|(key, value)| *key == "field" && value == "cuda_event.device"));
}

#[test]
fn prepared_table_export_rejects_cross_device_event_when_event_device_matches_producer() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let err = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, cross_device_completion_on_device(0))
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "event_device" && value == "0"));
    assert!(err.context().iter().any(|(key, value)| *key == "producer_device" && value == "0"));
}

#[test]
fn same_device_completion_handle_requires_explicit_device_identity() {
    let err = CudaStreamHandle::from_raw(fake_ptr(0x8000) as *mut core::ffi::c_void)
        .map(|stream| CompletionAuthority::SameDeviceStream { stream })
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert!(err.context().iter().any(|(key, value)| *key == "field" && value == "cuda_stream.device"));
}

#[test]
fn prepared_table_export_rejects_same_device_stream_completion_on_wrong_device() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let err = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion_on_device(1))
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);
    assert!(err.context().iter().any(|(key, value)| *key == "producer_device" && value == "0"));
    assert!(err.context().iter().any(|(key, value)| *key == "consumer_device" && value == "1"));
}

#[test]
fn column_lease_complete_returns_fallback_synchronized_for_synchronous_completion() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(lease.complete(), Ok(LeaseCompletionResult::FallbackSynchronized));
}

#[test]
fn table_lease_complete_returns_completed_witness_for_stream_completion() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    let result = lease.complete();
    assert_eq!(
        result,
        Ok(LeaseCompletionResult::Completed(rapids_interop::CompletionWitness::new(
            CompletionWitnessKind::ConsumerStream
        )))
    );
}

#[test]
fn table_lease_complete_returns_completed_witness_for_event_completion() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, cross_device_completion()).unwrap();

    let result = lease.complete();
    assert_eq!(
        result,
        Ok(LeaseCompletionResult::Completed(rapids_interop::CompletionWitness::new(
            CompletionWitnessKind::ConsumerEvent
        )))
    );
}

#[test]
fn table_lease_completed_witness_supports_ready_and_synchronize() {
    let (backend, _drop_calls, witness_state) = TestLeaseBackend::guaranteed_drop_fallback();
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap()
    .with_lease_backend(backend);

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    let witness = match lease.complete() {
        Ok(LeaseCompletionResult::Completed(witness)) => witness,
        Ok(LeaseCompletionResult::FallbackSynchronized) => {
            panic!("expected witness-backed completion for same-device stream completion")
        },
        Err(error) => panic!("unexpected completion error: {error}"),
    };

    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    assert!(!witness.is_ready().unwrap());
    witness.synchronize().unwrap();
    assert!(witness.is_ready().unwrap());
    assert_eq!(witness_state.sync_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_column_export_drop_fallback_releases_guard_and_invokes_backend() {
    let (backend, drop_calls, _witness_state) = TestLeaseBackend::guaranteed_drop_fallback();
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap()
    .with_lease_backend(backend);
    let cloned = export.clone();

    {
        let lease =
            export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
        assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);
    }

    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
    let reacquired = cloned
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(reacquired.complete(), Ok(LeaseCompletionResult::FallbackSynchronized));
}

#[test]
fn prepared_column_export_activation_invokes_backend_and_releases_activation_state_on_drop() {
    let (backend, activation_calls, activation_drop_calls, drop_calls) =
        TestLeaseBackend::guaranteed_drop_fallback_with_activation();
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap()
    .with_lease_backend(backend);

    {
        let lease =
            export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
        assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);
        assert_eq!(activation_calls.load(Ordering::SeqCst), 1);
        assert_eq!(activation_drop_calls.load(Ordering::SeqCst), 0);
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
    }

    assert_eq!(activation_drop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_column_export_activation_failure_releases_guard() {
    let (backend, activation_calls) = TestLeaseBackend::activation_failure();
    let (success_backend, _drop_calls, _witness_state) = TestLeaseBackend::guaranteed_drop_fallback();
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap()
    .with_lease_backend(backend);
    let cloned = export.clone().with_lease_backend(success_backend);

    let err =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert_eq!(activation_calls.load(Ordering::SeqCst), 1);

    let reacquired = cloned
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(reacquired.complete(), Ok(LeaseCompletionResult::FallbackSynchronized));
}

#[test]
fn prepared_column_export_explicit_completion_required_reports_behavior_without_drop_fallback() {
    let (backend, drop_calls) = TestLeaseBackend::explicit_completion_required();
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap()
    .with_lease_backend(backend);
    let cloned = export.clone();

    {
        let lease =
            export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
        assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);
    }

    assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
    let reacquired = cloned
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(reacquired.complete(), Ok(LeaseCompletionResult::FallbackSynchronized));
}

#[test]
fn prepared_column_export_builtin_explicit_completion_backend_is_runnable_without_drop_fallback() {
    let export = PreparedColumnExport::try_new(
        fixed_width_column(1, 10, 3, 0),
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_readiness(),
        prepared_diagnostics(),
    )
    .unwrap()
    .with_lease_backend(LeaseBackend::semantic_explicit_completion_required());
    let cloned = export.clone();

    {
        let lease =
            export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
        assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);
    }

    let reacquired =
        cloned.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    let witness = match reacquired.complete().unwrap() {
        LeaseCompletionResult::Completed(witness) => witness,
        LeaseCompletionResult::FallbackSynchronized => {
            panic!("expected built-in explicit backend to produce a completion witness")
        },
    };
    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    assert!(witness.is_ready().unwrap());
}

#[test]
fn prepared_table_export_with_event_readiness_activates_with_same_device_stream_completion() {
    let export = PreparedTableExport::try_new(
        SemanticTable::new(TableId::new(1), 3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        prepared_event_readiness(),
        prepared_diagnostics(),
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    assert_eq!(lease.completion(), same_device_completion());
}
