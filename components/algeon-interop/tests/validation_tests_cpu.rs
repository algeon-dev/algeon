use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout,
    CompletionAuthority, CompletionWitness, CompletionWitnessBackend, CudaEventHandle, CudaStreamHandle, DataType,
    InteropError, InteropErrorKind, LeaseActivation, LeaseBackend, LeaseCompletion, PhysicalValidity, Placement,
    PreparedAccess, PreparedColumnExport, PreparedExportProfile, PreparedTableExport, ProducerReadiness,
    ReadStartAuthority, Result as InteropResult, SemanticColumn, SemanticTable, TypeId,
};

fn fake_ptr(addr: usize) -> *const core::ffi::c_void {
    addr as *const core::ffi::c_void
}

fn prepared_readiness() -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceStreamOrdered {
        stream: unsafe { CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, 0) }.unwrap(),
    }
}

fn prepared_readiness_on_device(device: i32) -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceStreamOrdered {
        stream: unsafe { CudaStreamHandle::from_raw_on_device(fake_ptr(0x5000) as *mut core::ffi::c_void, device) }
            .unwrap(),
    }
}

fn prepared_event_readiness() -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceEventOrdered {
        event: unsafe { CudaEventHandle::from_raw_on_device(fake_ptr(0x5100) as *mut core::ffi::c_void, 0) }.unwrap(),
    }
}

fn prepared_event_readiness_on_device(device: i32) -> ProducerReadiness<'static> {
    ProducerReadiness::DeviceEventOrdered {
        event: unsafe { CudaEventHandle::from_raw_on_device(fake_ptr(0x5100) as *mut core::ffi::c_void, device) }
            .unwrap(),
    }
}

fn device_access(readiness: ProducerReadiness<'static>) -> PreparedAccess<'static> {
    PreparedAccess::Device { readiness, backend: TestLeaseBackend::with_witness().0 }
}

fn prepared_fixed_width_table_export_on_device(
    column_id: u32,
    buffer_id: u32,
    addr: usize,
    device: i32,
    num_rows: usize,
) -> PreparedTableExport<'static> {
    mock_table_export(
        SemanticTable::new(num_rows, vec![ColumnId::new(column_id)], None),
        vec![fixed_width_column(column_id, buffer_id, num_rows, 0)],
        vec![device_binding(buffer_id, addr, num_rows * std::mem::size_of::<i32>(), device)],
        device_access(prepared_readiness_on_device(device)),
    )
    .unwrap()
}

#[test]
fn prepared_table_export_lookup_helpers_resolve_top_level_names_ids_and_bindings() {
    let root_src = fixed_width_column(1, 11, 3, 0);
    let root_neighbors = list_column(2, 12, 3, 3, 0);
    let child = fixed_width_column(3, 13, 5, 0);
    let export = mock_table_export(
        SemanticTable::new(3, vec![root_src.id(), root_neighbors.id()], Some(vec!["src".into(), "neighbors".into()])),
        vec![root_src.clone(), root_neighbors.clone(), child.clone()],
        vec![
            device_binding(11, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x6100, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(13, 0x6200, 5 * std::mem::size_of::<i32>(), 0),
        ],
        device_access(prepared_readiness()),
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
    let export = prepared_fixed_width_table_export_on_device(1, 11, 0x6000, 0, 3);

    assert_eq!(export.top_level_column_by_name("src"), None);
}

fn same_device_completion() -> CompletionAuthority<'static> {
    CompletionAuthority::SameDeviceStream {
        stream: unsafe { CudaStreamHandle::from_raw_on_device(fake_ptr(0x8000) as *mut core::ffi::c_void, 0) }.unwrap(),
    }
}

fn same_device_completion_on_device(device: i32) -> CompletionAuthority<'static> {
    CompletionAuthority::SameDeviceStream {
        stream: unsafe { CudaStreamHandle::from_raw_on_device(fake_ptr(0x8000) as *mut core::ffi::c_void, device) }
            .unwrap(),
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

// These CPU fixtures exercise graph validation and lifecycle state. Their
// mock backends never pass the sentinel descriptors to a native reader.
fn mock_column_export<'a>(
    root: ColumnId,
    columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    access: PreparedAccess<'a>,
) -> InteropResult<PreparedColumnExport<'a>> {
    // SAFETY: no fixture consumer dereferences these descriptors.
    unsafe { PreparedColumnExport::try_new(root, columns, bindings, access) }
}

fn mock_table_export<'a>(
    table: SemanticTable,
    columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    access: PreparedAccess<'a>,
) -> InteropResult<PreparedTableExport<'a>> {
    // SAFETY: no fixture consumer dereferences these descriptors.
    unsafe { PreparedTableExport::try_new(table, columns, bindings, access) }
}

fn fixed_width_column(id: u32, values: u32, len: usize, offset: usize) -> SemanticColumn {
    SemanticColumn::new(
        ColumnId::new(id),
        DataType::new(TypeId::Int32),
        len,
        offset,
        PhysicalValidity::NoValidityBuffer,
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
        ColumnLayout::Dictionary { indices: BufferId::new(indices), keys: ColumnId::new(keys) },
    )
}

#[derive(Default)]
struct TestWitnessState {
    ready: AtomicBool,
    sync_calls: AtomicUsize,
    fail_sync: AtomicBool,
    quarantine_calls: AtomicUsize,
}

unsafe impl CompletionWitnessBackend for TestWitnessState {
    fn is_ready(&self) -> rapids_interop::Result<bool> {
        Ok(self.ready.load(Ordering::SeqCst))
    }

    fn synchronize(&self) -> rapids_interop::Result<()> {
        self.sync_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_sync.load(Ordering::SeqCst) {
            return Err(InteropError::new(InteropErrorKind::ReadinessUnavailable, "test witness failure"));
        }
        self.ready.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn quarantine(&self, _error: &InteropError) {
        self.quarantine_calls.fetch_add(1, Ordering::SeqCst);
    }
}

struct TestLeaseBackend {
    activation_calls: Arc<AtomicUsize>,
    activation_drop_calls: Arc<AtomicUsize>,
    fail_activation: bool,
    fail_completion: bool,
    fail_drop: bool,
    drop_calls: Arc<AtomicUsize>,
    witness_state: Arc<TestWitnessState>,
}

impl TestLeaseBackend {
    fn with_witness() -> (LeaseBackend, Arc<AtomicUsize>, Arc<TestWitnessState>) {
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let witness_state = Arc::new(TestWitnessState::default());
        (
            LeaseBackend::new(Self {
                activation_calls: Arc::new(AtomicUsize::new(0)),
                activation_drop_calls: Arc::new(AtomicUsize::new(0)),
                fail_activation: false,
                fail_completion: false,
                fail_drop: false,
                drop_calls: Arc::clone(&drop_calls),
                witness_state: Arc::clone(&witness_state),
            }),
            drop_calls,
            witness_state,
        )
    }

    fn with_activation() -> (LeaseBackend, Arc<AtomicUsize>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let activation_calls = Arc::new(AtomicUsize::new(0));
        let activation_drop_calls = Arc::new(AtomicUsize::new(0));
        let drop_calls = Arc::new(AtomicUsize::new(0));
        (
            LeaseBackend::new(Self {
                activation_calls: Arc::clone(&activation_calls),
                activation_drop_calls: Arc::clone(&activation_drop_calls),
                fail_activation: false,
                fail_completion: false,
                fail_drop: false,
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
                activation_calls: Arc::clone(&activation_calls),
                activation_drop_calls: Arc::new(AtomicUsize::new(0)),
                fail_activation: true,
                fail_completion: false,
                fail_drop: false,
                drop_calls: Arc::new(AtomicUsize::new(0)),
                witness_state: Arc::new(TestWitnessState::default()),
            }),
            activation_calls,
        )
    }

    fn completion_failure() -> (LeaseBackend, Arc<AtomicUsize>) {
        let drop_calls = Arc::new(AtomicUsize::new(0));
        (
            LeaseBackend::new(Self {
                activation_calls: Arc::new(AtomicUsize::new(0)),
                activation_drop_calls: Arc::new(AtomicUsize::new(0)),
                fail_activation: false,
                fail_completion: true,
                fail_drop: false,
                drop_calls: Arc::clone(&drop_calls),
                witness_state: Arc::new(TestWitnessState::default()),
            }),
            drop_calls,
        )
    }

    fn drop_failure(fail_completion: bool) -> (LeaseBackend, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let activation_drop_calls = Arc::new(AtomicUsize::new(0));
        (
            LeaseBackend::new(Self {
                activation_calls: Arc::default(),
                activation_drop_calls: Arc::clone(&activation_drop_calls),
                fail_activation: false,
                fail_completion,
                fail_drop: true,
                drop_calls: Arc::clone(&drop_calls),
                witness_state: Arc::new(TestWitnessState {
                    fail_sync: AtomicBool::new(true),
                    ..TestWitnessState::default()
                }),
            }),
            drop_calls,
            activation_drop_calls,
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

unsafe impl ActiveLeaseBackend for TestLeaseBackend {
    fn activate(
        &self,
        _readiness: ProducerReadiness<'_>,
        _read_start: ReadStartAuthority<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<LeaseActivation> {
        self.activation_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_activation {
            return Err(InteropError::new(InteropErrorKind::ReadinessUnavailable, "test activation failure"));
        }
        Ok(LeaseActivation::with_state(Arc::new(TestActivationState {
            drop_calls: Arc::clone(&self.activation_drop_calls),
        })))
    }

    fn complete(
        &self,
        _readiness: ProducerReadiness<'_>,
        completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<Option<CompletionWitness>> {
        if self.fail_completion {
            return Err(InteropError::new(InteropErrorKind::ReadinessUnavailable, "test completion failure"));
        }
        Ok(match completion {
            CompletionAuthority::SynchronousByReturn => None,
            CompletionAuthority::SameDeviceStream { .. } => {
                Some(CompletionWitness::new(Arc::clone(&self.witness_state) as Arc<dyn CompletionWitnessBackend>))
            },
        })
    }

    fn drop_incomplete(
        &self,
        _readiness: ProducerReadiness<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<()> {
        self.drop_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_drop {
            return Err(InteropError::new(InteropErrorKind::ReadinessUnavailable, "test fallback failure"));
        }
        Ok(())
    }
}

#[test]
fn prepared_column_export_accepts_fixed_width_device_binding() {
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap();

    assert_eq!(export.column().id(), ColumnId::new(1));
    assert_eq!(export.column().layout(), &ColumnLayout::FixedWidth { values: BufferId::new(10) });
    assert_eq!(export.bindings().len(), 1);
    assert!(matches!(export.access(), PreparedAccess::Device { readiness, .. } if *readiness == prepared_readiness()));
    assert_eq!(export.profile(), PreparedExportProfile::Device);
}

#[test]
fn prepared_column_export_rejects_missing_root_semantic_column() {
    let root = fixed_width_column(1, 10, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![fixed_width_column(2, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
}

#[test]
fn prepared_column_export_rejects_missing_values_binding() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "10"));
}

#[test]
fn prepared_column_export_rejects_duplicate_buffer_ids() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(10, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "10"));
}

#[test]
fn prepared_column_export_rejects_validity_mask_binding_on_wrong_device() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        )],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 1, 1)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_missing_validity_mask_binding() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        )],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_insufficient_value_buffer_range() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 1)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "required_bytes" && value == "16"));
}

#[test]
fn prepared_column_export_rejects_validity_bit_range_smaller_than_offset_plus_len() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column_with_validity(
            1,
            10,
            7,
            2,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        )],
        vec![device_binding(10, 0x6000, 36, 0), device_binding(11, 0x7000, 1, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "required_bits" && value == "9"));
}

#[test]
fn prepared_column_export_accepts_binary_like_device_bindings() {
    let export = mock_column_export(
        ColumnId::new(1),
        vec![binary_like_column(1, 10, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 16, 0), device_binding(11, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
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
    let err = mock_column_export(
        ColumnId::new(1),
        vec![binary_like_column(1, 10, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 16, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_insufficient_binary_like_offsets_range() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![binary_like_column(1, 10, 11, 3, 1)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 32, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "required_bytes" && value == "20"));
}

#[test]
fn prepared_column_export_rejects_binary_like_data_binding_on_wrong_device() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![binary_like_column(1, 10, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 16, 0), device_binding(11, 0x7000, 12, 1)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_host_column_export_accepts_fixed_width_host_binding() {
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![host_binding(10, 0x6000, 12)],
        PreparedAccess::Host,
    )
    .unwrap();

    assert_eq!(export.column().id(), ColumnId::new(1));
    assert_eq!(export.bindings().len(), 1);
    assert_eq!(export.profile(), PreparedExportProfile::Host);
}

#[test]
fn prepared_host_column_export_rejects_missing_validity_mask_binding() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(1) },
        )],
        vec![host_binding(10, 0x6000, 12)],
        PreparedAccess::Host,
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_host_column_export_rejects_cuda_device_binding() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        PreparedAccess::Host,
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "placement" && value == "cuda_device"));
}

#[test]
fn prepared_host_column_export_accepts_binary_like_host_bindings() {
    let export = mock_column_export(
        ColumnId::new(1),
        vec![binary_like_column(1, 10, 11, 2, 0)],
        vec![host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()), host_binding(11, 0x7000, "alicebob".len())],
        PreparedAccess::Host,
    )
    .unwrap();

    assert_eq!(export.column().id(), ColumnId::new(1));
    assert!(matches!(export.column().layout(), ColumnLayout::BinaryLike { .. }));
    assert_eq!(export.profile(), PreparedExportProfile::Host);
}

#[test]
fn prepared_host_column_export_accepts_list_layout_with_fixed_width_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = fixed_width_column(2, 11, 3, 0);
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), child.clone()],
        vec![host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()), host_binding(11, 0x7000, 12)],
        PreparedAccess::Host,
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child]);
}

#[test]
fn prepared_host_column_export_rejects_cyclic_graph() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![list_column(1, 10, 1, 2, 0)],
        vec![host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>())],
        PreparedAccess::Host,
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
}

#[test]
fn prepared_host_column_export_accepts_list_layout_with_binary_like_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = binary_like_column(2, 11, 12, 3, 0);
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), child.clone()],
        vec![
            host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 8),
        ],
        PreparedAccess::Host,
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
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), fixed.clone(), names.clone(), lists.clone(), list_child.clone()],
        vec![
            host_binding(10, 0x6000, 3 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 11),
            host_binding(13, 0x9000, 4 * std::mem::size_of::<i32>()),
            host_binding(14, 0xa000, 3 * std::mem::size_of::<i32>()),
        ],
        PreparedAccess::Host,
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, fixed, names, lists, list_child]);
}

#[test]
fn prepared_host_column_export_accepts_dictionary_layout_with_current_supported_keys() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let keys = binary_like_column(2, 11, 12, 3, 0);
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), keys.clone()],
        vec![
            host_binding(10, 0x6000, 4 * std::mem::size_of::<i32>()),
            host_binding(11, 0x7000, 4 * std::mem::size_of::<i32>()),
            host_binding(12, 0x8000, 14),
        ],
        PreparedAccess::Host,
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
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), nested.clone(), leaf.clone()],
        vec![host_binding(10, 0x6000, 2 * std::mem::size_of::<i32>())],
        PreparedAccess::Host,
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, nested, leaf]);
}

#[test]
fn prepared_table_export_accepts_uniform_device_and_readiness() {
    let export = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1), ColumnId::new(2)], Some(vec!["a".into(), "b".into()])),
        vec![fixed_width_column(1, 10, 3, 0), fixed_width_column(2, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap();

    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(export.table().columns(), &[ColumnId::new(1), ColumnId::new(2)]);
    assert_eq!(export.columns().len(), 2);
}

#[test]
fn prepared_table_export_rejects_row_count_mismatch() {
    let err = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 2, 0)],
        vec![device_binding(10, 0x6000, 8, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "column_len" && value == "2"));
}

#[test]
fn prepared_table_export_rejects_mixed_binding_devices() {
    let err = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1), ColumnId::new(2)], None),
        vec![fixed_width_column(1, 10, 3, 0), fixed_width_column(2, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 12, 1)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_table_export_rejects_missing_binding_for_column_buffer() {
    let err = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "10"));
}

#[test]
fn prepared_table_export_rejects_missing_column_id_in_table_schema() {
    let err = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1), ColumnId::new(2)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "2"));
}

#[test]
fn prepared_table_export_rejects_schema_column_missing_from_graph() {
    let err = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "1"));
}

#[test]
fn prepared_table_export_rejects_unreferenced_graph_column() {
    let err = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0), fixed_width_column(2, 99, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "2"));
}

#[test]
fn prepared_host_table_export_activates_with_synchronous_completion() {
    let export = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![host_binding(10, 0x6000, 12)],
        PreparedAccess::Host,
    )
    .unwrap();

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert!(lease.complete().unwrap().is_ready().unwrap());
}

#[test]
fn prepared_host_table_export_rejects_stream_completion_authority() {
    let export = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![host_binding(10, 0x6000, 12)],
        PreparedAccess::Host,
    )
    .unwrap();

    let err =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "completion" && value == "same_device_stream"));
}

#[test]
fn prepared_column_export_rejects_null_count_exceeding_length() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column_with_validity(
            1,
            10,
            3,
            0,
            PhysicalValidity::ValidityMask { mask: BufferId::new(11), null_count: Some(4) },
        )],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 1, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "null_count" && value == "4"));
    assert!(err.context().iter().any(|(key, value)| *key == "len" && value == "3"));
}

#[test]
fn prepared_column_export_rejects_readiness_stream_on_wrong_device() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness_on_device(1)),
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
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), child.clone()],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child]);
}

#[test]
fn prepared_column_export_accepts_list_layout_with_binary_like_child() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = binary_like_column(2, 11, 12, 3, 0);
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), child.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 8, 0),
        ],
        device_access(prepared_readiness()),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child]);
}

#[test]
fn prepared_column_export_rejects_fixed_width_layout_with_non_fixed_width_logical_type() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![SemanticColumn::new(
            ColumnId::new(1),
            DataType::new(TypeId::String),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            ColumnLayout::FixedWidth { values: BufferId::new(10) },
        )],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "string"));
}

#[test]
fn prepared_column_export_rejects_binary_like_layout_with_non_string_logical_type() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![SemanticColumn::new(
            ColumnId::new(1),
            DataType::new(TypeId::Int32),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            ColumnLayout::BinaryLike { offsets: BufferId::new(10), data: BufferId::new(11) },
        )],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 8, 0)],
        device_access(prepared_readiness()),
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
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), left.clone(), right.clone()],
        vec![
            device_binding(10, 0x6000, 12, 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 8, 0),
        ],
        device_access(prepared_readiness()),
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
        ColumnLayout::List { offsets: BufferId::new(10), child: ColumnId::new(2) },
    );
    let child = fixed_width_column(2, 11, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![
            SemanticColumn::new(
                ColumnId::new(1),
                DataType::new(TypeId::String),
                2,
                0,
                PhysicalValidity::NoValidityBuffer,
                ColumnLayout::List { offsets: BufferId::new(10), child: ColumnId::new(2) },
            ),
            child,
        ],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
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
        ColumnLayout::Struct { children: vec![ColumnId::new(2)] },
    );
    let child = fixed_width_column(2, 10, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![
            SemanticColumn::new(
                ColumnId::new(1),
                DataType::new(TypeId::List),
                3,
                0,
                PhysicalValidity::NoValidityBuffer,
                ColumnLayout::Struct { children: vec![ColumnId::new(2)] },
            ),
            child,
        ],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "list"));
}

#[test]
fn prepared_column_export_accepts_dictionary_layout_with_fixed_width_keys() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let keys = fixed_width_column(2, 11, 3, 0);
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), keys.clone()],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
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
        ColumnLayout::Dictionary { indices: BufferId::new(10), keys: ColumnId::new(2) },
    );
    let keys = fixed_width_column(2, 11, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![
            SemanticColumn::new(
                ColumnId::new(1),
                DataType::new(TypeId::Struct),
                4,
                0,
                PhysicalValidity::NoValidityBuffer,
                ColumnLayout::Dictionary { indices: BufferId::new(10), keys: ColumnId::new(2) },
            ),
            keys,
        ],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "struct"));
}

#[test]
fn prepared_column_export_accepts_dictionary_layout_with_binary_like_keys() {
    let root = dictionary_column(1, 10, 2, 3, 0);
    let keys = binary_like_column(2, 11, 12, 2, 0);
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), keys.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 8, 0),
        ],
        device_access(prepared_readiness()),
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
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), left.clone(), right.clone(), list_child.clone()],
        vec![
            device_binding(10, 0x6000, 12, 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(12, 0x8000, 5 * std::mem::size_of::<i32>(), 0),
            device_binding(13, 0x9000, 10, 0),
        ],
        device_access(prepared_readiness()),
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
    let export = mock_column_export(
        root.id(),
        vec![root.clone(), child.clone(), left.clone(), right.clone()],
        vec![
            device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0),
            device_binding(11, 0x7000, 12, 0),
            device_binding(12, 0x8000, 4 * std::mem::size_of::<i32>(), 0),
            device_binding(13, 0x9000, 8, 0),
        ],
        device_access(prepared_readiness()),
    )
    .unwrap();

    assert_eq!(export.column(), &root);
    assert_eq!(export.semantic_columns(), &[root, child, left, right]);
}

#[test]
fn prepared_column_export_rejects_missing_list_child_semantic_column() {
    let root = list_column(1, 10, 2, 2, 0);
    let err = mock_column_export(
        root.id(),
        vec![list_column(1, 10, 2, 2, 0)],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "2"));
}

#[test]
fn prepared_column_export_rejects_missing_struct_child_semantic_column() {
    let root = struct_column(1, &[2, 3], 3, 0);
    let left = fixed_width_column(2, 10, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![struct_column(1, &[2, 3], 3, 0), left],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "3"));
}

#[test]
fn prepared_column_export_rejects_missing_dictionary_keys_semantic_column() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let err = mock_column_export(
        root.id(),
        vec![dictionary_column(1, 10, 2, 4, 0)],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::NotFound);
    assert!(err.context().iter().any(|(key, value)| *key == "child_column_id" && value == "2"));
}

#[test]
fn prepared_column_export_rejects_list_child_on_wrong_device() {
    let root = list_column(1, 10, 2, 2, 0);
    let child = fixed_width_column(2, 11, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![list_column(1, 10, 2, 2, 0), child],
        vec![device_binding(10, 0x6000, 3 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 1)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "2"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_struct_child_on_wrong_device() {
    let root = struct_column(1, &[2, 3], 3, 0);
    let left = fixed_width_column(2, 10, 3, 0);
    let right = binary_like_column(3, 11, 12, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![struct_column(1, &[2, 3], 3, 0), left, right],
        vec![
            device_binding(10, 0x6000, 12, 0),
            device_binding(11, 0x7000, 4 * std::mem::size_of::<i32>(), 1),
            device_binding(12, 0x8000, 8, 1),
        ],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "3"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_rejects_dictionary_keys_on_wrong_device() {
    let root = dictionary_column(1, 10, 2, 4, 0);
    let keys = fixed_width_column(2, 11, 3, 0);
    let err = mock_column_export(
        root.id(),
        vec![dictionary_column(1, 10, 2, 4, 0), keys],
        vec![device_binding(10, 0x6000, 4 * std::mem::size_of::<i32>(), 0), device_binding(11, 0x7000, 12, 1)],
        device_access(prepared_readiness()),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    assert!(err.context().iter().any(|(key, value)| *key == "column_id" && value == "2"));
    assert!(err.context().iter().any(|(key, value)| *key == "buffer_id" && value == "11"));
}

#[test]
fn prepared_column_export_accepts_event_ordered_readiness() {
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_event_readiness()),
    )
    .unwrap();

    assert!(matches!(
        export.access(),
        PreparedAccess::Device { readiness, .. } if *readiness == prepared_event_readiness()
    ));
}

#[test]
fn prepared_column_export_rejects_readiness_event_on_wrong_device() {
    let err = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_event_readiness_on_device(1)),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);
    assert!(err.context().iter().any(|(key, value)| *key == "producer_device" && value == "0"));
    assert!(err.context().iter().any(|(key, value)| *key == "event_device" && value == "1"));
}

#[test]
fn prepared_column_export_activates_with_synchronous_completion() {
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
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
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap();

    let first = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    let second = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert!(first.complete().unwrap().is_ready().unwrap());
    assert!(second.complete().unwrap().is_ready().unwrap());
}

#[test]
fn prepared_table_export_activates_with_same_device_stream_completion() {
    let export = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1), ColumnId::new(2)], None),
        vec![fixed_width_column(1, 10, 3, 0), fixed_width_column(2, 11, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0), device_binding(11, 0x7000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    assert_eq!(lease.completion(), same_device_completion());
}

#[test]
fn prepared_table_export_clone_allows_overlapping_leases_and_reacquire_after_partial_drop() {
    let export = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
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
fn prepared_table_export_rejects_same_device_stream_completion_on_wrong_device() {
    let export = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
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
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_readiness()),
    )
    .unwrap();

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert!(lease.complete().unwrap().is_ready().unwrap());
}

fn device_column_export_with_backend(backend: LeaseBackend) -> PreparedColumnExport<'static> {
    mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        PreparedAccess::Device { readiness: prepared_readiness(), backend },
    )
    .unwrap()
}

#[test]
fn additional_retained_owner_preserves_live_host_storage() {
    let owner = Arc::new(vec![1i32, 2, 3]);
    let observed = Arc::downgrade(&owner);
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![host_binding(10, owner.as_ptr() as usize, 12)],
        PreparedAccess::Host,
    )
    .unwrap();
    let export = unsafe { export.with_retained_owner(owner).with_retained_owner(Arc::new(())) };
    assert_eq!(observed.upgrade().unwrap().as_slice(), &[1, 2, 3]);
    let clone = export.clone();
    drop(export);
    assert!(observed.upgrade().is_some());
    drop(clone);
    assert!(observed.upgrade().is_none());
}

#[test]
fn incomplete_fallback_failure_keeps_owner() {
    let (backend, drop_calls, activation_drop_calls) = TestLeaseBackend::drop_failure(false);
    let owner = Arc::new(vec![1i32, 2, 3]);
    let observed = Arc::downgrade(&owner);
    let export = unsafe { device_column_export_with_backend(backend).with_retained_owner(owner) };
    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    drop(lease);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
    drop(export);
    assert!(observed.upgrade().is_some());
    assert_eq!(
        activation_drop_calls.load(Ordering::SeqCst),
        0,
        "failed cleanup must retain backend activation resources"
    );
}

#[test]
fn backend_completion_and_fallback_failure_keep_owner() {
    let (backend, drop_calls, _) = TestLeaseBackend::drop_failure(true);
    let owner = Arc::new(vec![1i32, 2, 3]);
    let observed = Arc::downgrade(&owner);
    let export = unsafe { device_column_export_with_backend(backend).with_retained_owner(owner) };
    let error = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion())
        .unwrap()
        .complete()
        .unwrap_err();
    assert_eq!(error.kind(), InteropErrorKind::ReadinessUnavailable);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
    drop(export);
    assert_eq!(observed.upgrade().unwrap().as_slice(), &[1, 2, 3]);
}

#[test]
fn additional_table_owner_survives_clones_and_pending_completion() {
    let (backend, _, witness) = TestLeaseBackend::with_witness();
    let owner = Arc::new(vec![1i32, 2, 3]);
    let observed = Arc::downgrade(&owner);
    // SAFETY: this backend models device reads without dereferencing storage.
    let export = unsafe {
        PreparedTableExport::try_new(
            SemanticTable::new(3, vec![ColumnId::new(1)], None),
            vec![fixed_width_column(1, 10, 3, 0)],
            vec![device_binding(10, owner.as_ptr() as usize, 12, 0)],
            PreparedAccess::Device { readiness: prepared_readiness(), backend },
        )
        .unwrap()
        .with_retained_owner(owner)
        .with_retained_owner(Arc::new(()))
    };
    let clone = export.clone();
    let mut completion = clone
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion())
        .unwrap()
        .complete()
        .unwrap();
    drop(export);
    drop(clone);
    assert_eq!(observed.upgrade().unwrap().as_slice(), &[1, 2, 3]);
    assert_eq!(witness.sync_calls.load(Ordering::SeqCst), 0);
    completion.synchronize().unwrap();
    assert!(observed.upgrade().is_none());
}

#[test]
fn forgotten_active_lease_retains_storage_without_destructor() {
    let (backend, drop_calls, _) = TestLeaseBackend::with_witness();
    let owner = Arc::new(vec![1i32, 2, 3]);
    let observed = Arc::downgrade(&owner);
    let export = unsafe { device_column_export_with_backend(backend).with_retained_owner(owner) };
    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    std::mem::forget(lease);
    drop(export);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observed.upgrade().unwrap().as_slice(), &[1, 2, 3]);
}

#[test]
fn borrowed_witness_sync_failure_runs_incomplete_fallback() {
    let (backend, drop_calls, witness) = TestLeaseBackend::with_witness();
    witness.fail_sync.store(true, Ordering::SeqCst);
    let export = device_column_export_with_backend(backend);
    let error = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion())
        .unwrap()
        .complete()
        .unwrap_err();
    assert_eq!(error.kind(), InteropErrorKind::ReadinessUnavailable);
    assert_eq!(witness.sync_calls.load(Ordering::SeqCst), 1);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn borrowed_cleanup_failure_aborts_before_storage_can_be_released() {
    use std::{env, os::unix::process::ExitStatusExt, process::Command};

    const CHILD_MODE: &str = "ALGEON_INTEROP_BORROWED_FAILURE";
    if let Ok(mode) = env::var(CHILD_MODE) {
        let (backend, _, _) = TestLeaseBackend::drop_failure(false);
        let export = device_column_export_with_backend(backend);
        let lease =
            export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
        match mode.as_str() {
            "drop" => drop(lease),
            "witness" => {
                // Exercise backend success followed by a failed borrowed wait
                // and a failed fallback, with no owner to quarantine.
                let _ = lease.complete();
            },
            _ => panic!("unexpected child mode"),
        }
        panic!("cleanup returned without proving borrowed storage is safe");
    }
    for mode in ["drop", "witness"] {
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", "borrowed_cleanup_failure_aborts_before_storage_can_be_released", "--nocapture"])
            .env(CHILD_MODE, mode)
            .output()
            .unwrap();
        assert_eq!(output.status.signal(), Some(6), "{mode}: {:?}", output);
    }
}

/// Completes one stream-ordered lease over an export that retains `owner`,
/// then drops the export so only the completion can keep `owner` alive.
fn complete_with_retained_owner(backend: LeaseBackend, owner: Arc<()>) -> LeaseCompletion {
    let export = unsafe { device_column_export_with_backend(backend).with_retained_owner(owner) };
    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    lease.complete().unwrap()
}

#[test]
fn lease_completion_without_retained_owner_synchronizes_before_return() {
    let (backend, _drop_calls, witness_state) = TestLeaseBackend::with_witness();
    let export = device_column_export_with_backend(backend);
    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    let completion = lease.complete().unwrap();

    assert_eq!(witness_state.sync_calls.load(Ordering::SeqCst), 1);
    assert!(completion.is_ready().unwrap());
    drop(completion);
    assert_eq!(witness_state.sync_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn lease_completion_holds_retained_owner_until_synchronized() {
    let (backend, _drop_calls, witness_state) = TestLeaseBackend::with_witness();
    let owner = Arc::new(());
    let observed = Arc::downgrade(&owner);

    let mut completion = complete_with_retained_owner(backend, owner);

    assert!(observed.upgrade().is_some());
    assert_eq!(witness_state.sync_calls.load(Ordering::SeqCst), 0);
    assert!(!completion.is_ready().unwrap());
    completion.synchronize().unwrap();
    assert_eq!(witness_state.sync_calls.load(Ordering::SeqCst), 1);
    assert!(observed.upgrade().is_none());
    assert!(completion.is_ready().unwrap());
}

#[test]
fn lease_completion_drop_keeps_owner_after_failed_fence() {
    let (backend, _drop_calls, witness_state) = TestLeaseBackend::with_witness();
    witness_state.fail_sync.store(true, Ordering::SeqCst);
    let owner = Arc::new(());
    let observed = Arc::downgrade(&owner);

    let mut completion = complete_with_retained_owner(backend, owner);

    assert_eq!(completion.synchronize().unwrap_err().kind(), InteropErrorKind::ReadinessUnavailable);
    assert!(observed.upgrade().is_some());
    assert_eq!(witness_state.quarantine_calls.load(Ordering::SeqCst), 0);
    drop(completion);
    assert_eq!(witness_state.quarantine_calls.load(Ordering::SeqCst), 1);
    assert_eq!(witness_state.sync_calls.load(Ordering::SeqCst), 2);
    assert!(observed.upgrade().is_some());
    let observed_witness = Arc::downgrade(&witness_state);
    drop(witness_state);
    assert!(observed_witness.upgrade().is_some(), "failed cleanup must retain the completion witness");
}

#[test]
fn lease_completion_release_stream_ordered_drops_owner_without_waiting() {
    let (backend, _drop_calls, witness_state) = TestLeaseBackend::with_witness();
    let owner = Arc::new(());
    let observed = Arc::downgrade(&owner);

    // SAFETY: the mock backend submits no reads.
    unsafe { complete_with_retained_owner(backend, owner).release_stream_ordered() };

    assert!(observed.upgrade().is_none());
    assert_eq!(witness_state.sync_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn prepared_column_export_drop_fallback_releases_guard_and_invokes_backend() {
    let (backend, drop_calls, _witness_state) = TestLeaseBackend::with_witness();
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        PreparedAccess::Device { readiness: prepared_readiness(), backend },
    )
    .unwrap();
    let cloned = export.clone();

    {
        let _lease =
            export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    }

    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
    let reacquired = cloned
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert!(reacquired.complete().unwrap().is_ready().unwrap());
}

#[test]
fn prepared_column_export_activation_invokes_backend_and_releases_activation_state_on_drop() {
    let (backend, activation_calls, activation_drop_calls, drop_calls) = TestLeaseBackend::with_activation();
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        PreparedAccess::Device { readiness: prepared_readiness(), backend },
    )
    .unwrap();

    {
        let _lease =
            export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
        assert_eq!(activation_calls.load(Ordering::SeqCst), 1);
        assert_eq!(activation_drop_calls.load(Ordering::SeqCst), 0);
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
    }

    assert_eq!(activation_drop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_column_export_activation_failure_returns_backend_error() {
    let (backend, activation_calls) = TestLeaseBackend::activation_failure();
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        PreparedAccess::Device { readiness: prepared_readiness(), backend },
    )
    .unwrap();

    let err =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert_eq!(activation_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_column_export_completion_failure_runs_drop_fallback() {
    let (backend, drop_calls) = TestLeaseBackend::completion_failure();
    let export = mock_column_export(
        ColumnId::new(1),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        PreparedAccess::Device { readiness: prepared_readiness(), backend },
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();
    let err = lease.complete().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
    assert_eq!(drop_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_table_export_with_event_readiness_activates_with_same_device_stream_completion() {
    let export = mock_table_export(
        SemanticTable::new(3, vec![ColumnId::new(1)], None),
        vec![fixed_width_column(1, 10, 3, 0)],
        vec![device_binding(10, 0x6000, 12, 0)],
        device_access(prepared_event_readiness()),
    )
    .unwrap();

    let lease =
        export.try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, same_device_completion()).unwrap();

    assert_eq!(lease.completion(), same_device_completion());
}
