use std::{
    any::Any,
    ffi::{CStr, CString},
    mem::ManuallyDrop,
    os::raw::{c_char, c_void},
    ptr::{self, NonNull},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    DataFrame, Series,
    core::{
        Column, Table,
        error::{self, Error, ErrorKind, Result},
        types::TypeId,
    },
    ops::dictionary,
    runtime::Runtime,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceArrowExportMaterialization {
    ProducerRetainedView,
    BridgeOwnedDeviceCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceArrowImportMaterialization {
    ForeignRetainedView,
    BridgeOwnedDeviceCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceArrowDeviceType {
    Cuda,
    CudaHost,
    CudaManaged,
    Other(i32),
}

impl DeviceArrowDeviceType {
    fn from_raw(raw: i32) -> Self {
        match raw {
            cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA => Self::Cuda,
            cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_HOST => Self::CudaHost,
            cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_MANAGED => Self::CudaManaged,
            _ => Self::Other(raw),
        }
    }

    fn raw_code(self) -> i32 {
        match self {
            Self::Cuda => cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA,
            Self::CudaHost => cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_HOST,
            Self::CudaManaged => cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_MANAGED,
            Self::Other(raw) => raw,
        }
    }
}

fn export_materialization_from_raw(
    raw: cudf_sys::cudf_arrow_device_export_materialization_t,
) -> DeviceArrowExportMaterialization {
    match raw {
        cudf_sys::cudf_arrow_device_export_materialization_t::CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW => {
            DeviceArrowExportMaterialization::ProducerRetainedView
        },
        cudf_sys::cudf_arrow_device_export_materialization_t::CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY => {
            DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy
        },
    }
}

fn import_materialization_from_raw(
    raw: cudf_sys::cudf_arrow_device_import_materialization_t,
) -> DeviceArrowImportMaterialization {
    match raw {
        cudf_sys::cudf_arrow_device_import_materialization_t::CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW => {
            DeviceArrowImportMaterialization::ForeignRetainedView
        },
        cudf_sys::cudf_arrow_device_import_materialization_t::CUDF_ARROW_DEVICE_IMPORT_BRIDGE_OWNED_DEVICE_COPY => {
            DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy
        },
    }
}

fn foreign_import_materialization_for_column_view(
    view: &crate::core::column::ColumnView<'_>,
) -> Result<DeviceArrowImportMaterialization> {
    match view.data_type()?.id() {
        TypeId::Bool8 => Ok(DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy),
        TypeId::List => foreign_import_materialization_for_column_view(&view.list_sliced_child()?),
        TypeId::Struct => {
            for index in 0..view.num_children()? {
                if foreign_import_materialization_for_column_view(&view.child(index)?)?
                    == DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy
                {
                    return Ok(DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);
                }
            }
            Ok(DeviceArrowImportMaterialization::ForeignRetainedView)
        },
        TypeId::Dictionary32 => foreign_import_materialization_for_column_view(&dictionary::keys(view)?),
        _ => Ok(DeviceArrowImportMaterialization::ForeignRetainedView),
    }
}

fn foreign_import_materialization_for_table_view(
    view: &crate::core::table::TableView<'_>,
) -> Result<DeviceArrowImportMaterialization> {
    for index in 0..view.num_columns()? {
        if foreign_import_materialization_for_column_view(&view.column(index)?)?
            == DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy
        {
            return Ok(DeviceArrowImportMaterialization::BridgeOwnedDeviceCopy);
        }
    }
    Ok(DeviceArrowImportMaterialization::ForeignRetainedView)
}

fn validate_same_device(input_device: i32, runtime: &Runtime, operation: &'static str) -> Result<()> {
    let runtime_device = runtime.device().ordinal() as i32;
    if input_device != runtime_device {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "device-arrow import requires the foreign input and destination runtime to use the same CUDA device",
        )
        .with_operation(operation)
        .with_context("input_device", input_device.to_string())
        .with_context("runtime_device", runtime_device.to_string()));
    }
    Ok(())
}

fn validate_supported_device_type(input_type: DeviceArrowDeviceType, operation: &'static str) -> Result<()> {
    if !matches!(
        input_type,
        DeviceArrowDeviceType::Cuda | DeviceArrowDeviceType::CudaHost | DeviceArrowDeviceType::CudaManaged
    ) {
        return Err(Error::new(
            ErrorKind::InvalidArgument,
            "device-arrow import currently accepts only ARROW_DEVICE_CUDA, ARROW_DEVICE_CUDA_HOST, or ARROW_DEVICE_CUDA_MANAGED roots",
        )
        .with_operation(operation)
        .with_context("device_type", input_type.raw_code().to_string()));
    }
    Ok(())
}

#[repr(C)]
struct ArrowSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut ArrowSchema,
    dictionary: *mut ArrowSchema,
    release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    private_data: *mut c_void,
}

#[repr(C)]
struct ArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut ArrowArray,
    dictionary: *mut ArrowArray,
    release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    private_data: *mut c_void,
}

#[repr(C)]
struct ArrowDeviceArray {
    array: ArrowArray,
    device_id: i64,
    device_type: i32,
    sync_event: *mut c_void,
    reserved: [i64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceArrowAbiOwnership {
    DestroyShim,
    ArrowRelease,
}

struct DeviceArrowAbiTransferState {
    remaining_releases: AtomicUsize,
    schema_release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    schema_private_data: *mut c_void,
    array_release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    array_private_data: *mut c_void,
    _retained_owner: Option<Arc<dyn Any + Send + Sync>>,
}

unsafe impl Send for DeviceArrowAbiTransferState {}
unsafe impl Sync for DeviceArrowAbiTransferState {}

unsafe fn release_arrow_schema_handle(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }
    if let Some(release) = (*schema).release {
        release(schema);
    }
}

unsafe fn release_arrow_device_array_handle(array: *mut ArrowDeviceArray) {
    if array.is_null() {
        return;
    }
    if let Some(release) = (*array).array.release {
        release(ptr::addr_of_mut!((*array).array));
    }
}

unsafe fn finish_transfer_state(state: *mut DeviceArrowAbiTransferState) {
    if state.is_null() {
        return;
    }
    if (*state).remaining_releases.fetch_sub(1, Ordering::AcqRel) == 1 {
        drop(Box::from_raw(state));
    }
}

unsafe extern "C" fn release_owned_arrow_schema(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }

    let state = (*schema).private_data.cast::<DeviceArrowAbiTransferState>();
    if !state.is_null() {
        (*schema).release = (*state).schema_release;
        (*schema).private_data = (*state).schema_private_data;
    }
    cudf_sys::cudf_arrow_device_schema_destroy(schema.cast());
    finish_transfer_state(state);
}

unsafe extern "C" fn release_owned_arrow_array(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }

    let state = (*array).private_data.cast::<DeviceArrowAbiTransferState>();
    if !state.is_null() {
        (*array).release = (*state).array_release;
        (*array).private_data = (*state).array_private_data;
    }
    cudf_sys::cudf_arrow_device_array_destroy(array.cast());
    finish_transfer_state(state);
}

fn validate_raw_abi_parts(
    schema: *mut c_void,
    array: *mut c_void,
    operation: &'static str,
) -> Result<(NonNull<c_void>, NonNull<c_void>)> {
    let schema = NonNull::new(schema).ok_or_else(|| {
        Error::new(ErrorKind::InvalidArgument, "Arrow device schema pointer must not be null").with_operation(operation)
    })?;
    let array = NonNull::new(array).ok_or_else(|| {
        Error::new(ErrorKind::InvalidArgument, "Arrow device array pointer must not be null").with_operation(operation)
    })?;
    Ok((schema, array))
}

struct DeviceArrowAbiRootInner {
    schema: NonNull<c_void>,
    array: NonNull<c_void>,
    materialization: DeviceArrowImportMaterialization,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    ownership: DeviceArrowAbiOwnership,
}

impl Drop for DeviceArrowAbiRootInner {
    fn drop(&mut self) {
        unsafe {
            match self.ownership {
                DeviceArrowAbiOwnership::DestroyShim => {
                    cudf_sys::cudf_arrow_device_schema_destroy(self.schema.as_ptr());
                    cudf_sys::cudf_arrow_device_array_destroy(self.array.as_ptr());
                },
                DeviceArrowAbiOwnership::ArrowRelease => {
                    release_arrow_schema_handle(self.schema.as_ptr().cast());
                    release_arrow_device_array_handle(self.array.as_ptr().cast());
                },
            }
        }
    }
}

// SAFETY: The wrapper owns immutable raw Arrow ABI handles plus an optional
// retained Rust owner token. It does not expose interior mutability and only
// destroys the raw handles once on drop.
unsafe impl Send for DeviceArrowAbiRootInner {}
unsafe impl Sync for DeviceArrowAbiRootInner {}

impl DeviceArrowAbiRootInner {
    fn into_owned_abi(mut self) -> Self {
        if self.ownership == DeviceArrowAbiOwnership::ArrowRelease {
            return self;
        }

        unsafe {
            let schema = self.schema.as_ptr().cast::<ArrowSchema>();
            let array = self.array.as_ptr().cast::<ArrowDeviceArray>();
            let transfer_state = Box::into_raw(Box::new(DeviceArrowAbiTransferState {
                remaining_releases: AtomicUsize::new(2),
                schema_release: (*schema).release,
                schema_private_data: (*schema).private_data,
                array_release: (*array).array.release,
                array_private_data: (*array).array.private_data,
                _retained_owner: self.retained_owner.take(),
            }));

            (*schema).release = Some(release_owned_arrow_schema);
            (*schema).private_data = transfer_state.cast();
            (*array).array.release = Some(release_owned_arrow_array);
            (*array).array.private_data = transfer_state.cast();
        }

        self.ownership = DeviceArrowAbiOwnership::ArrowRelease;
        self
    }

    fn device_id(&self, operation: &'static str) -> i32 {
        self.try_device_id(operation).unwrap_or(-1)
    }

    fn try_device_id(&self, operation: &'static str) -> Result<i32> {
        let mut out = -1_i64;
        error::check_status_in(
            unsafe { cudf_sys::cudf_arrow_device_array_device_id(self.array.as_ptr(), &mut out) },
            operation,
        )?;
        i32::try_from(out).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "Arrow device id exceeds i32::MAX")
                .with_operation(operation)
                .with_context("device_id", out.to_string())
        })
    }

    fn device_type(&self, operation: &'static str) -> DeviceArrowDeviceType {
        self.try_device_type(operation).unwrap_or_else(|_| DeviceArrowDeviceType::from_raw(-1))
    }

    fn try_device_type(&self, operation: &'static str) -> Result<DeviceArrowDeviceType> {
        let mut out = -1_i32;
        error::check_status_in(
            unsafe { cudf_sys::cudf_arrow_device_array_device_type(self.array.as_ptr(), &mut out) },
            operation,
        )?;
        Ok(DeviceArrowDeviceType::from_raw(out))
    }

    fn has_sync_event(&self, operation: &'static str) -> bool {
        self.try_has_sync_event(operation).unwrap_or(false)
    }

    fn try_has_sync_event(&self, operation: &'static str) -> Result<bool> {
        let mut out = 0_i32;
        error::check_status_in(
            unsafe { cudf_sys::cudf_arrow_device_array_has_sync_event(self.array.as_ptr(), &mut out) },
            operation,
        )?;
        Ok(out != 0)
    }

    fn sync_event_raw_ptr(&self, operation: &'static str) -> *mut c_void {
        self.try_sync_event_raw_ptr(operation).unwrap_or(ptr::null_mut())
    }

    fn try_sync_event_raw_ptr(&self, operation: &'static str) -> Result<*mut c_void> {
        let mut out = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_arrow_device_array_sync_event_raw(self.array.as_ptr(), &mut out) },
            operation,
        )?;
        Ok(out)
    }

    fn fixed_width_values_ptr(&self, operation: &'static str) -> Result<*const c_void> {
        let mut out = ptr::null();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_arrow_device_column_fixed_width_values_ptr(
                    self.schema.as_ptr(),
                    self.array.as_ptr(),
                    &mut out,
                )
            },
            operation,
        )?;
        Ok(out)
    }

    fn table_fixed_width_values_ptr(&self, index: usize, operation: &'static str) -> Result<*const c_void> {
        let index = i64::try_from(index).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "table column index exceeds i64::MAX")
                .with_operation(operation)
                .with_context("column_index", index.to_string())
        })?;

        let mut out = ptr::null();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_arrow_device_table_fixed_width_values_ptr(
                    self.schema.as_ptr(),
                    self.array.as_ptr(),
                    index,
                    &mut out,
                )
            },
            operation,
        )?;
        Ok(out)
    }

    fn table_column_names(&self, operation: &'static str) -> Result<Vec<String>> {
        let mut out = 0_i64;
        error::check_status_in(
            unsafe { cudf_sys::cudf_arrow_device_table_num_columns(self.schema.as_ptr(), &mut out) },
            operation,
        )?;
        let count = usize::try_from(out).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "Arrow table column count must be >= 0")
                .with_operation(operation)
                .with_context("num_columns", out.to_string())
        })?;

        let mut names = Vec::with_capacity(count);
        for index in 0..count {
            let mut raw_name: *const c_char = ptr::null();
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_arrow_device_table_column_name(
                        self.schema.as_ptr(),
                        error::usize_to_i64(index, operation, "column_index")?,
                        &mut raw_name,
                    )
                },
                operation,
            )?;
            let name = if raw_name.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(raw_name) }.to_string_lossy().into_owned()
            };
            names.push(name);
        }
        Ok(names)
    }
}

fn unwrap_abi_root(inner: Arc<DeviceArrowAbiRootInner>) -> DeviceArrowAbiRootInner {
    match Arc::try_unwrap(inner) {
        Ok(inner) => inner,
        Err(_) => {
            unreachable!("device-arrow export root should have exactly one strong reference when handing off raw parts")
        },
    }
}

fn export_device_arrow_column_from_view(
    view: &crate::core::column::ColumnView<'_>,
    exec_options: cudf_sys::cudf_exec_options_t,
    retained_owner: Arc<dyn Any + Send + Sync>,
    operation: &'static str,
    reported_materialization: Option<DeviceArrowExportMaterialization>,
) -> Result<DeviceArrowColumnExport> {
    let mut schema = ptr::null_mut();
    let mut array = ptr::null_mut();
    let mut materialization =
        cudf_sys::cudf_arrow_device_export_materialization_t::CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;

    error::check_status_in(
        unsafe {
            cudf_sys::cudf_arrow_device_column_export(
                view.raw_handle(),
                &exec_options,
                &mut schema,
                &mut array,
                &mut materialization,
            )
        },
        operation,
    )?;

    let export_materialization =
        reported_materialization.unwrap_or_else(|| export_materialization_from_raw(materialization));
    let import_materialization = foreign_import_materialization_for_column_view(view)?;
    let inner = Arc::new(DeviceArrowAbiRootInner {
        schema: error::require_non_null(schema, operation, "ArrowSchema")?,
        array: error::require_non_null(array, operation, "ArrowDeviceArray")?,
        materialization: import_materialization,
        retained_owner: Some(retained_owner),
        ownership: DeviceArrowAbiOwnership::DestroyShim,
    });

    Ok(DeviceArrowColumnExport { inner, materialization: export_materialization })
}

fn export_device_arrow_table_from_view(
    view: &crate::core::table::TableView<'_>,
    exec_options: cudf_sys::cudf_exec_options_t,
    raw_names: Option<&[*const c_char]>,
    retained_owner: Arc<dyn Any + Send + Sync>,
    operation: &'static str,
    reported_materialization: Option<DeviceArrowExportMaterialization>,
) -> Result<DeviceArrowTableExport> {
    let mut schema = ptr::null_mut();
    let mut array = ptr::null_mut();
    let mut materialization =
        cudf_sys::cudf_arrow_device_export_materialization_t::CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;
    let (raw_names_ptr, raw_names_len) = match raw_names {
        Some(raw_names) => (
            raw_names.as_ptr(),
            i32::try_from(raw_names.len()).map_err(|_| {
                Error::new(ErrorKind::InvalidOperation, "number of dataframe columns exceeds i32::MAX")
                    .with_operation(operation)
                    .with_context("num_columns", raw_names.len().to_string())
            })?,
        ),
        None => (ptr::null(), 0),
    };

    error::check_status_in(
        unsafe {
            cudf_sys::cudf_arrow_device_table_export(
                view.raw_handle(),
                &exec_options,
                raw_names_ptr,
                raw_names_len,
                &mut schema,
                &mut array,
                &mut materialization,
            )
        },
        operation,
    )?;

    let export_materialization =
        reported_materialization.unwrap_or_else(|| export_materialization_from_raw(materialization));
    let import_materialization = foreign_import_materialization_for_table_view(view)?;
    let inner = Arc::new(DeviceArrowAbiRootInner {
        schema: error::require_non_null(schema, operation, "ArrowSchema")?,
        array: error::require_non_null(array, operation, "ArrowDeviceArray")?,
        materialization: import_materialization,
        retained_owner: Some(retained_owner),
        ownership: DeviceArrowAbiOwnership::DestroyShim,
    });

    Ok(DeviceArrowTableExport { inner, materialization: export_materialization })
}

pub struct DeviceArrowColumnExport {
    inner: Arc<DeviceArrowAbiRootInner>,
    materialization: DeviceArrowExportMaterialization,
}

impl DeviceArrowColumnExport {
    pub fn materialization(&self) -> DeviceArrowExportMaterialization {
        self.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::DeviceArrowColumnExport::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::DeviceArrowColumnExport::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::DeviceArrowColumnExport::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::DeviceArrowColumnExport::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self) -> Result<*const c_void> {
        self.inner.fixed_width_values_ptr("interop::arrow_device::DeviceArrowColumnExport::fixed_width_values_ptr")
    }

    pub fn schema_ptr(&self) -> *const c_void {
        self.inner.schema.as_ptr()
    }

    pub fn device_array_ptr(&self) -> *const c_void {
        self.inner.array.as_ptr()
    }

    /// Consume this export and return an owned Rust handoff for the raw Arrow
    /// device handles plus the retained producer owner chain.
    ///
    /// The returned raw-parts wrapper is still responsible for destroying the
    /// Arrow schema/array handles and retaining any producer state required by
    /// producer-retained exports. This is not yet a standalone Arrow C ABI
    /// ownership transfer that can discard the Rust-side owner token.
    pub fn into_raw_parts(self) -> DeviceArrowColumnRawParts {
        DeviceArrowColumnRawParts { inner: unwrap_abi_root(self.inner), materialization: self.materialization }
    }

    /// Consume this export and return self-contained Arrow C ABI ownership.
    ///
    /// The returned wrapper has already moved any retained producer owner
    /// chain into Arrow `release` / `private_data` state, so the raw ABI parts
    /// can outlive this Rust wrapper once handed off.
    pub fn into_owned_abi_parts(self) -> DeviceArrowColumnOwnedAbiParts {
        DeviceArrowColumnOwnedAbiParts {
            inner: unwrap_abi_root(self.inner).into_owned_abi(),
            materialization: self.materialization,
        }
    }

    pub fn into_foreign(self) -> ForeignDeviceArrowColumn {
        ForeignDeviceArrowColumn { inner: Arc::clone(&self.inner) }
    }
}

pub struct DeviceArrowColumnRawParts {
    inner: DeviceArrowAbiRootInner,
    materialization: DeviceArrowExportMaterialization,
}

impl DeviceArrowColumnRawParts {
    pub fn materialization(&self) -> DeviceArrowExportMaterialization {
        self.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::DeviceArrowColumnRawParts::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::DeviceArrowColumnRawParts::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::DeviceArrowColumnRawParts::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::DeviceArrowColumnRawParts::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self) -> Result<*const c_void> {
        self.inner.fixed_width_values_ptr("interop::arrow_device::DeviceArrowColumnRawParts::fixed_width_values_ptr")
    }

    pub fn schema_ptr(&self) -> *mut c_void {
        self.inner.schema.as_ptr()
    }

    pub fn device_array_ptr(&self) -> *mut c_void {
        self.inner.array.as_ptr()
    }

    /// Re-wrap these owned raw parts as a foreign input for safe import into a
    /// same-device runtime.
    pub fn into_foreign(self) -> ForeignDeviceArrowColumn {
        ForeignDeviceArrowColumn { inner: Arc::new(self.inner) }
    }
}

pub struct DeviceArrowColumnOwnedAbiParts {
    inner: DeviceArrowAbiRootInner,
    materialization: DeviceArrowExportMaterialization,
}

impl DeviceArrowColumnOwnedAbiParts {
    pub fn materialization(&self) -> DeviceArrowExportMaterialization {
        self.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::DeviceArrowColumnOwnedAbiParts::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::DeviceArrowColumnOwnedAbiParts::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::DeviceArrowColumnOwnedAbiParts::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::DeviceArrowColumnOwnedAbiParts::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self) -> Result<*const c_void> {
        self.inner
            .fixed_width_values_ptr("interop::arrow_device::DeviceArrowColumnOwnedAbiParts::fixed_width_values_ptr")
    }

    pub fn schema_ptr(&self) -> *mut c_void {
        self.inner.schema.as_ptr()
    }

    pub fn device_array_ptr(&self) -> *mut c_void {
        self.inner.array.as_ptr()
    }

    /// Consume this wrapper and return raw Arrow C ABI ownership.
    ///
    /// The caller now owns the schema/array handles and must eventually run
    /// the Arrow `release` callbacks exactly once.
    pub fn into_raw_abi(self) -> (*mut c_void, *mut c_void, DeviceArrowImportMaterialization) {
        let this = ManuallyDrop::new(self);
        (this.inner.schema.as_ptr(), this.inner.array.as_ptr(), this.inner.materialization)
    }

    /// Re-wrap this self-contained ABI ownership as a safe foreign import
    /// wrapper.
    pub fn into_foreign(self) -> ForeignDeviceArrowColumn {
        ForeignDeviceArrowColumn { inner: Arc::new(self.inner) }
    }
}

#[derive(Clone)]
pub struct ForeignDeviceArrowColumn {
    inner: Arc<DeviceArrowAbiRootInner>,
}

impl ForeignDeviceArrowColumn {
    /// Take ownership of standalone Arrow C ABI parts produced by this
    /// bridge's owned-ABI handoff.
    ///
    /// # Safety
    ///
    /// `schema` and `array` must be live, owned Arrow C ABI roots produced by
    /// this bridge's self-contained handoff path. They must not also remain
    /// owned by any other Rust wrapper.
    pub unsafe fn from_raw_abi_parts(
        schema: *mut c_void,
        array: *mut c_void,
        materialization: DeviceArrowImportMaterialization,
    ) -> Result<Self> {
        let operation = "interop::arrow_device::ForeignDeviceArrowColumn::from_raw_abi_parts";
        let (schema, array) = validate_raw_abi_parts(schema, array, operation)?;
        Ok(Self {
            inner: Arc::new(DeviceArrowAbiRootInner {
                schema,
                array,
                materialization,
                retained_owner: None,
                ownership: DeviceArrowAbiOwnership::ArrowRelease,
            }),
        })
    }

    pub fn materialization(&self) -> DeviceArrowImportMaterialization {
        self.inner.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::ForeignDeviceArrowColumn::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::ForeignDeviceArrowColumn::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::ForeignDeviceArrowColumn::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::ForeignDeviceArrowColumn::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self) -> Result<*const c_void> {
        self.inner.fixed_width_values_ptr("interop::arrow_device::ForeignDeviceArrowColumn::fixed_width_values_ptr")
    }
}

pub struct DeviceArrowTableExport {
    inner: Arc<DeviceArrowAbiRootInner>,
    materialization: DeviceArrowExportMaterialization,
}

impl DeviceArrowTableExport {
    pub fn materialization(&self) -> DeviceArrowExportMaterialization {
        self.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::DeviceArrowTableExport::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::DeviceArrowTableExport::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::DeviceArrowTableExport::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::DeviceArrowTableExport::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self, index: usize) -> Result<*const c_void> {
        self.inner.table_fixed_width_values_ptr(
            index,
            "interop::arrow_device::DeviceArrowTableExport::fixed_width_values_ptr",
        )
    }

    pub fn column_names(&self) -> Result<Vec<String>> {
        self.inner.table_column_names("interop::arrow_device::DeviceArrowTableExport::column_names")
    }

    pub fn schema_ptr(&self) -> *const c_void {
        self.inner.schema.as_ptr()
    }

    pub fn device_array_ptr(&self) -> *const c_void {
        self.inner.array.as_ptr()
    }

    /// Consume this export and return an owned Rust handoff for the raw Arrow
    /// device handles plus the retained producer owner chain.
    ///
    /// The returned raw-parts wrapper is still responsible for destroying the
    /// Arrow schema/array handles and retaining any producer state required by
    /// producer-retained exports. This is not yet a standalone Arrow C ABI
    /// ownership transfer that can discard the Rust-side owner token.
    pub fn into_raw_parts(self) -> DeviceArrowTableRawParts {
        DeviceArrowTableRawParts { inner: unwrap_abi_root(self.inner), materialization: self.materialization }
    }

    /// Consume this export and return self-contained Arrow C ABI ownership.
    ///
    /// The returned wrapper has already moved any retained producer owner
    /// chain into Arrow `release` / `private_data` state, so the raw ABI parts
    /// can outlive this Rust wrapper once handed off.
    pub fn into_owned_abi_parts(self) -> DeviceArrowTableOwnedAbiParts {
        DeviceArrowTableOwnedAbiParts {
            inner: unwrap_abi_root(self.inner).into_owned_abi(),
            materialization: self.materialization,
        }
    }

    pub fn into_foreign(self) -> ForeignDeviceArrowTable {
        ForeignDeviceArrowTable { inner: Arc::clone(&self.inner) }
    }
}

pub struct DeviceArrowTableRawParts {
    inner: DeviceArrowAbiRootInner,
    materialization: DeviceArrowExportMaterialization,
}

impl DeviceArrowTableRawParts {
    pub fn materialization(&self) -> DeviceArrowExportMaterialization {
        self.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::DeviceArrowTableRawParts::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::DeviceArrowTableRawParts::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::DeviceArrowTableRawParts::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::DeviceArrowTableRawParts::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self, index: usize) -> Result<*const c_void> {
        self.inner.table_fixed_width_values_ptr(
            index,
            "interop::arrow_device::DeviceArrowTableRawParts::fixed_width_values_ptr",
        )
    }

    pub fn column_names(&self) -> Result<Vec<String>> {
        self.inner.table_column_names("interop::arrow_device::DeviceArrowTableRawParts::column_names")
    }

    pub fn schema_ptr(&self) -> *mut c_void {
        self.inner.schema.as_ptr()
    }

    pub fn device_array_ptr(&self) -> *mut c_void {
        self.inner.array.as_ptr()
    }

    /// Re-wrap these owned raw parts as a foreign input for safe import into a
    /// same-device runtime.
    pub fn into_foreign(self) -> ForeignDeviceArrowTable {
        ForeignDeviceArrowTable { inner: Arc::new(self.inner) }
    }
}

pub struct DeviceArrowTableOwnedAbiParts {
    inner: DeviceArrowAbiRootInner,
    materialization: DeviceArrowExportMaterialization,
}

impl DeviceArrowTableOwnedAbiParts {
    pub fn materialization(&self) -> DeviceArrowExportMaterialization {
        self.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::DeviceArrowTableOwnedAbiParts::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::DeviceArrowTableOwnedAbiParts::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::DeviceArrowTableOwnedAbiParts::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::DeviceArrowTableOwnedAbiParts::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self, index: usize) -> Result<*const c_void> {
        self.inner.table_fixed_width_values_ptr(
            index,
            "interop::arrow_device::DeviceArrowTableOwnedAbiParts::fixed_width_values_ptr",
        )
    }

    pub fn column_names(&self) -> Result<Vec<String>> {
        self.inner.table_column_names("interop::arrow_device::DeviceArrowTableOwnedAbiParts::column_names")
    }

    pub fn schema_ptr(&self) -> *mut c_void {
        self.inner.schema.as_ptr()
    }

    pub fn device_array_ptr(&self) -> *mut c_void {
        self.inner.array.as_ptr()
    }

    /// Consume this wrapper and return raw Arrow C ABI ownership.
    ///
    /// The caller now owns the schema/array handles and must eventually run
    /// the Arrow `release` callbacks exactly once.
    pub fn into_raw_abi(self) -> (*mut c_void, *mut c_void, DeviceArrowImportMaterialization) {
        let this = ManuallyDrop::new(self);
        (this.inner.schema.as_ptr(), this.inner.array.as_ptr(), this.inner.materialization)
    }

    /// Re-wrap this self-contained ABI ownership as a safe foreign import
    /// wrapper.
    pub fn into_foreign(self) -> ForeignDeviceArrowTable {
        ForeignDeviceArrowTable { inner: Arc::new(self.inner) }
    }
}

#[derive(Clone)]
pub struct ForeignDeviceArrowTable {
    inner: Arc<DeviceArrowAbiRootInner>,
}

impl ForeignDeviceArrowTable {
    /// Take ownership of standalone Arrow C ABI parts produced by this
    /// bridge's owned-ABI handoff.
    ///
    /// # Safety
    ///
    /// `schema` and `array` must be live, owned Arrow C ABI roots produced by
    /// this bridge's self-contained handoff path. They must not also remain
    /// owned by any other Rust wrapper.
    pub unsafe fn from_raw_abi_parts(
        schema: *mut c_void,
        array: *mut c_void,
        materialization: DeviceArrowImportMaterialization,
    ) -> Result<Self> {
        let operation = "interop::arrow_device::ForeignDeviceArrowTable::from_raw_abi_parts";
        let (schema, array) = validate_raw_abi_parts(schema, array, operation)?;
        Ok(Self {
            inner: Arc::new(DeviceArrowAbiRootInner {
                schema,
                array,
                materialization,
                retained_owner: None,
                ownership: DeviceArrowAbiOwnership::ArrowRelease,
            }),
        })
    }

    pub fn materialization(&self) -> DeviceArrowImportMaterialization {
        self.inner.materialization
    }

    pub fn device_id(&self) -> i32 {
        self.inner.device_id("interop::arrow_device::ForeignDeviceArrowTable::device_id")
    }

    pub fn device_type(&self) -> DeviceArrowDeviceType {
        self.inner.device_type("interop::arrow_device::ForeignDeviceArrowTable::device_type")
    }

    pub fn sync_event_present(&self) -> bool {
        self.inner.has_sync_event("interop::arrow_device::ForeignDeviceArrowTable::sync_event_present")
    }

    pub fn sync_event_raw_ptr(&self) -> *mut c_void {
        self.inner.sync_event_raw_ptr("interop::arrow_device::ForeignDeviceArrowTable::sync_event_raw_ptr")
    }

    pub fn fixed_width_values_ptr(&self, index: usize) -> Result<*const c_void> {
        self.inner.table_fixed_width_values_ptr(
            index,
            "interop::arrow_device::ForeignDeviceArrowTable::fixed_width_values_ptr",
        )
    }

    pub fn column_names(&self) -> Result<Vec<String>> {
        self.inner.table_column_names("interop::arrow_device::ForeignDeviceArrowTable::column_names")
    }
}

impl Series {
    pub fn try_prepare_device_arrow_export(&self) -> Result<DeviceArrowColumnExport> {
        let operation = "interop::Series::try_prepare_device_arrow_export";
        let view = self.as_column().view().map_err(|error| error.with_operation(operation))?;
        let exec_options = view.execution_domain().exec_options_raw();
        let retained_owner: Arc<dyn Any + Send + Sync> = self.arc();

        export_device_arrow_column_from_view(&view, exec_options, retained_owner, operation, None)
    }
}

impl Column {
    /// Consume this advanced low-level column and export one column-shaped
    /// Arrow C Device root.
    ///
    /// This is the `Column`-level counterpart to
    /// [`Series::try_prepare_device_arrow_export`]. It consumes `self` so the
    /// bridge can retain honest producer ownership without hiding a deep copy.
    pub fn try_prepare_device_arrow_export(self) -> Result<DeviceArrowColumnExport> {
        let operation = "interop::Column::try_prepare_device_arrow_export";
        let retained_column = Arc::new(self);
        let view = retained_column.view().map_err(|error| error.with_operation(operation))?;
        let exec_options = retained_column.execution_domain().exec_options_raw();
        let retained_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&retained_column) as Arc<dyn Any + Send + Sync>;

        export_device_arrow_column_from_view(&view, exec_options, retained_owner, operation, None)
    }

    /// Borrow this advanced low-level column and export one column-shaped
    /// Arrow C Device root after first deep-copying the column.
    ///
    /// This path exists for callers that need a borrow-preserving low-level
    /// export without pretending the export is zero-copy. The returned export
    /// reports [`DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy`] to
    /// reflect that this public path copied device data before building the
    /// Arrow device root.
    pub fn try_prepare_device_arrow_export_with_deep_copy(&self) -> Result<DeviceArrowColumnExport> {
        let operation = "interop::Column::try_prepare_device_arrow_export_with_deep_copy";
        let retained_column = Arc::new(self.deep_copy().map_err(|error| error.with_operation(operation))?);
        let view = retained_column.view().map_err(|error| error.with_operation(operation))?;
        let exec_options = retained_column.execution_domain().exec_options_raw();
        let retained_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&retained_column) as Arc<dyn Any + Send + Sync>;

        export_device_arrow_column_from_view(
            &view,
            exec_options,
            retained_owner,
            operation,
            Some(DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy),
        )
    }
}

impl Table {
    pub fn try_prepare_device_arrow_export(self) -> Result<DeviceArrowTableExport> {
        let operation = "interop::Table::try_prepare_device_arrow_export";
        let retained_table = Arc::new(self);
        let view = retained_table.view().map_err(|error| error.with_operation(operation))?;
        let exec_options = retained_table.execution_domain().exec_options_raw();
        let retained_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&retained_table) as Arc<dyn Any + Send + Sync>;

        export_device_arrow_table_from_view(&view, exec_options, None, retained_owner, operation, None)
    }

    /// Borrow this advanced low-level table and export one table-shaped Arrow
    /// C Device root after first deep-copying the table.
    ///
    /// This keeps the original table alive and unchanged while making the copy
    /// cost explicit in the API contract. The returned export reports
    /// [`DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy`] because the
    /// public path materialized fresh device storage before exporting.
    pub fn try_prepare_device_arrow_export_with_deep_copy(&self) -> Result<DeviceArrowTableExport> {
        let operation = "interop::Table::try_prepare_device_arrow_export_with_deep_copy";
        let retained_table = Arc::new(self.deep_copy().map_err(|error| error.with_operation(operation))?);
        let view = retained_table.view().map_err(|error| error.with_operation(operation))?;
        let exec_options = retained_table.execution_domain().exec_options_raw();
        let retained_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&retained_table) as Arc<dyn Any + Send + Sync>;

        export_device_arrow_table_from_view(
            &view,
            exec_options,
            None,
            retained_owner,
            operation,
            Some(DeviceArrowExportMaterialization::BridgeOwnedDeviceCopy),
        )
    }
}

impl DataFrame {
    pub fn try_prepare_device_arrow_export(&self) -> Result<DeviceArrowTableExport> {
        let operation = "interop::DataFrame::try_prepare_device_arrow_export";
        let retained_df = Arc::new(self.clone());
        let view = retained_df.build_table_view().map_err(|error| error.with_operation(operation))?;
        let exec_options = retained_df.execution_domain().exec_options_raw();
        let names = retained_df.column_names_owned();
        let c_names = names
            .iter()
            .map(|name| {
                CString::new(name.as_str()).map_err(|_| {
                    Error::new(ErrorKind::InvalidArgument, "dataframe column names must not contain interior NUL bytes")
                        .with_operation(operation)
                        .with_context("column_name", name.clone())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let raw_names: Vec<*const c_char> = c_names.iter().map(|name| name.as_ptr()).collect();
        let retained_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&retained_df) as Arc<dyn Any + Send + Sync>;

        export_device_arrow_table_from_view(&view, exec_options, Some(&raw_names), retained_owner, operation, None)
    }
}

impl Column {
    pub fn try_import_from_device_arrow_column_into_runtime(
        input: &ForeignDeviceArrowColumn,
        runtime: &Runtime,
    ) -> Result<Self> {
        let operation = "interop::Column::try_import_from_device_arrow_column_into_runtime";
        validate_supported_device_type(input.inner.try_device_type(operation)?, operation)?;
        validate_same_device(input.inner.try_device_id(operation)?, runtime, operation)?;

        let exec_options = runtime.execution_domain().exec_options_raw();
        let mut handle = ptr::null_mut();
        let mut materialization =
            cudf_sys::cudf_arrow_device_import_materialization_t::CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW;

        error::check_status_in(
            unsafe {
                cudf_sys::cudf_arrow_device_column_import(
                    input.inner.schema.as_ptr(),
                    input.inner.array.as_ptr(),
                    &exec_options,
                    &mut handle,
                    &mut materialization,
                )
            },
            operation,
        )?;

        let retained_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&input.inner) as Arc<dyn Any + Send + Sync>;
        let column = unsafe {
            Column::from_raw_with_domain_and_owner(handle, Arc::clone(runtime.execution_domain()), retained_owner)
        }?;

        debug_assert_eq!(import_materialization_from_raw(materialization), input.materialization());
        Ok(column)
    }
}

impl Series {
    /// Import one column-shaped Arrow C Device root into a `Series` on the
    /// provided runtime.
    ///
    /// Column-shaped device Arrow import currently carries no separate high-level
    /// `Series` name contract, so the returned series is unnamed even when the
    /// producing side started from a named `Series`.
    pub fn try_import_from_device_arrow_column_into_runtime(
        input: &ForeignDeviceArrowColumn,
        runtime: &Runtime,
    ) -> Result<Self> {
        let operation = "interop::Series::try_import_from_device_arrow_column_into_runtime";
        let column = Column::try_import_from_device_arrow_column_into_runtime(input, runtime)
            .map_err(|error| error.with_operation(operation))?;
        Series::from_column(column).map_err(|error| error.with_operation(operation))
    }
}

impl Table {
    pub fn try_import_from_device_arrow_table_into_runtime(
        input: &ForeignDeviceArrowTable,
        runtime: &Runtime,
    ) -> Result<Self> {
        let operation = "interop::Table::try_import_from_device_arrow_table_into_runtime";
        validate_supported_device_type(input.inner.try_device_type(operation)?, operation)?;
        validate_same_device(input.inner.try_device_id(operation)?, runtime, operation)?;

        let exec_options = runtime.execution_domain().exec_options_raw();
        let mut handle = ptr::null_mut();
        let mut materialization =
            cudf_sys::cudf_arrow_device_import_materialization_t::CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW;

        error::check_status_in(
            unsafe {
                cudf_sys::cudf_arrow_device_table_import(
                    input.inner.schema.as_ptr(),
                    input.inner.array.as_ptr(),
                    &exec_options,
                    &mut handle,
                    &mut materialization,
                )
            },
            operation,
        )?;

        let retained_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&input.inner) as Arc<dyn Any + Send + Sync>;
        let columns = unsafe {
            Table::from_raw_with_domain_and_owners(
                handle,
                Arc::clone(runtime.execution_domain()),
                vec![Some(retained_owner); input.column_names()?.len()],
            )
        }?;

        debug_assert_eq!(import_materialization_from_raw(materialization), input.materialization());
        Ok(columns)
    }
}

impl DataFrame {
    pub fn try_import_from_device_arrow_table_into_runtime(
        input: &ForeignDeviceArrowTable,
        runtime: &Runtime,
    ) -> Result<Self> {
        let operation = "interop::DataFrame::try_import_from_device_arrow_table_into_runtime";
        let names = input.column_names().map_err(|error| error.with_operation(operation))?;
        let table = Table::try_import_from_device_arrow_table_into_runtime(input, runtime)
            .map_err(|error| error.with_operation(operation))?;
        DataFrame::from_table(table, names).map_err(|error| error.with_operation(operation))
    }
}
