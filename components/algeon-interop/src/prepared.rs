use std::{
    any::Any,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    ActiveColumnReadLease, ActiveTableReadLease, BufferBinding, BufferId, ColumnId, CompletionAuthority,
    ExportDiagnostics, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseBackend, LeaseExportContext,
    PreparedExportProfile, ProducerReadiness, ReadStartAuthority, Result, SemanticColumn, SemanticTable,
    contract::RetainedHostBuffers,
    lease::LeaseRootState,
    validation::{
        binding_map, completion_binding, first_buffer_id, lease_readiness_binding, prepared_profile_error,
        read_start_binding, semantic_column_map, validate_completion_authority,
        validate_prepared_column_against_bindings, validate_prepared_host_column_against_bindings,
        validate_prepared_profile, validate_prepared_readiness, validate_read_start_authority,
    },
};

/// Producer-rooted prepared export for one semantic column.
#[derive(Debug, Clone)]
pub struct PreparedColumnExport<'a> {
    column: SemanticColumn,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    readiness: ProducerReadiness<'a>,
    profile: PreparedExportProfile,
    diagnostics: ExportDiagnostics,
    retained_host_buffers: RetainedHostBuffers,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    /// Cached producer device ordinal for device-backed exports.
    producer_device: Option<i32>,
    lease_state: LeaseRootState,
    lease_backend: LeaseBackend,
}

/// Structural equality over all observable export state.  `lease_backend` is
/// compared by its `incomplete_drop_behavior()` rather than identity because
/// backends are trait objects without a meaningful equality relation; the drop
/// behavior is the only semantically relevant property for export comparison.
impl PartialEq for PreparedColumnExport<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.column == other.column
            && self.semantic_columns == other.semantic_columns
            && self.bindings == other.bindings
            && self.readiness == other.readiness
            && self.profile == other.profile
            && self.diagnostics == other.diagnostics
            && self.producer_device == other.producer_device
            && self.lease_backend.incomplete_drop_behavior() == other.lease_backend.incomplete_drop_behavior()
    }
}

impl Eq for PreparedColumnExport<'_> {}

impl<'a> PreparedColumnExport<'a> {
    pub fn try_new(
        column: SemanticColumn,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        Self::try_new_with_semantic_columns(column.clone(), vec![column], bindings, readiness, diagnostics)
    }

    pub fn try_new_with_semantic_columns(
        column: SemanticColumn,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        let profile = validate_prepared_profile(diagnostics)?;
        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let mapped_root = column_map.get(&column.id()).copied().ok_or_else(|| {
            InteropError::new(
                InteropErrorKind::NotFound,
                "prepared column export root must be present in the semantic column graph",
            )
            .with_context("column_id", column.id().get().to_string())
        })?;
        if mapped_root != &column {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared column export root must match the semantic column graph entry for its column id",
            )
            .with_context("column_id", column.id().get().to_string()));
        }
        let binding_map = binding_map(&bindings)?;
        let producer_device = match profile {
            PreparedExportProfile::Device => {
                let mut visited = HashMap::new();
                let producer_device = validate_prepared_column_against_bindings(
                    &column,
                    &column_map,
                    &binding_map,
                    InteropErrorKind::PlacementUnsupported,
                    &mut HashSet::new(),
                    &mut visited,
                )?;
                if visited.len() != column_map.len() {
                    let unreferenced = semantic_columns
                        .iter()
                        .find(|candidate| !visited.contains_key(&candidate.id()))
                        .expect("missing visited semantic column must exist");
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared column export included an unreferenced semantic column",
                    )
                    .with_context("column_id", unreferenced.id().get().to_string()));
                }
                Some(producer_device)
            },
            PreparedExportProfile::Host => {
                let mut validated = HashSet::new();
                validate_prepared_host_column_against_bindings(&column, &column_map, &binding_map, &mut validated)?;
                if validated.len() != column_map.len() {
                    let unreferenced = semantic_columns
                        .iter()
                        .find(|candidate| !validated.contains(&candidate.id()))
                        .expect("missing validated semantic column must exist");
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared column export included an unreferenced semantic column",
                    )
                    .with_context("column_id", unreferenced.id().get().to_string()));
                }
                None
            },
        };
        validate_prepared_readiness(profile, &readiness, producer_device)?;
        Ok(Self {
            column,
            semantic_columns,
            bindings,
            readiness,
            profile,
            diagnostics: profile.diagnostics(),
            retained_host_buffers: RetainedHostBuffers::default(),
            retained_owner: None,
            producer_device,
            lease_state: LeaseRootState::default(),
            lease_backend: LeaseBackend::semantic_guaranteed_fallback(),
        })
    }

    pub fn column(&self) -> &SemanticColumn {
        &self.column
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'a>] {
        &self.bindings
    }

    pub fn readiness(&self) -> &ProducerReadiness<'a> {
        &self.readiness
    }

    pub fn profile(&self) -> PreparedExportProfile {
        self.profile
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn with_lease_backend(mut self, lease_backend: LeaseBackend) -> Self {
        self.lease_backend = lease_backend;
        self
    }

    #[doc(hidden)]
    pub fn with_retained_host_buffers(mut self, retained_host_buffers: Vec<Arc<[u8]>>) -> Self {
        self.retained_host_buffers = RetainedHostBuffers::new(retained_host_buffers);
        self
    }

    #[doc(hidden)]
    pub fn with_retained_owner(mut self, retained_owner: Arc<dyn Any + Send + Sync>) -> Self {
        self.retained_owner = Some(retained_owner);
        self
    }

    #[doc(hidden)]
    pub fn with_retained_owner_arc<T>(self, retained_owner: Arc<T>) -> Self
    where T: Any + Send + Sync + 'static {
        self.with_retained_owner(retained_owner)
    }

    #[doc(hidden)]
    pub fn clone_retained_owner(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.retained_owner.as_ref().map(Arc::clone)
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    fn lease_export_context(&self) -> LeaseExportContext {
        LeaseExportContext::new(self.producer_device, lease_readiness_binding(self.readiness))
    }

    pub fn try_activate_read_lease<'export, 'consumer>(
        &'export self,
        read_start: ReadStartAuthority<'consumer>,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<ActiveColumnReadLease<'export, 'a, 'consumer>> {
        validate_read_start_authority(self.profile, read_start)?;
        validate_completion_authority(self.profile, completion, self.producer_device)?;
        let read_start_binding = read_start_binding(read_start);
        let completion_binding = completion_binding(completion);
        let export_context = self.lease_export_context();
        let registration = self.lease_state.try_acquire("column")?;
        let activation = self.lease_backend.activate(&export_context, read_start_binding, completion_binding)?;
        Ok(ActiveColumnReadLease {
            export: self,
            read_start,
            completion,
            completion_binding,
            export_context,
            lease_backend: self.lease_backend.clone(),
            _activation: activation,
            _registration: registration,
            completed: false,
        })
    }
}

/// Producer-rooted prepared export for one semantic table.
#[derive(Debug, Clone)]
pub struct PreparedTableExport<'a> {
    table: SemanticTable,
    columns: Vec<SemanticColumn>,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    readiness: ProducerReadiness<'a>,
    profile: PreparedExportProfile,
    diagnostics: ExportDiagnostics,
    retained_host_buffers: RetainedHostBuffers,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    /// Cached producer device ordinal for device-backed exports.
    producer_device: Option<i32>,
    lease_state: LeaseRootState,
    lease_backend: LeaseBackend,
}

impl PartialEq for PreparedTableExport<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.table == other.table
            && self.columns == other.columns
            && self.semantic_columns == other.semantic_columns
            && self.bindings == other.bindings
            && self.readiness == other.readiness
            && self.profile == other.profile
            && self.diagnostics == other.diagnostics
            && self.producer_device == other.producer_device
            && self.lease_backend.incomplete_drop_behavior() == other.lease_backend.incomplete_drop_behavior()
    }
}

impl Eq for PreparedTableExport<'_> {}

impl<'a> PreparedTableExport<'a> {
    pub fn try_new(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        Self::try_new_with_semantic_columns(table, columns.clone(), columns, bindings, readiness, diagnostics)
    }

    pub fn try_new_with_semantic_columns(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        readiness: ProducerReadiness<'a>,
        diagnostics: ExportDiagnostics,
    ) -> Result<Self> {
        let profile = validate_prepared_profile(diagnostics)?;

        if let Some(names) = table.names()
            && names.len() != table.columns().len()
        {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared table names must match the number of table schema columns",
            )
            .with_context("names_len", names.len().to_string())
            .with_context("num_columns", table.columns().len().to_string()));
        }

        let binding_map = binding_map(&bindings)?;
        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let mut top_level_map = HashMap::with_capacity(columns.len());
        for column in &columns {
            if top_level_map.insert(column.id(), column).is_some() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared table top-level columns must not reuse one semantic column id",
                )
                .with_context("column_id", column.id().get().to_string()));
            }
        }

        let mut seen = HashSet::with_capacity(table.columns().len());
        let mut table_device = None;
        let mut validated_devices = HashMap::new();
        let mut validated_columns = HashSet::new();
        for column_id in table.columns() {
            if !seen.insert(*column_id) {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared table schema must not repeat one semantic column id",
                )
                .with_context("column_id", column_id.get().to_string()));
            }
            let column = top_level_map.get(column_id).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::NotFound,
                    "prepared table schema referenced a missing top-level semantic column",
                )
                .with_context("column_id", column_id.get().to_string())
            })?;
            if column.len() != table.num_rows() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "column length does not match prepared table row count",
                )
                .with_context("column_id", column_id.get().to_string())
                .with_context("column_len", column.len().to_string())
                .with_context("num_rows", table.num_rows().to_string()));
            }

            match profile {
                PreparedExportProfile::Device => {
                    let column_device = validate_prepared_column_against_bindings(
                        column,
                        &column_map,
                        &binding_map,
                        InteropErrorKind::PlacementUnsupported,
                        &mut HashSet::new(),
                        &mut validated_devices,
                    )?;
                    match table_device {
                        None => table_device = Some(column_device),
                        Some(expected) if expected != column_device => {
                            let mut err = InteropError::new(
                                InteropErrorKind::MixedDeviceUnsupported,
                                "prepared table export requires all referenced bindings to live on one CUDA device",
                            )
                            .with_context("column_id", column_id.get().to_string())
                            .with_context("expected_device", expected.to_string())
                            .with_context("actual_device", column_device.to_string());
                            if let Some(buffer_id) = first_buffer_id(column) {
                                err = err.with_context("buffer_id", buffer_id.get().to_string());
                            }
                            return Err(err);
                        },
                        Some(_) => {},
                    }
                },
                PreparedExportProfile::Host => {
                    validate_prepared_host_column_against_bindings(
                        column,
                        &column_map,
                        &binding_map,
                        &mut validated_columns,
                    )?;
                },
            }
        }

        if seen.len() != top_level_map.len() {
            let unreferenced = top_level_map.keys().find(|column_id| !seen.contains(column_id)).copied().unwrap();
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared table export included an unreferenced top-level semantic column",
            )
            .with_context("column_id", unreferenced.get().to_string()));
        }
        if validated_devices.len() != column_map.len() && profile == PreparedExportProfile::Device {
            let unreferenced = semantic_columns
                .iter()
                .find(|candidate| !validated_devices.contains_key(&candidate.id()))
                .expect("missing validated semantic column must exist");
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared table export included an unreferenced semantic column",
            )
            .with_context("column_id", unreferenced.id().get().to_string()));
        }
        if profile == PreparedExportProfile::Host && validated_columns.len() != column_map.len() {
            let unreferenced = semantic_columns
                .iter()
                .find(|candidate| !validated_columns.contains(&candidate.id()))
                .expect("missing validated semantic column must exist");
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared table export included an unreferenced semantic column",
            )
            .with_context("column_id", unreferenced.id().get().to_string()));
        }

        let producer_device = match profile {
            PreparedExportProfile::Device => Some(match (table_device, readiness) {
                (Some(device), _) => device,
                (None, ProducerReadiness::DeviceStreamOrdered { stream }) => stream.device(),
                (None, ProducerReadiness::DeviceEventOrdered { event }) => event.device(),
                (None, ProducerReadiness::HostVisible) => {
                    return Err(prepared_profile_error(
                        profile,
                        "prepared device export requires device-ordered readiness",
                    )
                    .with_context("readiness", "host_visible"));
                },
            }),
            PreparedExportProfile::Host => None,
        };
        validate_prepared_readiness(profile, &readiness, producer_device)?;

        Ok(Self {
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            profile,
            diagnostics: profile.diagnostics(),
            retained_host_buffers: RetainedHostBuffers::default(),
            retained_owner: None,
            producer_device,
            lease_state: LeaseRootState::default(),
            lease_backend: LeaseBackend::semantic_guaranteed_fallback(),
        })
    }

    pub fn table(&self) -> &SemanticTable {
        &self.table
    }

    pub fn columns(&self) -> &[SemanticColumn] {
        &self.columns
    }

    /// Return one top-level semantic column by its schema column id.
    pub fn top_level_column(&self, column_id: ColumnId) -> Option<&SemanticColumn> {
        self.columns.iter().find(|column| column.id() == column_id)
    }

    /// Return one top-level semantic column by its schema name.
    ///
    /// This lookup follows the table schema order and returns `None` when the
    /// prepared table root does not carry names.
    pub fn top_level_column_by_name(&self, name: &str) -> Option<&SemanticColumn> {
        self.table
            .names()
            .and_then(|names| names.iter().position(|candidate| candidate == name))
            .and_then(|index| self.columns.get(index))
    }

    /// Return one semantic column from the full recursive semantic graph.
    pub fn semantic_column(&self, column_id: ColumnId) -> Option<&SemanticColumn> {
        self.semantic_columns.iter().find(|column| column.id() == column_id)
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'a>] {
        &self.bindings
    }

    /// Return one bound buffer by its symbolic buffer id.
    pub fn binding(&self, buffer_id: BufferId) -> Option<&BufferBinding<'a>> {
        self.bindings.iter().find(|binding| binding.id() == buffer_id)
    }

    pub fn readiness(&self) -> &ProducerReadiness<'a> {
        &self.readiness
    }

    pub fn profile(&self) -> PreparedExportProfile {
        self.profile
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn with_lease_backend(mut self, lease_backend: LeaseBackend) -> Self {
        self.lease_backend = lease_backend;
        self
    }

    #[doc(hidden)]
    pub fn with_retained_host_buffers(mut self, retained_host_buffers: Vec<Arc<[u8]>>) -> Self {
        self.retained_host_buffers = RetainedHostBuffers::new(retained_host_buffers);
        self
    }

    #[doc(hidden)]
    pub fn with_retained_owner(mut self, retained_owner: Arc<dyn Any + Send + Sync>) -> Self {
        self.retained_owner = Some(retained_owner);
        self
    }

    #[doc(hidden)]
    pub fn with_retained_owner_arc<T>(self, retained_owner: Arc<T>) -> Self
    where T: Any + Send + Sync + 'static {
        self.with_retained_owner(retained_owner)
    }

    #[doc(hidden)]
    pub fn clone_retained_owner(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.retained_owner.as_ref().map(Arc::clone)
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    fn lease_export_context(&self) -> LeaseExportContext {
        LeaseExportContext::new(self.producer_device, lease_readiness_binding(self.readiness))
    }

    pub fn try_activate_read_lease<'export, 'consumer>(
        &'export self,
        read_start: ReadStartAuthority<'consumer>,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<ActiveTableReadLease<'export, 'a, 'consumer>> {
        validate_read_start_authority(self.profile, read_start)?;
        validate_completion_authority(self.profile, completion, self.producer_device)?;
        let read_start_binding = read_start_binding(read_start);
        let completion_binding = completion_binding(completion);
        let export_context = self.lease_export_context();
        let registration = self.lease_state.try_acquire("table")?;
        let activation = self.lease_backend.activate(&export_context, read_start_binding, completion_binding)?;
        Ok(ActiveTableReadLease {
            export: self,
            read_start,
            completion,
            completion_binding,
            export_context,
            lease_backend: self.lease_backend.clone(),
            _activation: activation,
            _registration: registration,
            completed: false,
        })
    }
}
