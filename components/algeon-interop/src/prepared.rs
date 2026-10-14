use std::{any::Any, sync::Arc};

use crate::{
    ActiveColumnReadLease, ActiveTableReadLease, BufferBinding, BufferId, ColumnId, CompletionAuthority, LeaseBackend,
    PreparedExportProfile, ProducerReadiness, ReadStartAuthority, Result, SemanticColumn, SemanticTable,
    lease::{DeviceLease, LeaseCore},
    validation::{
        validate_column, validate_completion_authority, validate_host_authorities, validate_prepared_readiness,
        validate_table,
    },
};

/// How consumers reach the bytes of a prepared export.
#[derive(Debug, Clone)]
pub enum PreparedAccess<'a> {
    /// Host-visible bytes. Leases complete by return without a backend.
    Host,
    /// Device bytes ordered after `readiness`. `backend` performs lease work.
    Device { readiness: ProducerReadiness<'a>, backend: LeaseBackend },
}

impl PreparedAccess<'_> {
    pub fn profile(&self) -> PreparedExportProfile {
        match self {
            Self::Host => PreparedExportProfile::Host,
            Self::Device { .. } => PreparedExportProfile::Device,
        }
    }
}

/// Graph, bindings, access, and retained owner shared by column and table
/// exports.
#[derive(Debug, Clone)]
struct ExportRoot<'a> {
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
    access: PreparedAccess<'a>,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
}

impl<'a> ExportRoot<'a> {
    fn retain_owner(&mut self, owner: Arc<dyn Any + Send + Sync>) {
        self.retained_owner = Some(match self.retained_owner.take() {
            Some(previous) => Arc::new((previous, owner)),
            None => owner,
        });
    }

    /// Requires the bindings of a device export's graph, already validated
    /// with `graph_device` as their shared device, to share its readiness
    /// device.
    fn try_new(
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        access: PreparedAccess<'a>,
        graph_device: Option<i32>,
    ) -> Result<Self> {
        if let PreparedAccess::Device { readiness, .. } = &access {
            validate_prepared_readiness(readiness, graph_device)?;
        }
        Ok(Self { semantic_columns, bindings, access, retained_owner: None })
    }

    /// Validates the lease authorities against the access and, for a device
    /// export, starts backend lease work. A host lease has no backend state.
    fn activate<'consumer>(
        &self,
        read_start: ReadStartAuthority<'consumer>,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<LeaseCore<'a, 'consumer>> {
        let device = match &self.access {
            PreparedAccess::Host => {
                validate_host_authorities(read_start, completion)?;
                None
            },
            PreparedAccess::Device { readiness, backend } => {
                validate_completion_authority(readiness.device(), completion)?;
                let activation = backend.activate(*readiness, read_start, completion)?;
                Some(DeviceLease::new(backend.clone(), *readiness, activation))
            },
        };
        Ok(LeaseCore::new(read_start, completion, device, self.retained_owner.clone()))
    }
}

/// Producer-rooted prepared export for one semantic column.
#[derive(Debug, Clone)]
pub struct PreparedColumnExport<'a> {
    column: SemanticColumn,
    root: ExportRoot<'a>,
}

impl<'a> PreparedColumnExport<'a> {
    /// Validates the graph reachable from `root`. `semantic_columns` must
    /// contain `root` and every column reachable from it, and nothing else.
    ///
    /// # Safety
    ///
    /// Every binding must describe initialized, aligned storage readable on
    /// its declared placement with contents valid for the semantic layout.
    /// Storage and readiness handles must remain valid through all reads of
    /// this export and its clones. Borrowed storage must live for `'a`; owned
    /// storage may instead be attached with [`Self::with_retained_owner`]
    /// before the export is exposed to a consumer. Graph validation checks
    /// sizes and topology, not allocation liveness or buffer contents.
    /// Producer writes must precede reads through readiness; storage must not
    /// be mutated or reallocated while consumer reads are pending.
    ///
    /// ```compile_fail,E0133
    /// use rapids_interop::{ColumnId, PreparedAccess, PreparedColumnExport};
    /// let _ = PreparedColumnExport::try_new(ColumnId::new(0), vec![], vec![], PreparedAccess::Host);
    /// ```
    pub unsafe fn try_new(
        root: ColumnId,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        access: PreparedAccess<'a>,
    ) -> Result<Self> {
        let (column, graph_device) = validate_column(root, &semantic_columns, &bindings, access.profile().into())?;
        let column = column.clone();
        Ok(Self { column, root: ExportRoot::try_new(semantic_columns, bindings, access, graph_device)? })
    }

    pub fn column(&self) -> &SemanticColumn {
        &self.column
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.root.semantic_columns
    }

    /// Borrows descriptors for as long as this export keeps storage alive.
    ///
    /// ```compile_fail
    /// use rapids_interop::{BufferBinding, PreparedColumnExport};
    /// fn escape(export: &PreparedColumnExport<'static>) -> Vec<BufferBinding<'static>> {
    ///     export.bindings().to_vec()
    /// }
    /// ```
    pub fn bindings(&self) -> &[BufferBinding<'_>] {
        &self.root.bindings
    }

    pub fn access(&self) -> &PreparedAccess<'_> {
        &self.root.access
    }

    pub fn profile(&self) -> PreparedExportProfile {
        self.root.access.profile()
    }

    /// Retains an owner in addition to all previously attached owners.
    /// Clones, active leases and pending completions share these owners.
    ///
    /// # Safety
    ///
    /// The combined owners must keep all bound storage and producer readiness
    /// handles valid until consumer reads finish, including after the original
    /// producer borrow ends. An unrelated value cannot establish this guarantee.
    ///
    /// ```compile_fail,E0133
    /// use std::sync::Arc;
    /// use rapids_interop::PreparedColumnExport;
    /// fn attach(export: PreparedColumnExport<'_>) {
    ///     let _ = export.with_retained_owner(Arc::new(()));
    /// }
    /// ```
    pub unsafe fn with_retained_owner(mut self, retained_owner: Arc<dyn Any + Send + Sync>) -> Self {
        self.root.retain_owner(retained_owner);
        self
    }

    pub fn retained_owner(&self) -> Option<&Arc<dyn Any + Send + Sync>> {
        self.root.retained_owner.as_ref()
    }

    pub fn try_activate_read_lease<'export, 'consumer>(
        &'export self,
        read_start: ReadStartAuthority<'consumer>,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<ActiveColumnReadLease<'export, 'a, 'consumer>> {
        Ok(ActiveColumnReadLease { export: self, core: self.root.activate(read_start, completion)? })
    }
}

/// Producer-rooted prepared export for one semantic table.
#[derive(Debug, Clone)]
pub struct PreparedTableExport<'a> {
    table: SemanticTable,
    columns: Vec<SemanticColumn>,
    root: ExportRoot<'a>,
}

impl<'a> PreparedTableExport<'a> {
    /// Validates the table schema and the graph reachable from its columns.
    /// `semantic_columns` must contain every schema column and every column
    /// reachable from them, and nothing else.
    ///
    /// # Safety
    ///
    /// Bindings, contents and readiness must satisfy the storage and lifetime
    /// requirements of [`PreparedColumnExport::try_new`].
    ///
    /// ```compile_fail,E0133
    /// use rapids_interop::{PreparedAccess, PreparedTableExport, SemanticTable};
    /// let _ = PreparedTableExport::try_new(SemanticTable::new(0, vec![], None), vec![], vec![], PreparedAccess::Host);
    /// ```
    pub unsafe fn try_new(
        table: SemanticTable,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        access: PreparedAccess<'a>,
    ) -> Result<Self> {
        let (columns, graph_device) = validate_table(&table, &semantic_columns, &bindings, access.profile().into())?;
        let columns = columns.into_iter().cloned().collect();
        Ok(Self { table, columns, root: ExportRoot::try_new(semantic_columns, bindings, access, graph_device)? })
    }

    pub fn table(&self) -> &SemanticTable {
        &self.table
    }

    /// Top-level semantic columns in schema order.
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
        self.root.semantic_columns.iter().find(|column| column.id() == column_id)
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.root.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'_>] {
        &self.root.bindings
    }

    /// Return one bound buffer by its symbolic buffer id.
    pub fn binding(&self, buffer_id: BufferId) -> Option<&BufferBinding<'_>> {
        self.root.bindings.iter().find(|binding| binding.id() == buffer_id)
    }

    pub fn access(&self) -> &PreparedAccess<'_> {
        &self.root.access
    }

    pub fn profile(&self) -> PreparedExportProfile {
        self.root.access.profile()
    }

    /// Retains an owner in addition to all previously attached owners.
    /// Clones, active leases and pending completions share these owners.
    ///
    /// # Safety
    ///
    /// The combined owners must satisfy the storage and readiness lifetime
    /// requirements of [`PreparedColumnExport::with_retained_owner`].
    pub unsafe fn with_retained_owner(mut self, retained_owner: Arc<dyn Any + Send + Sync>) -> Self {
        self.root.retain_owner(retained_owner);
        self
    }

    pub fn retained_owner(&self) -> Option<&Arc<dyn Any + Send + Sync>> {
        self.root.retained_owner.as_ref()
    }

    pub fn try_activate_read_lease<'export, 'consumer>(
        &'export self,
        read_start: ReadStartAuthority<'consumer>,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<ActiveTableReadLease<'export, 'a, 'consumer>> {
        Ok(ActiveTableReadLease { export: self, core: self.root.activate(read_start, completion)? })
    }
}
