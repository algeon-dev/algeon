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
        Ok(LeaseCore::new(read_start, completion, device))
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
    pub fn try_new(
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

    pub fn bindings(&self) -> &[BufferBinding<'a>] {
        &self.root.bindings
    }

    pub fn access(&self) -> &PreparedAccess<'a> {
        &self.root.access
    }

    pub fn profile(&self) -> PreparedExportProfile {
        self.root.access.profile()
    }

    /// Attaches the value that keeps this export's storage alive, replacing
    /// any earlier one. Clones of the export share it, and a completed lease
    /// holds it while its witness is pending (see [`LeaseCompletion`](crate::LeaseCompletion)).
    pub fn with_retained_owner(mut self, retained_owner: Arc<dyn Any + Send + Sync>) -> Self {
        self.root.retained_owner = Some(retained_owner);
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
    pub fn try_new(
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

    pub fn bindings(&self) -> &[BufferBinding<'a>] {
        &self.root.bindings
    }

    /// Return one bound buffer by its symbolic buffer id.
    pub fn binding(&self, buffer_id: BufferId) -> Option<&BufferBinding<'a>> {
        self.root.bindings.iter().find(|binding| binding.id() == buffer_id)
    }

    pub fn access(&self) -> &PreparedAccess<'a> {
        &self.root.access
    }

    pub fn profile(&self) -> PreparedExportProfile {
        self.root.access.profile()
    }

    /// Attaches the value that keeps this export's storage alive, replacing
    /// any earlier one. Clones of the export share it, and a completed lease
    /// holds it while its witness is pending (see [`LeaseCompletion`](crate::LeaseCompletion)).
    pub fn with_retained_owner(mut self, retained_owner: Arc<dyn Any + Send + Sync>) -> Self {
        self.root.retained_owner = Some(retained_owner);
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
