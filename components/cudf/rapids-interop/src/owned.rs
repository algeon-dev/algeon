use std::collections::{HashMap, HashSet};

use crate::{
    BufferBinding, ExportDiagnostics, ExportProfile, InteropError, InteropErrorKind, ProducerReadiness,
    ReleaseContract, Result, SemanticColumn, SemanticTable,
    contract::OwnedTransferOwner,
    validation::{
        binding_map, first_buffer_id, owned_transfer_profile_error, semantic_column_map,
        validate_owned_transfer_profile, validate_owned_transfer_readiness, validate_prepared_column_against_bindings,
    },
};

/// Ownership-bearing device transfer root for one semantic column.
#[derive(Debug)]
pub struct OwnedColumnTransfer {
    column: SemanticColumn,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
    readiness: ProducerReadiness<'static>,
    diagnostics: ExportDiagnostics,
    release_contract: ReleaseContract,
    _owner: OwnedTransferOwner,
}

impl OwnedColumnTransfer {
    pub fn try_new<Owner>(
        column: SemanticColumn,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        Self::try_new_with_semantic_columns(
            column.clone(),
            vec![column],
            bindings,
            readiness,
            diagnostics,
            release_contract,
            owner,
        )
    }

    pub fn try_new_with_semantic_columns<Owner>(
        column: SemanticColumn,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        validate_owned_transfer_profile(diagnostics)?;
        if release_contract.release_device() < 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned device transfer release device must be >= 0",
            )
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let mapped_root = column_map.get(&column.id()).copied().ok_or_else(|| {
            InteropError::new(
                InteropErrorKind::InvalidView,
                "owned column transfer root must be present in the semantic column graph",
            )
            .with_context("column_id", column.id().get().to_string())
        })?;
        if mapped_root != &column {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned column transfer root must match the semantic column graph entry for its column id",
            )
            .with_context("column_id", column.id().get().to_string()));
        }

        let binding_map = binding_map(&bindings)?;
        let producer_device = validate_prepared_column_against_bindings(
            &column,
            &column_map,
            &binding_map,
            InteropErrorKind::PlacementUnsupported,
            &mut HashSet::new(),
            &mut HashMap::new(),
        )?;
        if column_map.len() != 1 {
            let mut visited = HashMap::new();
            validate_prepared_column_against_bindings(
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
                    "owned column transfer included an unreferenced semantic column",
                )
                .with_context("column_id", unreferenced.id().get().to_string()));
            }
        }

        validate_owned_transfer_readiness(&readiness, producer_device)?;
        if release_contract.release_device() != producer_device {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned column transfer release device must match the producer CUDA device",
            )
            .with_context("producer_device", producer_device.to_string())
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        Ok(Self {
            column,
            semantic_columns,
            bindings,
            readiness,
            diagnostics,
            release_contract,
            _owner: OwnedTransferOwner::new(owner),
        })
    }

    pub fn column(&self) -> &SemanticColumn {
        &self.column
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'static>] {
        &self.bindings
    }

    pub fn readiness(&self) -> &ProducerReadiness<'static> {
        &self.readiness
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn release_contract(&self) -> &ReleaseContract {
        &self.release_contract
    }

    /// Consume this transfer and attempt to recover its transport-specific
    /// owner payload by type.
    pub fn try_into_owner<Owner>(self) -> core::result::Result<Owner, Self>
    where Owner: Send + Sync + 'static {
        match self._owner.try_into::<Owner>() {
            Ok(owner) => Ok(owner),
            Err(owner) => Err(Self {
                column: self.column,
                semantic_columns: self.semantic_columns,
                bindings: self.bindings,
                readiness: self.readiness,
                diagnostics: self.diagnostics,
                release_contract: self.release_contract,
                _owner: owner,
            }),
        }
    }
}

/// Ownership-bearing device transfer root for one semantic table.
#[derive(Debug)]
pub struct OwnedTableTransfer {
    table: SemanticTable,
    columns: Vec<SemanticColumn>,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
    readiness: ProducerReadiness<'static>,
    diagnostics: ExportDiagnostics,
    release_contract: ReleaseContract,
    _owner: OwnedTransferOwner,
}

impl OwnedTableTransfer {
    pub fn try_new<Owner>(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        Self::try_new_with_semantic_columns(
            table,
            columns.clone(),
            columns,
            bindings,
            readiness,
            diagnostics,
            release_contract,
            owner,
        )
    }

    pub fn try_new_with_semantic_columns<Owner>(
        table: SemanticTable,
        columns: Vec<SemanticColumn>,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'static>>,
        readiness: ProducerReadiness<'static>,
        diagnostics: ExportDiagnostics,
        release_contract: ReleaseContract,
        owner: Owner,
    ) -> Result<Self>
    where
        Owner: Send + Sync + 'static,
    {
        validate_owned_transfer_profile(diagnostics)?;
        if release_contract.release_device() < 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned device transfer release device must be >= 0",
            )
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        if let Some(names) = table.names() {
            if names.len() != table.columns().len() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "owned table transfer names must match the number of table schema columns",
                )
                .with_context("names_len", names.len().to_string())
                .with_context("num_columns", table.columns().len().to_string()));
            }
        }

        let top_level_map = semantic_column_map(&columns, "table_column_id")?;
        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let binding_map = binding_map(&bindings)?;
        let mut seen = HashSet::with_capacity(table.columns().len());
        let mut table_device = None;
        let mut validated_devices = HashMap::new();

        for column_id in table.columns() {
            let column = top_level_map.get(column_id).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "owned table transfer referenced a missing top-level semantic column",
                )
                .with_context("column_id", column_id.get().to_string())
            })?;
            seen.insert(*column_id);

            if column.len() != table.num_rows() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "owned table transfer row count must match every top-level semantic column length",
                )
                .with_context("column_id", column_id.get().to_string())
                .with_context("column_len", column.len().to_string())
                .with_context("num_rows", table.num_rows().to_string()));
            }

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
                        "owned table transfer requires all referenced bindings to live on one CUDA device",
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
        }

        if seen.len() != top_level_map.len() {
            let unreferenced = top_level_map.keys().find(|column_id| !seen.contains(column_id)).copied().unwrap();
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned table transfer included an unreferenced top-level semantic column",
            )
            .with_context("column_id", unreferenced.get().to_string()));
        }

        if validated_devices.len() != column_map.len() {
            let unreferenced = semantic_columns
                .iter()
                .find(|candidate| !validated_devices.contains_key(&candidate.id()))
                .expect("missing validated semantic column must exist");
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned table transfer included an unreferenced semantic column",
            )
            .with_context("column_id", unreferenced.id().get().to_string()));
        }

        let producer_device = match (table_device, readiness) {
            (Some(device), _) => device,
            (None, ProducerReadiness::DeviceStreamOrdered { stream }) => stream.device(),
            (None, ProducerReadiness::DeviceEventOrdered { event }) => event.device(),
            (None, ProducerReadiness::HostVisible) => {
                return Err(owned_transfer_profile_error(
                    ExportProfile::OwnedDeviceTransfer,
                    "owned device transfer requires device-ordered readiness",
                )
                .with_context("readiness", "host_visible"));
            },
        };

        validate_owned_transfer_readiness(&readiness, producer_device)?;
        if release_contract.release_device() != producer_device {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "owned table transfer release device must match the producer CUDA device",
            )
            .with_context("producer_device", producer_device.to_string())
            .with_context("release_device", release_contract.release_device().to_string()));
        }

        Ok(Self {
            table,
            columns,
            semantic_columns,
            bindings,
            readiness,
            diagnostics,
            release_contract,
            _owner: OwnedTransferOwner::new(owner),
        })
    }

    pub fn table(&self) -> &SemanticTable {
        &self.table
    }

    pub fn columns(&self) -> &[SemanticColumn] {
        &self.columns
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        &self.semantic_columns
    }

    pub fn bindings(&self) -> &[BufferBinding<'static>] {
        &self.bindings
    }

    pub fn readiness(&self) -> &ProducerReadiness<'static> {
        &self.readiness
    }

    pub fn diagnostics(&self) -> &ExportDiagnostics {
        &self.diagnostics
    }

    pub fn release_contract(&self) -> &ReleaseContract {
        &self.release_contract
    }

    /// Consume this transfer and attempt to recover its transport-specific
    /// owner payload by type.
    pub fn try_into_owner<Owner>(self) -> core::result::Result<Owner, Self>
    where Owner: Send + Sync + 'static {
        match self._owner.try_into::<Owner>() {
            Ok(owner) => Ok(owner),
            Err(owner) => Err(Self {
                table: self.table,
                columns: self.columns,
                semantic_columns: self.semantic_columns,
                bindings: self.bindings,
                readiness: self.readiness,
                diagnostics: self.diagnostics,
                release_contract: self.release_contract,
                _owner: owner,
            }),
        }
    }
}
