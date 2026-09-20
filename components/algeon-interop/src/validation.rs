use std::collections::{HashMap, HashSet};

use crate::{
    BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority, InteropError, InteropErrorKind,
    PhysicalValidity, Placement, PreparedExportProfile, ProducerReadiness, ReadStartAuthority, Result, SemanticColumn,
    SemanticTable, TypeId,
};

fn host_profile_error(message: &'static str) -> InteropError {
    InteropError::new(InteropErrorKind::ProfileUnsupported, message).with_context("profile", "prepared_host_export")
}

/// Requires the readiness device to match the device shared by the graph's
/// bindings. A graph without bindings (a table without columns) takes the
/// readiness device.
pub(crate) fn validate_prepared_readiness(readiness: &ProducerReadiness<'_>, graph_device: Option<i32>) -> Result<()> {
    let (readiness_name, device_key, message) = match readiness {
        ProducerReadiness::DeviceStreamOrdered { .. } => (
            "device_stream_ordered",
            "stream_device",
            "prepared export readiness stream must live on the producer CUDA device",
        ),
        ProducerReadiness::DeviceEventOrdered { .. } => (
            "device_event_ordered",
            "event_device",
            "prepared export readiness event must live on the producer CUDA device",
        ),
    };
    match graph_device {
        Some(producer_device) if producer_device != readiness.device() => {
            Err(InteropError::new(InteropErrorKind::ReadinessMismatch, message)
                .with_context("readiness", readiness_name)
                .with_context("producer_device", producer_device.to_string())
                .with_context(device_key, readiness.device().to_string()))
        },
        _ => Ok(()),
    }
}

/// Placement every binding of an export graph must have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpectedPlacement {
    Host,
    Device,
}

impl From<PreparedExportProfile> for ExpectedPlacement {
    fn from(profile: PreparedExportProfile) -> Self {
        match profile {
            PreparedExportProfile::Host => Self::Host,
            PreparedExportProfile::Device => Self::Device,
        }
    }
}

/// Validates a column root and the graph reachable from it. Returns the root
/// column and the CUDA device shared by all bindings in device mode.
pub(crate) fn validate_column<'g>(
    root: ColumnId,
    semantic_columns: &'g [SemanticColumn],
    bindings: &'g [BufferBinding<'_>],
    expected: ExpectedPlacement,
) -> Result<(&'g SemanticColumn, Option<i32>)> {
    let mut walk = GraphWalk::new(semantic_columns, bindings, expected)?;
    let column = walk.root(root)?;
    walk.visit(column)?;
    Ok((column, walk.finish()?))
}

/// Validates a table root: names, schema ids, row counts, then the graph
/// reachable from the schema columns in schema order. Returns the top-level
/// columns in schema order and the CUDA device shared by all bindings in
/// device mode.
pub(crate) fn validate_table<'g>(
    table: &SemanticTable,
    semantic_columns: &'g [SemanticColumn],
    bindings: &'g [BufferBinding<'_>],
    expected: ExpectedPlacement,
) -> Result<(Vec<&'g SemanticColumn>, Option<i32>)> {
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

    let mut walk = GraphWalk::new(semantic_columns, bindings, expected)?;
    let mut seen = HashSet::with_capacity(table.columns().len());
    let mut roots = Vec::with_capacity(table.columns().len());
    for column_id in table.columns() {
        if !seen.insert(*column_id) {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared table schema must not repeat one semantic column id",
            )
            .with_context("column_id", column_id.get().to_string()));
        }
        let column = walk.root(*column_id)?;
        if column.len() != table.num_rows() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "column length does not match prepared table row count",
            )
            .with_context("column_id", column_id.get().to_string())
            .with_context("column_len", column.len().to_string())
            .with_context("num_rows", table.num_rows().to_string()));
        }
        roots.push(column);
    }
    for root in &roots {
        walk.visit(root)?;
    }
    Ok((roots, walk.finish()?))
}

fn semantic_column_map(columns: &[SemanticColumn]) -> Result<HashMap<ColumnId, &SemanticColumn>> {
    let mut map = HashMap::with_capacity(columns.len());
    for column in columns {
        if map.insert(column.id(), column).is_some() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export semantic columns must not reuse one semantic column id",
            )
            .with_context("column_id", column.id().get().to_string()));
        }
    }
    Ok(map)
}

fn binding_map<'g, 'b>(bindings: &'g [BufferBinding<'b>]) -> Result<HashMap<BufferId, &'g BufferBinding<'b>>> {
    let mut map = HashMap::with_capacity(bindings.len());
    for binding in bindings {
        if map.insert(binding.id(), binding).is_some() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export bindings must not reuse one buffer id",
            )
            .with_context("buffer_id", binding.id().get().to_string()));
        }
    }
    Ok(map)
}

/// Depth-first walk over one export graph. `visiting` holds the current path
/// for cycle detection; `validated` memoizes columns shared by several parents
/// and records the closure checked against the graph by `finish`.
struct GraphWalk<'g, 'b> {
    semantic_columns: &'g [SemanticColumn],
    columns: HashMap<ColumnId, &'g SemanticColumn>,
    bindings: HashMap<BufferId, &'g BufferBinding<'b>>,
    expected: ExpectedPlacement,
    device: Option<i32>,
    visiting: HashSet<ColumnId>,
    validated: HashSet<ColumnId>,
}

impl<'g, 'b> GraphWalk<'g, 'b> {
    fn new(
        semantic_columns: &'g [SemanticColumn],
        bindings: &'g [BufferBinding<'b>],
        expected: ExpectedPlacement,
    ) -> Result<Self> {
        Ok(Self {
            semantic_columns,
            columns: semantic_column_map(semantic_columns)?,
            bindings: binding_map(bindings)?,
            expected,
            device: None,
            visiting: HashSet::new(),
            validated: HashSet::with_capacity(semantic_columns.len()),
        })
    }

    fn root(&self, column_id: ColumnId) -> Result<&'g SemanticColumn> {
        self.columns.get(&column_id).copied().ok_or_else(|| {
            InteropError::new(
                InteropErrorKind::NotFound,
                "prepared export root column must be present in the semantic column graph",
            )
            .with_context("column_id", column_id.get().to_string())
        })
    }

    /// Requires every graph column to have been reached from a root and
    /// returns the device shared by all bindings in device mode.
    fn finish(self) -> Result<Option<i32>> {
        if let Some(unreferenced) = self.semantic_columns.iter().find(|column| !self.validated.contains(&column.id())) {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export included an unreferenced semantic column",
            )
            .with_context("column_id", unreferenced.id().get().to_string()));
        }
        Ok(self.device)
    }

    fn visit(&mut self, column: &'g SemanticColumn) -> Result<()> {
        if self.validated.contains(&column.id()) {
            return Ok(());
        }
        if !self.visiting.insert(column.id()) {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export semantic column graph must not contain cycles",
            )
            .with_context("column_id", column.id().get().to_string()));
        }
        validate_semantic_layout_matches_data_type(column)?;

        match column.layout() {
            ColumnLayout::FixedWidth { values } => {
                let binding = self.binding(column, *values)?;
                require_byte_len(
                    column,
                    *values,
                    binding,
                    fixed_width_required_bytes(column)?,
                    "prepared export value binding is too small for the active logical range",
                )?;
            },
            ColumnLayout::BinaryLike { offsets, data } => {
                let binding = self.binding(column, *offsets)?;
                require_byte_len(
                    column,
                    *offsets,
                    binding,
                    binary_like_offsets_required_bytes(column)?,
                    "prepared export binary-like offsets binding is too small for the active logical range",
                )?;
                self.binding(column, *data)?;
            },
            ColumnLayout::List { offsets, child } => {
                let binding = self.binding(column, *offsets)?;
                require_byte_len(
                    column,
                    *offsets,
                    binding,
                    binary_like_offsets_required_bytes(column)?,
                    "prepared export list offsets binding is too small for the active logical range",
                )?;
                self.visit_child(
                    column,
                    *child,
                    "prepared export list layout referenced a missing child semantic column",
                )?;
            },
            ColumnLayout::Struct { children } => {
                if children.is_empty() {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export struct layout must reference at least one child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string()));
                }
                for child in children {
                    self.visit_child(
                        column,
                        *child,
                        "prepared export struct layout referenced a missing child semantic column",
                    )?;
                }
            },
            ColumnLayout::Dictionary { indices, keys } => {
                let binding = self.binding(column, *indices)?;
                require_byte_len(
                    column,
                    *indices,
                    binding,
                    dictionary_indices_required_bytes(column)?,
                    "prepared export dictionary indices binding is too small for the active logical range",
                )?;
                self.visit_child(
                    column,
                    *keys,
                    "prepared export dictionary layout referenced a missing keys semantic column",
                )?;
            },
        }

        if let PhysicalValidity::ValidityMask { mask, null_count } = column.validity() {
            self.validate_mask(column, *mask, *null_count)?;
        }

        self.visiting.remove(&column.id());
        self.validated.insert(column.id());
        Ok(())
    }

    fn visit_child(&mut self, parent: &SemanticColumn, child: ColumnId, missing_message: &'static str) -> Result<()> {
        let child_column = self.columns.get(&child).copied().ok_or_else(|| {
            InteropError::new(InteropErrorKind::NotFound, missing_message)
                .with_context("column_id", parent.id().get().to_string())
                .with_context("child_column_id", child.get().to_string())
        })?;
        self.visit(child_column)
    }

    fn validate_mask(&mut self, column: &SemanticColumn, mask: BufferId, null_count: Option<usize>) -> Result<()> {
        if let Some(null_count) = null_count
            && null_count > column.len()
        {
            return Err(InteropError::new(InteropErrorKind::InvalidView, "null count exceeds logical length")
                .with_context("column_id", column.id().get().to_string())
                .with_context("null_count", null_count.to_string())
                .with_context("len", column.len().to_string()));
        }

        let mask_binding = self.binding(column, mask)?;
        let required_bits = column.offset().checked_add(column.len()).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "required null-mask range overflowed")
                .with_context("column_id", column.id().get().to_string())
                .with_context("offset", column.offset().to_string())
                .with_context("len", column.len().to_string())
        })?;
        let bit_len = mask_binding.bytes().byte_len().checked_mul(8).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "null-mask bit range overflowed")
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", mask.get().to_string())
                .with_context("byte_len", mask_binding.bytes().byte_len().to_string())
        })?;
        if bit_len < required_bits {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export validity mask is too small for the active logical range",
            )
            .with_context("column_id", column.id().get().to_string())
            .with_context("buffer_id", mask.get().to_string())
            .with_context("bit_len", bit_len.to_string())
            .with_context("required_bits", required_bits.to_string()));
        }
        Ok(())
    }

    /// Resolves `buffer_id` and checks its placement. In device mode the first
    /// binding fixes the graph device; every later binding must match it.
    fn binding(&mut self, column: &SemanticColumn, buffer_id: BufferId) -> Result<&'g BufferBinding<'b>> {
        let binding = self.bindings.get(&buffer_id).copied().ok_or_else(|| {
            InteropError::new(InteropErrorKind::NotFound, "prepared export referenced a missing buffer binding")
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", buffer_id.get().to_string())
        })?;
        match (self.expected, *binding.placement()) {
            (ExpectedPlacement::Host, Placement::Host) => {},
            (ExpectedPlacement::Host, Placement::CudaDevice { device }) => {
                return Err(InteropError::new(
                    InteropErrorKind::PlacementUnsupported,
                    "prepared host export requires host buffer placement",
                )
                .with_context("buffer_id", buffer_id.get().to_string())
                .with_context("placement", "cuda_device")
                .with_context("device", device.to_string()));
            },
            (ExpectedPlacement::Device, Placement::Host) => {
                return Err(InteropError::new(
                    InteropErrorKind::PlacementUnsupported,
                    "prepared device export requires CUDA-device buffer placement",
                )
                .with_context("buffer_id", buffer_id.get().to_string())
                .with_context("placement", "host"));
            },
            (ExpectedPlacement::Device, Placement::CudaDevice { device }) => match self.device {
                None => self.device = Some(device),
                Some(expected) if expected != device => {
                    return Err(InteropError::new(
                        InteropErrorKind::MixedDeviceUnsupported,
                        "prepared export bindings must live on one CUDA device",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", buffer_id.get().to_string())
                    .with_context("expected_device", expected.to_string())
                    .with_context("actual_device", device.to_string()));
                },
                Some(_) => {},
            },
        }
        Ok(binding)
    }
}

fn require_byte_len(
    column: &SemanticColumn,
    buffer_id: BufferId,
    binding: &BufferBinding<'_>,
    required_bytes: usize,
    message: &'static str,
) -> Result<()> {
    let byte_len = binding.bytes().byte_len();
    if byte_len < required_bytes {
        return Err(InteropError::new(InteropErrorKind::InvalidView, message)
            .with_context("column_id", column.id().get().to_string())
            .with_context("buffer_id", buffer_id.get().to_string())
            .with_context("byte_len", byte_len.to_string())
            .with_context("required_bytes", required_bytes.to_string()));
    }
    Ok(())
}

fn fixed_width_required_bytes(column: &SemanticColumn) -> Result<usize> {
    let elem_size = column.data_type().id().fixed_width_size_bytes().ok_or_else(|| {
        InteropError::new(
            InteropErrorKind::UnsupportedType,
            "prepared export fixed-width layout requires a fixed-width logical type",
        )
        .with_context("column_id", column.id().get().to_string())
        .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase())
    })?;
    let active_elements = column.offset().checked_add(column.len()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export active element range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })?;
    active_elements.checked_mul(elem_size).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export active byte range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
            .with_context("elem_size", elem_size.to_string())
    })
}

fn binary_like_offsets_required_bytes(column: &SemanticColumn) -> Result<usize> {
    let required_offsets =
        column.offset().checked_add(column.len()).and_then(|v| v.checked_add(1)).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "prepared export binary-like offsets range overflowed")
                .with_context("column_id", column.id().get().to_string())
                .with_context("offset", column.offset().to_string())
                .with_context("len", column.len().to_string())
        })?;
    required_offsets.checked_mul(core::mem::size_of::<i32>()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export binary-like offsets byte range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })
}

fn dictionary_indices_required_bytes(column: &SemanticColumn) -> Result<usize> {
    if column.data_type().id() != TypeId::Dictionary32 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedType,
            "prepared export dictionary layout requires the dictionary32 logical type in the current slice",
        )
        .with_context("column_id", column.id().get().to_string())
        .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
    }

    let active_elements = column.offset().checked_add(column.len()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export dictionary index range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })?;
    active_elements.checked_mul(core::mem::size_of::<i32>()).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "prepared export dictionary index byte range overflowed")
            .with_context("column_id", column.id().get().to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("len", column.len().to_string())
    })
}

fn validate_semantic_layout_matches_data_type(column: &SemanticColumn) -> Result<()> {
    match column.layout() {
        ColumnLayout::FixedWidth { .. } => {},
        ColumnLayout::BinaryLike { .. } => {
            if column.data_type().id() != TypeId::String {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export binary-like layout requires the string logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
        ColumnLayout::List { .. } => {
            if column.data_type().id() != TypeId::List {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export list layout requires the list logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
        ColumnLayout::Struct { .. } => {
            if column.data_type().id() != TypeId::Struct {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export struct layout requires the struct logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
        ColumnLayout::Dictionary { .. } => {},
    }

    Ok(())
}

pub(crate) fn validate_completion_authority(producer_device: i32, completion: CompletionAuthority<'_>) -> Result<()> {
    match completion {
        CompletionAuthority::SynchronousByReturn => Ok(()),
        CompletionAuthority::SameDeviceStream { stream } => {
            let consumer_device = stream.device();
            if consumer_device != producer_device {
                return Err(InteropError::new(
                    InteropErrorKind::ReadinessMismatch,
                    "same-device stream completion must use a consumer stream on the producer CUDA device",
                )
                .with_context("completion", "same_device_stream")
                .with_context("producer_device", producer_device.to_string())
                .with_context("consumer_device", consumer_device.to_string()));
            }
            Ok(())
        },
    }
}

/// Host leases read host memory and finish before `complete` returns, so they
/// accept only synchronous authorities.
pub(crate) fn validate_host_authorities(
    read_start: ReadStartAuthority<'_>,
    completion: CompletionAuthority<'_>,
) -> Result<()> {
    if let ReadStartAuthority::ConsumerStream { .. } = read_start {
        return Err(host_profile_error("prepared host export does not support CUDA read-start authority")
            .with_context("read_start", "consumer_stream"));
    }
    if let CompletionAuthority::SameDeviceStream { .. } = completion {
        return Err(host_profile_error("prepared host export does not support CUDA completion authority")
            .with_context("completion", "same_device_stream"));
    }
    Ok(())
}
