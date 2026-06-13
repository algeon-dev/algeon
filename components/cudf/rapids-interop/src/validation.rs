use std::collections::{HashMap, HashSet};

use crate::{
    BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority, CompletionBinding, ExportDiagnostics,
    ExportProfile, InteropError, InteropErrorKind, LeaseReadiness, PhysicalValidity, Placement, PreparedExportProfile,
    ProducerReadiness, ReadStartAuthority, ReadStartBinding, ReleaseContract, ReleaseSynchronization, Result,
    SemanticColumn, TransferCompletionRule, TypeId,
};

fn export_profile_name(profile: impl Into<ExportProfile>) -> &'static str {
    match profile.into() {
        ExportProfile::BorrowedDeviceCompatibility => "borrowed_device_compatibility",
        ExportProfile::PreparedDeviceExport => "prepared_device_export",
        ExportProfile::PreparedHostExport => "prepared_host_export",
        ExportProfile::OwnedDeviceTransfer => "owned_device_transfer",
    }
}

fn profile_error(profile: impl Into<ExportProfile>, message: &'static str) -> InteropError {
    InteropError::new(InteropErrorKind::ProfileUnsupported, message)
        .with_context("profile", export_profile_name(profile))
}

pub(crate) fn prepared_profile_error(profile: impl Into<ExportProfile>, message: &'static str) -> InteropError {
    profile_error(profile, message)
}

pub(crate) fn owned_transfer_profile_error(profile: ExportProfile, message: &'static str) -> InteropError {
    profile_error(profile, message)
}

pub(crate) fn validate_prepared_profile(diagnostics: ExportDiagnostics) -> Result<PreparedExportProfile> {
    match diagnostics.prepared_profile() {
        Some(profile) => Ok(profile),
        None => Err(prepared_profile_error(
            diagnostics.profile(),
            "prepared export values must declare a prepared export profile in diagnostics",
        )),
    }
}

pub(crate) fn validate_owned_transfer_profile(diagnostics: ExportDiagnostics) -> Result<ExportProfile> {
    match diagnostics.profile() {
        ExportProfile::OwnedDeviceTransfer => Ok(diagnostics.profile()),
        other => Err(owned_transfer_profile_error(
            other,
            "owned transfer values must declare an owned device transfer profile in diagnostics",
        )),
    }
}

pub(crate) fn validate_release_contract(release_contract: ReleaseContract) -> Result<()> {
    if release_contract.release_device() < 0 {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            "owned device transfer release device must be >= 0",
        )
        .with_context("release_device", release_contract.release_device().to_string()));
    }

    match (release_contract.completion(), release_contract.synchronization()) {
        (
            TransferCompletionRule::CallerMustReleaseAfterConsumerWorkCompletes,
            ReleaseSynchronization::Asynchronous,
        ) => Err(owned_transfer_profile_error(
            ExportProfile::OwnedDeviceTransfer,
            "owned device transfer release contract cannot declare caller-proven completion with asynchronous release",
        )
        .with_context("completion", "caller_must_release_after_consumer_work_completes")
        .with_context("synchronization", "asynchronous")),
        (
            TransferCompletionRule::ReleaseWaitsOnConsumerSignal,
            ReleaseSynchronization::RequiresCallerSynchronization,
        ) => Err(owned_transfer_profile_error(
            ExportProfile::OwnedDeviceTransfer,
            "owned device transfer release contract cannot declare signal-waited release with caller-required synchronization",
        )
        .with_context("completion", "release_waits_on_consumer_signal")
        .with_context("synchronization", "requires_caller_synchronization")),
        _ => Ok(()),
    }
}

pub(crate) fn validate_prepared_readiness<'a>(
    profile: PreparedExportProfile,
    readiness: &ProducerReadiness<'a>,
    producer_device: Option<i32>,
) -> Result<()> {
    match profile {
        PreparedExportProfile::Device => {
            let producer_device = producer_device.expect("prepared device export must resolve a producer device");
            match readiness {
                ProducerReadiness::DeviceStreamOrdered { stream } => {
                    let stream_device = stream.device();
                    if stream_device != producer_device {
                        return Err(InteropError::new(
                            InteropErrorKind::ReadinessMismatch,
                            "prepared export readiness stream must live on the producer CUDA device",
                        )
                        .with_context("readiness", "device_stream_ordered")
                        .with_context("producer_device", producer_device.to_string())
                        .with_context("stream_device", stream_device.to_string()));
                    }
                    Ok(())
                },
                ProducerReadiness::HostVisible => {
                    Err(prepared_profile_error(profile, "prepared device export requires device-ordered readiness")
                        .with_context("readiness", "host_visible"))
                },
                ProducerReadiness::DeviceEventOrdered { event } => {
                    let event_device = event.device();
                    if event_device != producer_device {
                        return Err(InteropError::new(
                            InteropErrorKind::ReadinessMismatch,
                            "prepared export readiness event must live on the producer CUDA device",
                        )
                        .with_context("readiness", "device_event_ordered")
                        .with_context("producer_device", producer_device.to_string())
                        .with_context("event_device", event_device.to_string()));
                    }
                    Ok(())
                },
            }
        },
        PreparedExportProfile::Host => match readiness {
            ProducerReadiness::HostVisible => Ok(()),
            ProducerReadiness::DeviceStreamOrdered { .. } => {
                Err(prepared_profile_error(profile, "prepared host export requires host-visible readiness")
                    .with_context("readiness", "device_stream_ordered"))
            },
            ProducerReadiness::DeviceEventOrdered { .. } => {
                Err(prepared_profile_error(profile, "prepared host export requires host-visible readiness")
                    .with_context("readiness", "device_event_ordered"))
            },
        },
    }
}

pub(crate) fn validate_owned_transfer_readiness<'a>(
    readiness: &ProducerReadiness<'a>,
    producer_device: i32,
) -> Result<()> {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            let stream_device = stream.device();
            if stream_device != producer_device {
                return Err(InteropError::new(
                    InteropErrorKind::ReadinessMismatch,
                    "owned device transfer readiness stream must live on the producer CUDA device",
                )
                .with_context("readiness", "device_stream_ordered")
                .with_context("producer_device", producer_device.to_string())
                .with_context("stream_device", stream_device.to_string()));
            }
            Ok(())
        },
        ProducerReadiness::DeviceEventOrdered { event } => {
            let event_device = event.device();
            if event_device != producer_device {
                return Err(InteropError::new(
                    InteropErrorKind::ReadinessMismatch,
                    "owned device transfer readiness event must live on the producer CUDA device",
                )
                .with_context("readiness", "device_event_ordered")
                .with_context("producer_device", producer_device.to_string())
                .with_context("event_device", event_device.to_string()));
            }
            Ok(())
        },
        ProducerReadiness::HostVisible => Err(owned_transfer_profile_error(
            ExportProfile::OwnedDeviceTransfer,
            "owned device transfer requires device-ordered readiness",
        )
        .with_context("readiness", "host_visible")),
    }
}

pub(crate) fn binding_map<'a>(bindings: &'a [BufferBinding<'a>]) -> Result<HashMap<BufferId, &'a BufferBinding<'a>>> {
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

fn require_binding<'a>(
    map: &HashMap<BufferId, &'a BufferBinding<'a>>,
    buffer_id: BufferId,
    column: &SemanticColumn,
) -> Result<&'a BufferBinding<'a>> {
    map.get(&buffer_id).copied().ok_or_else(|| {
        InteropError::new(InteropErrorKind::NotFound, "prepared export referenced a missing buffer binding")
            .with_context("column_id", column.id().get().to_string())
            .with_context("buffer_id", buffer_id.get().to_string())
    })
}

fn fixed_width_required_bytes(column: &SemanticColumn) -> Result<usize> {
    let elem_size = column.data_type().id().fixed_width_size_bytes().ok_or_else(|| {
        InteropError::new(
            InteropErrorKind::UnsupportedType,
            "prepared device export supports fixed-width logical types only for fixed-width layouts",
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
            "prepared device export dictionary layout requires the dictionary32 logical type in the current slice",
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
        ColumnLayout::FixedWidth { .. } => {
            if column.data_type().id().fixed_width_size_bytes().is_none() {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export fixed-width layout requires a fixed-width logical type",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
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
        ColumnLayout::Dictionary { .. } => {
            if column.data_type().id() != TypeId::Dictionary32 {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedType,
                    "prepared export dictionary layout requires the dictionary32 logical type in the current slice",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            }
        },
    }

    Ok(())
}

pub(crate) fn semantic_column_map<'a>(
    columns: &'a [SemanticColumn],
    context: &'static str,
) -> Result<HashMap<ColumnId, &'a SemanticColumn>> {
    let mut map = HashMap::with_capacity(columns.len());
    for column in columns {
        if map.insert(column.id(), column).is_some() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "prepared export semantic columns must not reuse one semantic column id",
            )
            .with_context(context, column.id().get().to_string()));
        }
    }
    Ok(map)
}

fn require_cuda_device<'a>(
    binding: &'a BufferBinding<'a>,
    kind: InteropErrorKind,
) -> Result<(i32, &'a BufferBinding<'a>)> {
    match binding.placement() {
        Placement::CudaDevice { device } => Ok((*device, binding)),
        Placement::Host => Err(InteropError::new(kind, "prepared device export requires CUDA-device buffer placement")
            .with_context("buffer_id", binding.id().get().to_string())
            .with_context("placement", "host")),
    }
}

fn require_host_binding<'a>(binding: &'a BufferBinding<'a>, kind: InteropErrorKind) -> Result<&'a BufferBinding<'a>> {
    match binding.placement() {
        Placement::Host => Ok(binding),
        Placement::CudaDevice { device } => {
            Err(InteropError::new(kind, "prepared host export requires host buffer placement")
                .with_context("buffer_id", binding.id().get().to_string())
                .with_context("placement", "cuda_device")
                .with_context("device", device.to_string()))
        },
    }
}

pub(crate) fn validate_prepared_host_column_against_bindings<'a>(
    column: &SemanticColumn,
    columns: &HashMap<ColumnId, &'a SemanticColumn>,
    bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    validated_columns: &mut HashSet<ColumnId>,
) -> Result<()> {
    validate_semantic_layout_matches_data_type(column)?;

    fn validate_host_validity_against_bindings<'a>(
        column: &SemanticColumn,
        bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    ) -> Result<()> {
        if let PhysicalValidity::ValidityMask { mask, null_count } = column.validity() {
            if let Some(null_count) = null_count {
                if *null_count > column.len() {
                    return Err(InteropError::new(InteropErrorKind::InvalidView, "null count exceeds logical length")
                        .with_context("column_id", column.id().get().to_string())
                        .with_context("null_count", null_count.to_string())
                        .with_context("len", column.len().to_string()));
                }
            }

            let mask_binding = require_binding(bindings, *mask, column)?;
            require_host_binding(mask_binding, InteropErrorKind::PlacementUnsupported)?;

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
        }

        Ok(())
    }

    fn validate_host_leaf_or_string_column_against_bindings<'a>(
        column: &SemanticColumn,
        bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    ) -> Result<()> {
        match column.layout() {
            ColumnLayout::FixedWidth { values } => {
                let values_binding = require_binding(bindings, *values, column)?;
                require_host_binding(values_binding, InteropErrorKind::PlacementUnsupported)?;
                let required_bytes = fixed_width_required_bytes(column)?;
                if values_binding.bytes().byte_len() < required_bytes {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export value binding is too small for the active logical range",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", values.get().to_string())
                    .with_context("byte_len", values_binding.bytes().byte_len().to_string())
                    .with_context("required_bytes", required_bytes.to_string()));
                }
            },
            ColumnLayout::BinaryLike { offsets, data } => {
                let offsets_binding = require_binding(bindings, *offsets, column)?;
                require_host_binding(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
                let required_bytes = binary_like_offsets_required_bytes(column)?;
                if offsets_binding.bytes().byte_len() < required_bytes {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export binary-like offsets binding is too small for the active logical range",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", offsets.get().to_string())
                    .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                    .with_context("required_bytes", required_bytes.to_string()));
                }

                let data_binding = require_binding(bindings, *data, column)?;
                require_host_binding(data_binding, InteropErrorKind::PlacementUnsupported)?;
            },
            _ => {
                return Err(InteropError::new(
                    InteropErrorKind::UnsupportedLayout,
                    "prepared host export currently supports fixed-width and binary-like string column layouts only on the non-list path",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("type_id", format!("{:?}", column.data_type().id()).to_lowercase()));
            },
        }
        validate_host_validity_against_bindings(column, bindings)
    }

    fn validate_host_column_with_current_supported_child_projection<'a>(
        column: &SemanticColumn,
        columns: &HashMap<ColumnId, &'a SemanticColumn>,
        bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
        validated_columns: &mut HashSet<ColumnId>,
    ) -> Result<()> {
        validated_columns.insert(column.id());

        match column.layout() {
            ColumnLayout::List { offsets, child } => {
                let offsets_binding = require_binding(bindings, *offsets, column)?;
                require_host_binding(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
                let required_bytes = binary_like_offsets_required_bytes(column)?;
                if offsets_binding.bytes().byte_len() < required_bytes {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "prepared export list offsets binding is too small for the active logical range",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", offsets.get().to_string())
                    .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                    .with_context("required_bytes", required_bytes.to_string()));
                }

                let child_column = columns.get(child).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::NotFound,
                        "prepared host export list layout referenced a missing child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                })?;
                validate_prepared_host_column_against_bindings(child_column, columns, bindings, validated_columns)?;
                validate_host_validity_against_bindings(column, bindings)?;
            },
            _ => validate_host_leaf_or_string_column_against_bindings(column, bindings)?,
        }

        Ok(())
    }

    match column.layout() {
        ColumnLayout::Struct { children } => {
            validated_columns.insert(column.id());
            if children.is_empty() {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared host export struct layout must reference at least one child semantic column",
                )
                .with_context("column_id", column.id().get().to_string()));
            }

            for child in children {
                let child_column = columns.get(child).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::NotFound,
                        "prepared host export struct layout referenced a missing child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                })?;
                validate_prepared_host_column_against_bindings(child_column, columns, bindings, validated_columns)?;
            }
            validate_host_validity_against_bindings(column, bindings)
        },
        ColumnLayout::Dictionary { indices, keys } => {
            validated_columns.insert(column.id());

            let indices_binding = require_binding(bindings, *indices, column)?;
            require_host_binding(indices_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = dictionary_indices_required_bytes(column)?;
            if indices_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export dictionary indices binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", indices.get().to_string())
                .with_context("byte_len", indices_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let keys_column = columns.get(keys).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::NotFound,
                    "prepared host export dictionary layout referenced a missing keys semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", keys.get().to_string())
            })?;
            validate_prepared_host_column_against_bindings(keys_column, columns, bindings, validated_columns)?;
            validate_host_validity_against_bindings(column, bindings)
        },
        _ => validate_host_column_with_current_supported_child_projection(column, columns, bindings, validated_columns),
    }
}

pub(crate) fn validate_prepared_column_against_bindings<'a>(
    column: &SemanticColumn,
    columns: &HashMap<ColumnId, &'a SemanticColumn>,
    bindings: &HashMap<BufferId, &'a BufferBinding<'a>>,
    cross_binding_kind: InteropErrorKind,
    visiting: &mut HashSet<ColumnId>,
    validated_devices: &mut HashMap<ColumnId, i32>,
) -> Result<i32> {
    validate_semantic_layout_matches_data_type(column)?;

    if let Some(device) = validated_devices.get(&column.id()) {
        return Ok(*device);
    }
    if !visiting.insert(column.id()) {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            "prepared export semantic column graph must not contain cycles",
        )
        .with_context("column_id", column.id().get().to_string()));
    }

    let device = match column.layout() {
        ColumnLayout::FixedWidth { values } => {
            let values_binding = require_binding(bindings, *values, column)?;
            let (device, _) = require_cuda_device(values_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = fixed_width_required_bytes(column)?;
            if values_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export value binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", values.get().to_string())
                .with_context("byte_len", values_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }
            device
        },
        ColumnLayout::BinaryLike { offsets, data } => {
            let offsets_binding = require_binding(bindings, *offsets, column)?;
            let (device, _) = require_cuda_device(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = binary_like_offsets_required_bytes(column)?;
            if offsets_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export binary-like offsets binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", offsets.get().to_string())
                .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let data_binding = require_binding(bindings, *data, column)?;
            let (data_device, _) = require_cuda_device(data_binding, InteropErrorKind::PlacementUnsupported)?;
            if data_device != device {
                return Err(InteropError::new(
                    cross_binding_kind,
                    "prepared export bindings must live on one CUDA device",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", data.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", data_device.to_string()));
            }
            device
        },
        ColumnLayout::List { offsets, child } => {
            let offsets_binding = require_binding(bindings, *offsets, column)?;
            let (device, _) = require_cuda_device(offsets_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = binary_like_offsets_required_bytes(column)?;
            if offsets_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export list offsets binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", offsets.get().to_string())
                .with_context("byte_len", offsets_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let child_column = columns.get(child).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::NotFound,
                    "prepared export list layout referenced a missing child semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", child.get().to_string())
            })?;
            let child_device = validate_prepared_column_against_bindings(
                child_column,
                columns,
                bindings,
                cross_binding_kind,
                visiting,
                validated_devices,
            )?;
            if child_device != device {
                return Err(InteropError::new(
                    cross_binding_kind,
                    "prepared export bindings must live on one CUDA device",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", child.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", child_device.to_string()));
            }
            device
        },
        ColumnLayout::Struct { children } => {
            let mut child_iter = children.iter();
            let first_child = child_iter.next().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export struct layout must reference at least one child semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
            })?;
            let first_child_column = columns.get(first_child).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::NotFound,
                    "prepared export struct layout referenced a missing child semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", first_child.get().to_string())
            })?;
            let device = validate_prepared_column_against_bindings(
                first_child_column,
                columns,
                bindings,
                cross_binding_kind,
                visiting,
                validated_devices,
            )?;
            for child in child_iter {
                let child_column = columns.get(child).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::NotFound,
                        "prepared export struct layout referenced a missing child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                })?;
                let child_device = validate_prepared_column_against_bindings(
                    child_column,
                    columns,
                    bindings,
                    cross_binding_kind,
                    visiting,
                    validated_devices,
                )?;
                if child_device != device {
                    return Err(InteropError::new(
                        cross_binding_kind,
                        "prepared export bindings must live on one CUDA device",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                    .with_context("expected_device", device.to_string())
                    .with_context("actual_device", child_device.to_string()));
                }
            }
            device
        },
        ColumnLayout::Dictionary { indices, keys } => {
            let indices_binding = require_binding(bindings, *indices, column)?;
            let (device, _) = require_cuda_device(indices_binding, InteropErrorKind::PlacementUnsupported)?;
            let required_bytes = dictionary_indices_required_bytes(column)?;
            if indices_binding.bytes().byte_len() < required_bytes {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "prepared export dictionary indices binding is too small for the active logical range",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", indices.get().to_string())
                .with_context("byte_len", indices_binding.bytes().byte_len().to_string())
                .with_context("required_bytes", required_bytes.to_string()));
            }

            let keys_column = columns.get(keys).copied().ok_or_else(|| {
                InteropError::new(
                    InteropErrorKind::NotFound,
                    "prepared export dictionary layout referenced a missing keys semantic column",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", keys.get().to_string())
            })?;
            let keys_device = validate_prepared_column_against_bindings(
                keys_column,
                columns,
                bindings,
                cross_binding_kind,
                visiting,
                validated_devices,
            )?;
            if keys_device != device {
                return Err(InteropError::new(
                    cross_binding_kind,
                    "prepared export bindings must live on one CUDA device",
                )
                .with_context("column_id", column.id().get().to_string())
                .with_context("child_column_id", keys.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", keys_device.to_string()));
            }
            device
        },
    };

    if let PhysicalValidity::ValidityMask { mask, null_count } = column.validity() {
        if let Some(null_count) = null_count {
            if *null_count > column.len() {
                return Err(InteropError::new(InteropErrorKind::InvalidView, "null count exceeds logical length")
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("null_count", null_count.to_string())
                    .with_context("len", column.len().to_string()));
            }
        }

        let mask_binding = require_binding(bindings, *mask, column)?;
        let (mask_device, _) = require_cuda_device(mask_binding, InteropErrorKind::PlacementUnsupported)?;
        if mask_device != device {
            return Err(InteropError::new(cross_binding_kind, "prepared export bindings must live on one CUDA device")
                .with_context("column_id", column.id().get().to_string())
                .with_context("buffer_id", mask.get().to_string())
                .with_context("expected_device", device.to_string())
                .with_context("actual_device", mask_device.to_string()));
        }

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
    }

    visiting.remove(&column.id());
    validated_devices.insert(column.id(), device);
    Ok(device)
}

pub(crate) fn validate_completion_authority(
    profile: PreparedExportProfile,
    completion: CompletionAuthority<'_>,
    producer_device: Option<i32>,
) -> Result<()> {
    match profile {
        PreparedExportProfile::Host => match completion {
            CompletionAuthority::SynchronousByReturn => Ok(()),
            CompletionAuthority::SameDeviceStream { .. } => {
                Err(prepared_profile_error(profile, "prepared host export does not support CUDA completion authority")
                    .with_context("completion", "same_device_stream"))
            },
            CompletionAuthority::CrossDeviceEvent { .. } => {
                Err(prepared_profile_error(profile, "prepared host export does not support CUDA completion authority")
                    .with_context("completion", "cross_device_event"))
            },
        },
        PreparedExportProfile::Device => {
            let producer_device = producer_device.expect("prepared device export must resolve a producer device");
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
                CompletionAuthority::CrossDeviceEvent { event } => {
                    let event_device = event.device();
                    if event_device == producer_device {
                        return Err(InteropError::new(
                            InteropErrorKind::InvalidView,
                            "cross-device event completion authority must use an event on a non-producer CUDA device; use SameDeviceStream for same-device completion",
                        )
                        .with_context("completion", "cross_device_event")
                        .with_context("event_device", event_device.to_string())
                        .with_context("producer_device", producer_device.to_string()));
                    }

                    Ok(())
                },
            }
        },
    }
}

pub(crate) fn lease_readiness_binding(readiness: ProducerReadiness<'_>) -> LeaseReadiness {
    match readiness {
        ProducerReadiness::HostVisible => LeaseReadiness::HostVisible,
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            LeaseReadiness::DeviceStreamOrdered { raw: stream.as_raw(), device: stream.device() }
        },
        ProducerReadiness::DeviceEventOrdered { event } => {
            LeaseReadiness::DeviceEventOrdered { raw: event.as_raw(), device: event.device() }
        },
    }
}

pub(crate) fn validate_read_start_authority(
    profile: PreparedExportProfile,
    read_start: ReadStartAuthority<'_>,
    producer_device: Option<i32>,
) -> Result<()> {
    match profile {
        PreparedExportProfile::Host => match read_start {
            ReadStartAuthority::SynchronousByReturn => Ok(()),
            ReadStartAuthority::ConsumerStream { .. } => {
                Err(prepared_profile_error(profile, "prepared host export does not support CUDA read-start authority")
                    .with_context("read_start", "consumer_stream"))
            },
        },
        PreparedExportProfile::Device => {
            let _ = producer_device.expect("prepared device export must resolve a producer device");
            match read_start {
                ReadStartAuthority::SynchronousByReturn => Ok(()),
                ReadStartAuthority::ConsumerStream { .. } => Ok(()),
            }
        },
    }
}

pub(crate) fn read_start_binding(read_start: ReadStartAuthority<'_>) -> ReadStartBinding {
    match read_start {
        ReadStartAuthority::SynchronousByReturn => ReadStartBinding::SynchronousByReturn,
        ReadStartAuthority::ConsumerStream { stream } => {
            ReadStartBinding::ConsumerStream { raw: stream.as_raw(), device: stream.device() }
        },
    }
}

pub(crate) fn completion_binding(completion: CompletionAuthority<'_>) -> CompletionBinding {
    match completion {
        CompletionAuthority::SynchronousByReturn => CompletionBinding::SynchronousByReturn,
        CompletionAuthority::SameDeviceStream { stream } => {
            CompletionBinding::SameDeviceStream { raw: stream.as_raw(), device: stream.device() }
        },
        CompletionAuthority::CrossDeviceEvent { event } => {
            CompletionBinding::CrossDeviceEvent { raw: event.as_raw(), device: event.device() }
        },
    }
}

pub(crate) fn first_buffer_id(column: &SemanticColumn) -> Option<BufferId> {
    match column.layout() {
        ColumnLayout::FixedWidth { values } => Some(*values),
        ColumnLayout::BinaryLike { offsets, .. } => Some(*offsets),
        ColumnLayout::List { offsets, .. } => Some(*offsets),
        ColumnLayout::Struct { .. } => None,
        ColumnLayout::Dictionary { indices, .. } => Some(*indices),
    }
}
