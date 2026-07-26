use std::{any::Any, marker::PhantomData, mem, sync::Arc};

use cudf::frame::dataframe::{ValidatedFixedSizeListFloat32Export, ValidatedUniformListFloat32Export};
use rapids_interop::{
    ActiveColumnReadLease, ColumnLayout, CompletionAuthority, CompletionWitness, IncompleteDropBehavior,
    LeaseCompletionResult, PhysicalValidity, PreparedColumnExport, PreparedExportProfile, PreparedTableExport,
    ReadStartAuthority,
};

use crate::{
    device::{DeviceElement, DeviceTensor, MatrixLayout, OwnedFloat32Matrix, ValidatedDeviceTensor},
    error::{Result, from_cudf, from_interop, invalid_input},
    resources::CuvsResources,
};

/// An activated prepared-export read plus its lifetime-bound tensor view.
#[must_use = "prepared tensor reads must be completed explicitly after cuVS work has been enqueued"]
pub struct PreparedTensorLease<'export, 'producer, 'consumer, T: DeviceElement> {
    tensor: DeviceTensor<'export, T>,
    lease: Option<ActiveColumnReadLease<'export, 'producer, 'consumer>>,
    export: &'export PreparedColumnExport<'producer>,
    _consumer: PhantomData<&'consumer CuvsResources>,
}

impl<'export, 'producer, 'consumer, T: DeviceElement> PreparedTensorLease<'export, 'producer, 'consumer, T> {
    pub fn tensor(&self) -> &DeviceTensor<'export, T> {
        &self.tensor
    }

    /// Complete producer lifetime protection after all cuVS reads have been
    /// enqueued onto the runtime stream.
    pub fn complete(mut self) -> Result<PreparedReadCompletion> {
        let operation = "PreparedTensorLease::complete";
        let retained_owner = self.export.clone_retained_owner();
        let lease =
            self.lease.take().ok_or_else(|| invalid_input(operation, "prepared tensor lease was already completed"))?;
        PreparedReadCompletion::from_lease_result(
            lease.complete().map_err(|error| from_interop(operation, error))?,
            retained_owner,
            operation,
        )
    }
}

impl<T: DeviceElement> std::fmt::Debug for PreparedTensorLease<'_, '_, '_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedTensorLease")
            .field("tensor", &self.tensor)
            .field("completed", &self.lease.is_none())
            .finish_non_exhaustive()
    }
}

/// A proof-bearing prepared finite `FixedSizeList<Float32>` read and tensor view.
#[must_use = "prepared tensor reads must be completed explicitly after cuVS work has been enqueued"]
pub struct PreparedValidatedTensorLease<'export, 'producer, 'consumer> {
    tensor: ValidatedDeviceTensor<'export>,
    lease: Option<ActiveColumnReadLease<'export, 'producer, 'consumer>>,
    export: &'export PreparedColumnExport<'producer>,
    _consumer: PhantomData<&'consumer CuvsResources>,
}

impl<'export, 'producer, 'consumer> PreparedValidatedTensorLease<'export, 'producer, 'consumer> {
    pub fn tensor(&self) -> &ValidatedDeviceTensor<'export> {
        &self.tensor
    }

    pub fn complete(mut self) -> Result<PreparedReadCompletion> {
        let operation = "PreparedValidatedTensorLease::complete";
        let retained_owner = self.export.clone_retained_owner();
        let lease =
            self.lease.take().ok_or_else(|| invalid_input(operation, "prepared tensor lease was already completed"))?;
        PreparedReadCompletion::from_lease_result(
            lease.complete().map_err(|error| from_interop(operation, error))?,
            retained_owner,
            operation,
        )
    }
}

impl std::fmt::Debug for PreparedValidatedTensorLease<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedValidatedTensorLease")
            .field("tensor", &self.tensor)
            .field("completed", &self.lease.is_none())
            .finish_non_exhaustive()
    }
}

/// Owned completion state for one prepared read.
///
/// When the producer supplied a retained owner, this value keeps that owner
/// alive until the completion witness is synchronized. A producer that does
/// not supply an owner is synchronized before [`PreparedTensorLease::complete`]
/// returns, so this type never carries a witness without its storage owner.
pub struct PreparedReadCompletion {
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    witness: Option<CompletionWitness>,
}

impl PreparedReadCompletion {
    fn from_lease_result(
        result: LeaseCompletionResult,
        retained_owner: Option<Arc<dyn Any + Send + Sync>>,
        operation: &'static str,
    ) -> Result<Self> {
        match result {
            LeaseCompletionResult::FallbackSynchronized => Ok(Self { retained_owner: None, witness: None }),
            LeaseCompletionResult::Completed(witness) => match retained_owner {
                Some(retained_owner) => Ok(Self { retained_owner: Some(retained_owner), witness: Some(witness) }),
                None => {
                    witness.synchronize().map_err(|error| from_interop(operation, error))?;
                    Ok(Self { retained_owner: None, witness: None })
                },
            },
        }
    }

    pub fn witness(&self) -> Option<&CompletionWitness> {
        self.witness.as_ref()
    }

    pub fn retains_producer_owner(&self) -> bool {
        self.retained_owner.is_some()
    }

    pub fn is_ready(&self) -> Result<bool> {
        self.witness.as_ref().map_or(Ok(true), |witness| {
            witness.is_ready().map_err(|error| from_interop("PreparedReadCompletion::is_ready", error))
        })
    }

    /// Synchronize consumer work and release the retained producer owner.
    pub fn synchronize(mut self) -> Result<()> {
        self.synchronize_inner("PreparedReadCompletion::synchronize")?;
        self.retained_owner.take();
        Ok(())
    }

    /// Alias for [`synchronize`](Self::synchronize) that emphasizes owner
    /// release at this boundary.
    pub fn release(self) -> Result<()> {
        self.synchronize()
    }

    fn synchronize_inner(&mut self, operation: &'static str) -> Result<()> {
        if let Some(witness) = self.witness.take()
            && let Err(error) = witness.synchronize()
        {
            self.witness = Some(witness);
            return Err(from_interop(operation, error));
        }
        Ok(())
    }
}

impl Drop for PreparedReadCompletion {
    fn drop(&mut self) {
        if self.synchronize_inner("PreparedReadCompletion::drop").is_err() {
            if let Some(owner) = self.retained_owner.take() {
                // A failed completion fence cannot authorize releasing the
                // producer allocation. Leaking the retained owner is the only
                // sound drop-time fallback because Drop cannot report failure.
                mem::forget(owner);
            }
        } else {
            self.retained_owner.take();
        }
    }
}

impl std::fmt::Debug for PreparedReadCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedReadCompletion")
            .field("retains_producer_owner", &self.retains_producer_owner())
            .field("witness", &self.witness)
            .finish_non_exhaustive()
    }
}

impl CuvsResources {
    pub fn activate_prepared_tensor<'export, 'producer, 'consumer, T: DeviceElement>(
        &'consumer self,
        export: &'export PreparedColumnExport<'producer>,
        shape: &[usize],
        strides: Option<&[usize]>,
    ) -> Result<PreparedTensorLease<'export, 'producer, 'consumer, T>> {
        let operation = "CuvsResources::activate_prepared_tensor";
        self.validate_exact_binding(operation)?;
        require_device_export(export, operation)?;
        let column = export.column();
        if column.data_type().id() != T::INTEROP_TYPE {
            return Err(invalid_input(
                operation,
                format!(
                    "prepared column dtype {:?} does not match tensor element {}",
                    column.data_type().id(),
                    std::any::type_name::<T>()
                ),
            ));
        }
        if !matches!(column.validity(), PhysicalValidity::NoValidityBuffer) {
            return Err(invalid_input(operation, "cuVS tensor inputs must not carry a validity mask"));
        }
        let values = match column.layout() {
            ColumnLayout::FixedWidth { values } => *values,
            _ => return Err(invalid_input(operation, "cuVS prepared tensors require a fixed-width column layout")),
        };
        let binding = export
            .bindings()
            .iter()
            .find(|binding| binding.id() == values)
            .ok_or_else(|| invalid_input(operation, "prepared tensor values binding is missing"))?;
        let binding_device = require_binding_device(binding, self.device().ordinal() as i32, operation)?;
        let logical_elements = shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| invalid_input(operation, "prepared tensor shape overflowed usize"))
        })?;
        if logical_elements != column.len() {
            return Err(invalid_input(
                operation,
                format!("prepared tensor shape contains {logical_elements} elements, column contains {}", column.len()),
            ));
        }
        let byte_offset = column
            .offset()
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| invalid_input(operation, "prepared tensor byte offset overflowed usize"))?;

        let runtime_binding = self.runtime().execution_binding().map_err(|error| from_interop(operation, error))?;
        let stream = runtime_binding.cuda_stream();
        let lease = export
            .try_activate_read_lease(
                ReadStartAuthority::ConsumerStream { stream },
                CompletionAuthority::SameDeviceStream { stream },
            )
            .map_err(|error| from_interop(operation, error))?;
        let tensor = DeviceTensor::from_parts(
            binding.bytes().base_ptr() as *mut std::ffi::c_void,
            binding_device,
            shape,
            strides,
            binding.bytes().byte_len(),
            byte_offset,
            self.binding_id(),
            operation,
        )?;
        Ok(PreparedTensorLease { tensor, lease: Some(lease), export, _consumer: PhantomData })
    }

    /// Activate a structurally and finite-validated `FixedSizeList<Float32>`
    /// child as a row-major tensor under the retained root export's single
    /// read lease.
    ///
    /// The cudf witness proves every active list-offset delta equals the
    /// dimension; a caller-supplied dimension or equal total child length is
    /// never accepted as a substitute for that structural proof.
    pub fn activate_validated_fixed_size_list_tensor<'witness, 'producer, 'consumer>(
        &'consumer self,
        validated: &'witness ValidatedFixedSizeListFloat32Export<'producer>,
    ) -> Result<PreparedValidatedTensorLease<'witness, 'witness, 'consumer>> {
        let operation = "CuvsResources::activate_validated_fixed_size_list_tensor";
        self.validate_exact_binding(operation)?;
        let export = validated.export();
        require_device_export(export, operation)?;
        if validated.binding_id() != self.binding_id() {
            return Err(invalid_input(
                operation,
                "prepared tensor and cuVS resources must use the same exact runtime binding",
            )
            .with_context("tensor_binding_id", validated.binding_id())
            .with_context("resource_binding_id", self.binding_id()));
        }
        let binding = validated.child_values_binding();
        let binding_device = require_binding_device(binding, self.device().ordinal() as i32, operation)?;

        let runtime_binding = self.runtime().execution_binding().map_err(|error| from_interop(operation, error))?;
        let stream = runtime_binding.cuda_stream();
        let lease = export
            .try_activate_read_lease(
                ReadStartAuthority::ConsumerStream { stream },
                CompletionAuthority::SameDeviceStream { stream },
            )
            .map_err(|error| from_interop(operation, error))?;
        let tensor = DeviceTensor::from_parts(
            binding.bytes().base_ptr() as *mut std::ffi::c_void,
            binding_device,
            &[validated.rows(), validated.dimension()],
            None,
            binding.bytes().byte_len(),
            validated.child_byte_offset(),
            validated.binding_id(),
            operation,
        )
        .map(ValidatedDeviceTensor::new)?;
        Ok(PreparedValidatedTensorLease { tensor, lease: Some(lease), export, _consumer: PhantomData })
    }

    /// Activate a structurally and finite-validated runtime-uniform
    /// `List<Float32>` child as a row-major tensor under the retained root
    /// export's single read lease.
    ///
    /// The cuDF witness derives the positive dimension from device-reduced
    /// active list-offset widths. A caller cannot substitute a declared
    /// dimension or total child length for that proof.
    pub fn activate_validated_uniform_list_tensor<'witness, 'producer, 'consumer>(
        &'consumer self,
        validated: &'witness ValidatedUniformListFloat32Export<'producer>,
    ) -> Result<PreparedValidatedTensorLease<'witness, 'witness, 'consumer>> {
        let operation = "CuvsResources::activate_validated_uniform_list_tensor";
        self.validate_exact_binding(operation)?;
        let export = validated.export();
        require_device_export(export, operation)?;
        if validated.binding_id() != self.binding_id() {
            return Err(invalid_input(
                operation,
                "prepared tensor and cuVS resources must use the same exact runtime binding",
            )
            .with_context("tensor_binding_id", validated.binding_id())
            .with_context("resource_binding_id", self.binding_id()));
        }
        let binding = validated.child_values_binding();
        let binding_device = require_binding_device(binding, self.device().ordinal() as i32, operation)?;

        let runtime_binding = self.runtime().execution_binding().map_err(|error| from_interop(operation, error))?;
        let stream = runtime_binding.cuda_stream();
        let lease = export
            .try_activate_read_lease(
                ReadStartAuthority::ConsumerStream { stream },
                CompletionAuthority::SameDeviceStream { stream },
            )
            .map_err(|error| from_interop(operation, error))?;
        let tensor = DeviceTensor::from_parts(
            binding.bytes().base_ptr() as *mut std::ffi::c_void,
            binding_device,
            &[validated.rows(), validated.dimension()],
            None,
            binding.bytes().byte_len(),
            validated.child_byte_offset(),
            validated.binding_id(),
            operation,
        )
        .map(ValidatedDeviceTensor::new)?;
        Ok(PreparedValidatedTensorLease { tensor, lease: Some(lease), export, _consumer: PhantomData })
    }

    /// Copy a proof-validated `FixedSizeList<Float32>` child into a private
    /// matrix in the requested layout on this exact runtime binding.
    pub fn pack_validated_fixed_size_list_float32(
        &self,
        validated: &ValidatedFixedSizeListFloat32Export<'_>,
        layout: MatrixLayout,
    ) -> Result<OwnedFloat32Matrix> {
        self.pack_validated_list_float32_child(
            validated
                .copy_active_child_into_runtime(self.runtime())
                .map_err(|error| from_cudf("CuvsResources::pack_validated_fixed_size_list_float32", error))?,
            validated.rows(),
            validated.dimension(),
            validated.binding_id(),
            layout,
            "CuvsResources::pack_validated_fixed_size_list_float32",
        )
    }

    /// Copy a proof-validated runtime-uniform `List<Float32>` child into a
    /// private matrix in the requested layout on this exact runtime binding.
    pub fn pack_validated_uniform_list_float32(
        &self,
        validated: &ValidatedUniformListFloat32Export<'_>,
        layout: MatrixLayout,
    ) -> Result<OwnedFloat32Matrix> {
        self.pack_validated_list_float32_child(
            validated
                .copy_active_child_into_runtime(self.runtime())
                .map_err(|error| from_cudf("CuvsResources::pack_validated_uniform_list_float32", error))?,
            validated.rows(),
            validated.dimension(),
            validated.binding_id(),
            layout,
            "CuvsResources::pack_validated_uniform_list_float32",
        )
    }

    /// Atomically import one prepared table root and pack all top-level columns
    /// into an owned Float32 matrix on this exact runtime binding.
    pub fn pack_prepared_float32_table(
        &self,
        export: &PreparedTableExport<'_>,
        layout: MatrixLayout,
    ) -> Result<OwnedFloat32Matrix> {
        let operation = "CuvsResources::pack_prepared_float32_table";
        self.validate_exact_binding(operation)?;
        let table = cudf::core::Table::try_import_from_prepared_table_export_into_runtime(export, self.runtime())
            .map_err(|error| from_interop(operation, error))?;
        let columns = table.into_columns().map_err(|error| from_cudf(operation, error))?;
        let column_refs = columns.iter().collect::<Vec<_>>();
        let packed = self
            .runtime()
            .pack_float32_columns(&column_refs, layout.into())
            .map_err(|error| from_cudf(operation, error))?;
        OwnedFloat32Matrix::from_finite_checked_packed(self, packed)
    }

    fn pack_validated_list_float32_child(
        &self,
        child: cudf::core::Column,
        rows: usize,
        dimension: usize,
        producer_binding_id: cudf::runtime::ExecutionBindingId,
        layout: MatrixLayout,
        operation: &'static str,
    ) -> Result<OwnedFloat32Matrix> {
        self.validate_exact_binding(operation)?;
        if producer_binding_id != self.binding_id() {
            return Err(invalid_input(
                operation,
                "validated list input and cuVS resources must use the same exact runtime binding",
            )
            .with_context("tensor_binding_id", producer_binding_id)
            .with_context("resource_binding_id", self.binding_id()));
        }
        let packed = self
            .runtime()
            .pack_flat_float32_column(&child, rows, dimension, layout.into())
            .map_err(|error| from_cudf(operation, error))?;
        OwnedFloat32Matrix::from_finite_checked_packed(self, packed)
    }
}

fn require_device_export(export: &PreparedColumnExport<'_>, operation: &'static str) -> Result<()> {
    if export.profile() != PreparedExportProfile::Device {
        return Err(invalid_input(operation, "cuVS requires a prepared device export"));
    }
    if export.incomplete_drop_behavior() != IncompleteDropBehavior::GuaranteedFallback {
        return Err(invalid_input(
            operation,
            "prepared tensor exports must guarantee safe fallback when an active lease is dropped without completion",
        ));
    }
    Ok(())
}

fn require_binding_device(
    binding: &rapids_interop::BufferBinding<'_>,
    expected_device: i32,
    operation: &'static str,
) -> Result<i32> {
    let device = match binding.placement() {
        rapids_interop::Placement::CudaDevice { device } => *device,
        rapids_interop::Placement::Host => {
            return Err(invalid_input(operation, "cuVS prepared tensor binding must be CUDA device memory"));
        },
    };
    if device != expected_device {
        return Err(invalid_input(operation, "prepared tensor and cuVS resources must use the same CUDA device"));
    }
    Ok(device)
}
