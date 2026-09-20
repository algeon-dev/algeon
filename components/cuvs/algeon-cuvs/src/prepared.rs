use std::{marker::PhantomData, mem};

use cudf::{
    TypeId,
    core::Table,
    frame::dataframe::{ValidatedFixedSizeListExport, ValidatedUniformListExport},
    interop::contract::{
        ActiveColumnReadLease, BufferBinding, ColumnLayout, CompletionAuthority, LeaseCompletion, PhysicalValidity,
        Placement, PreparedColumnExport, PreparedExportProfile, PreparedTableExport, ReadStartAuthority,
    },
};

use crate::{
    device::{DeviceElement, DeviceTensor, MatrixLayout, OwnedMatrix, ValidatedDeviceTensor},
    error::{Result, from_cudf, from_interop, invalid_input},
    resources::CuvsResources,
};

/// An activated prepared-export read plus its lifetime-bound tensor view.
#[must_use = "prepared tensor reads must be completed explicitly after cuVS work has been enqueued"]
pub struct PreparedTensorLease<'export, 'producer, 'consumer, T: DeviceElement> {
    tensor: DeviceTensor<'export, T>,
    lease: Option<ActiveColumnReadLease<'export, 'producer, 'consumer>>,
    _consumer: PhantomData<&'consumer CuvsResources>,
}

impl<'export, T: DeviceElement> PreparedTensorLease<'export, '_, '_, T> {
    pub fn tensor(&self) -> &DeviceTensor<'export, T> {
        &self.tensor
    }

    /// Complete producer lifetime protection after all cuVS reads have been
    /// enqueued onto the runtime stream.
    pub fn complete(mut self) -> Result<LeaseCompletion> {
        let operation = "PreparedTensorLease::complete";
        let lease =
            self.lease.take().ok_or_else(|| invalid_input(operation, "prepared tensor lease was already completed"))?;
        lease.complete().map_err(|error| from_interop(operation, error))
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

/// A prepared list read with floating finite validation or integer structural validation.
#[must_use = "prepared tensor reads must be completed explicitly after cuVS work has been enqueued"]
pub struct PreparedValidatedTensorLease<'export, 'producer, 'consumer, T: DeviceElement> {
    tensor: ValidatedDeviceTensor<'export, T>,
    lease: Option<ActiveColumnReadLease<'export, 'producer, 'consumer>>,
    _consumer: PhantomData<&'consumer CuvsResources>,
}

impl<'export, T: DeviceElement> PreparedValidatedTensorLease<'export, '_, '_, T> {
    pub fn tensor(&self) -> &ValidatedDeviceTensor<'export, T> {
        &self.tensor
    }

    pub fn complete(mut self) -> Result<LeaseCompletion> {
        let operation = "PreparedValidatedTensorLease::complete";
        let lease =
            self.lease.take().ok_or_else(|| invalid_input(operation, "prepared tensor lease was already completed"))?;
        lease.complete().map_err(|error| from_interop(operation, error))
    }
}

impl<T: DeviceElement> std::fmt::Debug for PreparedValidatedTensorLease<'_, '_, '_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedValidatedTensorLease")
            .field("tensor", &self.tensor)
            .field("completed", &self.lease.is_none())
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
        Ok(PreparedTensorLease { tensor, lease: Some(lease), _consumer: PhantomData })
    }

    /// Activate a structurally validated numeric `FixedSizeList`
    /// child as a row-major tensor under the retained root export's single
    /// read lease.
    ///
    /// The cudf witness proves every active list-offset delta equals the
    /// dimension; a caller-supplied dimension or equal total child length is
    /// never accepted as a substitute for that structural proof.
    pub fn activate_validated_fixed_size_list_tensor<'witness, 'producer, 'consumer, T: DeviceElement>(
        &'consumer self,
        validated: &'witness ValidatedFixedSizeListExport<'producer>,
    ) -> Result<PreparedValidatedTensorLease<'witness, 'witness, 'consumer, T>> {
        let operation = "CuvsResources::activate_validated_fixed_size_list_tensor";
        self.validate_exact_binding(operation)?;
        require_element::<T>(validated.element(), operation)?;
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
        Ok(PreparedValidatedTensorLease { tensor, lease: Some(lease), _consumer: PhantomData })
    }

    /// Activate a structurally validated runtime-uniform
    /// numeric `List` child as a row-major tensor under the retained root
    /// export's single read lease.
    ///
    /// The cuDF witness derives the positive dimension from device-reduced
    /// active list-offset widths. A caller cannot substitute a declared
    /// dimension or total child length for that proof.
    pub fn activate_validated_uniform_list_tensor<'witness, 'producer, 'consumer, T: DeviceElement>(
        &'consumer self,
        validated: &'witness ValidatedUniformListExport<'producer>,
    ) -> Result<PreparedValidatedTensorLease<'witness, 'witness, 'consumer, T>> {
        let operation = "CuvsResources::activate_validated_uniform_list_tensor";
        self.validate_exact_binding(operation)?;
        require_element::<T>(validated.element(), operation)?;
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
        Ok(PreparedValidatedTensorLease { tensor, lease: Some(lease), _consumer: PhantomData })
    }

    /// Copy a validated numeric `FixedSizeList` child into a private
    /// matrix in the requested layout on this exact runtime binding.
    pub fn pack_validated_fixed_size_list<T: DeviceElement>(
        &self,
        validated: &ValidatedFixedSizeListExport<'_>,
        layout: MatrixLayout,
    ) -> Result<OwnedMatrix<T>> {
        let operation = "CuvsResources::pack_validated_fixed_size_list";
        self.validate_exact_binding(operation)?;
        require_element::<T>(validated.element(), operation)?;
        if validated.binding_id() != self.binding_id() {
            return Err(invalid_input(
                operation,
                "validated list input and cuVS resources must use the same exact runtime binding",
            )
            .with_context("tensor_binding_id", validated.binding_id())
            .with_context("resource_binding_id", self.binding_id()));
        }
        let packed = validated
            .pack_active_child_into_runtime(self.runtime(), layout.into())
            .map_err(|error| from_cudf(operation, error))?;
        OwnedMatrix::from_packed(self, packed)
    }

    /// Copy a validated runtime-uniform numeric `List` child into a
    /// private matrix in the requested layout on this exact runtime binding.
    pub fn pack_validated_uniform_list<T: DeviceElement>(
        &self,
        validated: &ValidatedUniformListExport<'_>,
        layout: MatrixLayout,
    ) -> Result<OwnedMatrix<T>> {
        let operation = "CuvsResources::pack_validated_uniform_list";
        self.validate_exact_binding(operation)?;
        require_element::<T>(validated.element(), operation)?;
        if validated.binding_id() != self.binding_id() {
            return Err(invalid_input(
                operation,
                "validated list input and cuVS resources must use the same exact runtime binding",
            )
            .with_context("tensor_binding_id", validated.binding_id())
            .with_context("resource_binding_id", self.binding_id()));
        }
        let packed = validated
            .pack_active_child_into_runtime(self.runtime(), layout.into())
            .map_err(|error| from_cudf(operation, error))?;
        OwnedMatrix::from_packed(self, packed)
    }

    /// Atomically import one prepared table root and pack all top-level columns
    /// into an owned numeric matrix on this exact runtime binding.
    pub fn pack_prepared_table<T: DeviceElement>(
        &self,
        export: &PreparedTableExport<'_>,
        layout: MatrixLayout,
    ) -> Result<OwnedMatrix<T>> {
        let operation = "CuvsResources::pack_prepared_table";
        self.validate_exact_binding(operation)?;
        let table = Table::try_import_from_prepared_table_export_into_runtime(export, self.runtime())
            .map_err(|error| from_interop(operation, error))?;
        let columns = table.into_columns().map_err(|error| from_cudf(operation, error))?;
        let column_refs = columns.iter().collect::<Vec<_>>();
        let packed =
            self.runtime().pack_columns(&column_refs, layout.into()).map_err(|error| from_cudf(operation, error))?;
        OwnedMatrix::from_packed(self, packed)
    }
}

fn require_element<T: DeviceElement>(element: TypeId, operation: &'static str) -> Result<()> {
    if element != T::TYPE_ID {
        return Err(invalid_input(operation, "validated element does not match requested cuVS element")
            .with_context("validated_element", format!("{element:?}"))
            .with_context("requested_element", format!("{:?}", T::TYPE_ID)));
    }
    Ok(())
}

fn require_device_export(export: &PreparedColumnExport<'_>, operation: &'static str) -> Result<()> {
    if export.profile() != PreparedExportProfile::Device {
        return Err(invalid_input(operation, "cuVS requires a prepared device export"));
    }
    Ok(())
}

fn require_binding_device(binding: &BufferBinding<'_>, expected_device: i32, operation: &'static str) -> Result<i32> {
    let device = match binding.placement() {
        Placement::CudaDevice { device } => *device,
        Placement::Host => {
            return Err(invalid_input(operation, "cuVS prepared tensor binding must be CUDA device memory"));
        },
    };
    if device != expected_device {
        return Err(invalid_input(operation, "prepared tensor and cuVS resources must use the same CUDA device"));
    }
    Ok(device)
}
