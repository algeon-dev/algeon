use std::{marker::PhantomData, ptr};

use crate::device::{DeviceElement, DeviceTensor, DeviceTensorMut, TensorParts};

pub(crate) struct DlpackTensor<'a> {
    tensor: cuvs_sys::DLManagedTensor,
    _shape: Box<[i64]>,
    _strides: Option<Box<[i64]>>,
    _borrow: PhantomData<&'a ()>,
}

impl<'a> DlpackTensor<'a> {
    pub(crate) fn from_const<T: DeviceElement>(tensor: &'a DeviceTensor<'_, T>) -> Self {
        Self::from_parts::<T>(tensor.parts())
    }

    pub(crate) fn from_mut<T: DeviceElement>(tensor: &'a mut DeviceTensorMut<'_, T>) -> Self {
        Self::from_parts::<T>(tensor.parts())
    }

    fn from_parts<T: DeviceElement>(parts: TensorParts<'_, T>) -> Self {
        debug_assert!(parts.byte_len > 0);
        let mut shape = parts.shape.to_vec().into_boxed_slice();
        let mut strides = parts.strides.map(|strides| strides.to_vec().into_boxed_slice());
        let strides_ptr = strides.as_mut().map_or(ptr::null_mut(), |strides| strides.as_mut_ptr());
        // Validation bounds and aligns the offset; pinned cuVS reads data directly and select_k requires zero byte_offset.
        let data = parts.data.as_ptr().wrapping_byte_add(parts.byte_offset);
        let tensor = cuvs_sys::DLManagedTensor {
            dl_tensor: cuvs_sys::DLTensor {
                data,
                device: cuvs_sys::DLDevice { device_type: cuvs_sys::DLDeviceType::kDLCUDA, device_id: parts.device },
                ndim: shape.len() as i32,
                dtype: cuvs_sys::DLDataType { code: T::DLPACK_CODE, bits: T::DLPACK_BITS, lanes: 1 },
                shape: shape.as_mut_ptr(),
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        };
        Self { tensor, _shape: shape, _strides: strides, _borrow: PhantomData }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        &mut self.tensor
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use super::*;

    #[test]
    fn descriptor_folds_offset_into_data_and_preserves_device_dtype_shape_strides() {
        let runtime = cudf::Runtime::new().expect("runtime binding");
        let mut backing = [0_f32; 16];
        let data = backing.as_mut_ptr().cast::<c_void>();
        let tensor = DeviceTensor::<f32>::from_parts(
            data,
            7,
            &[2, 3],
            Some(&[5, 1]),
            64,
            8,
            runtime.binding_id(),
            "dlpack::tests::descriptor_preserves_metadata",
        )
        .expect("valid tensor descriptor");
        let mut dlpack = DlpackTensor::from_const(&tensor);
        let raw = unsafe { &(*dlpack.as_mut_ptr()).dl_tensor };

        assert_eq!(raw.data, data.wrapping_byte_add(8));
        assert_eq!(raw.device.device_type, cuvs_sys::DLDeviceType::kDLCUDA);
        assert_eq!(raw.device.device_id, 7);
        assert_eq!(raw.ndim, 2);
        assert_eq!(raw.dtype.code, cuvs_sys::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(raw.dtype.bits, 32);
        assert_eq!(raw.dtype.lanes, 1);
        assert_eq!(unsafe { std::slice::from_raw_parts(raw.shape, 2) }, &[2, 3]);
        assert_eq!(unsafe { std::slice::from_raw_parts(raw.strides, 2) }, &[5, 1]);
        assert_eq!(raw.byte_offset, 0);
    }
}
