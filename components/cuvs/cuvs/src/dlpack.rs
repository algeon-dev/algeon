use std::{ffi::c_void, ptr};

use crate::{device::MatrixLayout, error::Result};

pub struct DlpackTensor {
    tensor: cuvs_sys::DLManagedTensor,
    _shape: Box<[i64]>,
    _strides: Option<Box<[i64]>>,
}

impl DlpackTensor {
    pub(crate) fn device_matrix_f32(data: *mut f32, rows: usize, cols: usize, layout: MatrixLayout) -> Result<Self> {
        let strides = match layout {
            MatrixLayout::RowMajor => None,
            MatrixLayout::ColumnMajor => Some(vec![1_i64, rows as i64]),
        };
        Ok(Self::new(
            data.cast::<c_void>(),
            cuvs_sys::DLDeviceType::kDLCUDA,
            cuvs_sys::DLDataType { code: cuvs_sys::DLDataTypeCode::kDLFloat as u8, bits: 32, lanes: 1 },
            vec![rows as i64, cols as i64],
            strides,
        ))
    }

    pub(crate) fn device_vector_f32(data: *mut f32, len: usize) -> Self {
        Self::new(
            data.cast::<c_void>(),
            cuvs_sys::DLDeviceType::kDLCUDA,
            cuvs_sys::DLDataType { code: cuvs_sys::DLDataTypeCode::kDLFloat as u8, bits: 32, lanes: 1 },
            vec![len as i64],
            None,
        )
    }

    pub(crate) fn device_vector_i32(data: *mut i32, len: usize) -> Self {
        Self::new(
            data.cast::<c_void>(),
            cuvs_sys::DLDeviceType::kDLCUDA,
            cuvs_sys::DLDataType { code: cuvs_sys::DLDataTypeCode::kDLInt as u8, bits: 32, lanes: 1 },
            vec![len as i64],
            None,
        )
    }

    pub(crate) fn device_matrix_i64(data: *mut i64, rows: usize, cols: usize) -> Self {
        Self::new(
            data.cast::<c_void>(),
            cuvs_sys::DLDeviceType::kDLCUDA,
            cuvs_sys::DLDataType { code: cuvs_sys::DLDataTypeCode::kDLInt as u8, bits: 64, lanes: 1 },
            vec![rows as i64, cols as i64],
            None,
        )
    }

    pub(crate) fn new(
        data: *mut c_void,
        device_type: cuvs_sys::DLDeviceType,
        dtype: cuvs_sys::DLDataType,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
    ) -> Self {
        let mut shape = shape.into_boxed_slice();
        let mut strides = strides.map(Vec::into_boxed_slice);
        let strides_ptr = strides.as_mut().map_or(ptr::null_mut(), |strides| strides.as_mut_ptr());

        let tensor = cuvs_sys::DLManagedTensor {
            dl_tensor: cuvs_sys::DLTensor {
                data,
                device: cuvs_sys::DLDevice { device_type, device_id: 0 },
                ndim: shape.len() as i32,
                dtype,
                shape: shape.as_mut_ptr(),
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        };

        Self { tensor, _shape: shape, _strides: strides }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        &mut self.tensor
    }
}
