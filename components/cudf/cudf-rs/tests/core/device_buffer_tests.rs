//! Integration tests for DeviceBuffer and zero-copy Column adoption.

use cudf::{
    ErrorKind, TypeId,
    core::{Column, DeviceBuffer},
};

// ---------------------------------------------------------------------------
// DeviceBuffer lifecycle
// ---------------------------------------------------------------------------

#[test]
fn new_allocates_requested_size() {
    let buf = DeviceBuffer::new(1024).unwrap();
    assert_eq!(buf.size_bytes().unwrap(), 1024);
}

#[test]
fn new_zero_bytes_succeeds() {
    let buf = DeviceBuffer::new(0).unwrap();
    assert!(buf.is_empty().unwrap());
    assert_eq!(buf.size_bytes().unwrap(), 0);
}

#[test]
fn data_ptr_non_null_for_non_empty() {
    let buf = DeviceBuffer::new(64).unwrap();
    assert!(!buf.data_ptr().unwrap().is_null());
}

#[test]
fn from_host_data_creates_correct_size() {
    let data: &[i32] = &[1, 2, 3];
    let buf = DeviceBuffer::from_host_data(data).unwrap();
    assert_eq!(buf.size_bytes().unwrap(), 12); // 3 * 4 bytes
}

#[test]
fn from_host_data_empty_slice() {
    let data: &[i32] = &[];
    let buf = DeviceBuffer::from_host_data(data).unwrap();
    assert!(buf.is_empty().unwrap());
    assert_eq!(buf.size_bytes().unwrap(), 0);
}

// ---------------------------------------------------------------------------
// Zero-copy adoption — non-nullable
// ---------------------------------------------------------------------------

#[test]
fn adopt_i32_roundtrip() {
    let data: &[i32] = &[10, 20, 30];
    let buf = DeviceBuffer::from_host_data(data).unwrap();
    let col = Column::from_device_buffer(buf, 3, TypeId::Int32).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn adopt_f64_roundtrip() {
    let data: &[f64] = &[1.5, 2.5, 3.5];
    let buf = DeviceBuffer::from_host_data(data).unwrap();
    let col = Column::from_device_buffer(buf, 3, TypeId::Float64).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.to_vec::<f64>().unwrap(), vec![1.5, 2.5, 3.5]);
}

#[test]
fn adopt_proves_zero_copy() {
    let data: &[i32] = &[42, 99];
    let buf = DeviceBuffer::from_host_data(data).unwrap();
    let original_ptr = buf.data_ptr().unwrap();
    let col = Column::from_device_buffer(buf, 2, TypeId::Int32).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.device_data_ptr().unwrap(), original_ptr as *const _);
}

#[test]
fn adopt_zero_elements() {
    let buf = DeviceBuffer::new(0).unwrap();
    let col = Column::from_device_buffer(buf, 0, TypeId::Int32).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 0);
    assert!(view.is_empty().unwrap());
}

#[test]
fn adopt_rejects_string_type() {
    let buf = DeviceBuffer::new(64).unwrap();
    let err = Column::from_device_buffer(buf, 4, TypeId::String).err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn adopt_rejects_size_mismatch() {
    // 12 bytes = 3 i32 elements, but claim 4 rows
    let data: &[i32] = &[1, 2, 3];
    let buf = DeviceBuffer::from_host_data(data).unwrap();
    let err = Column::from_device_buffer(buf, 4, TypeId::Int32).err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn adopt_rejects_row_count_exceeding_i32_max() {
    let buf = DeviceBuffer::new(0).unwrap();
    let err = Column::from_device_buffer(buf, (i32::MAX as usize) + 1, TypeId::Int32).err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// Zero-copy adoption — nullable
// ---------------------------------------------------------------------------

#[test]
fn adopt_nullable_with_nulls() {
    // Data: [10, 20, 30, 40] with validity [true, false, true, false] → null_count = 2
    let data: &[i32] = &[10, 20, 30, 40];
    let data_buf = DeviceBuffer::from_host_data(data).unwrap();

    // Build a bitmask: bit 0 and bit 2 set → 0b0000_0101 = 0x05
    // cudf requires 64-byte aligned bitmask buffers.
    let mut mask = [0u8; 64];
    mask[0] = 0x05;
    let mask_buf = DeviceBuffer::from_host_data(&mask).unwrap();

    let col = Column::from_device_buffer_nullable(data_buf, mask_buf, 2, 4, TypeId::Int32).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 2);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30, 40]);

    let validity = view.validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true, false]);
}

#[test]
fn adopt_nullable_all_valid() {
    let data: &[i32] = &[1, 2, 3];
    let data_buf = DeviceBuffer::from_host_data(data).unwrap();

    // All valid: bits 0,1,2 set → 0b0000_0111 = 0x07
    // cudf requires 64-byte aligned bitmask buffers.
    let mut mask = [0u8; 64];
    mask[0] = 0x07;
    let mask_buf = DeviceBuffer::from_host_data(&mask).unwrap();

    let col = Column::from_device_buffer_nullable(data_buf, mask_buf, 0, 3, TypeId::Int32).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.null_count().unwrap(), 0);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn adopt_nullable_rejects_null_count_exceeding_row_count() {
    let data_buf = DeviceBuffer::from_host_data(&[1i32, 2]).unwrap();
    let mask_buf = DeviceBuffer::from_host_data(&[0u8; 64]).unwrap();

    let err =
        Column::from_device_buffer_nullable(data_buf, mask_buf, 5, 2, TypeId::Int32).err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn adopt_nullable_rejects_mask_too_small() {
    // 4 rows need at least 1 byte of bitmask. Provide 0 bytes.
    let data_buf = DeviceBuffer::from_host_data(&[1i32, 2, 3, 4]).unwrap();
    let mask_buf = DeviceBuffer::new(0).unwrap();

    let err =
        Column::from_device_buffer_nullable(data_buf, mask_buf, 1, 4, TypeId::Int32).err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn adopt_nullable_rejects_mask_smaller_than_cudf_allocation_size() {
    let data_buf = DeviceBuffer::from_host_data(&[1i32]).unwrap();
    let mask_buf = DeviceBuffer::from_host_data(&[0xffu8; 63]).unwrap();

    let err =
        Column::from_device_buffer_nullable(data_buf, mask_buf, 0, 1, TypeId::Int32).err().expect("expected error");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}
