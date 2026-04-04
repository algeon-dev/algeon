use core::ptr;

use rapids_interop::{
    BorrowedColumnView, CudaStreamHandle, DataType, DeviceMemoryResourceHandle, ExecutionBinding, InteropErrorKind,
    NullMask, TypeId,
};

fn fake_ptr(addr: usize) -> *const core::ffi::c_void {
    addr as *const core::ffi::c_void
}

fn explicit_binding() -> ExecutionBinding<'static> {
    ExecutionBinding::explicit(
        0,
        CudaStreamHandle::from_raw(fake_ptr(0x3000) as *mut core::ffi::c_void).unwrap(),
        DeviceMemoryResourceHandle::from_raw(fake_ptr(0x4000) as *mut core::ffi::c_void).unwrap(),
    )
}

#[test]
fn borrowed_column_view_rejects_list_type() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::List),
        0,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "list"));
}

#[test]
fn borrowed_column_view_rejects_struct_type() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Struct),
        0,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "struct"));
}

#[test]
fn borrowed_column_view_rejects_dictionary32_type() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Dictionary32),
        0,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::UnsupportedType);
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "dictionary32"));
}

#[test]
fn borrowed_column_view_rejects_null_data_pointer_for_non_empty_view() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        3,
        0,
        ptr::null(),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "len" && value == "3"));
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "int32"));
}

#[test]
fn borrowed_column_view_rejects_active_range_overflow() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        1,
        usize::MAX,
        fake_ptr(0x1000),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "offset" && value == &usize::MAX.to_string()));
    assert!(err.context().iter().any(|(key, value)| *key == "elem_size" && value == "4"));
}

#[test]
fn borrowed_column_view_rejects_misaligned_active_address() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        2,
        0,
        fake_ptr(0x1001),
        NullMask::absent(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "alignment" && value == "4"));
    assert!(err.context().iter().any(|(key, value)| *key == "type_id" && value == "int32"));
}

#[test]
fn borrowed_column_view_rejects_null_count_exceeding_len() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        2,
        0,
        fake_ptr(0x1000),
        NullMask::present(fake_ptr(0x2000), 2, 3).unwrap(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "null_count" && value == "3"));
    assert!(err.context().iter().any(|(key, value)| *key == "len" && value == "2"));
}

#[test]
fn borrowed_column_view_rejects_undersized_null_mask() {
    let err = BorrowedColumnView::try_new(
        DataType::new(TypeId::Int32),
        3,
        2,
        fake_ptr(0x1000),
        NullMask::present(fake_ptr(0x2000), 4, 1).unwrap(),
        explicit_binding(),
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::InvalidView);
    assert!(err.context().iter().any(|(key, value)| *key == "bit_len" && value == "4"));
    assert!(err.context().iter().any(|(key, value)| *key == "required_bits" && value == "5"));
}
