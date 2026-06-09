use crate::{
    core::{column::Column, error::Result, types::CudfType},
    runtime::Runtime,
};

mod private {
    use crate::core::types::CudfType;

    pub trait Sealed {}

    impl<T: CudfType> Sealed for T {}
    impl<'a> Sealed for &'a str {}
}

/// Host slice element types supported by the high-level `Series` and `Runtime`
/// constructors.
///
/// This trait is sealed because host-to-device construction semantics are part
/// of cudf-rs's safety and performance contract.
pub trait HostColumnData: private::Sealed {
    /// Create a GPU-backed column from a non-null host slice.
    fn create_column(data: &[Self]) -> Result<Column>
    where Self: Sized;

    /// Create a GPU-backed column from a host slice plus validity mask.
    fn create_column_nullable(data: &[Self], validity: &[bool]) -> Result<Column>
    where Self: Sized;

    /// Create a GPU-backed column inside an explicit runtime.
    fn create_column_in_runtime(runtime: &Runtime, data: &[Self]) -> Result<Column>
    where Self: Sized;

    /// Create a nullable GPU-backed column inside an explicit runtime.
    fn create_column_nullable_in_runtime(runtime: &Runtime, data: &[Self], validity: &[bool]) -> Result<Column>
    where Self: Sized;
}

impl<T: CudfType> HostColumnData for T {
    fn create_column(data: &[Self]) -> Result<Column> {
        Column::from_slice(data)
    }

    fn create_column_nullable(data: &[Self], validity: &[bool]) -> Result<Column> {
        Column::from_slice_nullable(data, validity)
    }

    fn create_column_in_runtime(runtime: &Runtime, data: &[Self]) -> Result<Column> {
        Column::from_slice_with_runtime(runtime, data)
    }

    fn create_column_nullable_in_runtime(runtime: &Runtime, data: &[Self], validity: &[bool]) -> Result<Column> {
        Column::from_slice_nullable_with_runtime(runtime, data, validity)
    }
}

impl<'a> HostColumnData for &'a str {
    fn create_column(data: &[Self]) -> Result<Column> {
        Column::from_strs(data)
    }

    fn create_column_nullable(data: &[Self], validity: &[bool]) -> Result<Column> {
        Column::from_strs_nullable(data, validity)
    }

    fn create_column_in_runtime(runtime: &Runtime, data: &[Self]) -> Result<Column> {
        Column::from_strs_with_runtime(runtime, data)
    }

    fn create_column_nullable_in_runtime(runtime: &Runtime, data: &[Self], validity: &[bool]) -> Result<Column> {
        Column::from_strs_nullable_with_runtime(runtime, data, validity)
    }
}
