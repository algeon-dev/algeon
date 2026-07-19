use crate::{
    core::{column::Column, error::Result, types::CudfType},
    runtime::Runtime,
};

mod private {
    use crate::core::types::CudfType;

    pub trait Sealed {}

    impl<T: CudfType> Sealed for T {}
    impl Sealed for &str {}
}

/// Host slice element types supported by the high-level `Series` and `Runtime`
/// constructors.
///
/// This trait is sealed because host-to-device construction semantics are part
/// of cudf's safety and performance contract.
pub trait HostColumnData: private::Sealed {
    /// Create a GPU-backed column inside an explicit runtime.
    fn create_column_in_runtime(runtime: &Runtime, data: &[Self]) -> Result<Column>
    where Self: Sized;

    /// Create a nullable GPU-backed column inside an explicit runtime.
    fn create_column_nullable_in_runtime(runtime: &Runtime, data: &[Self], validity: &[bool]) -> Result<Column>
    where Self: Sized;
}

impl<T: CudfType> HostColumnData for T {
    fn create_column_in_runtime(runtime: &Runtime, data: &[Self]) -> Result<Column> {
        Column::from_slice_with_runtime(runtime, data)
    }

    fn create_column_nullable_in_runtime(runtime: &Runtime, data: &[Self], validity: &[bool]) -> Result<Column> {
        Column::from_slice_nullable_with_runtime(runtime, data, validity)
    }
}

impl HostColumnData for &str {
    fn create_column_in_runtime(runtime: &Runtime, data: &[Self]) -> Result<Column> {
        Column::from_strs_with_runtime(runtime, data)
    }

    fn create_column_nullable_in_runtime(runtime: &Runtime, data: &[Self], validity: &[bool]) -> Result<Column> {
        Column::from_strs_nullable_with_runtime(runtime, data, validity)
    }
}
