//! Native cuDF dtype conversion, delegating the conversion table to the
//! component's own Arrow interop contract (`cudf::interop::arrow_host`).
//!
//! Every other Arrow type support and cast policy helper lives in
//! [`query_planning::datatypes`] and is re-exported here unchanged.

use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use cudf::{DataType as CudfDataType, TimestampResolution};
pub use query_planning::datatypes::{
    arrow_data_types_match_except_list_child_names, arrow_decimal_scale_to_cudf_scale,
    arrow_field_metadata_matches_except_parquet_field_id, arrow_fields_match_except_name, arrow_fixed_width_byte_size,
    diagnostic_arrow_type, is_arrow_fixed_width, is_narrow_decimal_carry_type, is_supported_arrow_cast,
    is_supported_arrow_implicit_cast, is_supported_list_carry_type, is_supported_numeric_widening_cast,
    is_timestamp_with_timezone,
};

/// Converts a supported Arrow type to its cuDF dtype.
///
/// Delegates to the component's own Arrow interop conversion table
/// (`cudf::interop::arrow_host::cudf_dtype_from_arrow_type`), which owns the
/// conversion contract. Unsupported nested types and decimal shapes outside
/// the native policy return `None`. Timezone-aware timestamps and supported
/// list columns are admitted only as carry types; operator validation
/// decides where their semantics are unsupported.
pub fn cudf_dtype_from_arrow_type(data_type: &ArrowDataType) -> Option<CudfDataType> {
    cudf::interop::arrow_host::cudf_dtype_from_arrow_type(data_type).ok()
}

/// Returns whether this Arrow type has a native cuDF physical representation.
pub fn is_supported_arrow_type(data_type: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(data_type).is_some()
}

/// Converts an Arrow temporal unit to the matching cuDF timestamp resolution.
pub fn timestamp_unit_to_cudf_resolution(unit: &TimeUnit) -> TimestampResolution {
    match unit {
        TimeUnit::Second => TimestampResolution::Seconds,
        TimeUnit::Millisecond => TimestampResolution::Milliseconds,
        TimeUnit::Microsecond => TimestampResolution::Microseconds,
        TimeUnit::Nanosecond => TimestampResolution::Nanoseconds,
    }
}
