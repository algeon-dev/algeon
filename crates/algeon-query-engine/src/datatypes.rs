//! Native cuDF dtype conversion facade.
//!
//! Arrow-to-cuDF dtype conversion and native type support are re-exported
//! from the component's own Arrow interop contract
//! (`cudf::interop::arrow_host`). A rejected conversion returns the cuDF error
//! with its code and `reason` detail. Timezone-aware timestamps and supported
//! list columns convert only as carry types; operator validation rejects them
//! where their semantics are unsupported.
//!
//! Every other Arrow type support and cast policy helper lives in
//! [`query_planning::datatypes`] and is re-exported here unchanged.

use arrow::datatypes::TimeUnit;
use cudf::TimestampResolution;
pub use cudf::interop::arrow_host::{cudf_dtype_from_arrow_type, is_supported_arrow_type};
pub use query_planning::datatypes::{
    arrow_data_types_match_except_list_child_names, arrow_decimal_scale_to_cudf_scale,
    arrow_field_metadata_matches_except_parquet_field_id, arrow_fields_match_except_name, arrow_fixed_width_byte_size,
    diagnostic_arrow_type, is_arrow_fixed_width, is_narrow_decimal_carry_type, is_supported_arrow_cast,
    is_supported_arrow_implicit_cast, is_supported_list_carry_type, is_supported_numeric_widening_cast,
    is_timestamp_with_timezone,
};

/// Converts an Arrow temporal unit to the matching cuDF timestamp resolution.
pub fn timestamp_unit_to_cudf_resolution(unit: &TimeUnit) -> TimestampResolution {
    match unit {
        TimeUnit::Second => TimestampResolution::Seconds,
        TimeUnit::Millisecond => TimestampResolution::Milliseconds,
        TimeUnit::Microsecond => TimestampResolution::Microseconds,
        TimeUnit::Nanosecond => TimestampResolution::Nanoseconds,
    }
}
