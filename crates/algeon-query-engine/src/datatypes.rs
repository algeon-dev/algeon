//! Native cuDF dtype conversion facade.
//!
//! Arrow-to-cuDF dtype conversion and native type support are re-exported
//! from the component's own Arrow interop contract
//! (`cudf::interop::arrow_host`). A rejected conversion returns the cuDF error
//! with its code and `reason` detail. Timezone-aware timestamps and supported
//! list columns convert only as carry types; operator validation rejects them
//! where their semantics are unsupported.
//!
//! The Arrow type support and cast policy helpers exposed here live in
//! [`query_planning::datatypes`] and are re-exported unchanged.

use arrow::datatypes::TimeUnit;
use cudf::TimestampResolution;
pub use cudf::interop::arrow_host::{cudf_dtype_from_arrow_type, is_supported_arrow_type};
pub(crate) use query_planning::datatypes::{
    arrow_decimal_scale_to_cudf_scale, arrow_fixed_width_byte_size, is_arrow_fixed_width,
};
pub use query_planning::datatypes::{
    diagnostic_arrow_type, is_narrow_decimal_carry_type, is_supported_arrow_cast, is_supported_arrow_implicit_cast,
    is_supported_list_carry_type, is_timestamp_with_timezone,
};

/// Converts an Arrow temporal unit to the matching cuDF timestamp resolution.
pub(crate) fn timestamp_unit_to_cudf_resolution(unit: &TimeUnit) -> TimestampResolution {
    match unit {
        TimeUnit::Second => TimestampResolution::Seconds,
        TimeUnit::Millisecond => TimestampResolution::Milliseconds,
        TimeUnit::Microsecond => TimestampResolution::Microseconds,
        TimeUnit::Nanosecond => TimestampResolution::Nanoseconds,
    }
}
