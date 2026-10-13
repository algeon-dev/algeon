use super::*;
use crate::ops::datetime::DatetimeRoundingFrequency;

/// Datetime expression methods on timestamp columns.
///
/// Construct via [`Expr::dt`]. Methods on this type produce row-aligned
/// expressions and preserve the input column name. Component extraction
/// returns `INT16`; calendrical offsets return the input timestamp type.
pub struct ExprDatetimeNamespace(pub(super) Expr);

impl ExprDatetimeNamespace {
    /// Extract the calendar year.
    pub fn year(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Year }
    }

    /// Extract the one-based calendar month.
    pub fn month(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Month }
    }

    /// Extract the one-based day of month.
    pub fn day(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Day }
    }

    /// Extract the hour of the day, 0 through 23.
    pub fn hour(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Hour }
    }

    /// Extract the minute of the hour, 0 through 59.
    pub fn minute(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Minute }
    }

    /// Extract the second of the minute, 0 through 59.
    pub fn second(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Second }
    }

    /// Extract the millisecond segment within the second, 0 through 999.
    pub fn millisecond(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Millisecond }
    }

    /// Extract the microsecond segment within the millisecond, 0 through 999.
    pub fn microsecond(self) -> Expr {
        Expr::DateTimeComponent { input: Box::new(self.0), component: DatetimeComponent::Microsecond }
    }

    /// Floor timestamp values to the requested frequency.
    pub fn floor(self, frequency: DatetimeRoundingFrequency) -> Expr {
        Expr::DateTimeFloor { input: Box::new(self.0), frequency }
    }

    /// Add a calendrical interval to a timestamp expression.
    ///
    /// Month offsets use libcudf calendrical month semantics. Day and
    /// nanosecond offsets are applied as fixed durations in the input
    /// timestamp's resolution.
    pub fn offset(self, months: i32, days: i32, nanoseconds: i64) -> Expr {
        Expr::DateTimeOffset { input: Box::new(self.0), months, days, nanoseconds }
    }
}
