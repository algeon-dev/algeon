//! `Series` datetime namespace for GPU datetime operations.

use super::{context::SeriesOps, owned::Series};
use crate::{core::error::Result, ops::datetime::DatetimeComponent, runtime::Exec};

impl<'a> SeriesOps<'a> {
    /// Enter the datetime namespace for timestamp-specific series methods.
    pub fn dt(&self) -> SeriesDatetimeOps<'a> {
        SeriesDatetimeOps { series: self.series, ctx: self.ctx }
    }
}

/// Datetime operations of one [`Series`], run where an [`Exec`] selects.
#[must_use]
pub struct SeriesDatetimeOps<'a> {
    series: &'a Series,
    ctx: Exec<'a>,
}

impl SeriesDatetimeOps<'_> {
    /// Extract the calendar year as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements.
    ///
    /// Equivalent to cuDF Python's `Series.dt.year`.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn year(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Year, self.ctx, "series::dt::year")
    }

    /// Extract the calendar month as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Months are one-based.
    ///
    /// Equivalent to cuDF Python's `Series.dt.month`.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn month(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Month, self.ctx, "series::dt::month")
    }

    /// Extract the day of month as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Days are one-based.
    ///
    /// Equivalent to cuDF Python's `Series.dt.day`.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn day(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Day, self.ctx, "series::dt::day")
    }

    /// Extract the hour of the day as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Hours range from 0 through 23.
    ///
    /// Equivalent to cuDF Python's `Series.dt.hour`.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn hour(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Hour, self.ctx, "series::dt::hour")
    }

    /// Extract the minute of the hour as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Minutes range from 0 through 59.
    ///
    /// Equivalent to cuDF Python's `Series.dt.minute`.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn minute(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Minute, self.ctx, "series::dt::minute")
    }

    /// Extract the second of the minute as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Seconds range from 0 through 59.
    ///
    /// Equivalent to cuDF Python's `Series.dt.second`.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn second(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Second, self.ctx, "series::dt::second")
    }

    /// Extract the millisecond segment within the second as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Values range from 0 through 999 and are not a total of
    /// milliseconds since midnight.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn millisecond(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Millisecond, self.ctx, "series::dt::millisecond")
    }

    /// Extract the microsecond segment within the millisecond as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Values range from 0 through 999 and are not a total of
    /// microseconds since midnight.
    ///
    /// Equivalent to cuDF Python's `Series.dt.microsecond`.
    ///
    /// # Cost
    /// Allocates a new `INT16` device column.
    pub fn microsecond(&self) -> Result<Series> {
        self.series.datetime_component_with_context(DatetimeComponent::Microsecond, self.ctx, "series::dt::microsecond")
    }
}
