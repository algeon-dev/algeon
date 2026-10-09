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
}
