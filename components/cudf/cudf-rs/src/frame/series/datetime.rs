//! `Series` datetime namespace for GPU datetime operations.

use super::{context::default_execution_context, owned::Series};
use crate::{core::error::Result, ops::datetime::DatetimeComponent, runtime::ExecutionOverrides};

impl Series {
    /// Enter the datetime namespace for timestamp-specific series methods.
    pub fn dt(&self) -> SeriesDatetimeNamespace<'_> {
        SeriesDatetimeNamespace(self)
    }
}

/// Datetime namespace for [`Series`].
#[must_use]
pub struct SeriesDatetimeNamespace<'series>(&'series Series);

impl SeriesDatetimeNamespace<'_> {
    /// Extract the calendar year as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements.
    ///
    /// Equivalent to cuDF Python's `Series.dt.year`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `INT16` device column.
    pub fn year(&self) -> Result<Series> {
        self.0.datetime_component_with_context(
            DatetimeComponent::Year,
            &default_execution_context(),
            "series::dt::year",
        )
    }

    /// Extract the calendar year using the explicit execution-context override.
    pub fn year_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.datetime_component_with_context(DatetimeComponent::Year, ctx, "series::dt::year_in")
    }

    /// Extract the calendar month as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Months are one-based.
    ///
    /// Equivalent to cuDF Python's `Series.dt.month`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `INT16` device column.
    pub fn month(&self) -> Result<Series> {
        self.0.datetime_component_with_context(
            DatetimeComponent::Month,
            &default_execution_context(),
            "series::dt::month",
        )
    }

    /// Extract the calendar month using the explicit execution-context override.
    pub fn month_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.datetime_component_with_context(DatetimeComponent::Month, ctx, "series::dt::month_in")
    }

    /// Extract the day of month as an `INT16` Series.
    ///
    /// The input must be timestamp typed. Null input elements produce null
    /// output elements. Days are one-based.
    ///
    /// Equivalent to cuDF Python's `Series.dt.day`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `INT16` device column.
    pub fn day(&self) -> Result<Series> {
        self.0.datetime_component_with_context(DatetimeComponent::Day, &default_execution_context(), "series::dt::day")
    }

    /// Extract the day of month using the explicit execution-context override.
    pub fn day_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.datetime_component_with_context(DatetimeComponent::Day, ctx, "series::dt::day_in")
    }
}
