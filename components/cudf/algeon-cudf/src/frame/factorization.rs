use std::{ops::Deref, sync::Arc};

#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::Runtime;
use crate::{
    core::{
        column::{Column, position_sequence_in_domain},
        error::{Error, ErrorCode, FACT_DISTINCT_KEYS_COUNT, FACT_KEYS_COUNT, Result},
        scalar::Scalar,
    },
    frame::series::{IntoSeries, Series},
    ops::{copying, lookup},
    runtime::{Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryKeyOrder {
    Sorted,
    FirstSeen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DictionaryEncodingOptions {
    key_order: DictionaryKeyOrder,
}

impl Default for DictionaryEncodingOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DictionaryEncodingOptions {
    pub fn new() -> Self {
        Self { key_order: DictionaryKeyOrder::Sorted }
    }

    pub fn key_order(mut self, key_order: DictionaryKeyOrder) -> Self {
        self.key_order = key_order;
        self
    }

    pub(crate) fn key_order_mode(&self) -> DictionaryKeyOrder {
        self.key_order
    }
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_codebook_key_count_limit_error_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    position_sequence_in_domain((i32::MAX as usize) + 1, runtime.execution_domain(), "codebook::from_keys").map(|_| ())
}

/// A series of encoded dictionary indices.
///
/// This wraps a regular [`Series`] but preserves the semantic distinction
/// between row-aligned encoded indices and decoded values.
#[derive(Clone)]
pub struct EncodedSeries {
    series: Series,
}

impl EncodedSeries {
    pub(crate) fn new(series: Series) -> Self {
        Self { series }
    }

    pub fn as_series(&self) -> &Series {
        &self.series
    }

    pub fn into_series(self) -> Series {
        self.series
    }
}

impl AsRef<Series> for EncodedSeries {
    fn as_ref(&self) -> &Series {
        self.as_series()
    }
}

impl IntoSeries for EncodedSeries {
    fn into_series(self) -> Result<Series> {
        Ok(self.series)
    }
}

impl Deref for EncodedSeries {
    type Target = Series;

    fn deref(&self) -> &Self::Target {
        self.as_series()
    }
}

/// A reusable mapping between distinct values and dense integer codes.
#[derive(Clone)]
pub struct Codebook {
    keys: Series,
    codes: EncodedSeries,
}

impl Codebook {
    pub fn from_keys(keys: Series) -> Result<Self> {
        let operation = "codebook::from_keys";
        let distinct_keys = keys.on(Exec::RETAINED).drop_duplicates().map_err(|e| e.reanchor(operation))?;
        if distinct_keys.len() != keys.len() {
            return Err(Error::new_with(ErrorCode::DuplicateKey, operation, "codebook keys must be unique")
                .with_fact(FACT_KEYS_COUNT, keys.len())
                .with_fact(FACT_DISTINCT_KEYS_COUNT, distinct_keys.len())
                .with_detail("remedy", "deduplicate the key series before constructing a reusable Codebook"));
        }

        let codes = Series::from_column(position_sequence_in_domain(
            keys.len(),
            keys.as_column().execution_domain(),
            operation,
        )?)?;
        let codes = EncodedSeries::new(codes);
        Ok(Self { keys, codes })
    }

    pub fn keys(&self) -> &Series {
        &self.keys
    }

    pub fn codes(&self) -> &EncodedSeries {
        &self.codes
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn encode(&self, values: &Series, ctx: Exec<'_>) -> Result<EncodedSeries> {
        let encoded = lookup::probe_prevalidated_in(
            &values.view()?,
            &self.keys.view()?,
            &self.codes.view()?,
            ctx,
            "codebook::encode",
        )?;
        let encoded = Series::from_column(encoded)?;
        Ok(EncodedSeries::new(encoded))
    }

    pub fn decode(&self, codes: &EncodedSeries, ctx: Exec<'_>) -> Result<Series> {
        self.decode_series(codes.as_series(), ctx).map_err(|e| e.reanchor("codebook::decode"))
    }

    pub fn decode_series(&self, codes: &Series, ctx: Exec<'_>) -> Result<Series> {
        if codes.null_count() == 0 {
            return self.keys.on(ctx).gather(codes).map_err(|e| e.reanchor("codebook::decode_series"));
        }

        let filled_codes = codes.on(ctx).fillna(0i32).map_err(|e| e.reanchor("codebook::decode_series"))?;
        if self.keys.is_empty() {
            return self.all_null_keys_like_in(codes.len(), ctx);
        }

        let decoded = self.keys.on(ctx).gather(&filled_codes).map_err(|e| e.reanchor("codebook::decode_series"))?;
        let nulls = null_series_like_in_domain(
            &decoded,
            decoded.len(),
            decoded.as_column().execution_domain(),
            "codebook::decode_series",
        )?;
        let valid_codes = codes.on(ctx).is_not_null().map_err(|e| e.reanchor("codebook::decode_series"))?;
        let decoded_view = decoded.view().map_err(|e| e.reanchor("codebook::decode_series"))?;
        let nulls_view = nulls.view().map_err(|e| e.reanchor("codebook::decode_series"))?;
        let valid_codes_view = valid_codes.view().map_err(|e| e.reanchor("codebook::decode_series"))?;
        let restored = copying::copy_if_else_in(&decoded_view, &nulls_view, &valid_codes_view, ctx)
            .map_err(|e| e.reanchor("codebook::decode_series"))?;
        Series::from_column(restored).map_err(|e| e.reanchor("codebook::decode_series"))
    }

    pub fn into_parts(self) -> (Series, EncodedSeries) {
        (self.keys, self.codes)
    }

    fn all_null_keys_like_in(&self, len: usize, ctx: Exec<'_>) -> Result<Series> {
        let operation = "codebook::decode_series";
        let execution = ResolvedExecution::unary(self.keys.as_column().execution_domain(), ctx, operation)?;
        null_series_like_in_domain(&self.keys, len, execution.output_domain(), operation)
    }
}

fn null_series_like_in_domain(
    series: &Series,
    len: usize,
    output_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Series> {
    let dtype = series.dtype().map_err(|e| e.reanchor(operation))?;
    let null_scalar = Scalar::null_in_domain(dtype, output_domain).map_err(|e| e.reanchor(operation))?;
    let column = Column::from_scalar(&null_scalar, len).map_err(|e| e.reanchor(operation))?;
    Series::from_column(column).map_err(|e| e.reanchor(operation))
}

/// The result of factorizing one logical value series.
#[derive(Clone)]
pub struct Factorized {
    row_codes: EncodedSeries,
    codebook: Codebook,
}

impl Factorized {
    pub(crate) fn new(row_codes: EncodedSeries, codebook: Codebook) -> Self {
        Self { row_codes, codebook }
    }

    pub fn row_codes(&self) -> &EncodedSeries {
        &self.row_codes
    }

    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    pub fn decode(&self, exec: Exec<'_>) -> Result<Series> {
        self.codebook.decode(&self.row_codes, exec)
    }

    pub fn into_parts(self) -> (EncodedSeries, Codebook) {
        (self.row_codes, self.codebook)
    }
}
