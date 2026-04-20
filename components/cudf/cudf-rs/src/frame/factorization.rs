use std::{ops::Deref, sync::Arc};

use crate::{
    core::{
        column::Column,
        error::{Error, ErrorKind, Result},
    },
    frame::series::{IntoSeries, Series},
    ops::lookup,
    runtime::execution_domain::ExecutionDomain,
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

fn positional_code_series(len: usize, domain: &Arc<ExecutionDomain>) -> Result<EncodedSeries> {
    let values: Vec<i32> = (0..len)
        .map(|idx| {
            i32::try_from(idx).map_err(|_| {
                Error::new(ErrorKind::InvalidOperation, "codebook length exceeds i32::MAX")
                    .with_operation("factorization::positional_code_series")
                    .with_context("len", len.to_string())
            })
        })
        .collect::<Result<_>>()?;
    let column = Column::from_slice_in_domain(&values, domain)?;
    let series = Series::from_column(column)?;
    Ok(EncodedSeries::new(series))
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
        let distinct_keys = keys.drop_duplicates().map_err(|e| e.with_operation(operation))?;
        if distinct_keys.len() != keys.len() {
            return Err(Error::new(ErrorKind::InvalidOperation, "codebook keys must be unique")
                .with_operation(operation)
                .with_context("key_count", keys.len().to_string())
                .with_context("distinct_key_count", distinct_keys.len().to_string())
                .with_context("remedy", "deduplicate the key series before constructing a reusable Codebook"));
        }

        let codes = positional_code_series(keys.len(), keys.as_column().execution_domain())?;
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

    pub fn encode(&self, values: &Series) -> Result<EncodedSeries> {
        let encoded =
            lookup::probe_prevalidated(&values.view()?, &self.keys.view()?, &self.codes.view()?, "codebook::encode")?;
        let encoded = Series::from_column(encoded)?;
        Ok(EncodedSeries::new(encoded))
    }

    pub fn decode(&self, codes: &EncodedSeries) -> Result<Series> {
        self.keys.gather(codes.as_series())
    }

    pub fn decode_series(&self, codes: &Series) -> Result<Series> {
        self.keys.gather(codes)
    }

    pub fn into_parts(self) -> (Series, EncodedSeries) {
        (self.keys, self.codes)
    }
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

    pub fn decode(&self) -> Result<Series> {
        self.codebook.decode(&self.row_codes)
    }

    pub fn into_parts(self) -> (EncodedSeries, Codebook) {
        (self.row_codes, self.codebook)
    }
}
