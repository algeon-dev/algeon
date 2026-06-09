use std::{ops::Deref, sync::Arc};

use crate::{
    core::{
        column::Column,
        error::{Error, ErrorKind, Result},
    },
    frame::series::Series,
    runtime::execution_domain::ExecutionDomain,
};

fn positional_code_series(
    len: usize,
    domain: &Arc<ExecutionDomain>,
    name: Option<&str>,
    decoded_name: Option<&str>,
) -> Result<EncodedSeries> {
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
    let series = match name {
        Some(name) => series.rename(name),
        None => series,
    };
    Ok(EncodedSeries::new(series, decoded_name.map(str::to_string)))
}

/// A series of encoded dictionary indices.
///
/// This wraps a regular [`Series`] but preserves the semantic distinction
/// between row-aligned encoded indices and decoded values.
#[derive(Clone)]
pub struct EncodedSeries {
    series: Series,
    decoded_name: Option<String>,
}

impl EncodedSeries {
    pub(crate) fn new(series: Series, decoded_name: Option<String>) -> Self {
        Self { series, decoded_name }
    }

    pub fn as_series(&self) -> &Series {
        &self.series
    }

    pub fn into_series(self) -> Series {
        self.series
    }

    pub fn decoded_name(&self) -> Option<&str> {
        self.decoded_name.as_deref()
    }
}

impl AsRef<Series> for EncodedSeries {
    fn as_ref(&self) -> &Series {
        self.as_series()
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
    fallback_decoded_name: Option<String>,
}

impl Codebook {
    pub(crate) fn from_keys(keys: Series, fallback_decoded_name: Option<String>) -> Result<Self> {
        let code_name = keys.name().map(|name| format!("{name}_codes"));
        let codes = positional_code_series(
            keys.len(),
            keys.as_column().execution_domain(),
            code_name.as_deref(),
            fallback_decoded_name.as_deref(),
        )?;
        Ok(Self { keys, codes, fallback_decoded_name })
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
        let encoded = values.as_column_view()?.map(&self.keys.as_column_view()?, &self.codes.as_column_view()?)?;
        let encoded = Series::from_column(encoded)?;
        let encoded = match values.name() {
            Some(name) => encoded.rename(format!("{name}_codes")),
            None => encoded,
        };
        Ok(EncodedSeries::new(encoded, values.name().map(str::to_string)))
    }

    pub fn decode(&self, codes: &EncodedSeries) -> Result<Series> {
        let decoded = self.keys.gather(codes.as_series())?;
        let name = codes.decoded_name().or(self.fallback_decoded_name.as_deref());
        Ok(match name {
            Some(name) => decoded.rename(name),
            None => decoded,
        })
    }

    pub fn decode_series(&self, codes: &Series) -> Result<Series> {
        let decoded = self.keys.gather(codes)?;
        Ok(match self.fallback_decoded_name.as_deref() {
            Some(name) => decoded.rename(name),
            None => decoded,
        })
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
