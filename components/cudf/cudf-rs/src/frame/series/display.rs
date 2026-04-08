use super::owned::Series;

struct SeriesDisplayMetadata {
    dtype: String,
    nullable: Option<bool>,
}

impl SeriesDisplayMetadata {
    fn unknown() -> Self {
        Self { dtype: "?".to_string(), nullable: None }
    }
}

fn build_display_metadata(series: &Series) -> SeriesDisplayMetadata {
    let cv = match series.column.view() {
        Ok(cv) => cv,
        Err(_) => return SeriesDisplayMetadata::unknown(),
    };

    let dtype = match cv.data_type() {
        Ok(dtype) => format!("{:?}", dtype.id()),
        Err(_) => return SeriesDisplayMetadata::unknown(),
    };

    SeriesDisplayMetadata { dtype, nullable: cv.nullable().ok() }
}

impl std::fmt::Debug for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let meta = build_display_metadata(self);

        let mut ds = f.debug_struct("Series");
        ds.field("name", &self.name);
        ds.field("dtype", &meta.dtype);
        ds.field("len", &self.cached_len);
        ds.field("nullable", &meta.nullable.unwrap_or(false));
        ds.field("null_count", &self.cached_null_count);
        ds.finish()
    }
}

impl std::fmt::Display for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let meta = build_display_metadata(self);
        let name = self.name.as_deref().unwrap_or("<unnamed>");

        write!(f, "Series [name={name}, dtype={}, len={}", meta.dtype, self.cached_len)?;

        if matches!(meta.nullable, Some(true)) {
            write!(f, ", nullable")?;
        }

        if self.cached_null_count > 0 {
            write!(f, ", nulls={}", self.cached_null_count)?;
        }

        write!(f, "]")
    }
}
