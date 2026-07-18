use super::owned::DataFrame;

impl std::fmt::Debug for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Debug shows schema, row count, and column types — NO GPU data copy.
        let num_cols = self.columns.len();
        let num_rows = self.num_rows();

        let mut ds = f.debug_struct("DataFrame");
        ds.field("num_columns", &num_cols);
        ds.field("num_rows", &num_rows);

        // Build a columns summary: [(name, type)] without GPU data copy.
        let columns_info: Vec<(&str, String)> = self
            .columns
            .iter()
            .map(|(name, col)| {
                let type_str = col
                    .view()
                    .and_then(|cv| cv.data_type())
                    .map(|dt| format!("{:?}", dt.id()))
                    .unwrap_or_else(|_| "?".to_string());
                (name.as_str(), type_str)
            })
            .collect();
        ds.field("columns", &columns_info);
        ds.finish()
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display_dataframe(self, f)
    }
}

/// Cached schema metadata for one display column.
struct DisplayColumnMetadata {
    type_str: String,
    nullable: Option<bool>,
    null_count: Option<usize>,
}

impl DisplayColumnMetadata {
    fn unknown() -> Self {
        Self { type_str: "?".to_string(), nullable: None, null_count: None }
    }

    fn details(&self) -> Option<String> {
        let mut details = Vec::new();

        if matches!(self.nullable, Some(true)) {
            details.push("nullable".to_string());
        }

        if let Some(null_count) = self.null_count
            && null_count > 0
        {
            details.push(format!("nulls={null_count}"));
        }

        (!details.is_empty()).then(|| details.join(", "))
    }
}

/// Build one display column schema summary from a named DataFrame column.
fn build_display_column_metadata(df: &DataFrame, name: &str) -> DisplayColumnMetadata {
    let col = match df.columns.get(name) {
        Some(col) => col,
        None => return DisplayColumnMetadata::unknown(),
    };

    let cv = match col.view() {
        Ok(cv) => cv,
        Err(_) => return DisplayColumnMetadata::unknown(),
    };

    let type_id = match cv.data_type() {
        Ok(dt) => dt.id(),
        Err(_) => return DisplayColumnMetadata::unknown(),
    };

    DisplayColumnMetadata {
        type_str: format!("{:?}", type_id),
        nullable: cv.nullable().ok(),
        null_count: cv.null_count().ok(),
    }
}

/// Format a DataFrame as a human-readable metadata summary.
///
/// Shows row and column counts plus per-column schema metadata. This is
/// intentionally metadata-only and never materializes cell values to host.
fn display_dataframe(df: &DataFrame, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let num_cols = df.num_columns();
    let num_rows = df.num_rows();
    write!(f, "DataFrame [{} columns x {} rows]", num_cols, num_rows)?;

    if num_cols == 0 {
        return Ok(());
    }

    let names: Vec<&str> = df.column_names();
    let display_cols: Vec<DisplayColumnMetadata> =
        names.iter().map(|&name| build_display_column_metadata(df, name)).collect();

    write!(f, "\ncolumns:")?;
    for (name, meta) in names.iter().zip(display_cols.iter()) {
        write!(f, "\n- {name}: {}", meta.type_str)?;
        if let Some(details) = meta.details() {
            write!(f, " ({details})")?;
        }
    }

    Ok(())
}
