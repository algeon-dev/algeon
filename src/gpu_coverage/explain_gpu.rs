//! Parsing for the small Flight SQL `EXPLAIN GPU` extension.

use crate::error::{Error, ErrorCode, Result};

/// The shared coverage-row projection requested by an `EXPLAIN GPU` statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainGpuOutputMode {
    /// Return the summary row followed by candidate rows.
    Rows,
    /// Return the summary row, whose `coverage_json` contains the complete evidence.
    Json,
}

/// A parsed `EXPLAIN GPU` Flight SQL extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainGpuStatement {
    pub inner_sql: String,
    pub output_mode: ExplainGpuOutputMode,
}

/// Parse the narrow `EXPLAIN GPU` prefix without parsing or rewriting the inner SQL.
///
/// `None` means the statement is an ordinary DataFusion statement. Once the
/// exact `EXPLAIN GPU` prefix is present, malformed extension options produce
/// a structured request error instead of being delegated to DataFusion's
/// unrelated `EXPLAIN` implementation.
pub fn parse_explain_gpu(sql: &str) -> Result<Option<ExplainGpuStatement>> {
    let Some(after_explain) = consume_keyword(sql, "EXPLAIN") else {
        return Ok(None);
    };
    let Some(after_gpu) = consume_keyword(after_explain, "GPU") else {
        return Ok(None);
    };

    let after_gpu = required_inner_sql(after_gpu, "EXPLAIN GPU requires a query after the GPU prefix")?;
    if let Some(after_format) = consume_keyword(after_gpu, "FORMAT") {
        let Some(after_json) = consume_keyword(after_format, "JSON") else {
            return Err(invalid_syntax(
                "EXPLAIN GPU FORMAT requires JSON; supported forms are EXPLAIN GPU [VERBOSE|FORMAT JSON] <query>",
            ));
        };
        let inner_sql = required_inner_sql(after_json, "EXPLAIN GPU FORMAT JSON requires a query")?;
        reject_repeated_option(inner_sql)?;
        return Ok(Some(ExplainGpuStatement {
            inner_sql: inner_sql.to_string(),
            output_mode: ExplainGpuOutputMode::Json,
        }));
    }
    if let Some(after_verbose) = consume_keyword(after_gpu, "VERBOSE") {
        let inner_sql = required_inner_sql(after_verbose, "EXPLAIN GPU VERBOSE requires a query")?;
        reject_repeated_option(inner_sql)?;
        return Ok(Some(ExplainGpuStatement {
            inner_sql: inner_sql.to_string(),
            output_mode: ExplainGpuOutputMode::Rows,
        }));
    }

    Ok(Some(ExplainGpuStatement { inner_sql: after_gpu.to_string(), output_mode: ExplainGpuOutputMode::Rows }))
}

fn consume_keyword<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let input = input.trim_start();
    let token_end = input
        .char_indices()
        .find_map(|(index, character)| (!character.is_ascii_alphabetic()).then_some(index))
        .unwrap_or(input.len());
    let token = &input[..token_end];
    (token.eq_ignore_ascii_case(keyword)
        && (token_end == input.len() || input[token_end..].chars().next().is_some_and(char::is_whitespace)))
    .then_some(&input[token_end..])
}

fn required_inner_sql<'a>(input: &'a str, message: &'static str) -> Result<&'a str> {
    let inner_sql = input.trim_start();
    if inner_sql.is_empty() { Err(invalid_syntax(message)) } else { Ok(inner_sql) }
}

fn reject_repeated_option(inner_sql: &str) -> Result<()> {
    if consume_keyword(inner_sql, "FORMAT").is_some() || consume_keyword(inner_sql, "VERBOSE").is_some() {
        return Err(invalid_syntax(
            "EXPLAIN GPU accepts one option only; supported forms are EXPLAIN GPU [VERBOSE|FORMAT JSON] <query>",
        ));
    }
    Ok(())
}

fn invalid_syntax(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.gpu_coverage.explain_gpu.parse", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_forms_without_rewriting_the_inner_query() {
        let default = parse_explain_gpu("EXPLAIN GPU SELECT 'GPU' AS label").unwrap().unwrap();
        assert_eq!(default.output_mode, ExplainGpuOutputMode::Rows);
        assert_eq!(default.inner_sql, "SELECT 'GPU' AS label");

        let verbose = parse_explain_gpu(" explain\n gpu\tverbose\nSELECT 1 ").unwrap().unwrap();
        assert_eq!(verbose.output_mode, ExplainGpuOutputMode::Rows);
        assert_eq!(verbose.inner_sql, "SELECT 1 ");

        let json = parse_explain_gpu("EXPLAIN GPU FORMAT JSON SELECT 1").unwrap().unwrap();
        assert_eq!(json.output_mode, ExplainGpuOutputMode::Json);
        assert_eq!(json.inner_sql, "SELECT 1");
    }

    #[test]
    fn leaves_regular_explain_to_datafusion() {
        assert!(parse_explain_gpu("EXPLAIN SELECT 1").unwrap().is_none());
        assert!(parse_explain_gpu("EXPLAIN GPUSELECT 1").unwrap().is_none());
        assert!(parse_explain_gpu("SELECT 'EXPLAIN GPU SELECT 1'").unwrap().is_none());
    }

    #[test]
    fn rejects_malformed_or_ambiguous_extension_options() {
        for sql in [
            "EXPLAIN GPU",
            "EXPLAIN GPU VERBOSE",
            "EXPLAIN GPU FORMAT",
            "EXPLAIN GPU FORMAT TEXT SELECT 1",
            "EXPLAIN GPU VERBOSE FORMAT JSON SELECT 1",
            "EXPLAIN GPU FORMAT JSON VERBOSE SELECT 1",
        ] {
            assert!(parse_explain_gpu(sql).is_err(), "{sql}");
        }
    }
}
