use super::TpchScaleFactor;

/// Number of vendored TPC-H queries.
pub const TPCH_QUERY_COUNT: usize = 22;

/// Canonical TPC-H query table in query-id order.
pub const TPCH_QUERIES: &[TpchQuery] = &[
    TpchQuery::fixed("q01", include_str!("queries/q01.sql")),
    TpchQuery::fixed("q02", include_str!("queries/q02.sql")),
    TpchQuery::fixed("q03", include_str!("queries/q03.sql")),
    TpchQuery::fixed("q04", include_str!("queries/q04.sql")),
    TpchQuery::fixed("q05", include_str!("queries/q05.sql")),
    TpchQuery::fixed("q06", include_str!("queries/q06.sql")),
    TpchQuery::fixed("q07", include_str!("queries/q07.sql")),
    TpchQuery::fixed("q08", include_str!("queries/q08.sql")),
    TpchQuery::fixed("q09", include_str!("queries/q09.sql")),
    TpchQuery::fixed("q10", include_str!("queries/q10.sql")),
    TpchQuery::with_fraction("q11", include_str!("queries/q11.sql")),
    TpchQuery::fixed("q12", include_str!("queries/q12.sql")),
    TpchQuery::fixed("q13", include_str!("queries/q13.sql")),
    TpchQuery::fixed("q14", include_str!("queries/q14.sql")),
    TpchQuery::fixed("q15", include_str!("queries/q15.sql")),
    TpchQuery::fixed("q16", include_str!("queries/q16.sql")),
    TpchQuery::fixed("q17", include_str!("queries/q17.sql")),
    TpchQuery::fixed("q18", include_str!("queries/q18.sql")),
    TpchQuery::fixed("q19", include_str!("queries/q19.sql")),
    TpchQuery::fixed("q20", include_str!("queries/q20.sql")),
    TpchQuery::fixed("q21", include_str!("queries/q21.sql")),
    TpchQuery::fixed("q22", include_str!("queries/q22.sql")),
];

/// Q11 substitution parameter, named as in specification clause 2.4.11.
const FRACTION_PARAMETER: &str = "[FRACTION]";

/// One vendored TPC-H query.
///
/// Q11's text depends on the dataset's scale factor, so every query's SQL is
/// produced by [`TpchQuery::sql`] for an explicit scale.
pub struct TpchQuery {
    pub id: &'static str,
    template: &'static str,
    substitutes_fraction: bool,
}

impl TpchQuery {
    const fn fixed(id: &'static str, template: &'static str) -> Self {
        assert!(occurrences(template, "[") == 0, "a TPC-H query without parameters has a bracketed placeholder");
        Self { id, template, substitutes_fraction: false }
    }

    const fn with_fraction(id: &'static str, template: &'static str) -> Self {
        assert!(
            occurrences(template, FRACTION_PARAMETER) == 1 && occurrences(template, "[") == 1,
            "a TPC-H query with the FRACTION parameter must contain exactly one [FRACTION] placeholder"
        );
        Self { id, template, substitutes_fraction: true }
    }

    /// Returns the query's SQL with qgen's substitution values for a dataset
    /// of scale `scale`.
    pub fn sql(&self, scale: TpchScaleFactor) -> String {
        if self.substitutes_fraction {
            self.template.replacen(FRACTION_PARAMETER, &scale.q11_fraction(), 1)
        } else {
            self.template.to_string()
        }
    }
}

const fn occurrences(haystack: &str, needle: &str) -> usize {
    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();
    let mut count = 0;
    let mut start = 0;
    while start + needle.len() <= haystack.len() {
        let mut offset = 0;
        while offset < needle.len() && haystack[start + offset] == needle[offset] {
            offset += 1;
        }
        if offset == needle.len() {
            count += 1;
        }
        start += 1;
    }
    count
}

pub fn tpch_query_ids() -> impl Iterator<Item = &'static str> {
    TPCH_QUERIES.iter().map(|query| query.id)
}
