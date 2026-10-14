/// Number of vendored TPC-H queries.
pub const TPCH_QUERY_COUNT: usize = 22;

/// Canonical TPC-H query table in query-id order.
pub const TPCH_QUERIES: &[TpchQuery] = &[
    TpchQuery { id: "q01", sql: include_str!("queries/q01.sql") },
    TpchQuery { id: "q02", sql: include_str!("queries/q02.sql") },
    TpchQuery { id: "q03", sql: include_str!("queries/q03.sql") },
    TpchQuery { id: "q04", sql: include_str!("queries/q04.sql") },
    TpchQuery { id: "q05", sql: include_str!("queries/q05.sql") },
    TpchQuery { id: "q06", sql: include_str!("queries/q06.sql") },
    TpchQuery { id: "q07", sql: include_str!("queries/q07.sql") },
    TpchQuery { id: "q08", sql: include_str!("queries/q08.sql") },
    TpchQuery { id: "q09", sql: include_str!("queries/q09.sql") },
    TpchQuery { id: "q10", sql: include_str!("queries/q10.sql") },
    TpchQuery { id: "q11", sql: include_str!("queries/q11.sql") },
    TpchQuery { id: "q12", sql: include_str!("queries/q12.sql") },
    TpchQuery { id: "q13", sql: include_str!("queries/q13.sql") },
    TpchQuery { id: "q14", sql: include_str!("queries/q14.sql") },
    TpchQuery { id: "q15", sql: include_str!("queries/q15.sql") },
    TpchQuery { id: "q16", sql: include_str!("queries/q16.sql") },
    TpchQuery { id: "q17", sql: include_str!("queries/q17.sql") },
    TpchQuery { id: "q18", sql: include_str!("queries/q18.sql") },
    TpchQuery { id: "q19", sql: include_str!("queries/q19.sql") },
    TpchQuery { id: "q20", sql: include_str!("queries/q20.sql") },
    TpchQuery { id: "q21", sql: include_str!("queries/q21.sql") },
    TpchQuery { id: "q22", sql: include_str!("queries/q22.sql") },
];

pub struct TpchQuery {
    pub id: &'static str,
    pub sql: &'static str,
}

pub fn tpch_query_ids() -> impl Iterator<Item = &'static str> {
    TPCH_QUERIES.iter().map(|query| query.id)
}
