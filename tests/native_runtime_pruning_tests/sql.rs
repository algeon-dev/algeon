pub(super) const SQL_SUM_EXPRESSION_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_MIN_MAX_COUNT_QUERY: &str = "
SELECT key, MIN(m00) AS min_m00, MAX(m01) AS max_m01, COUNT(payload) AS count_payload
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

pub(super) const SQL_COUNT_STAR_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00, COUNT(*) AS count_rows
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

pub(super) const SQL_COUNT_ONLY_QUERY: &str = "
SELECT COUNT(*) AS count_rows
FROM bench
";

pub(super) const SQL_AVG_AGGREGATE_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 3
";

pub(super) const SQL_SELECTIVE_READER_PUSHDOWN_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 900
GROUP BY key
ORDER BY key ASC
LIMIT 8
";

pub(super) const SQL_AVG_BY_KEY_NO_FILTER_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 3
";

pub(super) const SQL_AVG_ORDER_BY_VALUE_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 3
";

pub(super) const SQL_FILTERED_GLOBAL_COUNT_QUERY: &str = "
SELECT COUNT(*) FILTER (WHERE payload >= 10) AS count_rows
FROM bench
";

pub(super) const SQL_Q6_SHAPED_QUERY: &str = "
SELECT
    SUM(l_extendedprice * l_discount) AS revenue
FROM bench
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24
";

pub(super) const SQL_GLOBAL_DECIMAL_AVG_QUERY: &str = "
SELECT
    AVG(l_discount) AS avg_discount
FROM bench
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
";

pub(super) const SQL_PHASE2_SCALAR_EXPRESSIONS_QUERY: &str = "
SELECT status_prefix, ship_year, ship_month, status_label, SUM(order_mod) AS sum_mod
FROM (
    SELECT
        SUBSTRING(status FROM 1 FOR 1) AS status_prefix,
        EXTRACT(YEAR FROM shipdate) AS ship_year,
        shipdate + INTERVAL '1' MONTH AS ship_month,
        CASE WHEN status IN ('O', 'P') THEN 'open' ELSE 'other' END AS status_label,
        orderkey % 2 AS order_mod
    FROM bench
    WHERE status LIKE 'O%'
      AND status IS NOT NULL
      AND orderkey IN (1, 3)
      AND orderkey % 2 = 1
) phase2
GROUP BY status_prefix, ship_year, ship_month, status_label
ORDER BY status_prefix ASC, ship_year ASC, ship_month ASC, status_label ASC
LIMIT 10
";

pub(super) const SQL_PHASE2_NULLABLE_SCALAR_EXPRESSIONS_QUERY: &str = "
SELECT
    status_prefix,
    ship_year,
    ship_month,
    status_label,
    not_in_label,
    like_match,
    not_like_match,
    status_is_null,
    status_is_not_null,
    SUM(order_mod) AS sum_mod
FROM (
    SELECT
        SUBSTRING(status FROM 1 FOR 1) AS status_prefix,
        EXTRACT(YEAR FROM shipdate) AS ship_year,
        shipdate + INTERVAL '1' MONTH AS ship_month,
        CASE
            WHEN status IS NULL THEN 'missing'
            WHEN status IN ('O', 'P') THEN 'listed'
            WHEN status NOT LIKE 'O%' THEN 'not_open'
            ELSE 'other'
        END AS status_label,
        CASE WHEN status NOT IN ('F') THEN 'not_f' ELSE 'f_or_null' END AS not_in_label,
        status LIKE 'O%' AS like_match,
        status NOT LIKE 'O%' AS not_like_match,
        status IS NULL AS status_is_null,
        status IS NOT NULL AS status_is_not_null,
        orderkey % 2 AS order_mod
    FROM bench
) phase2
GROUP BY
    status_prefix,
    ship_year,
    ship_month,
    status_label,
    not_in_label,
    like_match,
    not_like_match,
    status_is_null,
    status_is_not_null
ORDER BY
    status_is_null ASC,
    status_prefix ASC NULLS FIRST,
    ship_year ASC NULLS FIRST,
    ship_month ASC NULLS FIRST,
    status_label ASC,
    not_in_label ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_LIKE_QUERY: &str = "
SELECT row_id, like_match, not_like_match, COUNT(*) AS row_count
FROM (
    SELECT row_id, status LIKE 'O%' AS like_match, status NOT LIKE 'O%' AS not_like_match
    FROM bench
) null_semantics
GROUP BY row_id, like_match, not_like_match
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_SUBSTRING_QUERY: &str = "
SELECT row_id, status_prefix, COUNT(*) AS row_count
FROM (
    SELECT row_id, SUBSTRING(status FROM 1 FOR 1) AS status_prefix
    FROM bench
) null_semantics
GROUP BY row_id, status_prefix
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_STRING_CASE_QUERY: &str = "
SELECT row_id, status_lower, status_upper, COUNT(*) AS row_count
FROM (
    SELECT row_id, LOWER(status) AS status_lower, UPPER(status) AS status_upper
    FROM bench
) null_semantics
GROUP BY row_id, status_lower, status_upper
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_IN_LIST_QUERY: &str = "
SELECT row_id, status_in_list, status_not_in_list, COUNT(*) AS row_count
FROM (
    SELECT row_id, status IN ('O', 'P') AS status_in_list, status NOT IN ('F') AS status_not_in_list
    FROM bench
) null_semantics
GROUP BY row_id, status_in_list, status_not_in_list
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_CASE_QUERY: &str = "
SELECT row_id, status_label, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        CASE
            WHEN status IS NULL THEN 'missing'
            WHEN status IN ('O', 'P') THEN 'listed'
            ELSE 'other'
        END AS status_label
    FROM bench
) null_semantics
GROUP BY row_id, status_label
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_CASE_NO_ELSE_QUERY: &str = "
SELECT row_id, open_code, COUNT(*) AS row_count
FROM (
    SELECT row_id, CASE WHEN status = 'O' THEN 1 END AS open_code
    FROM bench
) null_semantics
GROUP BY row_id, open_code
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_EXTRACT_QUERY: &str = "
SELECT row_id, ship_year, ship_month, ship_day, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        EXTRACT(YEAR FROM shipdate) AS ship_year,
        EXTRACT(MONTH FROM shipdate) AS ship_month,
        EXTRACT(DAY FROM shipdate) AS ship_day
    FROM bench
) null_semantics
GROUP BY row_id, ship_year, ship_month, ship_day
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_INTERVAL_QUERY: &str = "
SELECT row_id, shipdate_plus_month, shipdate_minus_day, event_ts_plus_second, event_ts_minus_second, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        shipdate + INTERVAL '1' MONTH AS shipdate_plus_month,
        shipdate - INTERVAL '1' DAY AS shipdate_minus_day,
        event_ts + INTERVAL '1' SECOND AS event_ts_plus_second,
        event_ts - INTERVAL '1' SECOND AS event_ts_minus_second
    FROM bench
) null_semantics
GROUP BY row_id, shipdate_plus_month, shipdate_minus_day, event_ts_plus_second, event_ts_minus_second
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_MODULO_QUERY: &str = "
SELECT row_id, order_mod, COUNT(*) AS row_count
FROM (
    SELECT row_id, orderkey % 2 AS order_mod
    FROM bench
) null_semantics
GROUP BY row_id, order_mod
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE7_BITWISE_SCALAR_QUERY: &str = "
SELECT row_id, flags_and, flags_or, flags_shift_left, flags_shift_right, flags_grouping, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        orderkey & 3 AS flags_and,
        orderkey | 8 AS flags_or,
        orderkey << 1 AS flags_shift_left,
        orderkey >> 1 AS flags_shift_right,
        CAST((orderkey & 2) >> 1 AS INT) AS flags_grouping
    FROM bench
) bitwise_scalars
GROUP BY row_id, flags_and, flags_or, flags_shift_left, flags_shift_right, flags_grouping
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE2_NULL_SEMANTICS_IS_NULL_QUERY: &str = "
SELECT row_id, status_is_null, status_is_not_null, shipdate_is_null, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        status IS NULL AS status_is_null,
        status IS NOT NULL AS status_is_not_null,
        shipdate IS NULL AS shipdate_is_null
    FROM bench
) null_semantics
GROUP BY row_id, status_is_null, status_is_not_null, shipdate_is_null
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE5_ROUND_SCALAR_QUERY: &str = "
SELECT row_id, rounded_orderkey, COUNT(*) AS row_count
FROM (
    SELECT row_id, ROUND(CAST(orderkey AS DOUBLE) + 0.5, 0) AS rounded_orderkey
    FROM bench
) round_scalars
GROUP BY row_id, rounded_orderkey
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_PHASE7_ABS_SCALAR_QUERY: &str = "
SELECT row_id, abs_order_delta, COUNT(*) AS row_count
FROM (
    SELECT row_id, ABS(CAST(orderkey AS DOUBLE) - 4.5) AS abs_order_delta
    FROM bench
) abs_scalars
GROUP BY row_id, abs_order_delta
ORDER BY row_id ASC
LIMIT 20
";

pub(super) const SQL_INVALID_SUBSTRING_ZERO_START_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 0 FOR 1) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

pub(super) const SQL_INVALID_SUBSTRING_DYNAMIC_START_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM orderkey FOR 1) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

pub(super) const SQL_INVALID_SUBSTRING_DYNAMIC_LENGTH_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 1 FOR orderkey) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

pub(super) const SQL_INVALID_SUBSTRING_NEGATIVE_LENGTH_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 1 FOR -1) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

pub(super) const SQL_SUBSTRING_MAX_START_LENGTH_ONE_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 2147483647 FOR 1) AS status_prefix
    FROM bench
) boundary_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

pub(super) const SQL_INVALID_SUBSTRING_STOP_OVERFLOW_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 2147483647 FOR 2) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

pub(super) const SQL_SORT_NULLS_ASC_FIRST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 ASC NULLS FIRST, tie_key DESC
LIMIT 8
";

pub(super) const SQL_SORT_NULLS_ASC_LAST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 ASC NULLS LAST, tie_key DESC
LIMIT 8
";

pub(super) const SQL_SORT_NULLS_DESC_FIRST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 DESC NULLS FIRST, tie_key DESC
LIMIT 8
";

pub(super) const SQL_SORT_NULLS_DESC_LAST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 DESC NULLS LAST, tie_key DESC
LIMIT 8
";

pub(super) const SQL_SORT_NULLS_TOPK_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 ASC NULLS LAST, tie_key DESC
LIMIT 4
";
