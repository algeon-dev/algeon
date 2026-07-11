pub(super) const SQL_BENCHMARK_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_NO_FILTER_BENCHMARK_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_NATIVE_AGGREGATE_PARITY_QUERY: &str = "
SELECT key, MIN(m00) AS min_m00, MAX(m01) AS max_m01, COUNT(payload) AS count_payload
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

pub(super) const SQL_NATIVE_STATISTICAL_AGGREGATE_QUERY: &str = "
SELECT key,
       STDDEV_POP(m00) AS stddev_pop_m00,
       VAR_SAMP(m00) AS var_samp_m00,
       VAR_POP(m00) AS var_pop_m00,
       VAR(m01) AS var_m01
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

pub(super) const SQL_COUNT_STAR_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00, COUNT(*) AS count_rows
FROM bench
WHERE payload >= 10
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
LIMIT 2
";

pub(super) const SQL_GLOBAL_AVG_QUERY: &str = "
SELECT AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
";

pub(super) const SQL_MEDIAN_AGGREGATE_QUERY: &str = "
SELECT key, MEDIAN(m00) AS median_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY median_m00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_GLOBAL_MEDIAN_QUERY: &str = "
SELECT MEDIAN(m00) AS median_m00
FROM bench
WHERE payload >= 10
";

pub(super) const SQL_ROLLUP_AVG_QUERY: &str = "
SELECT key, payload, AVG(m00) AS avg_m00
FROM bench
GROUP BY ROLLUP(key, payload)
ORDER BY avg_m00 DESC NULLS LAST, key ASC NULLS LAST, payload ASC NULLS LAST
LIMIT 10
";

pub(super) const SQL_CUBE_AVG_QUERY: &str = "
SELECT key, payload, AVG(m00) AS avg_m00
FROM bench
GROUP BY CUBE(key, payload)
ORDER BY avg_m00 DESC NULLS LAST, key ASC NULLS LAST, payload ASC NULLS LAST
LIMIT 10
";

pub(super) const SQL_GROUPING_SETS_AVG_QUERY: &str = "
SELECT key, payload, AVG(m00) AS avg_m00
FROM bench
GROUP BY GROUPING SETS ((key, payload), (key), ())
ORDER BY avg_m00 DESC NULLS LAST, key ASC NULLS LAST, payload ASC NULLS LAST
LIMIT 10
";

pub(super) const SQL_GLOBAL_AGGREGATE_PARITY_QUERY: &str = "
SELECT
    SUM(m00) AS sum_m00,
    MIN(m00) AS min_m00,
    MAX(m01) AS max_m01,
    COUNT(payload) AS count_payload,
    AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
";

pub(super) const SQL_ROUND_DECIMAL_INTEGRAL_CAST_QUERY: &str = "
SELECT key, CAST(ROUND(SUM(amount) / CAST(50 AS DECIMAL(20, 0))) AS INT) AS segment
FROM bench
GROUP BY key
ORDER BY segment ASC, key ASC
LIMIT 10
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

pub(super) const SQL_FILTERED_AVG_QUERY: &str = "
SELECT key, AVG(m00) FILTER (WHERE payload >= 10) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_DISTINCT_AVG_QUERY: &str = "
SELECT key, AVG(DISTINCT m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_FILTERED_AGGREGATE_QUERY: &str = "
SELECT key, SUM(m00) FILTER (WHERE payload >= 10) AS sum_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_m00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_DISTINCT_AGGREGATE_QUERY: &str = "
SELECT key, COUNT(DISTINCT payload) AS distinct_payload
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY distinct_payload DESC, key ASC
LIMIT 2
";

pub(super) const SQL_SOURCE_COLUMN_FILTER_REQUIRED_COLUMNS_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE label IS NOT NULL
  AND CASE WHEN label IS NULL THEN false ELSE true END
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_SOURCE_COLUMN_PRE_PROJECTION_REQUIRED_COLUMNS_QUERY: &str = "
SELECT key, SUM(x00) AS sum_x00
FROM (
    SELECT key, CASE WHEN label IN ('a', 'b') THEN m00 ELSE m01 END AS x00
    FROM bench
) projected
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_SOURCE_COLUMN_AGGREGATE_ARGUMENT_REQUIRED_COLUMNS_QUERY: &str = "
SELECT key, SUM(CASE WHEN label LIKE 'a%' THEN m00 ELSE m01 END) AS sum_x00
FROM bench
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

pub(super) const SQL_SOURCE_COLUMN_NULLIF_ARGUMENT_REQUIRED_COLUMNS_QUERY: &str = "
SELECT key, SUM(NULLIF(m00, m01)) AS sum_nullif
FROM bench
GROUP BY key
ORDER BY sum_nullif DESC, key ASC
LIMIT 2
";

pub(super) const SQL_SOURCE_COLUMN_GROUP_REQUIRED_COLUMNS_QUERY: &str = "
SELECT
    SUBSTRING(label FROM 1 FOR 1) AS label_prefix,
    EXTRACT(YEAR FROM shipdate) AS ship_year,
    SUM(m00) AS sum_m00
FROM bench
GROUP BY SUBSTRING(label FROM 1 FOR 1), EXTRACT(YEAR FROM shipdate)
ORDER BY label_prefix ASC, ship_year ASC
LIMIT 2
";

pub(super) const SQL_SOURCE_COLUMN_PUSHED_PROJECTION_REQUIRED_COLUMNS_QUERY: &str = "
SELECT label_prefix, ship_year, SUM(m00) AS sum_m00
FROM (
    SELECT SUBSTRING(label FROM 1 FOR 1) AS label_prefix, EXTRACT(YEAR FROM shipdate) AS ship_year, m00
    FROM bench
) projected
GROUP BY label_prefix, ship_year
ORDER BY label_prefix ASC, ship_year ASC
LIMIT 2
";

pub(super) const SQL_DATE64_EXTRACT_QUERY: &str = "
SELECT event_year, COUNT(*) AS row_count
FROM (
    SELECT EXTRACT(YEAR FROM event_date64) AS event_year
    FROM bench
) projected
GROUP BY event_year
ORDER BY event_year ASC
LIMIT 2
";
