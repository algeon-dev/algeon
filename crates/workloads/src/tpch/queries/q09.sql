SELECT
    nation,
    o_year,
    SUM(amount) AS sum_profit
FROM (
    SELECT
        n_name AS nation,
        EXTRACT(YEAR FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM part
    JOIN lineitem ON p_partkey = l_partkey
    JOIN partsupp ON ps_suppkey = l_suppkey AND ps_partkey = l_partkey
    JOIN supplier ON s_suppkey = l_suppkey
    JOIN orders ON o_orderkey = l_orderkey
    JOIN nation ON s_nationkey = n_nationkey
    WHERE p_name LIKE '%green%'
) profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC
