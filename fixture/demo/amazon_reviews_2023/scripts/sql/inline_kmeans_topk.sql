WITH complaint_features AS (
    SELECT
        p.product_id AS id,
        COUNT(*) AS negative_reviews,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%battery%' THEN 1.0 ELSE 0.0 END) AS REAL) AS battery,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%connect%' OR LOWER(r.text) LIKE '%bluetooth%' THEN 1.0 ELSE 0.0 END) AS REAL) AS connection,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%sound%' OR LOWER(r.text) LIKE '%volume%' THEN 1.0 ELSE 0.0 END) AS REAL) AS sound,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%comfort%' OR LOWER(r.text) LIKE '%fit%' THEN 1.0 ELSE 0.0 END) AS REAL) AS comfort,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%broke%' OR LOWER(r.text) LIKE '%stopped working%' THEN 1.0 ELSE 0.0 END) AS REAL) AS durability,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%charg%' THEN 1.0 ELSE 0.0 END) AS REAL) AS charging,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%microphone%' OR LOWER(r.text) LIKE '%mic %' THEN 1.0 ELSE 0.0 END) AS REAL) AS microphone,
        CAST(AVG(CASE WHEN LOWER(r.text) LIKE '%return%' OR LOWER(r.text) LIKE '%refund%' THEN 1.0 ELSE 0.0 END) AS REAL) AS returns
    FROM reviews AS r
    JOIN products AS p
        ON r.product_id = p.product_id
    WHERE p.category_l2 = 'Headphones, Earbuds & Accessories'
      AND r.review_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND r.review_time <  TIMESTAMP '2023-01-01 00:00:00'
      AND r.rating <= 2
      AND r.text IS NOT NULL
    GROUP BY p.product_id
    HAVING COUNT(*) >= 10
),

clusters AS (
    SELECT
        id,
        cluster_id
    FROM cuvs_kmeans(
        input => (
            SELECT
                id,
                COALESCE(battery, CAST(0 AS REAL))    AS battery,
                COALESCE(connection, CAST(0 AS REAL)) AS connection,
                COALESCE(sound, CAST(0 AS REAL))      AS sound,
                COALESCE(comfort, CAST(0 AS REAL))    AS comfort,
                COALESCE(durability, CAST(0 AS REAL)) AS durability,
                COALESCE(charging, CAST(0 AS REAL))   AS charging,
                COALESCE(microphone, CAST(0 AS REAL)) AS microphone,
                COALESCE(returns, CAST(0 AS REAL))    AS returns
            FROM complaint_features
            ORDER BY id
        ),
        n_clusters => 6,
        n_init => 5
    )
),

profiles AS (
    SELECT
        CAST(k.cluster_id AS BIGINT) AS id,
        COALESCE(CAST(AVG(c.battery) AS REAL), CAST(0 AS REAL))    AS battery,
        COALESCE(CAST(AVG(c.connection) AS REAL), CAST(0 AS REAL)) AS connection,
        COALESCE(CAST(AVG(c.sound) AS REAL), CAST(0 AS REAL))      AS sound,
        COALESCE(CAST(AVG(c.comfort) AS REAL), CAST(0 AS REAL))    AS comfort,
        COALESCE(CAST(AVG(c.durability) AS REAL), CAST(0 AS REAL)) AS durability,
        COALESCE(CAST(AVG(c.charging) AS REAL), CAST(0 AS REAL))   AS charging,
        COALESCE(CAST(AVG(c.microphone) AS REAL), CAST(0 AS REAL)) AS microphone,
        COALESCE(CAST(AVG(c.returns) AS REAL), CAST(0 AS REAL))    AS returns
    FROM clusters AS k
    JOIN complaint_features AS c
        ON k.id = c.id
    GROUP BY k.cluster_id
)

SELECT
    t.id AS cluster_id,
    t.rank,
    CASE t.position
        WHEN 0 THEN 'battery'
        WHEN 1 THEN 'connection'
        WHEN 2 THEN 'sound'
        WHEN 3 THEN 'comfort'
        WHEN 4 THEN 'durability'
        WHEN 5 THEN 'charging'
        WHEN 6 THEN 'microphone'
        WHEN 7 THEN 'returns'
    END AS issue,
    t.value AS mean_product_mention_rate
FROM cuvs_top_k(
    input => (
        SELECT
            id,
            battery,
            connection,
            sound,
            comfort,
            durability,
            charging,
            microphone,
            returns
        FROM profiles
    ),
    k => 3,
    select => 'max'
) AS t
ORDER BY
    t.id,
    t.rank;
