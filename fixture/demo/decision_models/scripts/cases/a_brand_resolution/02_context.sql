WITH target_items AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, category_path, name, price
  FROM merrec_items
  WHERE brand_id IN ({brand_ids})
), stats AS (
  SELECT brand_id, COUNT(*) AS listings, ROUND(AVG(price), 2) AS avg_price
  FROM target_items
  GROUP BY brand_id
), top_cat AS (
  SELECT brand_id, category_path,
         ROW_NUMBER() OVER (PARTITION BY brand_id ORDER BY COUNT(*) DESC, category_path) AS rk
  FROM target_items
  WHERE category_path IS NOT NULL
  GROUP BY brand_id, category_path
), sample_titles AS (
  SELECT brand_id, name,
         ROW_NUMBER() OVER (PARTITION BY brand_id ORDER BY name) AS trk
  FROM target_items
  WHERE name IS NOT NULL
)
SELECT s.brand_id, s.listings, s.avg_price,
       MAX(CASE WHEN tc.rk = 1 THEN tc.category_path END) AS top_category,
       MAX(CASE WHEN tc.rk = 2 THEN tc.category_path END) AS second_category,
       MAX(CASE WHEN st.trk = 5 THEN st.name END) AS sample_1,
       MAX(CASE WHEN st.trk = 25 THEN st.name END) AS sample_2,
       MAX(CASE WHEN st.trk = 50 THEN st.name END) AS sample_3
FROM stats s
LEFT JOIN top_cat tc ON tc.brand_id = s.brand_id AND tc.rk <= 2
LEFT JOIN sample_titles st ON st.brand_id = s.brand_id AND st.trk IN (5, 25, 50)
GROUP BY s.brand_id, s.listings, s.avg_price
ORDER BY s.brand_id
