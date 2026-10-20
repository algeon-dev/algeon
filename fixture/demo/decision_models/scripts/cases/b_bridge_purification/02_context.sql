WITH target_items AS (
  SELECT CAST(brand_id AS BIGINT) AS brand_id, c0_name, c1_name, category_path, name, price
  FROM merrec_items
  WHERE brand_id IN ({brand_ids})
), brand_stats AS (
  SELECT brand_id,
         COUNT(*) AS total_listings,
         COUNT(DISTINCT c0_name) AS distinct_c0,
         COUNT(DISTINCT c1_name) AS distinct_c1,
         ROUND(AVG(price), 2) AS avg_price
  FROM target_items
  GROUP BY brand_id
), top_cats AS (
  SELECT brand_id, category_path, COUNT(*) AS listings,
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
SELECT bs.brand_id, bs.total_listings, bs.distinct_c0, bs.distinct_c1, bs.avg_price,
       MAX(CASE WHEN tc.rk = 1 THEN tc.category_path END) AS cat1,
       MAX(CASE WHEN tc.rk = 1 THEN tc.listings END) AS cat1_listings,
       MAX(CASE WHEN tc.rk = 2 THEN tc.category_path END) AS cat2,
       MAX(CASE WHEN tc.rk = 2 THEN tc.listings END) AS cat2_listings,
       MAX(CASE WHEN tc.rk = 3 THEN tc.category_path END) AS cat3,
       MAX(CASE WHEN tc.rk = 3 THEN tc.listings END) AS cat3_listings,
       MAX(CASE WHEN st.trk = 10 THEN st.name END) AS title1,
       MAX(CASE WHEN st.trk = 50 THEN st.name END) AS title2,
       MAX(CASE WHEN st.trk = 100 THEN st.name END) AS title3,
       MAX(CASE WHEN st.trk = 200 THEN st.name END) AS title4
FROM brand_stats bs
LEFT JOIN top_cats tc ON tc.brand_id = bs.brand_id AND tc.rk <= 3
LEFT JOIN sample_titles st ON st.brand_id = bs.brand_id AND st.trk IN (10, 50, 100, 200)
GROUP BY bs.brand_id, bs.total_listings, bs.distinct_c0, bs.distinct_c1, bs.avg_price
ORDER BY bs.brand_id
