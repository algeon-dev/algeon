WITH shoppers AS (
  SELECT e.user_id AS id,
    SUM(CASE WHEN e.event_type = 'buy_comp' THEN 1 ELSE 0 END) AS purchases,
    SUM(CASE WHEN e.event_type = 'buy_comp' THEN i.price ELSE 0 END) AS gmv,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Women' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS women,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Toys & Collectibles' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS toys,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Kids' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS kids,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Men' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS men,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Home' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS home,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Electronics' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS electronics,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Beauty' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS beauty,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Vintage & collectibles' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS vintage,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Books' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS books,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Sports & outdoors' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS sports,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Other' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS other,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Handmade' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS handmade,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Garden & Outdoor' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS garden,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Arts & Crafts' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS arts,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Pet Supplies' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS pets,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Office' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS office,
    CAST(SQRT(SUM(CASE WHEN e.event_type = 'item_view' AND i.c0_name = 'Tools' THEN 1.0 ELSE 0.0 END) / SUM(CASE WHEN e.event_type = 'item_view' THEN 1.0 ELSE 0.0 END)) AS REAL) AS tools
  FROM merrec_events e JOIN merrec_items i ON i.item_id = e.item_id
  GROUP BY e.user_id
  HAVING SUM(CASE WHEN e.event_type = 'item_view' THEN 1 ELSE 0 END) >= 20
), assignments AS (
  SELECT id, cluster_id FROM cuvs_kmeans(
    input => (SELECT id, women, toys, kids, men, home, electronics, beauty, vintage, books, sports, other, handmade, garden, arts, pets, office, tools FROM shoppers ORDER BY id),
    n_clusters => 10, n_init => 3, max_iter => 300)
), personas AS (
  SELECT a.cluster_id AS id, COUNT(*) AS shoppers,
    AVG(CASE WHEN s.purchases > 0 THEN 1.0 ELSE 0.0 END) AS buyer_share,
    SUM(s.gmv) AS gmv,
    CAST(AVG(s.women * s.women) AS REAL) AS women,
    CAST(AVG(s.toys * s.toys) AS REAL) AS toys,
    CAST(AVG(s.kids * s.kids) AS REAL) AS kids,
    CAST(AVG(s.men * s.men) AS REAL) AS men,
    CAST(AVG(s.home * s.home) AS REAL) AS home,
    CAST(AVG(s.electronics * s.electronics) AS REAL) AS electronics,
    CAST(AVG(s.beauty * s.beauty) AS REAL) AS beauty,
    CAST(AVG(s.vintage * s.vintage) AS REAL) AS vintage,
    CAST(AVG(s.books * s.books) AS REAL) AS books,
    CAST(AVG(s.sports * s.sports) AS REAL) AS sports,
    CAST(AVG(s.other * s.other) AS REAL) AS other,
    CAST(AVG(s.handmade * s.handmade) AS REAL) AS handmade,
    CAST(AVG(s.garden * s.garden) AS REAL) AS garden,
    CAST(AVG(s.arts * s.arts) AS REAL) AS arts,
    CAST(AVG(s.pets * s.pets) AS REAL) AS pets,
    CAST(AVG(s.office * s.office) AS REAL) AS office,
    CAST(AVG(s.tools * s.tools) AS REAL) AS tools
  FROM assignments a JOIN shoppers s ON s.id = a.id
  GROUP BY a.cluster_id
), leading AS (
  SELECT id, rank, position, value FROM cuvs_top_k(
    input => (SELECT id, women, toys, kids, men, home, electronics, beauty, vintage, books, sports, other, handmade, garden, arts, pets, office, tools FROM personas ORDER BY id),
    k => 3, select => 'max')
)
SELECT p.id AS persona, p.shoppers, p.buyer_share, p.gmv, p.gmv / p.shoppers AS gmv_per_shopper,
  l.rank,
  CASE l.position
    WHEN 0 THEN 'Women'
    WHEN 1 THEN 'Toys & Collectibles'
    WHEN 2 THEN 'Kids'
    WHEN 3 THEN 'Men'
    WHEN 4 THEN 'Home'
    WHEN 5 THEN 'Electronics'
    WHEN 6 THEN 'Beauty'
    WHEN 7 THEN 'Vintage & collectibles'
    WHEN 8 THEN 'Books'
    WHEN 9 THEN 'Sports & outdoors'
    WHEN 10 THEN 'Other'
    WHEN 11 THEN 'Handmade'
    WHEN 12 THEN 'Garden & Outdoor'
    WHEN 13 THEN 'Arts & Crafts'
    WHEN 14 THEN 'Pet Supplies'
    WHEN 15 THEN 'Office'
    WHEN 16 THEN 'Tools'
  END AS category,
  l.value AS mean_view_share
FROM leading l JOIN personas p ON p.id = l.id
ORDER BY p.shoppers DESC, l.rank;
