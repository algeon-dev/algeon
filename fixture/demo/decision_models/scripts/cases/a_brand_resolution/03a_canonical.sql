SELECT vertex AS brand_id, component AS canonical_id
FROM cugraph_weakly_connected_components(
  edges => (SELECT brand_id_a AS src, brand_id_b AS dst FROM brand_pairs WHERE band = 'accept'),
  directed => false
)
ORDER BY canonical_id, brand_id
