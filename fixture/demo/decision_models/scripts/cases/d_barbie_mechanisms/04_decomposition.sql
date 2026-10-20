-- Share of each brand's July to August Barbie-fan likes on its audited bridge items, split by the model's
-- mechanism, by the policy band and by the title keyword. Shares divide by the brand's audited fan likes.
SELECT b.brand_id, b.brand_name,
       COUNT(*) AS items,
       SUM(b.fan_likes_after) AS fan_likes,
       SUM(CASE WHEN d.mechanism = 'official_barbie_collab' THEN b.fan_likes_after ELSE 0 END)
         / CAST(SUM(b.fan_likes_after) AS DOUBLE) AS official_share,
       SUM(CASE WHEN d.mechanism = 'barbiecore_pink_aesthetic' THEN b.fan_likes_after ELSE 0 END)
         / CAST(SUM(b.fan_likes_after) AS DOUBLE) AS pink_aesthetic_share,
       SUM(CASE WHEN d.mechanism = 'eras_tour_or_pop_fandom' THEN b.fan_likes_after ELSE 0 END)
         / CAST(SUM(b.fan_likes_after) AS DOUBLE) AS pop_fandom_share,
       SUM(CASE WHEN d.mechanism = 'general_brand_staple' THEN b.fan_likes_after ELSE 0 END)
         / CAST(SUM(b.fan_likes_after) AS DOUBLE) AS staple_share,
       SUM(CASE WHEN d.band = 'accept' THEN b.fan_likes_after ELSE 0 END)
         / CAST(SUM(b.fan_likes_after) AS DOUBLE) AS accept_share,
       SUM(CASE WHEN d.band = 'review' THEN b.fan_likes_after ELSE 0 END)
         / CAST(SUM(b.fan_likes_after) AS DOUBLE) AS review_share,
       SUM(CASE WHEN d.keyword_barbie THEN b.fan_likes_after ELSE 0 END)
         / CAST(SUM(b.fan_likes_after) AS DOUBLE) AS keyword_share
FROM {bridge_items} b JOIN {decisions} d ON d.item_id = b.item_id
GROUP BY b.brand_id, b.brand_name
ORDER BY b.brand_id;
