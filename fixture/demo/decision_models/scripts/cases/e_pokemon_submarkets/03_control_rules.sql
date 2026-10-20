WITH titled AS (
  SELECT d.item_id, i.category_path, ' ' || LOWER(i.name) || ' ' AS title
  FROM pokemon_items d
  JOIN merrec_items i ON i.item_id = d.item_id
)
SELECT item_id,
       CASE
         WHEN category_path LIKE '%Trading Cards%' AND (
              title LIKE '% lot %' OR title LIKE '%bundle%' OR title LIKE '%binder%'
              OR title LIKE '%bulk%' OR title LIKE '%collection%')
           THEN 'tcg_bundle_lot_or_binder'
         WHEN category_path LIKE '%Trading Cards%' THEN 'tcg_sealed_or_single_card'
         WHEN category_path LIKE '%Video games%' OR title LIKE '%nintendo%' OR title LIKE '%switch%'
              OR title LIKE '% ds %' OR title LIKE '% 3ds %' OR title LIKE '% gba %' OR title LIKE '%game boy%'
              OR title LIKE '%gameboy%' OR title LIKE '%cartridge%' OR title LIKE '%go plus%'
           THEN 'video_game_or_hardware'
         WHEN category_path LIKE '%Stuffed Animals%' OR title LIKE '%plush%' OR title LIKE '%squishmallow%'
              OR title LIKE '%figure%' OR title LIKE '%funko%' OR title LIKE '%shirt%' OR title LIKE '%hoodie%'
              OR title LIKE '%backpack%'
           THEN 'plush_toy_or_apparel'
         ELSE 'unclassified'
       END AS rule_label
FROM titled
ORDER BY item_id
