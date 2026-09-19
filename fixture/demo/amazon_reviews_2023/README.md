# Amazon Reviews 2023 (Electronics)

A relational demo dataset: 43.9M product reviews joined to a 1.6M-row product
catalog on one key. It exercises time-range pruning, large hash joins, and
high-cardinality group-bys on the native GPU path.

## Semantic SQL examples

The [Amazon demo walkthrough](../../../website/docs/guides/datasets/amazon-reviews.mdx)
uses this fixture for catalog search constrained by review history, changes
in a product's review topics, and a UMAP view linked to original reviews.
The [Top-K example](../../../website/docs/cuvs/_examples/cuvs_top_k.mdx)
derives keyword-mention rates directly from the raw tables and composes
KMeans with Top-K in one statement.

`scripts/amazon_reviews_semantic.py --profile bge-m3` prepares 1,010,000
product vectors and 1,000,000 review vectors with `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`. It uses complete fields after HTML
normalization, subject to the model's 8,192-token limit, and stores normalized
1,024-dimensional Float32 vectors. Model inference is an external preparation
step; SQL consumes the resulting Parquet relations.

The walkthrough contains the environment, preparation commands and server
configuration. Generated matrices and results live under
`target/amazon-reviews-bge-m3/`; SQL, checksums and research measurements live
under `docs/research/amazon_reviews_2023/semantic/bge-m3/`. Both directories
are ignored by Git. The default `minilm` profile retains the original
384-dimensional experiments in their existing directories.

## Source

[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
`Electronics` category. `raw/all_categories.txt` lists the other 33 categories;
`scripts/prepare.py --category NAME` converts any of them.

| File | Bytes | Records | SHA-256 |
| :--- | ---: | ---: | :--- |
| `raw/review_categories/Electronics.jsonl` | 22,616,233,652 | 43,886,944 | `bf11cf9c94c23ead7d3ffb174d18231fad8fcd76e72ad167f538a5599a38de8e` |
| `raw/meta_categories/meta_Electronics.jsonl` | 5,246,144,134 | 1,610,012 | `5eb4dabd539586c49654297319ef178b62a2fe8e43108376feda125f04455d93` |

```bash
hf download McAuley-Lab/Amazon-Reviews-2023 \
  raw/review_categories/Electronics.jsonl raw/meta_categories/meta_Electronics.jsonl \
  --repo-type dataset --local-dir fixture/demo/amazon_reviews_2023
```

`raw/sample/` holds the first 50,000 reviews and their 34,349 products as JSONL,
plus the original `analysis_report.json`. `raw/analysis_notes.md` is the
original README for those files. Its verified-purchase count (40,546,881) is 3
lower than the raw file, which has 40,546,884 `true` values.

## Conversion

```bash
fixture/fixture.sh amazon-reviews prepare
# same as: python3 fixture/demo/amazon_reviews_2023/scripts/prepare.py
```

The converter requires the `duckdb` CLI and `pyarrow`. On a 32-core host with
the raw files on EBS and the output on local NVMe, the full category took
2.5 to 5.5 minutes (depending on whether the source was in the page cache) at
81 GB peak RSS. DuckDB spills past `--duckdb-memory-limit` into
`--work-dir`, so point the work directory at fast local storage on smaller
hosts. `--force` replaces existing output. `--reviews-jsonl`, `--meta-jsonl`,
and `--out-dir` convert other inputs, such as the 50,000-review sample.

```
manifest.json          source sizes and SHA-256, counts, per-table layout
parquet/
  products.parquet     1,610,012 rows, 13 row groups, 1.15 GB
  reviews.parquet      43,886,944 rows, 42 row groups, 10.0 GB
```

The conversion fails without writing output if any of these source invariants
break:

- `parent_asin` is present and unique in the metadata.
- Every review resolves to a product.
- Every review has `parent_asin`, `asin`, `user_id`, and `timestamp`.
- Every rating is an integer from 0 to 5.

In the Electronics category, 152 products have no reviews. The reviews come
from 18,286,191 users and span 1996-11-18 to 2023-09-13.

## Tables

`products`, sorted by `product_id`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `product_id` | Int32, not null | dense 0-based rank of `parent_asin` |
| `parent_asin` | Utf8, not null | source key |
| `title`, `main_category`, `store` | Utf8 | source |
| `brand` | Utf8 | `details.Brand` |
| `category_l1` .. `category_l3` | Utf8 | first three `categories` levels; `category_l1` is the root department |
| `category_path` | Utf8 | all levels joined by ` > ` |
| `average_rating` | Float32 | source |
| `rating_number` | Int32 | source |
| `price` | Float64 | numeric source prices; the 316 string prices (`from 19.95`, a dash) become null |
| `date_first_available` | Date32 | parsed `details."Date First Available"` |
| `image_count`, `video_count` | Int16, not null | source list lengths |
| `features_text`, `description_text` | Utf8 | source lists joined by newlines |
| `details_json` | Utf8 | the source `details` object as JSON text |

`reviews`, sorted by `(review_time, review_id)`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `review_id` | Int64, not null | 0-based line number in the source JSONL |
| `review_time` | Timestamp(ms), not null | source epoch milliseconds, stored as UTC wall time without a timezone |
| `product_id` | Int32, not null | joins `products.product_id` |
| `user_key` | Int32, not null | dense 0-based rank of `user_id` |
| `rating` | Int8, not null | source (integer-valued) |
| `helpful_vote` | Int32, not null | source |
| `verified_purchase` | Boolean, not null | source |
| `image_count` | Int16, not null | source list length |
| `parent_asin`, `asin`, `user_id` | Utf8, not null | source keys |
| `title`, `text` | Utf8 | source |

Image and video URLs, `subtitle` and `author` (present on under 900 media
products), and `bought_together` (null on every Electronics product) are not
converted.

## Parquet layout for the GPU engine

- **Types.** Every column is a type the native engine can filter, group, and
  join on. Struct and Map columns send the whole scan to the CPU, List columns
  can only be carried, and timezone-aware timestamps cannot be compared. The
  converter therefore flattens nested source fields to strings or counts and
  writes `review_time` without a timezone.
- **Integer keys.** `product_id` and `user_key` let joins and group-bys hash 4
  bytes instead of 10- and 28-character strings. They also serve directly as
  vertex ids for a cuGraph user-to-product graph.
- **Pruning.** Sorting `reviews` by time makes `review_time` row-group
  statistics prune time ranges. `products` is sorted by `product_id`, which
  also orders `parent_asin`. Bloom filters cover `product_id`, `user_key`,
  `parent_asin`, and `user_id` on reviews and `store` and `brand` on products.
- **Encoding.** Snappy, format 2.6, and page indexes throughout. Dictionary
  encoding is limited to low-cardinality columns, and free-text columns carry
  no min/max statistics.
- **Row groups.** 1,048,576 rows per review row group. Products use 131,072
  because their text columns are wide.
- **Ordering.** As in `fixture/tpc`, the files do not declare
  `sorting_columns`: the physical order serves pruning and is not an
  output-ordering promise to DataFusion.

`algeon_tools validate-query` plans all of these queries as a single
`AlgeonNativeExec`:

```sql
SELECT p.category_l2, count(*) AS n, avg(r.rating) AS avg_rating
FROM reviews r JOIN products p ON r.product_id = p.product_id
WHERE r.review_time >= TIMESTAMP '2022-01-01 00:00:00' AND r.verified_purchase
GROUP BY p.category_l2 ORDER BY n DESC LIMIT 10;

SELECT user_key, count(*) AS n FROM reviews WHERE rating <= 2
GROUP BY user_key ORDER BY n DESC LIMIT 10;

SELECT brand, avg(price) AS avg_price, sum(rating_number) AS ratings
FROM products
WHERE date_first_available >= DATE '2020-01-01' AND price IS NOT NULL
GROUP BY brand ORDER BY ratings DESC LIMIT 20;
```
