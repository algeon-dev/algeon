#!/usr/bin/env python3
"""Prepare and inspect semantic Amazon demo inputs; keep generated data under target."""

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import html
from html.parser import HTMLParser
import importlib.metadata
import json
from pathlib import Path
import re
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "target/amazon-reviews-semantic"
REPORT = ROOT / "docs/research/amazon_reviews_2023/semantic"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSIONS = 384
FEATURES = [f"e{i:03d}" for i in range(DIMENSIONS)]
PROFILE = "minilm"
INTENTS = [
    "Wireless noise cancelling over ear headphones for long airplane flights",
    "A USB C docking station with two HDMI monitors and laptop charging",
    "A waterproof portable Bluetooth speaker for the shower and beach",
    "A dash camera that records the road in front and behind a car at night",
    "A USB microphone for recording podcasts and voice calls at a desk",
    "A compact portable projector for watching movies on a bedroom wall",
    "A WiFi mesh system to eliminate dead spots throughout a large house",
    "A battery powered security camera for outdoor monitoring without wires",
    "A fast microSD memory card for recording 4K video in an action camera",
    "A portable power bank that charges a phone and laptop using USB C",
    "A 27 inch computer monitor with a high refresh rate for gaming",
    "A webcam with a built in microphone for video meetings on a laptop",
]


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        elif tag in ("br", "p", "div", "li"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        elif tag in ("p", "div", "li"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    parser = PlainText()
    parser.feed(value)
    return " ".join(html.unescape("".join(parser.parts)).split())


def configure(profile):
    global PROFILE, DATA, REPORT, MODEL, MODEL_REVISION, DIMENSIONS, FEATURES, INTENTS
    PROFILE = profile
    if profile == "bge-m3":
        DATA = ROOT / "target/amazon-reviews-bge-m3"
        REPORT = ROOT / "docs/research/amazon_reviews_2023/semantic/bge-m3"
        MODEL = "BAAI/bge-m3"
        MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
        DIMENSIONS = 1024
        FEATURES = [f"e{i:03d}" for i in range(DIMENSIONS)]
        INTENTS = INTENTS[:12] + [
            "適合長途飛行的無線主動降噪耳罩式耳機",
            "Auriculares inalámbricos de diadema con cancelación de ruido para vuelos largos",
        ]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=16")
    products = ROOT / "fixture/demo/amazon_reviews_2023/parquet/products.parquet"
    reviews = ROOT / "fixture/demo/amazon_reviews_2023/parquet/reviews.parquet"
    statements = {
        "products": f"""
SELECT CAST(row_number() OVER (ORDER BY hash(product_id), product_id)-1 AS INTEGER) AS sample_rank,
       product_id AS id, parent_asin, title, brand, price, category_l2, category_l3,
       rating_number, average_rating,
       left(title,512) || '\n' || left(coalesce(features_text,''),1536) || '\n' ||
       left(coalesce(description_text,''),1024) AS embedding_text
FROM '{products}' WHERE length(trim(coalesce(title,'')))>0
ORDER BY hash(product_id), product_id LIMIT 1010000
""",
        "reviews": f"""
SELECT CAST(row_number() OVER (ORDER BY hash(r.review_id),r.review_id)-1 AS INTEGER) AS sample_rank,
       r.review_id AS id, r.product_id, r.review_time, r.rating, r.verified_purchase,
       r.title, r.text, p.category_l2, p.category_l3,
       left(coalesce(r.title,''),256) || '\n' || left(r.text,2048) AS embedding_text
FROM '{reviews}' r JOIN '{products}' p ON r.product_id=p.product_id
WHERE r.rating<=2 AND length(r.text)>=40
  AND r.review_time>=TIMESTAMP '2021-01-01' AND r.review_time<TIMESTAMP '2023-01-01'
ORDER BY hash(r.review_id), r.review_id LIMIT 1000000
""",
    }
    records = []
    for name, sql in statements.items():
        if PROFILE == "bge-m3":
            sql = re.sub(r"left\((coalesce\([^)]*\)|[\w.]+),\d+\)", r"\1", sql)
        started = time.perf_counter()
        path = DATA / f"{name}_text.parquet"
        if PROFILE == "bge-m3":
            reader = connection.execute(sql).to_arrow_reader(65536)
            with pq.ParquetWriter(path, reader.schema, compression="zstd") as writer:
                for batch in reader:
                    table = pa.Table.from_batches([batch])
                    index = table.schema.get_field_index("embedding_text")
                    texts = [clean_text(text) for text in table["embedding_text"].to_pylist()]
                    table = table.set_column(index, "embedding_text", pa.array(texts))
                    writer.write_table(table)
        else:
            connection.execute(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        record = {"name": name, "sql": sql, "elapsed_s": time.perf_counter()-started,
                  "rows": pq.ParquetFile(path).metadata.num_rows, "bytes": path.stat().st_size}
        records.append(record)
        print(json.dumps(record), flush=True)
    pq.write_table(pa.table({"id": np.arange(len(INTENTS), dtype=np.int32),
                             "embedding_text": INTENTS}), DATA / "intents_text.parquet")
    save_json(REPORT / "preparation.json", records)


def embed(names):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(8)
    revision = MODEL_REVISION
    started = time.perf_counter()
    model = SentenceTransformer(MODEL, revision=revision, device="cuda",
                                model_kwargs={"attn_implementation": "sdpa"})
    model.half()
    provenance = {"model": MODEL, "revision": revision, "dimensions": DIMENSIONS,
                  "max_sequence_length": model.max_seq_length, "inference_dtype": "float16",
                  "stored_dtype": "float32", "normalization": "L2 in float32 after encoding",
                  "batch_size": 32 if PROFILE == "bge-m3" else 512,
                  "model_load_s": time.perf_counter()-started,
                  "text_preparation": "full fields, HTML removed, entities decoded, whitespace collapsed"
                  if PROFILE == "bge-m3" else "legacy character truncation, HTML retained",
                  "packages": {p: importlib.metadata.version(p) for p in
                               ("torch", "sentence-transformers", "transformers", "numpy", "pyarrow", "duckdb", "faiss-cpu")},
                  "device": torch.cuda.get_device_name(), "runs": []}
    for name in names:
        source = DATA / f"{name}_text.parquet"
        parquet = pq.ParquetFile(source)
        count = parquet.metadata.num_rows
        matrix = np.lib.format.open_memmap(DATA / f"{name}.npy", mode="w+", dtype="float32",
                                           shape=(count, DIMENSIONS))
        started = time.perf_counter()
        offset = 0
        torch.cuda.reset_peak_memory_stats()
        vector_type = pa.list_(pa.field("element", pa.float32(), nullable=False), DIMENSIONS)
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False),
                            pa.field("sample_rank", pa.int32(), nullable=False)] +
                           ([pa.field("vector", vector_type, nullable=False)] if PROFILE == "bge-m3" else
                            [pa.field(column, pa.float32(), nullable=False) for column in FEATURES]))
        digest = hashlib.sha256()
        with pq.ParquetWriter(DATA / f"{name}_vectors.parquet", schema, compression="snappy",
                              use_dictionary=False) as writer:
            for batch in parquet.iter_batches(batch_size=4096 if PROFILE == "bge-m3" else 8192):
                table = pa.Table.from_batches([batch])
                texts = table["embedding_text"].to_pylist()
                values = model.encode(texts, batch_size=32 if PROFILE == "bge-m3" else 512, show_progress_bar=False,
                                      convert_to_numpy=True).astype(np.float32)
                values /= np.linalg.norm(values, axis=1, keepdims=True)
                assert values.shape == (len(texts), DIMENSIONS) and np.isfinite(values).all()
                matrix[offset:offset+len(values)] = values
                digest.update(values.tobytes())
                rank = (table["sample_rank"] if "sample_rank" in table.column_names else
                        pa.array(np.arange(offset, offset+len(values), dtype=np.int32)))
                arrays = [table["id"].cast(pa.int64()), rank]
                if PROFILE == "bge-m3":
                    arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), type=vector_type))
                else:
                    arrays += [pa.array(values[:, i]) for i in range(DIMENSIONS)]
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                offset += len(values)
                print(json.dumps({"embedding": name, "completed": offset, "total": count,
                                  "elapsed_s": round(time.perf_counter()-started, 2)}), flush=True)
        matrix.flush()
        record = {"name": name, "rows": count, "elapsed_s": time.perf_counter()-started,
                  "matrix_sha256": digest.hexdigest(), "matrix_bytes": matrix.nbytes,
                  "parquet_bytes": (DATA / f"{name}_vectors.parquet").stat().st_size,
                  "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        provenance["runs"].append(record)
        save_json(REPORT / "embedding_provenance.json", provenance)
        print(json.dumps(record), flush=True)


def feature_projection(prefix=""):
    if PROFILE == "bge-m3":
        return f"{prefix}vector"
    return ", ".join(f"COALESCE({prefix}{column}, CAST(0 AS REAL)) AS {column}" for column in FEATURES)


def generate_sql():
    directory = REPORT / "sql"
    directory.mkdir(parents=True, exist_ok=True)
    definitions = []
    for name in ("products", "reviews", "intents"):
        for kind in ("text", "vectors"):
            suffix = "_compact" if kind == "vectors" else ""
            path = DATA / f"{name}_{kind}{suffix}.parquet"
            definitions.append(f"CREATE EXTERNAL TABLE semantic_{name}_{kind} STORED AS PARQUET LOCATION '{path}';")
    for size in (100000, 1000000):
        definitions.append(f"CREATE VIEW product_corpus_{size} AS SELECT id, {feature_projection()} "
                           f"FROM semantic_products_vectors WHERE sample_rank < {size};")
        definitions.append(f"CREATE VIEW review_corpus_{size} AS SELECT id, {feature_projection()} "
                           f"FROM semantic_reviews_vectors WHERE sample_rank < {size};")
    for size in (1, 1000, 10000):
        definitions.append(f"CREATE VIEW product_queries_{size} AS SELECT id, {feature_projection()} "
                           f"FROM semantic_products_vectors WHERE sample_rank >= 1000000 AND sample_rank < {1000000+size};")
    definitions.append(f"CREATE VIEW intent_queries AS SELECT id, {feature_projection()} FROM semantic_intents_vectors;")
    definitions.append(f"CREATE VIEW headphone_reviews AS SELECT v.id, {feature_projection('v.')} "
                       "FROM semantic_reviews_vectors v JOIN semantic_reviews_text r ON v.id=r.id "
                       "WHERE r.category_l2='Headphones, Earbuds & Accessories';")
    connection = duckdb.connect()
    for size in (10000, 50000, 100000):
        boundary = connection.execute(
            f"SELECT sample_rank FROM '{DATA / 'reviews_text.parquet'}' "
            "WHERE category_l2='Headphones, Earbuds & Accessories' "
            f"ORDER BY sample_rank LIMIT 1 OFFSET {size-1}").fetchone()[0]
        definitions.append(f"CREATE VIEW headphone_reviews_{size} AS SELECT v.id, {feature_projection('v.')} "
                           "FROM semantic_reviews_vectors v JOIN semantic_reviews_text r ON v.id=r.id "
                           f"WHERE r.category_l2='Headphones, Earbuds & Accessories' AND v.sample_rank<={boundary};")
    definitions.append(f"CREATE VIEW budget_headphones AS SELECT v.id, {feature_projection('v.')} "
                       "FROM semantic_products_vectors v JOIN semantic_products_text p ON v.id=p.id "
                       "WHERE v.sample_rank<1000000 AND p.category_l3='Headphones & Earbuds' "
                       "AND p.price BETWEEN 20 AND 100 AND p.rating_number>=100 AND p.average_rating>=4;")
    queries = {}
    for size in (100000, 1000000):
        for count in (1, 1000, 10000):
            queries[f"exact_n{size}_q{count}"] = (
                "SELECT * FROM cuvs_brute_force_knn(\n"
                f"  dataset => (SELECT * FROM product_corpus_{size}),\n"
                f"  queries => (SELECT * FROM product_queries_{count}),\n"
                "  k => 10, metric => 'l2_expanded') ORDER BY query_id, rank")
    for name, options in {
        "ivf_flat_p32": "cuvs_ivf_flat|n_lists => 1024, n_probes => 32",
        "ivf_flat_p128": "cuvs_ivf_flat|n_lists => 1024, n_probes => 128",
        "ivf_pq": "cuvs_ivf_pq|n_lists => 1024, n_probes => 64, pq_dim => 48, pq_bits => 8",
        "cagra": "cuvs_cagra|graph_degree => 64, intermediate_graph_degree => 128",
    }.items():
        function, options = options.split("|")
        queries[name] = (f"SELECT * FROM {function}(\n"
                         "  dataset => (SELECT * FROM product_corpus_1000000),\n"
                         "  queries => (SELECT * FROM product_queries_1000),\n"
                         f"  k => 10, metric => 'l2_expanded', {options}) ORDER BY query_id, rank")
        queries[name + "_q10000"] = queries[name].replace("product_queries_1000)", "product_queries_10000)")
    queries["intent_search"] = """SELECT n.query_id, q.embedding_text AS request, n.rank,
       n.neighbor_id, p.title, p.category_l2, p.category_l3, p.brand, p.price,
       CAST(1 AS REAL) - n.distance / CAST(2 AS REAL) AS cosine_similarity
FROM cuvs_brute_force_knn(
  dataset => (SELECT * FROM product_corpus_1000000),
  queries => (SELECT * FROM intent_queries), k => 20, metric => 'l2_expanded') n
JOIN semantic_products_text p ON p.id=n.neighbor_id
JOIN semantic_intents_text q ON q.id=n.query_id
ORDER BY n.query_id, n.rank"""
    queries["budget_headphones"] = """SELECT n.rank, n.neighbor_id, p.title, p.brand,
       p.price, p.rating_number, p.average_rating,
       CAST(1 AS REAL)-n.distance/CAST(2 AS REAL) AS cosine_similarity
FROM cuvs_brute_force_knn(
  dataset => (SELECT * FROM budget_headphones),
  queries => (SELECT * FROM intent_queries WHERE id=0),
  k => 10, metric => 'l2_expanded') n
JOIN semantic_products_text p ON p.id=n.neighbor_id
ORDER BY n.rank"""
    queries["catalog_candidates"] = """SELECT n.query_id, q.title AS source_title, q.brand AS source_brand,
       n.rank, n.neighbor_id, p.title AS candidate_title, p.brand AS candidate_brand,
       q.category_l2, q.price AS source_price, p.price AS candidate_price,
       CAST(1 AS REAL)-n.distance/CAST(2 AS REAL) AS cosine_similarity
FROM cuvs_brute_force_knn(
  dataset => (SELECT * FROM product_corpus_1000000),
  queries => (SELECT * FROM product_queries_1000), k => 20, metric => 'l2_expanded') n
JOIN semantic_products_text p ON p.id=n.neighbor_id
JOIN semantic_products_text q ON q.id=n.query_id
WHERE p.category_l2=q.category_l2 AND p.brand<>q.brand
ORDER BY n.query_id, n.rank"""
    for size in (100000, 1000000):
        queries[f"kmeans_n{size}"] = (
            "SELECT * FROM cuvs_kmeans(\n"
            f"  input => (SELECT * FROM review_corpus_{size}),\n"
            "  n_clusters => 64, n_init => 1, max_iter => 30, tol => 0.0001)")
        queries[f"pca_kmeans_n{size}"] = (
            "SELECT * FROM cuvs_kmeans(\n"
            "  input => (SELECT id, " + ", ".join(f"pc_{i}" for i in range(32)) + "\n"
            "    FROM cuvs_pca(\n"
            f"      input => (SELECT * FROM review_corpus_{size}), n_components => 32)),\n"
            "  n_clusters => 64, n_init => 1, max_iter => 30, tol => 0.0001)")
    queries["pca_n1000000"] = ("SELECT * FROM cuvs_pca("
                                 "input => (SELECT * FROM review_corpus_1000000), n_components => 32)")
    queries["tsvd_n100000"] = ("SELECT * FROM cuml_tsvd("
                                "input => (SELECT * FROM review_corpus_100000), n_components => 32)")
    queries["headphone_topics"] = """SELECT m.cluster_id,
       EXTRACT(YEAR FROM r.review_time) AS review_year,
       EXTRACT(MONTH FROM r.review_time) AS review_month,
       COUNT(*) AS negative_reviews, COUNT(DISTINCT r.product_id) AS products,
       AVG(CAST(r.rating AS DOUBLE)) AS mean_rating
FROM cuvs_kmeans(input => (SELECT * FROM headphone_reviews),
                 n_clusters => 24, n_init => 1, max_iter => 30, tol => 0.0001) m
JOIN semantic_reviews_text r ON r.id=m.id
GROUP BY m.cluster_id, EXTRACT(YEAR FROM r.review_time), EXTRACT(MONTH FROM r.review_time)
ORDER BY m.cluster_id, review_year, review_month"""
    queries["headphone_assignments"] = ("SELECT * FROM cuvs_kmeans("
        "input => (SELECT * FROM headphone_reviews), n_clusters => 24, n_init => 1, max_iter => 30, tol => 0.0001)")
    queries["headphone_monthly_saved"] = """SELECT a.cluster_id,
       EXTRACT(YEAR FROM r.review_time) AS review_year,
       EXTRACT(MONTH FROM r.review_time) AS review_month, COUNT(*) AS negative_reviews
FROM semantic_headphone_clusters a JOIN semantic_reviews_text r ON a.id=r.id
GROUP BY a.cluster_id, EXTRACT(YEAR FROM r.review_time), EXTRACT(MONTH FROM r.review_time)
ORDER BY a.cluster_id, review_year, review_month"""
    queries["headphone_issue_shift"] = """WITH issues AS (
  SELECT r.product_id, a.cluster_id,
    SUM(CASE WHEN EXTRACT(YEAR FROM r.review_time)=2021 THEN 1 ELSE 0 END) AS n2021,
    SUM(CASE WHEN EXTRACT(YEAR FROM r.review_time)=2022 THEN 1 ELSE 0 END) AS n2022
  FROM semantic_headphone_clusters a JOIN semantic_reviews_text r ON a.id=r.id
  GROUP BY r.product_id, a.cluster_id
), totals AS (
  SELECT product_id, SUM(n2021) AS n2021, SUM(n2022) AS n2022 FROM issues GROUP BY product_id
)
SELECT i.product_id, p.title, i.cluster_id, i.n2021 AS issue_2021, i.n2022 AS issue_2022,
       t.n2021 AS negatives_2021, t.n2022 AS negatives_2022,
       CAST(i.n2021 AS DOUBLE)/t.n2021 AS share_2021,
       CAST(i.n2022 AS DOUBLE)/t.n2022 AS share_2022,
       CAST(i.n2022 AS DOUBLE)/t.n2022-CAST(i.n2021 AS DOUBLE)/t.n2021 AS share_change
FROM issues i JOIN totals t ON i.product_id=t.product_id
JOIN semantic_full_products p ON i.product_id=p.product_id
WHERE t.n2021>=30 AND t.n2022>=30 AND i.n2022>=10
  AND CAST(i.n2022 AS DOUBLE)/t.n2022-CAST(i.n2021 AS DOUBLE)/t.n2021>0.1
ORDER BY share_change DESC, issue_2022 DESC LIMIT 30"""
    queries["headphone_pca2"] = ("SELECT * FROM cuvs_pca("
        "input => (SELECT * FROM headphone_reviews), n_components => 2)")
    for size in (10000, 50000, 100000):
        queries[f"umap_hdbscan_{size}"] = f"""SELECT * FROM cuml_hdbscan(
  input => (SELECT id, dim_0, dim_1, dim_2, dim_3, dim_4 FROM cuml_umap(
    input => (SELECT * FROM headphone_reviews_{size}),
    n_components => 5, n_neighbors => 30, min_dist => 0.0, random_state => 42)),
  min_cluster_size => 50, min_samples => 10)"""
    if PROFILE == "bge-m3":
        definitions += [
            f"CREATE EXTERNAL TABLE products STORED AS PARQUET LOCATION '{ROOT / 'fixture/demo/amazon_reviews_2023/parquet/products.parquet'}';",
            f"CREATE EXTERNAL TABLE reviews STORED AS PARQUET LOCATION '{ROOT / 'fixture/demo/amazon_reviews_2023/parquet/reviews.parquet'}';",
        ]
        for name, sql in queries.items():
            sql = sql.replace("pq_dim => 48", "pq_dim => 64")
            if any(function in sql for function in ("cuvs_kmeans(", "cuvs_pca(", "cuml_umap(", "cuml_tsvd(")):
                sql = re.sub(r"(SELECT \* FROM (?:review_corpus_\d+|headphone_reviews(?:_\d+)?))\)",
                             r"\1 ORDER BY id)", sql)
                sql = sql.replace("n_init => 1, max_iter => 30", "n_init => 5, max_iter => 100")
                sql = sql.replace("n_components => 32)),", "n_components => 32) ORDER BY id),")
                sql = sql.replace("random_state => 42)),", "random_state => 42) ORDER BY id),")
            queries[name] = sql
        queries.update(demo_queries())
    (directory / "00_register.sql").write_text("\n\n".join(definitions) + "\n")
    for name, sql in queries.items():
        (directory / f"{name}.sql").write_text(sql + "\n")
    (directory / "01_register_topics.sql").write_text(
        "CREATE EXTERNAL TABLE semantic_headphone_clusters STORED AS PARQUET LOCATION '"
        + str(DATA / "results/headphone_assignments_r1.parquet") + "';\n"
        "CREATE EXTERNAL TABLE semantic_full_products STORED AS PARQUET LOCATION '"
        + str(ROOT / "fixture/demo/amazon_reviews_2023/parquet/products.parquet") + "';\n")
    print(json.dumps({"queries": len(queries), "views_and_tables": len(definitions)}))


def demo_queries():
    queries = {
        "catalog_batch": """SELECT n.query_id, q.title AS source_title, n.rank,
       n.neighbor_id, p.title AS candidate_title,
       1.0 - n.distance / 2.0 AS cosine_similarity
FROM cuvs_brute_force_knn(
  dataset => (SELECT * FROM product_corpus_1000000),
  queries => (SELECT * FROM product_queries_10000),
  k => 10, metric => 'l2_expanded') n
JOIN semantic_products_text q ON q.id = n.query_id
JOIN semantic_products_text p ON p.id = n.neighbor_id
ORDER BY n.query_id, n.rank""",
        "headphone_emerging": """WITH assignments AS (
  SELECT id, cluster_id
  FROM cuvs_kmeans(
    input => (SELECT v.id, v.vector
      FROM semantic_reviews_vectors v
      JOIN semantic_reviews_text r ON r.id = v.id
      WHERE r.category_l3 = 'Headphones & Earbuds'
      ORDER BY v.id),
    n_clusters => 24, n_init => 5, max_iter => 100, tol => 0.0001)
), issues AS (
  SELECT r.product_id, a.cluster_id,
         SUM(CASE WHEN EXTRACT(YEAR FROM r.review_time) = 2021 THEN 1 ELSE 0 END) AS n2021,
         SUM(CASE WHEN EXTRACT(YEAR FROM r.review_time) = 2022 THEN 1 ELSE 0 END) AS n2022,
         MIN(CASE WHEN EXTRACT(YEAR FROM r.review_time) = 2022 THEN r.id END) AS example_id
  FROM assignments a JOIN semantic_reviews_text r ON r.id = a.id
  GROUP BY r.product_id, a.cluster_id
), totals AS (
  SELECT *, SUM(n2021) OVER (PARTITION BY product_id) AS total2021,
            SUM(n2022) OVER (PARTITION BY product_id) AS total2022
  FROM issues
)
SELECT p.product_id, p.title, t.cluster_id,
       t.n2021, t.total2021, t.n2022, t.total2022,
       100.0 * t.n2021 / t.total2021 AS share2021_pct,
       100.0 * t.n2022 / t.total2022 AS share2022_pct,
       100.0 * t.n2022 / t.total2022 - 100.0 * t.n2021 / t.total2021 AS change_pp,
       t.example_id, r.embedding_text AS example_review
FROM totals t
JOIN products p ON p.product_id = t.product_id
JOIN semantic_reviews_text r ON r.id = t.example_id
WHERE t.total2021 >= 30 AND t.total2022 >= 30 AND t.n2022 >= 10
  AND 1.0 * t.n2022 / t.total2022 - 1.0 * t.n2021 / t.total2021 > 0.10
ORDER BY change_pp DESC, p.product_id, t.cluster_id
LIMIT 10""",
        "travel_shortlist": """WITH review_history AS (
  SELECT product_id, COUNT(*) AS reviews_2022,
         AVG(CASE WHEN rating <= 2 THEN 1.0 ELSE 0.0 END) AS low_rating_share
  FROM reviews
  WHERE review_time >= TIMESTAMP '2022-01-01'
    AND review_time < TIMESTAMP '2023-01-01' AND verified_purchase
  GROUP BY product_id
), eligible AS (
  SELECT v.id, v.vector
  FROM semantic_products_vectors v
  JOIN products p ON p.product_id = v.id
  JOIN review_history h ON h.product_id = v.id
  WHERE v.sample_rank < 1000000
    AND p.category_l3 = 'Headphones & Earbuds'
    AND p.price BETWEEN 20 AND 150
    AND h.reviews_2022 >= 100 AND h.low_rating_share <= 0.20
)
SELECT n.rank, p.product_id, p.title, p.price,
       h.reviews_2022, h.low_rating_share,
       1.0 - n.distance / 2.0 AS cosine_similarity
FROM cuvs_brute_force_knn(
  dataset => (SELECT id, vector FROM eligible),
  queries => (SELECT id, vector FROM intent_queries WHERE id = 0),
  k => 10, metric => 'l2_expanded') n
JOIN products p ON p.product_id = n.neighbor_id
JOIN review_history h ON h.product_id = p.product_id
ORDER BY n.rank""",
        "headphone_pca_topics": """SELECT m.cluster_id,
       EXTRACT(YEAR FROM r.review_time) AS review_year,
       EXTRACT(MONTH FROM r.review_time) AS review_month,
       COUNT(*) AS negative_reviews, COUNT(DISTINCT r.product_id) AS products
FROM cuvs_kmeans(
  input => (SELECT id, """ + ", ".join(f"pc_{i}" for i in range(32)) + """
    FROM cuvs_pca(
      input => (SELECT v.id, v.vector
        FROM semantic_reviews_vectors v
        JOIN semantic_reviews_text r ON r.id = v.id
        WHERE r.category_l3 = 'Headphones & Earbuds'
        ORDER BY v.id), n_components => 32)
    ORDER BY id),
  n_clusters => 24, n_init => 5, max_iter => 100, tol => 0.0001) m
JOIN semantic_reviews_text r ON r.id = m.id
GROUP BY m.cluster_id, EXTRACT(YEAR FROM r.review_time), EXTRACT(MONTH FROM r.review_time)
ORDER BY m.cluster_id, review_year, review_month""",
        "headphone_umap2": """SELECT u.id, u.dim_0, u.dim_1, r.product_id,
       p.title AS product_title, EXTRACT(YEAR FROM r.review_time) AS review_year,
       r.rating, r.title, r.embedding_text
FROM cuml_umap(
  input => (SELECT v.id, v.vector
    FROM semantic_reviews_vectors v
    JOIN semantic_reviews_text r ON r.id = v.id
    WHERE r.category_l3 = 'Headphones & Earbuds' AND v.sample_rank < 100000
    ORDER BY v.id),
  n_components => 2, n_neighbors => 30, min_dist => 0.1, random_state => 42) u
JOIN semantic_reviews_text r ON r.id = u.id
JOIN products p ON p.product_id = r.product_id
ORDER BY u.id""",
    }
    queries["inline_kmeans_topk"] = (Path(__file__).with_name("sql") / "inline_kmeans_topk.sql").read_text()
    return queries


def run_queries(names, register=False, repeats=1):
    import pyarrow.flight as flight
    import amazon_reviews_research as research

    research.REPORT = REPORT
    research.DATA = DATA / "results"
    client = research.ResearchClient("grpc://127.0.0.1:50062")
    client.options = flight.FlightCallOptions(timeout=900)
    if register:
        statements = (REPORT / "sql/00_register.sql").read_text().split(";")
        for statement in statements:
            words = statement.split()
            if words[:2] == ["CREATE", "VIEW"]:
                client.execute(f"DROP VIEW IF EXISTS {words[2]}")
        for statement in statements:
            words = statement.split()
            if words[:3] == ["CREATE", "EXTERNAL", "TABLE"]:
                client.execute(f"DROP TABLE IF EXISTS {words[3]}")
        for statement in statements:
            if statement.strip():
                client.execute(statement)
    for name in names:
        sql = (REPORT / "sql" / f"{name}.sql").read_text()
        for repeat in range(repeats):
            record = client.run(name + f"_r{repeat+1}", sql, explain=False)
            if record["status"] != "ok":
                raise RuntimeError(f"Query failed: {record['name']}: {record['error']}")


def cpu_search(repeats, sizes, counts):
    import faiss
    from threadpoolctl import threadpool_info

    faiss.omp_set_num_threads(16)
    matrix = np.load(DATA / "products.npy", mmap_mode="r")
    records = []
    for size in sizes:
        for count in counts:
            for repeat in range(repeats):
                started = time.perf_counter()
                dataset = np.array(matrix[:size], copy=True)
                queries = np.array(matrix[1000000:1000000+count], copy=True)
                materialized = time.perf_counter()
                index = faiss.IndexFlatL2(DIMENSIONS)
                index.add(dataset)
                built = time.perf_counter()
                distances, neighbors = index.search(queries, 10)
                finished = time.perf_counter()
                record = {"n": size, "q": count, "d": DIMENSIONS, "repeat": repeat+1,
                          "matrix_copy_s": materialized-started, "build_s": built-materialized,
                          "search_s": finished-built, "total_s": finished-started,
                          "threads": 16, "boundary": "resident NPY copy + index add + search; no SQL or Flight"}
                records.append(record)
                np.savez(DATA / f"cpu_exact_n{size}_q{count}.npz", distances=distances, neighbors=neighbors)
                save_json(REPORT / "cpu_search.json", {"runs": records, "threadpools": threadpool_info()})
                print(json.dumps(record), flush=True)


def cpu_clustering(sizes):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_info

    matrix = np.load(DATA / "reviews.npy", mmap_mode="r")
    records = []
    for size in sizes:
        started = time.perf_counter()
        values = np.array(matrix[:size], copy=True)
        copied = time.perf_counter()
        estimator = KMeans(n_clusters=64, n_init=1, max_iter=30, tol=0.0001,
                           random_state=42, algorithm="lloyd")
        labels = estimator.fit_predict(values)
        finished = time.perf_counter()
        np.save(DATA / f"cpu_kmeans_n{size}.npy", labels)
        record = {"name": "kmeans", "n": size, "matrix_copy_s": copied-started,
                  "fit_s": finished-copied, "total_s": finished-started,
                  "inertia": float(estimator.inertia_), "iterations": int(estimator.n_iter_)}
        records.append(record)
        save_json(REPORT / "cpu_clustering.json", {"runs": records, "threadpools": threadpool_info()})
        print(json.dumps(record), flush=True)
        started = time.perf_counter()
        reduction = PCA(n_components=32, svd_solver="covariance_eigh")
        projected = reduction.fit_transform(values)
        reduced = time.perf_counter()
        labels = estimator.fit_predict(projected)
        finished = time.perf_counter()
        np.save(DATA / f"cpu_pca_kmeans_n{size}.npy", labels)
        np.save(DATA / f"cpu_pca_components_n{size}.npy", reduction.components_)
        record = {"name": "pca_kmeans", "n": size, "pca_s": reduced-started,
                  "kmeans_s": finished-reduced, "total_s": finished-started,
                  "inertia": float(estimator.inertia_), "iterations": int(estimator.n_iter_),
                  "explained_variance_ratio": float(reduction.explained_variance_ratio_.sum())}
        records.append(record)
        save_json(REPORT / "cpu_clustering.json", {"runs": records, "threadpools": threadpool_info()})
        print(json.dumps(record), flush=True)


def cpu_thread_sweep():
    import faiss
    from threadpoolctl import threadpool_info, threadpool_limits

    matrix = np.load(DATA / "products.npy", mmap_mode="r")
    index = faiss.IndexFlatL2(DIMENSIONS)
    index.add(np.asarray(matrix[:1000000]))
    queries = np.array(matrix[1000000:1010000])
    records = []
    for threads in (4, 8, 16, 32):
        with threadpool_limits(limits=threads):
            faiss.omp_set_num_threads(threads)
            for count in (1000, 10000):
                started = time.perf_counter()
                index.search(queries[:count], 10)
                record = {"threads": threads, "q": count, "search_s": time.perf_counter()-started,
                          "threadpools": threadpool_info()}
                records.append(record)
                save_json(REPORT / "cpu_thread_sweep.json", records)
                print(json.dumps({k: v for k, v in record.items() if k!="threadpools"}), flush=True)


def compact_vectors():
    records = []
    for name in ("products", "reviews", "intents"):
        source = DATA / f"{name}_vectors.parquet"
        target = DATA / f"{name}_vectors_compact.parquet"
        parquet = pq.ParquetFile(source)
        started = time.perf_counter()
        with pq.ParquetWriter(target, parquet.schema_arrow, compression="snappy", use_dictionary=False) as writer:
            for batch in parquet.iter_batches(batch_size=131072):
                writer.write_batch(batch, row_group_size=131072)
        records.append({"name": name, "before_row_groups": parquet.num_row_groups,
                        "after_row_groups": pq.ParquetFile(target).num_row_groups,
                        "before_bytes": source.stat().st_size, "after_bytes": target.stat().st_size,
                        "elapsed_s": time.perf_counter()-started})
    save_json(REPORT / "compact_layout.json", records)
    print(json.dumps(records), flush=True)


def verify_inputs():
    provenance = json.loads((REPORT / "embedding_provenance.json").read_text())
    recorded = {run["name"]: run for run in provenance["runs"]}
    results = {}
    for name in ("products", "reviews", "intents"):
        source = pq.read_table(DATA / f"{name}_text.parquet", columns=["id"])["id"].to_numpy()
        matrix = np.load(DATA / f"{name}.npy", mmap_mode="r")
        parquet = pq.ParquetFile(DATA / f"{name}_vectors_compact.parquet")
        digest = hashlib.sha256()
        offset = 0
        norm_error = 0.0
        for batch in parquet.iter_batches(batch_size=8192):
            table = pa.Table.from_batches([batch])
            assert all(column.null_count == 0 for column in table.columns)
            if PROFILE == "bge-m3":
                vector = table["vector"].combine_chunks()
                assert vector.values.null_count == 0
                values = vector.values.to_numpy().reshape(len(table), DIMENSIONS)
            else:
                values = np.column_stack([table[column].to_numpy() for column in FEATURES])
            assert np.isfinite(values).all()
            assert np.array_equal(table["id"].to_numpy(), source[offset:offset+len(table)])
            assert np.array_equal(table["sample_rank"].to_numpy(), np.arange(offset, offset+len(table)))
            assert np.array_equal(values, matrix[offset:offset+len(table)])
            norm_error = max(norm_error, float(np.max(np.abs(np.linalg.norm(values, axis=1)-1))))
            digest.update(values.tobytes())
            offset += len(table)
        assert offset == len(source) == len(matrix)
        assert len(np.unique(source)) == len(source)
        assert norm_error < 1e-5
        assert digest.hexdigest() == recorded[name]["matrix_sha256"]
        results[name] = {"rows": offset, "dimensions": DIMENSIONS, "unique_ids": True,
                         "parquet_matches_npy_and_source_order": True,
                         "finite_and_non_null": True, "max_unit_norm_error": norm_error,
                         "matrix_sha256": digest.hexdigest(), "row_groups": parquet.num_row_groups}
    save_json(REPORT / "input_verification.json", results)
    print(json.dumps(results), flush=True)


def export_review_map():
    table = pq.read_table(DATA / "results/headphone_umap2_r1.parquet")
    # PyArrow 25 lacks take/sort kernels for StringView; select from exported Python rows.
    rows = sorted(table.to_pylist(), key=lambda row: row["id"])
    count = min(1200, len(table))
    rows = [rows[position] for position in np.linspace(0, len(rows)-1, count, dtype=np.int64)]
    points = [{"id": row["id"], "x": row["dim_0"], "y": row["dim_1"],
               "year": int(row["review_year"]), "rating": row["rating"],
               "product": row["product_title"], "text": row["embedding_text"]} for row in rows]
    record = json.loads((REPORT / "results/headphone_umap2_r1.json").read_text())
    # The capture date comes from the saved query artifact, including when re-exported later.
    captured = datetime.fromtimestamp((REPORT / "results/headphone_umap2_r1.json").stat().st_mtime,
                                      tz=timezone.utc).date().isoformat()
    output = ROOT / "website/src/data/amazonReviewMap.json"
    save_json(output, {"fittedRows": len(table), "captured": captured, "points": points})
    save_json(REPORT / "website_export.json", {"path": str(output.relative_to(ROOT)),
              "query_id": record["query_id"], "sql": record["sql"], "displayed_rows": count,
              "fitted_rows": len(table), "sampling": "evenly spaced positions after ORDER BY id",
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    print(json.dumps({"website_map": str(output), "displayed_rows": count, "fitted_rows": len(table)}))


def index_positions(ids, keys):
    order = np.argsort(ids)
    positions = order[np.searchsorted(ids[order], keys)]
    assert np.array_equal(ids[positions], keys)
    return positions


def verify_demo_results(timings):
    from sklearn.manifold import trustworthiness

    connection = duckdb.connect()
    connection.execute("SET threads=8")
    products_path = ROOT / "fixture/demo/amazon_reviews_2023/parquet/products.parquet"
    reviews_path = ROOT / "fixture/demo/amazon_reviews_2023/parquet/reviews.parquet"
    product_text = DATA / "products_text.parquet"
    review_text = DATA / "reviews_text.parquet"
    results = {}
    if "travel_shortlist" in timings:
        eligible = connection.execute(f"""WITH history AS (
          SELECT product_id, count(*) AS n,
                 sum(CAST(rating <= 2 AS BIGINT))::DOUBLE / count(*) AS low
          FROM '{reviews_path}'
          WHERE year(review_time) = 2022 AND verified_purchase = true GROUP BY product_id)
        SELECT p.product_id, h.n, h.low, p.price, s.sample_rank
        FROM '{products_path}' p JOIN history h USING (product_id)
        JOIN '{product_text}' s ON s.id = p.product_id
        WHERE s.sample_rank < 1000000 AND p.category_l3 = 'Headphones & Earbuds'
          AND p.price >= 20 AND p.price <= 150 AND h.n >= 100 AND h.low <= 0.2""").to_arrow_table()
        product_vectors = np.load(DATA / "products.npy", mmap_mode="r")
        query = np.load(DATA / "intents.npy")[0].astype(np.float64)
        values = product_vectors[eligible["sample_rank"].to_numpy()].astype(np.float64)
        distance = np.sum((values-query)**2, axis=1)
        ordered = np.argsort(distance)[:10]
        expected = eligible.take(ordered).to_pylist()
        actual = pq.read_table(DATA / "results/travel_shortlist_r1.parquet").to_pylist()
        assert [row["product_id"] for row in actual] == [row["product_id"] for row in expected]
        for row, reference, position in zip(actual, expected, ordered):
            assert row["reviews_2022"] == reference["n"]
            assert abs(row["low_rating_share"] - reference["low"]) < 1e-12
            assert row["price"] == reference["price"]
            assert abs(row["cosine_similarity"] - (1-distance[position]/2)) < 1e-5
        results["travel_shortlist"] = {"eligible_products": len(eligible), "rows": len(actual),
                                      "duckdb_history_and_float64_knn_match": True}
    if "headphone_emerging" in timings:
        totals = connection.execute(f"""SELECT product_id, year(review_time), count(*)
            FROM '{review_text}' WHERE category_l3 = 'Headphones & Earbuds' GROUP BY 1,2""").fetchall()
        totals = {(product, year): count for product, year, count in totals}
        actual = pq.read_table(DATA / "results/headphone_emerging_r1.parquet").to_pylist()
        ids = [row["example_id"] for row in actual]
        source_rows = connection.execute(f"SELECT * FROM '{review_text}' WHERE id IN (SELECT unnest(?))",
                                          [ids]).to_arrow_table().to_pylist()
        source = {row["id"]: row for row in source_rows}
        for row in actual:
            assert row["total2021"] == totals[row["product_id"], 2021]
            assert row["total2022"] == totals[row["product_id"], 2022]
            assert 0 <= row["n2021"] <= row["total2021"]
            assert 10 <= row["n2022"] <= row["total2022"]
            change = 100 * row["n2022"] / row["total2022"] - 100 * row["n2021"] / row["total2021"]
            assert abs(row["change_pp"]-change) < 1e-8
            example = source[row["example_id"]]
            assert example["product_id"] == row["product_id"] and example["review_time"].year == 2022
            assert example["embedding_text"] == row["example_review"]
        results["headphone_emerging"] = {"rows": len(actual), "duckdb_denominators_match": True,
                                        "examples_match_original_rows": True}
    if "headphone_umap2" in timings:
        actual = pq.read_table(DATA / "results/headphone_umap2_r1.parquet")
        source = connection.execute(f"""SELECT id, sample_rank FROM '{review_text}'
            WHERE category_l3 = 'Headphones & Earbuds' AND sample_rank < 100000 ORDER BY id""").to_arrow_table()
        assert np.array_equal(actual["id"].to_numpy(), source["id"].to_numpy())
        coordinates = np.column_stack([actual["dim_0"].to_numpy(), actual["dim_1"].to_numpy()])
        assert np.isfinite(coordinates).all()
        sample = np.random.default_rng(42).choice(len(actual), min(1500, len(actual)), replace=False)
        vectors = np.load(DATA / "reviews.npy", mmap_mode="r")[source["sample_rank"].to_numpy()[sample]]
        score = trustworthiness(vectors, coordinates[sample], n_neighbors=15)
        deltas = []
        for repeat in range(2, len(timings["headphone_umap2"])+1):
            other = pq.read_table(DATA / f"results/headphone_umap2_r{repeat}.parquet")
            assert np.array_equal(other["id"].to_numpy(), actual["id"].to_numpy())
            other_coordinates = np.column_stack([other["dim_0"].to_numpy(), other["dim_1"].to_numpy()])
            deltas.append(float(np.max(np.abs(coordinates-other_coordinates))))
        results["headphone_umap2"] = {"rows": len(actual), "cohort_ids_match": True,
                                     "finite_coordinates": True, "trustworthiness_1500_at_15": float(score),
                                     "repeat_max_coordinate_deltas": deltas}
    return results


def analyze():
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    records = [json.loads(p.read_text()) for p in sorted((REPORT / "results").glob("*_r*.json"))]
    timings = {}
    for record in records:
        name = re.sub(r"_r\d+$", "", record["name"])
        timings.setdefault(name, []).append(record)
    summary = {"queries": {name: {"runs": len(group), "statuses": [r["status"] for r in group],
                                  "median_s": float(np.median([r["elapsed_s"] for r in group])),
                                  "min_s": min(r["elapsed_s"] for r in group),
                                  "max_s": max(r["elapsed_s"] for r in group),
                                  "rows": group[0].get("rows")}
                           for name, group in timings.items()}}
    for name, group in timings.items():
        observations = []
        for record in group:
            with gzip.open(REPORT / "results" / f"{record['name']}.stats.txt.gz", "rt") as source:
                lines = source.read().splitlines()
            position = lines.index("[execution]")
            fields = dict(zip(lines[position+1].split("\t"), lines[position+2].split("\t")))
            observation = {key: fields[key] for key in (
                "query_id", "post_native_execution_path", "host_datafusion_boundary_count",
                "device_ordinal", "query_grant_bytes", "max_task_observed_peak_bytes",
                "gpu_resident_input_handoffs", "host_to_device_uploaded_bytes",
                "device_to_host_materializations", "device_to_host_materialized_bytes",
                "allocator_high_water_bytes")}
            planning_position = lines.index("[planning]")
            planning_header = lines[planning_position+1].split("\t")
            reasons = set()
            for line in lines[planning_position+2:]:
                if not line or line.startswith("["):
                    break
                planning = dict(zip(planning_header, line.split("\t")))
                if planning["reason_code"] != "none":
                    reasons.add(planning["reason_code"])
            observation["planning_reason_codes"] = sorted(reasons)
            observations.append(observation)
        summary["queries"][name]["observations"] = observations
    product_ids = pq.read_table(DATA / "products_text.parquet", columns=["id"])["id"].to_numpy()
    products = np.load(DATA / "products.npy", mmap_mode="r")
    summary["neighbors"] = {}
    for name in timings:
        if not (name.startswith("exact_") or name.startswith("ivf_") or name.startswith("cagra")):
            continue
        match = re.match(r"exact_n(\d+)_q(\d+)", name)
        size, count = map(int, match.groups()) if match else (1000000, 10000 if name.endswith("q10000") else 1000)
        reference = np.load(DATA / f"cpu_exact_n{size}_q{count}.npz")
        table = pq.read_table(DATA / f"results/{name}_r1.parquet")
        query_ids = table["query_id"].to_numpy().reshape(count, 10)[:, 0]
        query_positions = index_positions(product_ids, query_ids)
        actual_ids = table["neighbor_id"].to_numpy().reshape(count, 10)
        actual_positions = index_positions(product_ids, actual_ids.reshape(-1)).reshape(count, 10)
        expected = reference["neighbors"][query_positions-1000000]
        recall = np.mean([len(set(a) & set(b))/10 for a, b in zip(actual_positions, expected)])
        distances = np.sum((products[actual_positions].astype(np.float64) -
                            products[query_positions, None].astype(np.float64))**2, axis=2)
        threshold = reference["distances"][query_positions-1000000, -1]
        reported = table["distance"].to_numpy().reshape(count, 10)
        summary["neighbors"][name] = {
            "recall_at_10_ids": float(recall),
            "fraction_within_exact_top10_radius_plus_1e_5": float(np.mean(distances <= threshold[:, None]+1e-5)),
            "max_reported_distance_error": float(np.max(np.abs(distances-reported))),
            "mean_reported_distance_error": float(np.mean(np.abs(distances-reported))),
            "reported_zero_with_true_distance_gt_1e_5": int(np.sum((reported==0) & (distances>1e-5))),
            "true_distance_rank_inversions_gt_1e_5": int(np.sum(np.diff(distances, axis=1)<-1e-5)),
            "corpus_membership_ok": bool(np.all(actual_positions<size)),
            "queries_disjoint_from_corpus": bool(np.all(query_positions>=1000000)),
        }
    review_ids = pq.read_table(DATA / "reviews_text.parquet", columns=["id"])["id"].to_numpy()
    reviews = np.load(DATA / "reviews.npy", mmap_mode="r")
    summary["clusters"] = {}
    for name in ("kmeans_n100000", "kmeans_n1000000", "pca_kmeans_n100000", "pca_kmeans_n1000000",
                 "headphone_assignments", "umap_hdbscan_10000", "umap_hdbscan_50000", "umap_hdbscan_100000"):
        if name not in timings:
            continue
        table = pq.read_table(DATA / f"results/{name}_r1.parquet")
        ids = table["id"].to_numpy()
        labels = table["cluster_id"].to_numpy()
        positions = index_positions(review_ids, ids)
        values = np.array(reviews[positions])
        clusters, counts = np.unique(labels, return_counts=True)
        record = {"rows": len(ids), "clusters_excluding_noise": int(sum(clusters>=0)),
                  "noise_rows": int(sum(labels<0)), "largest_cluster": int(max(counts)),
                  "unique_ids": int(len(np.unique(ids))), "repeat_ari": []}
        if "hdbscan" not in name:
            centroids = np.vstack([values[labels==c].mean(axis=0, dtype=np.float64) for c in clusters])
            total_squares = np.einsum("ij,ij->", values, values, dtype=np.float64)
            record["original_space_inertia"] = float(total_squares - np.sum(counts*np.sum(centroids**2, axis=1)))
            record["silhouette_2000"] = float(silhouette_score(values, labels, sample_size=2000, random_state=42))
        for repeat in (2, 3):
            path = DATA / f"results/{name}_r{repeat}.parquet"
            if path.exists():
                other = pq.read_table(path)
                aligned = index_positions(other["id"].to_numpy(), ids)
                record["repeat_ari"].append(float(adjusted_rand_score(labels, other["cluster_id"].to_numpy()[aligned])))
        summary["clusters"][name] = record
    if "pca_n1000000" in timings:
        pca = pq.read_table(DATA / "results/pca_n1000000_r1.parquet")
        projected = np.column_stack([pca[f"pc_{i}"].to_numpy() for i in range(32)])
        mean = reviews.mean(axis=0, dtype=np.float64)
        total = np.einsum("ij,ij->", reviews, reviews, dtype=np.float64)-len(reviews)*np.dot(mean, mean)
        summary["pca_32_centered_variance_ratio"] = float(np.sum(projected.astype(np.float64)**2)/total)
    if PROFILE == "bge-m3":
        summary["demo_verification"] = verify_demo_results(timings)
    save_json(REPORT / "analysis.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def summarize_topics():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    connection = duckdb.connect()
    connection.execute("SET threads=16")
    assignments = DATA / "results/headphone_assignments_r1.parquet"
    source = DATA / "reviews_text.parquet"
    products = ROOT / "fixture/demo/amazon_reviews_2023/parquet/products.parquet"
    table = connection.execute(f"""SELECT a.id, a.cluster_id, r.sample_rank, r.product_id,
        r.review_time, r.rating, r.title, r.text, r.embedding_text, p.title AS product_title
        FROM '{assignments}' a JOIN '{source}' r ON a.id=r.id
        JOIN '{products}' p ON r.product_id=p.product_id ORDER BY a.id"""
    ).to_arrow_table()
    rows = table.to_pylist()
    labels = table["cluster_id"].to_numpy()
    values = np.load(DATA / "reviews.npy", mmap_mode="r")[table["sample_rank"].to_numpy()]
    stop_words = sorted(set(ENGLISH_STOP_WORDS) | {
        "headphones", "earbuds", "headphone", "product", "amazon", "bought", "buy",
        "use", "used", "work", "works", "good", "great", "just", "really", "like",
        "don", "didn", "ve", "ll", "doesn", "star", "stars", "got", "pair", "thing"})
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=10, max_df=0.85,
                                 max_features=20000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(table["embedding_text"].to_pylist())
    terms = vectorizer.get_feature_names_out()
    years = np.array([r["review_time"].year for r in rows])
    monthly = pq.read_table(DATA / "results/headphone_monthly_saved_r1.parquet").to_pylist()
    topics = []
    for cluster in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels==cluster)
        centroid = values[indices].mean(axis=0)
        distances = np.sum((values[indices]-centroid)**2, axis=1)
        representative_positions = indices[np.argsort(distances)[:5]]
        weights = np.asarray(tfidf[indices].mean(axis=0)).ravel()
        top_terms = terms[np.argsort(weights)[-12:][::-1]].tolist()
        count_2021 = int(sum(years[indices]==2021))
        count_2022 = int(sum(years[indices]==2022))
        representatives = [{k: rows[i][k] for k in
                            ("id", "product_id", "product_title", "review_time", "rating", "title", "text")}
                           for i in representative_positions]
        topics.append({"cluster_id": int(cluster), "rows": len(indices), "top_terms": top_terms,
                       "count_2021": count_2021, "count_2022": count_2022,
                       "share_2021": count_2021/int(sum(years==2021)),
                       "share_2022": count_2022/int(sum(years==2022)),
                       "representatives": representatives,
                       "monthly": [m for m in monthly if m["cluster_id"]==cluster]})
    expected = connection.execute(f"""SELECT a.cluster_id, year(r.review_time) AS review_year,
        month(r.review_time) AS review_month, count(*) AS negative_reviews
        FROM '{assignments}' a JOIN '{source}' r ON a.id=r.id GROUP BY 1,2,3 ORDER BY 1,2,3"""
    ).fetchall()
    actual = [tuple(m[k] for k in ("cluster_id", "review_year", "review_month", "negative_reviews")) for m in monthly]
    assert expected == actual
    pca = pq.read_table(DATA / "results/headphone_pca2_r1.parquet")
    selected = np.linspace(0, len(rows)-1, 3000, dtype=int)
    pca_positions = index_positions(pca["id"].to_numpy(), table["id"].to_numpy()[selected])
    points = [{"x": float(pca["pc_0"][p].as_py()), "y": float(pca["pc_1"][p].as_py()),
               "cluster": int(labels[i]), "id": int(rows[i]["id"]),
               "text": rows[i]["text"][:500], "product": rows[i]["product_title"]}
              for i, p in zip(selected, pca_positions)]
    result = {"source_assignment": "headphone_assignments_r1", "rows": len(rows),
              "year_totals": {str(year): int(sum(years==year)) for year in (2021, 2022)},
              "monthly_cpu_oracle_equal": True, "topics": topics, "points": points,
              "shifts": pq.read_table(DATA / "results/headphone_issue_shift_r1.parquet").to_pylist()}
    save_json(REPORT / "topics.json", result)
    print(json.dumps([{k: t[k] for k in ("cluster_id", "rows", "top_terms", "share_2021", "share_2022")}
                      for t in topics], indent=2), flush=True)


def render_report():
    summary = json.loads((REPORT / "analysis.json").read_text())
    cpu = json.loads((REPORT / "cpu_search.json").read_text())["runs"]
    cpu_summary = []
    for size in (100000, 1000000):
        for count in (1, 1000, 10000):
            group = [r for r in cpu if r["n"]==size and r["q"]==count]
            cpu_summary.append({"n": size, "q": count,
                                "search_s": float(np.median([r["search_s"] for r in group]))})
    catalog = pq.read_table(DATA / "results/catalog_candidates_r1.parquet")
    payload = {"intents": INTENTS, "labels": json.loads((REPORT / "topic_labels.json").read_text())["labels"],
               "topics": json.loads((REPORT / "topics.json").read_text()),
               "search": pq.read_table(DATA / "results/intent_search_r1.parquet").to_pylist(),
               "budget": pq.read_table(DATA / "results/budget_headphones_r1.parquet").to_pylist(),
               "catalog": {"rows": catalog.num_rows, "queries": len(set(catalog["query_id"].to_pylist()))},
               "timings": summary["queries"], "cpu": cpu_summary,
               "ml": [r for r in json.loads((REPORT / "cpu_clustering.json").read_text())["runs"] if r["n"]==1000000],
               "sql": {name: (REPORT / "sql" / filename).read_text() for name, filename in
                       (("budget", "budget_headphones.sql"), ("catalog", "catalog_candidates.sql"),
                        ("topics", "headphone_topics.sql"))}}
    encoded = json.dumps(payload, ensure_ascii=False, default=str).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (REPORT / "dashboard.template.html").read_text()
    font = base64.b64encode((REPORT / "ResearchCJK-subset.otf").read_bytes()).decode()
    license_text = html.escape((REPORT / "FONT_LICENSE.txt").read_text())
    (REPORT / "demo.html").write_text(template.replace("__PAYLOAD__", encoded)
                                    .replace("__FONT_BASE64__", font)
                                    .replace("__FONT_LICENSE__", license_text))
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    exports = REPORT / "exports"
    exports.mkdir(exist_ok=True)
    names = ["intent_search", "catalog_candidates", "budget_headphones", "headphone_assignments",
             "headphone_monthly_saved", "headphone_issue_shift", "kmeans_n1000000", "pca_kmeans_n1000000",
             "exact_n1000000_q10000", "ivf_flat_p32_q10000", "ivf_pq_q10000", "cagra_q10000",
             "90_l2_near_unit_pairwise", "91_l2_near_unit_knn", "92_cosine_near_unit_pairwise", "93_l2_near_unit_top1"]
    for name in names:
        source = DATA / f"results/{name}_r1.parquet"
        target = exports / f"{name}.csv.gz"
        connection.execute(f"COPY (SELECT * FROM '{source}') TO '{target}' (FORMAT CSV, COMPRESSION GZIP)")
    manifest = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size}
                for p in sorted(DATA.rglob("*.parquet"))]
    save_json(REPORT / "artifact_manifest.json", manifest)
    print(json.dumps({"dashboard_bytes": (REPORT / "demo.html").stat().st_size,
                      "exported_tables": len(names), "large_artifacts": len(manifest)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("minilm", "bge-m3"), default="minilm")
    parser.add_argument("stage", choices=("prepare", "embed", "sql", "run", "cpu-search", "cpu-clustering", "cpu-threads", "compact", "verify-inputs", "export-review-map", "analyze", "topics", "render"))
    parser.add_argument("--names", nargs="+", default=["intents", "products", "reviews"])
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 1000, 10000])
    args = parser.parse_args()
    configure(args.profile)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "embed":
        embed(args.names)
    elif args.stage == "sql":
        generate_sql()
    elif args.stage == "run":
        run_queries(args.names, args.register, args.repeats)
    elif args.stage == "cpu-search":
        cpu_search(args.repeats, args.sizes, args.counts)
    elif args.stage == "cpu-clustering":
        cpu_clustering(args.sizes)
    elif args.stage == "cpu-threads":
        cpu_thread_sweep()
    elif args.stage == "compact":
        compact_vectors()
    elif args.stage == "verify-inputs":
        verify_inputs()
    elif args.stage == "export-review-map":
        export_review_map()
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "topics":
        summarize_topics()
    elif args.stage == "render":
        render_report()


if __name__ == "__main__":
    main()
