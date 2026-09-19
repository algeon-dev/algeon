#!/usr/bin/env python3
"""Prepare and inspect semantic Amazon demo inputs; keep generated data under target."""

import argparse
import base64
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "target/amazon-reviews-semantic"
REPORT = ROOT / "docs/research/amazon_reviews_2023/semantic"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSIONS = 384
FEATURES = [f"e{i:03d}" for i in range(DIMENSIONS)]
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
        started = time.perf_counter()
        path = DATA / f"{name}_text.parquet"
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
    model = SentenceTransformer(MODEL, revision=revision, device="cuda")
    model.half()
    provenance = {"model": MODEL, "revision": revision, "dimensions": DIMENSIONS,
                  "max_sequence_length": model.max_seq_length, "inference_dtype": "float16",
                  "stored_dtype": "float32", "normalization": "L2 in float32 after encoding",
                  "model_load_s": time.perf_counter()-started,
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
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False),
                            pa.field("sample_rank", pa.int32(), nullable=False)] +
                           [pa.field(column, pa.float32(), nullable=False) for column in FEATURES])
        digest = hashlib.sha256()
        with pq.ParquetWriter(DATA / f"{name}_vectors.parquet", schema, compression="snappy",
                              use_dictionary=False) as writer:
            for batch in parquet.iter_batches(batch_size=8192):
                table = pa.Table.from_batches([batch])
                texts = table["embedding_text"].to_pylist()
                values = model.encode(texts, batch_size=512, show_progress_bar=False,
                                      convert_to_numpy=True).astype(np.float32)
                values /= np.linalg.norm(values, axis=1, keepdims=True)
                assert values.shape == (len(texts), DIMENSIONS) and np.isfinite(values).all()
                matrix[offset:offset+len(values)] = values
                digest.update(values.tobytes())
                rank = (table["sample_rank"] if "sample_rank" in table.column_names else
                        pa.array(np.arange(offset, offset+len(values), dtype=np.int32)))
                arrays = [table["id"].cast(pa.int64()), rank]
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
    (directory / "00_register.sql").write_text("\n\n".join(definitions) + "\n")
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
    for name, sql in queries.items():
        (directory / f"{name}.sql").write_text(sql + "\n")
    (directory / "01_register_topics.sql").write_text(
        "CREATE EXTERNAL TABLE semantic_headphone_clusters STORED AS PARQUET LOCATION '"
        + str(DATA / "results/headphone_assignments_r1.parquet") + "';\n"
        "CREATE EXTERNAL TABLE semantic_full_products STORED AS PARQUET LOCATION '"
        + str(ROOT / "fixture/demo/amazon_reviews_2023/parquet/products.parquet") + "';\n")
    print(json.dumps({"queries": len(queries), "views_and_tables": len(definitions)}))


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
            client.run(name + f"_r{repeat+1}", sql, explain=False)


def cpu_search(repeats):
    import faiss
    from threadpoolctl import threadpool_info

    faiss.omp_set_num_threads(16)
    matrix = np.load(DATA / "products.npy", mmap_mode="r")
    records = []
    for size in (100000, 1000000):
        for count in (1, 1000, 10000):
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


def index_positions(ids, keys):
    order = np.argsort(ids)
    positions = order[np.searchsorted(ids[order], keys)]
    assert np.array_equal(ids[positions], keys)
    return positions


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
        distances = np.sum((products[actual_positions] - products[query_positions, None])**2, axis=2)
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
    pca = pq.read_table(DATA / "results/pca_n1000000_r1.parquet")
    projected = np.column_stack([pca[f"pc_{i}"].to_numpy() for i in range(32)])
    mean = reviews.mean(axis=0, dtype=np.float64)
    total = np.einsum("ij,ij->", reviews, reviews, dtype=np.float64)-len(reviews)*np.dot(mean, mean)
    summary["pca_32_centered_variance_ratio"] = float(np.sum(projected.astype(np.float64)**2)/total)
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
    parser.add_argument("stage", choices=("prepare", "embed", "sql", "run", "cpu-search", "cpu-clustering", "cpu-threads", "compact", "analyze", "topics", "render"))
    parser.add_argument("--names", nargs="+", default=["products", "reviews", "intents"])
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 1000000])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "embed":
        embed(args.names)
    elif args.stage == "sql":
        generate_sql()
    elif args.stage == "run":
        run_queries(args.names, args.register, args.repeats)
    elif args.stage == "cpu-search":
        cpu_search(args.repeats)
    elif args.stage == "cpu-clustering":
        cpu_clustering(args.sizes)
    elif args.stage == "cpu-threads":
        cpu_thread_sweep()
    elif args.stage == "compact":
        compact_vectors()
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "topics":
        summarize_topics()
    elif args.stage == "render":
        render_report()


if __name__ == "__main__":
    main()
