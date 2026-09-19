#!/usr/bin/env python3
"""CPU references and measurements for the persisted Amazon SQL experiments."""

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, silhouette_score,
)
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/research/amazon_reviews_2023"
DATA = ROOT / "target/amazon-reviews-research"
ISSUES = "battery connection sound comfort durability charging microphone returns".split()
RISK = "log_volume low_rate mean_stars five_rate verified_rate image_rate q4_share three_rate".split()


def read(name):
    return pq.read_table(DATA / f"{name}.parquet")


def matrix(table, columns):
    return np.column_stack([np.array(table[c]) for c in columns])


def write(name, value):
    (REPORT / "results" / f"{name}.json").write_text(json.dumps(value, indent=2, default=str) + "\n")


def train():
    table = read("input_cohort")
    order = np.argsort(np.array(table["id"]))
    x = matrix(table, RISK)[order]
    y = np.array(table["label"])[order]
    ids = np.array(table["id"])[order]
    training = np.array(table["feature_year"])[order] == 2021
    assert np.isfinite(x).all()
    assert not set(ids[training]) & set(ids[~training])
    started = time.perf_counter()
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=.05, subsample=1,
        colsample_bytree=1, tree_method="hist", n_jobs=8, random_state=42,
    )
    model.fit(x[training], y[training])
    model.save_model(REPORT / "models/quality_risk_xgboost.json")
    pq.write_table(pa.table({"id": ids[~training], "prediction": model.predict_proba(x[~training])[:, 1]}),
                   DATA / "xgboost_cpu_predictions.parquet")
    write("model_training", {
        "xgboost_version": xgb.__version__, "parameters": model.get_params(), "features": RISK,
        "train_rows": int(training.sum()), "test_rows": int((~training).sum()),
        "training_positive_rate": float(y[training].mean()), "test_positive_rate": float(y[~training].mean()),
        "fit_and_export_s": time.perf_counter() - started,
    })


def features():
    db = duckdb.connect()
    for name in ("products", "reviews"):
        db.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{ROOT}/fixture/demo/amazon_reviews_2023/parquet/{name}.parquet')")
    checks = []
    for name, keys in (("00_review_rollup", ["id", "feature_year"]), ("01_headphone_complaints", ["id"])):
        started = time.perf_counter()
        reference = db.sql((REPORT / "sql" / f"{name}.sql").read_text()).to_arrow_table()
        elapsed = time.perf_counter() - started
        actual = read(name)
        a_order = np.lexsort(tuple(np.array(actual[k]) for k in reversed(keys)))
        b_order = np.lexsort(tuple(np.array(reference[k]) for k in reversed(keys)))
        errors = {c: float(np.max(np.abs(np.array(actual[c], dtype=float)[a_order] -
                                         np.array(reference[c], dtype=float)[b_order]))) for c in actual.column_names}
        assert len(actual) == len(reference)
        assert max(errors.values()) < 1e-6, errors
        checks.append({"name": name, "rows": len(actual), "null_counts": {c: actual[c].null_count for c in actual.column_names},
                       "max_abs_errors": errors, "duckdb_reference_s": elapsed})
    write("feature_correctness", checks)


def neighbors():
    table = read("17_cuvs_pca")
    vectors = dict(zip(table["id"].to_pylist(), matrix(table, ISSUES).astype(float)))
    ids = np.array(list(vectors))
    values = np.array(list(vectors.values()))
    report = {}
    baseline = {}
    for name in ("10_cuvs_brute_force_knn", "11_cuvs_ivf_flat", "12_cuvs_ivf_pq", "13_cuvs_cagra"):
        rows = read(name).to_pylist()
        peers = {}
        distance_errors = []
        incorrect_exact = 0
        for row in rows:
            q, n = row["query_id"], row["neighbor_id"]
            true = float(np.sum((vectors[q] - vectors[n]) ** 2))
            distance_errors.append(abs(true - row["distance"]))
            if q != n:
                peers.setdefault(q, []).append(n)
            if name == "10_cuvs_brute_force_knn":
                threshold = np.partition(np.sum((values - vectors[q]) ** 2, axis=1), 10)[10]
                incorrect_exact += true > threshold + 1e-6
        peers = {q: set(ns[:10]) for q, ns in peers.items()}
        if not baseline:
            baseline = peers
            assert incorrect_exact == 0
            assert max(distance_errors) < 1e-5
        report[name] = {"dataset_rows": len(ids), "query_rows": len(peers), "rows": len(rows),
                        "peer_recall_at_10_excluding_self": float(np.mean([len(ns & baseline[q]) / 10 for q, ns in peers.items()])),
                        "max_distance_error": max(distance_errors),
                        "incorrect_exact_neighbors": int(incorrect_exact) if name == "10_cuvs_brute_force_knn" else None}
    pairwise_errors = [abs(row["distance"] - np.sum((vectors[row["x_id"]] - vectors[row["y_id"]]) ** 2))
                       for row in read("14_cuvs_pairwise_distance").to_pylist()]
    assert max(pairwise_errors) < 1e-5
    report["pairwise"] = {"rows": len(pairwise_errors), "max_abs_error": float(max(pairwise_errors))}
    topk_errors = []
    for row in read("15_cuvs_top_k").to_pylist():
        vector = vectors[row["id"]]
        topk_errors.append(abs(row["mention_rate"] - vector[row["position"]]))
        assert row["mention_rate"] >= np.sort(vector)[-3] - 1e-7
    assert max(topk_errors) == 0
    report["topk"] = {"rows": len(topk_errors), "max_abs_error": float(max(topk_errors))}
    scale_base = {}
    for name in ("50_scale_cuvs_brute_force_knn", "51_scale_cuvs_ivf_flat", "52_scale_cuvs_ivf_pq", "53_scale_cuvs_cagra"):
        peers = {}
        for row in read(name).to_pylist():
            if row["query_id"] != row["neighbor_id"]:
                peers.setdefault(row["query_id"], []).append(row["neighbor_id"])
        peers = {q: set(ns[:10]) for q, ns in peers.items()}
        if not scale_base:
            scale_base = peers
        report[name] = {"queries": len(peers), "dataset_rows": int(np.sum(np.array(read("input_rollup")["feature_year"]) == 2022)),
                        "peer_recall_at_10_excluding_self": float(np.mean([len(ns & scale_base[q]) / 10 for q, ns in peers.items()]))}
    write("neighbor_correctness", report)


def geometry():
    table = read("17_cuvs_pca")
    x = matrix(table, ISSUES).astype(float)
    report = {}
    for name, centered in (("17_cuvs_pca", True), ("18_cuml_tsvd", False)):
        source = x - x.mean(axis=0) if centered else x
        u, singular, _ = np.linalg.svd(source, full_matrices=False)
        reference = u[:, :2] * singular[:2]
        actual = matrix(read(name), ["pc_0", "pc_1"]).astype(float)
        gram = reference @ reference.T
        relative = float(np.linalg.norm(actual @ actual.T - gram) / np.linalg.norm(gram))
        assert relative < 1e-4, (name, relative)
        report[name] = {"relative_gram_error": relative,
                        "two_component_energy_fraction": float(np.sum(singular[:2] ** 2) / np.sum(singular ** 2))}
    kmeans = np.array(read("16_cuvs_kmeans")["cluster_id"])
    centroids = {k: x[kmeans == k].mean(axis=0) for k in set(kmeans)}
    inertia = sum(float(np.sum((v - centroids[k]) ** 2)) for v, k in zip(x, kmeans))
    cpu = KMeans(n_clusters=6, n_init=5, random_state=42).fit(x)
    report["kmeans"] = {"gpu_inertia": inertia, "sklearn_inertia": float(cpu.inertia_),
                        "silhouette": float(silhouette_score(x, kmeans)), "cluster_sizes": dict(Counter(map(int, kmeans)))}
    for name in ("20_cuml_hdbscan", "25_pca_hdbscan", "26_umap_hdbscan", "47_umap_hdbscan_ordered"):
        t = read(name)
        labels = np.array(t["cluster_id"])
        probabilities = np.array(t["probability"])
        assert np.isfinite(probabilities).all() and np.min(probabilities) >= 0 and np.max(probabilities) <= 1
        report[name] = {"cluster_sizes": dict(Counter(map(int, labels))), "noise_fraction": float(np.mean(labels == -1))}
    for name in ("19_cuml_umap", "45_umap_ordered", "49_umap_ordered_limit"):
        z = matrix(read(name), ["dim_0", "dim_1"])
        z2 = matrix(read(name + "_r2"), ["dim_0", "dim_1"])
        report[name] = {"trustworthiness_15": float(trustworthiness(x, z, n_neighbors=15)),
                        "repeat_max_abs_coordinate_change": float(np.max(np.abs(z-z2))),
                        "repeat_rows_changed": int(np.sum(np.any(z != z2, axis=1)))}
    for name in ("16_cuvs_kmeans", "46_kmeans_ordered", "56_kmeans_ordered_limit", "20_cuml_hdbscan",
                 "26_umap_hdbscan", "47_umap_hdbscan_ordered", "57_umap_hdbscan_ordered_limit"):
        report.setdefault(name, {})["repeat_adjusted_rand"] = float(adjusted_rand_score(
            np.array(read(name)["cluster_id"]), np.array(read(name + "_r2")["cluster_id"])))
    for name in ("44_input_order", "48_input_order_limit"):
        ids = np.array(read(name)["id"])
        report[name] = {"rows": len(ids), "descending_adjacent_ids": int(np.sum(np.diff(ids) < 0)), "first_ten_ids": ids[:10].tolist()}
    write("geometry_correctness", report)


def supervised():
    source = read("21_cuml_logistic_regression")
    actual = np.array(source["actual_risk"])
    predictions = {"majority": np.zeros(len(actual)), "persistence": np.array(source["persistence_prediction"]) >= .30}
    for name in ("21_cuml_logistic_regression", "22_cuml_random_forest_classifier", "24_cuml_forest_predict"):
        predictions[name] = np.array(read(name)["prediction"]) >= .5
    report = {}
    for name, predicted in predictions.items():
        report[name] = {"accuracy": float(accuracy_score(actual, predicted)),
                        "precision": float(precision_score(actual, predicted, zero_division=0)),
                        "recall": float(recall_score(actual, predicted)), "f1": float(f1_score(actual, predicted)),
                        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                        "confusion_matrix": confusion_matrix(actual, predicted).tolist()}
    forest = read("24_cuml_forest_predict")
    scores = np.array(forest["prediction"])
    cpu = read("xgboost_cpu_predictions")
    cpu_order = np.argsort(np.array(cpu["id"]))
    error = float(np.max(np.abs(scores - np.array(cpu["prediction"])[cpu_order])))
    assert error < 1e-6
    report["forest_probability"] = {"cpu_max_abs_error": error, "roc_auc": float(roc_auc_score(actual, scores)),
                                    "average_precision": float(average_precision_score(actual, scores)),
                                    "precision_top_100": float(actual[np.argsort(-scores)[:100]].mean()),
                                    "persistence_precision_top_100": float(actual[np.argsort(-np.array(source["persistence_prediction"]))[:100]].mean())}
    target = np.array(source["future_low_rate"])
    for name, prediction in (("persistence_regression", np.array(source["persistence_prediction"])),
                             ("23_cuml_random_forest_regressor", np.array(read("23_cuml_random_forest_regressor")["prediction"]))):
        report[name] = {"mae": float(mean_absolute_error(target, prediction)), "r2": float(r2_score(target, prediction))}
    write("supervised_metrics", report)


def observations():
    report = []
    for path in sorted((REPORT / "sql").glob("*.sql")):
        records = [json.loads(p.read_text()) for suffix in ("", "_r2", "_r3")
                   if (p := REPORT / "results" / f"{path.stem}{suffix}.json").exists()]
        if not records:
            continue
        item = {"name": path.stem, "status": records[0]["status"],
                "elapsed_s": [r["elapsed_s"] for r in records], "rows": records[0].get("rows")}
        item["median_elapsed_s"] = float(np.median(item["elapsed_s"]))
        stats = REPORT / "results" / f"{path.stem}.stats.txt.gz"
        if stats.exists():
            lines = gzip.open(stats, "rt").read().splitlines()
            item["attempt"] = dict(line[2:].split("=", 1) for line in lines if line.startswith("# attempt_"))
            if "[execution]" in lines:
                i = lines.index("[execution]")
                values = dict(zip(lines[i+1].split("\t"), lines[i+2].split("\t")))
                item["execution"] = {k: values[k] for k in (
                    "post_native_execution_path", "host_to_device_uploaded_bytes", "device_to_host_materialized_bytes",
                    "allocator_high_water_bytes", "gpu_resident_input_handoffs", "reason_code", "unsupported_category",
                    "device_ordinal", "query_grant_bytes", "native_runtime_started", "output_rows") if k in values}
        report.append(item)
    write("experiment_index", report)


def interop():
    table = read("24_cuml_forest_predict")
    result = {"pyarrow_version": pa.__version__, "title_type": str(table.schema.field("title").type),
              "operation": "pq.read_table('24_cuml_forest_predict.parquet').sort_by([('id', 'ascending')])"}
    try:
        table.sort_by([("id", "ascending")])
        result["status"] = "ok"
    except pa.ArrowNotImplementedError as error:
        result.update(status="error", error=str(error))
    write("arrow_interop", result)


def export():
    for path in sorted((REPORT / "sql").glob("*.sql")):
        source = DATA / f"{path.stem}.parquet"
        if not source.exists() or int(path.stem[:2]) < 10:
            continue
        table = pq.read_table(source)
        with gzip.open(REPORT / "results" / f"{path.stem}.csv.gz", "wt", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=table.column_names)
            writer.writeheader()
            writer.writerows(table.to_pylist())
    points = read("49_umap_ordered_limit").to_pylist()
    pca = {r["id"]: r for r in read("17_cuvs_pca").to_pylist()}
    km = {r["id"]: r["cluster_id"] for r in read("16_cuvs_kmeans").to_pylist()}
    hd = {r["id"]: r["cluster_id"] for r in read("20_cuml_hdbscan").to_pylist()}
    for point in points:
        point.update(pc_0=pca[point["id"]]["pc_0"], pc_1=pca[point["id"]]["pc_1"],
                     kmeans=km[point["id"]], hdbscan=hd[point["id"]])
    payload = json.dumps(points, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Amazon Electronics | Headphone review issue map</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0d1420;color:#e5edf8;font:15px system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:28px}h1{font-size:28px;margin:6px 0}p{color:#acbdd2;line-height:1.7}label{display:inline-flex;gap:10px;align-items:center;margin:6px 20px 10px 0}select,input{padding:10px;background:#182338;color:#fff;border:1px solid #41506a;border-radius:6px}input{min-width:280px}.grid{display:grid;grid-template-columns:minmax(400px,2fr) minmax(300px,1fr);gap:24px}.panel{background:#142033;border:1px solid #2b3d56;border-radius:12px;padding:18px}canvas{width:100%;height:620px;cursor:crosshair}h2{font-size:20px}#title{line-height:1.5}#bars>div{display:grid;grid-template-columns:95px 1fr 48px;gap:8px;align-items:center;margin:11px 0}.track{background:#26364e;height:13px;border-radius:5px}.bar{background:#55bcd0;height:13px;border-radius:5px}button{color:#e5edf8;background:#23344d;border:0;border-radius:6px;padding:9px;cursor:pointer;text-align:left;width:100%;margin:4px 0}.muted{font-size:13px;color:#a8bbd3}#counts{margin:6px 0 14px}a{color:#83d7e5}@media(max-width:850px){.grid{grid-template-columns:1fr}canvas{height:440px}}
</style>
<main><div class="muted">ALGEON RESEARCH · AMAZON REVIEWS 2023 · ELECTRONICS</div>
<h1>Review issue map for 935 headphone products</h1>
<p>Reviews from 2022; each product has at least 20 one- or two-star reviews with text. Coordinates come from SQL UDTFs, and point size represents the negative review count. Select a product to see eight keyword rates.</p>
<label>Coordinates<select id="layout"><option value="dim">UMAP 2D</option><option value="pc">PCA 2D</option></select></label>
<label>Color<select id="color"><option value="kmeans">KMeans: 6 clusters</option><option value="hdbscan">HDBSCAN; gray indicates noise</option></select></label>
<input id="search" aria-label="Search products or brands" placeholder="Search products, brands, or product IDs"><div id="counts"></div>
<div class="grid"><div class="panel"><canvas id="map" aria-label="Headphone product scatter plot"></canvas>
<p class="muted">UMAP axes have no direct business units. The two PCA axes retain 58.9% of centered variance. Cluster IDs apply only to this query; noise does not indicate a defective product.</p></div>
<aside class="panel"><h2 id="title">Select a product</h2><p id="meta"></p><div id="bars"></div><h2>Matching products with the most negative reviews</h2><div id="list"></div></aside></div>
<p class="muted">These rates measure keyword matches. Negation, comparisons, and multilingual text are not handled; a match does not establish a defect. The data contains reviews but no sales or return records. UMAP uses the fixed input ordering snapshot from diagnostic query 49; FINDINGS.md AR-001 documents the ordering issue.</p>
<p><a href="README.md">Research report</a> · <a href="FINDINGS.md">Findings and reproductions</a></p></main>
<script>
const data=__POINTS__, issues=['battery','connection','sound','comfort','durability','charging','microphone','returns'];
const palette=['#ffb86b','#59c9c0','#8b99ff','#f08abd','#b0d77a','#e6d678'];
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');let visible=[],projected=[],selected=null;
const el=id=>document.getElementById(id);
function show(p){selected=p.id;el('title').textContent=p.title;el('meta').textContent=`Product ${p.id} · ${p.brand||'Brand unavailable'} · ${p.negative_reviews} negative reviews · KMeans ${p.kmeans} · HDBSCAN ${p.hdbscan}`;el('bars').replaceChildren();
issues.map(k=>[k,p[k]]).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{const row=document.createElement('div'),name=document.createElement('span'),track=document.createElement('div'),bar=document.createElement('div'),n=document.createElement('span');name.textContent=k;track.className='track';bar.className='bar';bar.style.width=`${v*100}%`;n.textContent=`${(v*100).toFixed(1)}%`;track.append(bar);row.append(name,track,n);el('bars').append(row)});draw()}
function draw(){const box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=box.width*dpr;canvas.height=box.height*dpr;ctx.scale(dpr,dpr);const w=box.width,h=box.height,q=el('search').value.toLowerCase(),mode=el('layout').value,color=el('color').value;
visible=data.filter(p=>`${p.id} ${p.title} ${p.brand}`.toLowerCase().includes(q));el('counts').textContent=`Showing ${visible.length} / ${data.length} products`;
const xs=data.map(p=>p[mode+'_0']),ys=data.map(p=>p[mode+'_1']),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);projected=[];
ctx.strokeStyle='#26374d';ctx.lineWidth=1;for(let i=1;i<5;i++){ctx.beginPath();ctx.moveTo(w*i/5,20);ctx.lineTo(w*i/5,h-20);ctx.stroke();ctx.beginPath();ctx.moveTo(20,h*i/5);ctx.lineTo(w-20,h*i/5);ctx.stroke()}
for(const p of visible){const x=30+(p[mode+'_0']-xmin)/(xmax-xmin)*(w-60),y=h-30-(p[mode+'_1']-ymin)/(ymax-ymin)*(h-60),r=2+Math.log1p(p.negative_reviews)/2;projected.push({p,x,y,r});ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=p[color]<0?'#718297':palette[p[color]%palette.length];ctx.globalAlpha=p.id===selected?1:.68;ctx.fill();if(p.id===selected){ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke()}}ctx.globalAlpha=1;
el('list').replaceChildren();visible.slice().sort((a,b)=>b.negative_reviews-a.negative_reviews).slice(0,8).forEach(p=>{const b=document.createElement('button');b.textContent=`${p.negative_reviews} reviews · ${p.title.slice(0,82)}`;b.onclick=()=>show(p);el('list').append(b)})}
canvas.onclick=e=>{const b=canvas.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top;const nearest=projected.map(v=>({...v,d:Math.hypot(v.x-x,v.y-y)})).sort((a,b)=>a.d-b.d)[0];if(nearest&&nearest.d<25)show(nearest.p)};
['search','layout','color'].forEach(id=>el(id).addEventListener('input',draw));window.addEventListener('resize',draw);draw();show(data.reduce((a,b)=>a.negative_reviews>b.negative_reviews?a:b));
</script></html>"""
    (REPORT / "complaint_map.html").write_text(template.replace("__POINTS__", payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
        return
    for function in (features, neighbors, geometry, supervised, observations, interop, export):
        function()
        print(function.__name__, "complete", flush=True)


if __name__ == "__main__":
    main()
