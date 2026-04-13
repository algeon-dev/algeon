#!/usr/bin/env python3
"""Pandas+NetworkX equivalents of the cuGraph website example workloads.

Each task mirrors one documented example end-to-end: read the same Parquet the
server reads, build the same (sub)graph, run the equivalent algorithm, and
produce the same final rows. Run one task per invocation (subprocess isolation:
`bench_nx.py <task>`); the caller applies wall-clock caps.
"""

import json
import resource
import sys
import time

import pandas as pd

DATA = "/home/ubuntu/code/datafusion-nexus/fixture/data/demo"
CIT = f"{DATA}/citation_network/v1/parquet"

AI_FOS = [
    "Deep learning", "Artificial neural network", "Convolutional neural network",
    "Recurrent neural network", "Natural language processing",
    "Reinforcement learning", "Image segmentation", "Feature extraction",
    "Object detection", "Speech recognition",
]

marks = []


def mark(label):
    marks.append((label, time.perf_counter()))


def load_ai_edges():
    mark("start")
    papers = pd.read_parquet(f"{CIT}/profiles_by_paper_id.parquet",
                             columns=["paper_id", "year", "primary_fos", "title"])
    edges = pd.read_parquet(f"{CIT}/edges_by_src.parquet", columns=["src", "dst"])
    mark("parquet_read")
    ai = papers[(papers.year >= 2010) & (papers.primary_fos.isin(AI_FOS))]
    ids = set(ai.paper_id.to_numpy())
    e = edges[edges.src.isin(ids) & edges.dst.isin(ids)]
    mark("subgraph_filter")
    return papers, e


def task_pagerank_full():
    import networkx as nx
    mark("start")
    papers = pd.read_parquet(f"{CIT}/profiles_by_paper_id.parquet",
                             columns=["paper_id", "title", "year", "n_citation"])
    edges = pd.read_parquet(f"{CIT}/edges_by_src.parquet", columns=["src", "dst"])
    mark("parquet_read")
    g = nx.from_pandas_edgelist(edges, "src", "dst", create_using=nx.DiGraph)
    mark("graph_build")
    pr = nx.pagerank(g, alpha=0.85)
    mark("pagerank")
    top = sorted(pr.items(), key=lambda kv: -kv[1])[:5]
    ranked = pd.DataFrame(top, columns=["paper_id", "pagerank"])
    out = ranked.merge(papers, on="paper_id")[["title", "year", "n_citation", "pagerank"]]
    mark("join_top5")
    return {"top5": out.title.tolist(), "top_value": top[0][1]}


def task_triangle_full():
    import networkx as nx
    mark("start")
    edges = pd.read_parquet(f"{CIT}/edges_by_src.parquet", columns=["src", "dst"])
    mark("parquet_read")
    g = nx.from_pandas_edgelist(edges, "src", "dst")  # undirected
    mark("graph_build")
    tri = nx.triangles(g)
    mark("triangles")
    total = sum(tri.values())
    peak = max(tri.values())
    mark("aggregate")
    return {"vertices": len(tri), "triangle_sum": total, "max": peak}


def task_betweenness_ai_k200():
    import networkx as nx
    papers, e = load_ai_edges()
    g = nx.from_pandas_edgelist(e, "src", "dst", create_using=nx.DiGraph)
    mark("graph_build")
    bc = nx.betweenness_centrality(g, k=200, normalized=False, seed=0)
    mark("betweenness_k200")
    top = sorted(bc.items(), key=lambda kv: -kv[1])[:6]
    ranked = pd.DataFrame(top, columns=["paper_id", "paths_through"])
    out = ranked.merge(papers, on="paper_id").title.tolist()
    mark("join_top6")
    return {"sampled_sources": 200, "graph_nodes": g.number_of_nodes(),
            "graph_edges": g.number_of_edges(), "top6": out}


def task_louvain_ai():
    import networkx as nx
    papers, e = load_ai_edges()
    g = nx.from_pandas_edgelist(e, "src", "dst")  # undirected, like cugraph louvain
    mark("graph_build")
    comms = nx.community.louvain_communities(g, seed=0)
    mark("louvain")
    sizes = sorted((len(c) for c in comms), reverse=True)
    return {"communities": len(comms), "largest": sizes[0], "ge100": sum(1 for s in sizes if s >= 100)}


def task_mst_citation_era():
    import networkx as nx
    mark("start")
    papers = pd.read_parquet(f"{CIT}/profiles_by_paper_id.parquet",
                             columns=["paper_id", "year", "primary_fos", "title"])
    edges = pd.read_parquet(f"{CIT}/edges_by_src.parquet", columns=["src", "dst"])
    mark("parquet_read")
    nodes = papers[(papers.year.between(1900, 2020)) & (papers.primary_fos.isin(AI_FOS))]
    years = dict(zip(nodes.paper_id, nodes.year))
    e = edges[edges.src.isin(years.keys()) & edges.dst.isin(years.keys())].copy()
    e["year_gap"] = (e.src.map(years) - e.dst.map(years)).abs().astype(float)
    mark("subgraph_filter")
    g = nx.Graph()
    g.add_weighted_edges_from(e[["src", "dst", "year_gap"]].itertuples(index=False, name=None))
    mark("graph_build")
    mst = nx.minimum_spanning_tree(g)  # spanning forest for disconnected input
    mark("mst")
    kept = pd.DataFrame(mst.edges(), columns=["a", "b"])
    kept["gap"] = (kept.a.map(years) - kept.b.map(years)).abs()
    top = kept.nlargest(5, "gap")
    titles = dict(zip(papers.paper_id, papers.title))
    mark("join_top5")
    return {"mst_edges": mst.number_of_edges(),
            "vertices": mst.number_of_nodes(),
            "max_gap": int(top.iloc[0]["gap"]),
            "top_pair": [str(titles[top.iloc[0]["a"]])[:38], str(titles[top.iloc[0]["b"]])[:38]]}


TASKS = {
    "pagerank_full": task_pagerank_full,
    "triangle_full": task_triangle_full,
    "betweenness_ai_k200": task_betweenness_ai_k200,
    "louvain_ai": task_louvain_ai,
    "mst_citation_era": task_mst_citation_era,
}


def main():
    task = sys.argv[1]
    t0 = time.perf_counter()
    result = TASKS[task]()
    elapsed = time.perf_counter() - t0
    stages = {}
    prev = None
    for label, t in marks:
        if prev is not None:
            stages[label] = round(t - prev, 3)
        prev = t
    print(json.dumps({
        "task": task,
        "elapsed_s": round(elapsed, 3),
        "stages_s": stages,
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        "result": result,
    }, default=str))


if __name__ == "__main__":
    main()
