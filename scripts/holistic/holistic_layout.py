#!/usr/bin/env python3
"""GPU ForceAtlas2 layout + security-prioritized export for the AD/Entra BloodHound graph.

Reads the generic `bh_nodes` / `bh_edges` Parquet slice (produced by
`fixture/scripts/gen_ad_entra_bh.py`), lays the graph out on the GPU with
cuGraph ForceAtlas2, then emits a `graph.json` (graphology serialization) that
`index.html` renders with sigma.js.

Why a *slice* and not the whole thing
-------------------------------------
At `bh_sf100` the graph is ~1.08M nodes / ~8.67M edges. cuGraph FA2 lays that
out on the GPU in seconds, but no browser renders 8.67M edges. So we split the
two concerns:

* HOLISTIC LAYOUT (the GPU showcase): FA2 over the *entire* graph. Saved to
  `out/positions_full.parquet` (node_idx, x, y). Enabled by default; set
  `HOLISTIC_FULL_LAYOUT=0` to skip.
* SECURITY RENDER (what sigma.js draws): a bounded, attack-path-shaped subgraph.
  The default `crown_ego` strategy keeps the crown jewels (domains, tenants,
  DCs, admin principals, high-value groups, privileged service principals) plus,
  for each, the cheapest incoming escalation edges (who can take it over) and
  cheapest outgoing edges (what it controls) -- i.e. bounded attack
  neighborhoods -- plus a sample of the hybrid `ADSyncedTo` bridges that pivot
  on-prem <-> cloud. The slice is re-laid-out with a second, fast FA2 for
  readability (`HOLISTIC_RENDER_COORDS=refined`, the default), or pinned to its
  true full-graph coordinates (`=holistic`).

Security property choices (what we keep / drop)
-----------------------------------------------
Nodes keep: identity label, `primary_kind` (drives color), `source_kind`
(on-prem AD vs cloud Entra vs hybrid), `high_value` / `risk_tier` (crown-jewel
gradient), computed total degree (hub-ness), and the raw `node_id` (SID / GUID)
for forensic traceability in the tooltip. Edges keep: `edge_kind` (the privilege
type -- the whole attack story), `weight` (traversal/escalation cost; lower =
cheaper = more dangerous -> thicker edge), and `source_kind` (so the `hybrid`
bridge stands out). Dropped: `collected_at` (telemetry noise) and `traversable`
(uniformly true in this fixture, so it carries no signal -- in real BloodHound
data it would gate config-only edges).

Env:
  HOLISTIC_EDGES         path to bh_edges.parquet
  HOLISTIC_NODES         path to bh_nodes.parquet
  HOLISTIC_OUT_DIR       output dir (default: <this dir>/out)
  HOLISTIC_FULL_LAYOUT   1 (default) run full-graph FA2 + cache positions; 0 skip
  HOLISTIC_RENDER_COORDS refined (default) | holistic
  HOLISTIC_REUSE         1 reuse cached out/positions_full.parquet if present
  HOLISTIC_STRATEGY      crown_ego (default) | danger_edges
  HOLISTIC_CROWN_BUDGET  crown-jewel seeds PER principal kind; domains/tenants
                         are always taken in full (default 350)
  HOLISTIC_EGO_K         cheapest in/out edges kept per crown seed (default 22)
  HOLISTIC_BRIDGE_BUDGET ADSyncedTo hybrid edges sampled in (default 4000)
  HOLISTIC_NODE_BUDGET   hard cap on rendered nodes (default 32000)
  HOLISTIC_EDGE_BUDGET   hard cap on rendered edges (default 90000)
  HOLISTIC_FA2_ITERS     subgraph FA2 iterations (default 350)
  HOLISTIC_FULL_ITERS    full-graph FA2 iterations (default 120)
  HOLISTIC_SEED          RNG seed for sampling / jitter (default 1337)

Run (GPU is a shared resource -- always serialize):
  flock /tmp/cudf-gpu.lock scripts/holistic/run.sh
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import cudf
import cugraph
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = "/opt/dlami/nvme/graph/ad_entra/bh_sf100/parquet"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _envi(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


# --------------------------------------------------------------------------
# Security visual vocabulary.
# --------------------------------------------------------------------------
# Node fill by object kind. AD (on-prem) families lean cool/structural; Entra
# (cloud) families lean warm; the two domain/tenant "roots" are red crowns.
NODE_COLORS = {
    "ADUser": "#5b8def",
    "ADGroup": "#9b6dff",
    "ADComputer": "#3fb950",
    "ADDomain": "#f85149",
    "ADCertTemplate": "#d29922",
    "AZUser": "#2dd4bf",
    "AZGroup": "#c084fc",
    "AZApp": "#e3b341",
    "AZServicePrincipal": "#f778ba",
    "AZVM": "#56d364",
    "AZTenant": "#ff7b72",
    "AZSubscription": "#a371f7",
    "AZResourceGroup": "#79c0ff",
}
NODE_COLOR_DEFAULT = "#8b949e"

# AD ACE-derived privilege edges (full-control rights = cheap takeover).
AD_ACE_KINDS = {
    "ADGenericAll",
    "ADGenericWrite",
    "ADWriteDacl",
    "ADWriteOwner",
    "ADOwns",
    "ADAllExtendedRights",
    "ADAddMember",
}


def edge_color(kind: str) -> str:
    if kind == "ADSyncedTo":
        return "#ff2d95"  # hybrid bridge -- the on-prem<->cloud pivot, headline
    if kind in AD_ACE_KINDS:
        return "#f85149"  # ACL takeover -- dangerous
    if kind.startswith("AZHasRole_"):
        return "#ffa657"  # Azure RBAC grant
    if kind == "AZRunsAs":
        return "#bc8cff"  # app -> service principal
    if kind in ("ADMemberOf", "AZMemberOf"):
        return "#2ea043"  # group membership
    if kind == "ADHasSession":
        return "#39c5cf"  # live logon -> credential theft surface
    if kind in ("ADContains", "AZContains"):
        return "#3a4048"  # structural containment (muted)
    return "#6e7681"


def edge_tier(kind: str) -> int:
    """Lower = more security-interesting -> selected first."""
    if kind == "ADSyncedTo":
        return 0
    if kind in AD_ACE_KINDS:
        return 1
    if kind.startswith("AZHasRole_") or kind == "AZRunsAs":
        return 2
    if kind in ("ADMemberOf", "AZMemberOf", "ADHasSession"):
        return 3
    return 4  # containment / structure


def log(msg: str) -> None:
    print(f"[holistic] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# FA2 helpers.
# --------------------------------------------------------------------------
def run_fa2(edges: cudf.DataFrame, iters: int, ewi: float) -> cudf.DataFrame:
    """ForceAtlas2 over `edges` (cols src,dst,fa2_w). Returns vertex,x,y."""
    g = cugraph.Graph(directed=False)
    g.from_cudf_edgelist(
        edges, source="src", destination="dst", edge_attr="fa2_w", renumber=True
    )
    t0 = time.time()
    pos = cugraph.force_atlas2(
        g,
        max_iter=iters,
        edge_weight_influence=ewi,
        scaling_ratio=2.0,
        gravity=1.0,
        barnes_hut_optimize=True,
        outbound_attraction_distribution=True,
    )
    log(f"  FA2 {g.number_of_vertices():,} verts / {g.number_of_edges():,} edges, "
        f"{iters} iters in {time.time() - t0:.1f}s")
    return pos


def main() -> int:
    seed = _envi("HOLISTIC_SEED", 1337)
    rng = np.random.default_rng(seed)
    edges_path = _env("HOLISTIC_EDGES", f"{DEFAULT_DATA}/bh_edges.parquet")
    nodes_path = _env("HOLISTIC_NODES", f"{DEFAULT_DATA}/bh_nodes.parquet")
    out_dir = Path(_env("HOLISTIC_OUT_DIR", str(HERE / "out")))
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = _env("HOLISTIC_STRATEGY", "crown_ego")
    crown_budget = _envi("HOLISTIC_CROWN_BUDGET", 350)
    ego_k = _envi("HOLISTIC_EGO_K", 22)
    bridge_budget = _envi("HOLISTIC_BRIDGE_BUDGET", 4000)
    node_budget = _envi("HOLISTIC_NODE_BUDGET", 32000)
    edge_budget = _envi("HOLISTIC_EDGE_BUDGET", 90000)
    fa2_iters = _envi("HOLISTIC_FA2_ITERS", 350)
    full_iters = _envi("HOLISTIC_FULL_ITERS", 120)
    render_coords = _env("HOLISTIC_RENDER_COORDS", "refined")
    do_full = _env("HOLISTIC_FULL_LAYOUT", "1") == "1"
    reuse = _env("HOLISTIC_REUSE", "0") == "1"

    log(f"edges={edges_path}")
    log(f"nodes={nodes_path}")
    log(f"strategy={strategy} render_coords={render_coords} full_layout={do_full}")

    # ---- load (GPU) --------------------------------------------------------
    t0 = time.time()
    edges = cudf.read_parquet(
        edges_path, columns=["src", "dst", "edge_kind", "weight", "source_kind"]
    )
    nodes = cudf.read_parquet(
        nodes_path,
        columns=[
            "node_idx", "node_id", "primary_kind", "source_kind",
            "name", "displayname", "risk_tier", "high_value", "environment_id",
        ],
    )
    log(f"loaded {len(nodes):,} nodes / {len(edges):,} edges in {time.time() - t0:.1f}s")

    # FA2 attraction weight: cheap (low-cost) escalation edges pull endpoints
    # together, so attack-reachable sets cluster visually.
    edges["fa2_w"] = (1.0 / edges["weight"]).astype("float64")

    # total degree per node (hub-ness)
    deg = (
        cudf.concat([edges["src"], edges["dst"]])
        .value_counts()
        .rename("degree")
        .reset_index()
        .rename(columns={"index": "node_idx"})
    )
    deg.columns = ["node_idx", "degree"]
    nodes = nodes.merge(deg, on="node_idx", how="left")
    nodes["degree"] = nodes["degree"].fillna(0).astype("int64")

    # ---- optional: holistic full-graph FA2 (the GPU showcase) --------------
    full_path = out_dir / "positions_full.parquet"
    if do_full:
        if reuse and full_path.exists():
            log(f"reusing cached {full_path.name}")
        else:
            log("full-graph ForceAtlas2 (holistic layout)...")
            pos_full = run_fa2(edges[["src", "dst", "fa2_w"]], full_iters, 1.0)
            pos_full = pos_full.rename(columns={"vertex": "node_idx"})
            pos_full.to_parquet(full_path)
            log(f"wrote {full_path}")

    # ---- security selection ------------------------------------------------
    # Per-edge "danger" tier via a small lookup merge (cudf Series has no map).
    kinds = edges["edge_kind"].unique().to_arrow().to_pylist()
    tier_df = cudf.DataFrame(
        {"edge_kind": kinds, "tier": [edge_tier(k) for k in kinds]}
    )
    edges = edges.merge(tier_df, on="edge_kind", how="left")

    # Crown-jewel seeding is STRATIFIED by object kind, not global top-degree.
    # AD has ~15k admin users each carrying many ACE edges, so a pure
    # degree ranking buries every Azure crown jewel and the cloud half vanishes
    # from the picture. Instead: always take every domain/tenant (structural
    # roots, a handful each), then the highest-degree `crown_budget` seeds
    # *within* each principal kind so AD and Entra are both represented.
    crown_all = nodes[nodes["high_value"]]
    quota = {
        "ADDomain": 10_000, "AZTenant": 10_000,   # roots: take all
        "ADGroup": crown_budget, "ADComputer": crown_budget,
        "AZServicePrincipal": crown_budget, "AZUser": crown_budget,
        "ADUser": crown_budget,
    }
    parts = []
    for kind, q in quota.items():
        sub = crown_all[crown_all["primary_kind"] == kind]
        if len(sub):
            parts.append(sub.sort_values("degree", ascending=False).head(q))
    crown = cudf.concat(parts)
    crown_ids = crown["node_idx"]
    by_kind = crown["primary_kind"].value_counts().to_pandas().to_dict()
    log(f"crown jewels: {len(crown_all):,} total, {len(crown):,} seeded "
        f"({', '.join(f'{k}:{v}' for k, v in by_kind.items())})")

    if strategy == "danger_edges":
        sel = edges.sort_values(["tier", "weight"]).head(edge_budget)
    else:  # crown_ego
        crown_set = crown_ids
        inc = edges[edges["dst"].isin(crown_set)].sort_values(["dst", "weight"])
        inc["_rk"] = inc.groupby("dst").cumcount()
        inc = inc[inc["_rk"] < ego_k].drop(columns=["_rk"])
        out = edges[edges["src"].isin(crown_set)].sort_values(["src", "weight"])
        out["_rk"] = out.groupby("src").cumcount()
        out = out[out["_rk"] < ego_k].drop(columns=["_rk"])
        sel = cudf.concat([inc, out])
        # Always carry a sample of the hybrid bridge.
        bridges = edges[edges["edge_kind"] == "ADSyncedTo"]
        if len(bridges) > bridge_budget:
            take = rng.choice(len(bridges), size=bridge_budget, replace=False)
            bridges = bridges.iloc[np.sort(take)]
        sel = cudf.concat([sel, bridges]).drop_duplicates(subset=["src", "dst", "edge_kind"])

    if len(sel) > edge_budget:
        sel = sel.sort_values(["tier", "weight"]).head(edge_budget)

    # --- assemble the node set under a security-aware budget ----------------
    # Slice-LOCAL degree (connectivity *within* the render slice). Clamping by
    # this -- not global degree -- avoids keeping globally-busy nodes that have
    # one stray edge here and would render as disconnected dust.
    sdeg = cudf.concat([sel["src"], sel["dst"]]).value_counts().reset_index()
    sdeg.columns = ["node_idx", "sdeg"]
    # Connected endpoints plus the crown jewels (kept even if isolated, so a
    # high-value target with no cheap path in still appears).
    node_pool = cudf.concat([sdeg["node_idx"], crown_ids]).unique().to_frame("node_idx")
    cand = node_pool.merge(nodes, on="node_idx", how="inner").merge(
        sdeg, on="node_idx", how="left"
    )
    cand["sdeg"] = cand["sdeg"].fillna(0).astype("int64")
    # Protected = crown jewels + endpoints of the hybrid ADSyncedTo bridge.
    # These are the security headline and must survive the budget regardless of
    # local degree.
    br = sel[sel["edge_kind"] == "ADSyncedTo"]
    protected = cudf.concat([crown_ids, br["src"], br["dst"]]).unique()
    cand["protected"] = cand["node_idx"].isin(protected)
    if len(cand) > node_budget:
        keep_prot = cand[cand["protected"]]
        rest = cand[~cand["protected"]].sort_values("sdeg", ascending=False)
        rest = rest.head(max(0, node_budget - len(keep_prot)))
        cand = cudf.concat([keep_prot, rest])
    keep = cand["node_idx"]
    sel = sel[sel["src"].isin(keep) & sel["dst"].isin(keep)]
    sel = sel.drop_duplicates(subset=["src", "dst", "edge_kind"])
    sub_nodes = nodes.merge(keep.to_frame("node_idx"), on="node_idx", how="inner")
    log(f"selected subgraph: {len(sub_nodes):,} nodes / {len(sel):,} edges")

    # ---- render coordinates ------------------------------------------------
    if render_coords == "holistic":
        if not full_path.exists():
            log("holistic coords requested but no full layout cached; "
                "falling back to refined")
            render_coords = "refined"
        else:
            pos = cudf.read_parquet(full_path)
    if render_coords != "holistic":
        log("subgraph ForceAtlas2 (render layout)...")
        sub_e = sel[["src", "dst", "fa2_w"]]
        pos = run_fa2(sub_e, fa2_iters, 1.0).rename(columns={"vertex": "node_idx"})

    sub_nodes = sub_nodes.merge(pos, on="node_idx", how="left")

    # ---- to host + build graphology JSON -----------------------------------
    nd = sub_nodes.to_pandas()
    ed = sel.to_pandas()

    # jitter any node FA2 didn't place (isolated within the slice)
    nd["x"] = nd["x"].astype("float64")
    nd["y"] = nd["y"].astype("float64")
    miss = nd["x"].isna() | nd["y"].isna()
    if miss.any():
        span = float(np.nanmax(np.abs(nd[["x", "y"]].to_numpy())) or 1.0)
        nd.loc[miss, "x"] = rng.uniform(-span, span, int(miss.sum()))
        nd.loc[miss, "y"] = rng.uniform(-span, span, int(miss.sum()))
        log(f"jittered {int(miss.sum())} unplaced node(s)")

    deg_max = max(int(nd["degree"].max()), 1)

    def node_size(row) -> float:
        base = 2.2 + 6.0 * math.sqrt(row["degree"] / deg_max)
        if row["primary_kind"] in ("ADDomain", "AZTenant"):
            return base + 12.0
        if row["high_value"]:
            return base + 4.5
        return base

    json_nodes = []
    for row in nd.itertuples(index=False):
        d = row._asdict()
        json_nodes.append({
            "key": str(int(d["node_idx"])),
            "attributes": {
                "x": float(d["x"]),
                "y": float(d["y"]),
                "size": round(node_size(d), 2),
                "color": NODE_COLORS.get(d["primary_kind"], NODE_COLOR_DEFAULT),
                "label": str(d["displayname"] or d["name"]),
                "kind": d["primary_kind"],
                "source": d["source_kind"],
                "scope": d["environment_id"],
                "high_value": bool(d["high_value"]),
                "risk_tier": int(d["risk_tier"]),
                "degree": int(d["degree"]),
                "nid": d["node_id"],
            },
        })

    w_min = float(ed["weight"].min()) if len(ed) else 0.1

    def edge_size(kind: str, weight: float) -> float:
        if kind == "ADSyncedTo":
            return 2.6  # bridge stands out
        # cheaper escalation -> thicker
        return max(0.35, min(2.4, (w_min / max(weight, 1e-6)) * 1.4))

    json_edges = []
    for i, row in enumerate(ed.itertuples(index=False)):
        d = row._asdict()
        json_edges.append({
            "key": f"e{i}",
            "source": str(int(d["src"])),
            "target": str(int(d["dst"])),
            "attributes": {
                "color": edge_color(d["edge_kind"]),
                "size": round(edge_size(d["edge_kind"], float(d["weight"])), 2),
                "kind": d["edge_kind"],
                "weight": round(float(d["weight"]), 3),
                "source_kind": d["source_kind"],
            },
        })

    graph = {
        "attributes": {"name": "AD/Entra BloodHound — security holistic view"},
        "options": {"type": "directed", "multi": True, "allowSelfLoops": True},
        "nodes": json_nodes,
        "edges": json_edges,
    }
    out_json = out_dir / "graph.json"
    with open(out_json, "w") as f:
        json.dump(graph, f)
    log(f"wrote {out_json} ({out_json.stat().st_size / 1e6:.1f} MB)")

    # ---- legend / stats sidecar (drives the HTML legend + headline counts) -
    kind_counts = nd["primary_kind"].value_counts().to_dict()
    edge_kind_counts = ed["edge_kind"].value_counts().to_dict()
    src_counts = nd["source_kind"].value_counts().to_dict()
    stats = {
        "data": {"edges_path": edges_path, "nodes_path": nodes_path},
        "full_graph": {"nodes": int(len(nodes)), "edges": int(len(edges))},
        "rendered": {"nodes": int(len(nd)), "edges": int(len(ed))},
        "render_coords": render_coords,
        "strategy": strategy,
        "crown_jewels_rendered": int(nd["high_value"].sum()),
        "hybrid_bridges_rendered": int((ed["edge_kind"] == "ADSyncedTo").sum()),
        "node_kind_counts": {str(k): int(v) for k, v in kind_counts.items()},
        "edge_kind_counts": {str(k): int(v) for k, v in edge_kind_counts.items()},
        "source_kind_counts": {str(k): int(v) for k, v in src_counts.items()},
        "node_colors": NODE_COLORS,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log(f"wrote {out_dir / 'stats.json'}")
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
