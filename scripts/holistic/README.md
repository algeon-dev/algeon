# Holistic AD/Entra attack-surface view

GPU **ForceAtlas2** layout (cuGraph) of the AD + Entra BloodHound graph, rendered
in the browser with **sigma.js**. Built for the `bh_sf100` fixture produced by
`fixture/scripts/gen_ad_entra_bh.py` (~1.08M nodes / ~8.67M edges).

```bash
# from repo root — computes the layout on the GPU, then serves the viz
scripts/holistic/run.sh
# open http://localhost:8000/
```

`run.sh` subcommands: `layout` (GPU compute only), `serve [port]`, `shot [port]`
(headless-Chrome screenshot → `out/preview.png`), or no arg for layout-then-serve.

## What it does

```
bh_nodes.parquet ─┐
                  ├─►  cudf (GPU load + security selection)  ─►  cuGraph FA2  ─►  graph.json  ─►  sigma.js
bh_edges.parquet ─┘
```

1. **Holistic layout (the GPU headline).** cuGraph ForceAtlas2 over the *entire*
   graph — 1.06M positioned vertices / 8.66M edges in ~1 s on a Blackwell card.
   Cached to `out/positions_full.parquet`. (`HOLISTIC_FULL_LAYOUT=0` skips it.)
2. **Security render slice.** A browser cannot draw 8.67M edges, so we render a
   bounded, attack-path-shaped subgraph (default ~32k nodes / ~30k edges) and
   re-lay-it-out with a second, fast FA2 for readability.

## Why a slice, and how it's chosen (the security thinking)

The full graph is a hairball; the *interesting* part is the escalation structure
around the crown jewels. The default `crown_ego` strategy:

- **Seeds the crown jewels**, stratified by object kind so on-prem **and** cloud
  are both represented — every `ADDomain`/`AZTenant`, plus the highest-degree
  high-value `ADGroup`s, domain controllers (`ADComputer`), privileged
  `AZServicePrincipal`s, and admin `ADUser`/`AZUser`s. (A naive global
  top-degree seeding buries the whole Azure half under AD's ~15k admin users,
  each carrying thousands of ACE edges.)
- **Grows bounded attack neighborhoods**: for each crown jewel, the cheapest
  *incoming* escalation edges (who can take it over) and cheapest *outgoing*
  edges (what it controls), ranked by `weight` (traversal cost; lower = cheaper
  = more dangerous).
- **Always carries the hybrid bridge**: a sample of `ADSyncedTo` edges — the
  on-prem ↔ cloud identity-sync pivot, the highest-value lateral-movement path
  in a hybrid estate.
- **Budgets by slice-local degree**, protecting crown jewels and bridge
  endpoints, so the render stays connected instead of dissolving into dust.

## Properties kept (and dropped)

I picked these with an attack-path lens — enough to read the graph as a security
analyst, nothing that's pure telemetry noise.

**Nodes** — `primary_kind` (object type → node color), `source_kind`
(on-prem AD vs cloud Entra vs hybrid), `high_value` + `risk_tier` (crown-jewel
gradient → size/spotlight), computed total `degree` (hub-ness), the identity
label, and the raw `node_id` (SID / GUID, in the tooltip for forensic
traceability).
**Edges** — `edge_kind` (the *privilege type* — the whole attack story → edge
color), `weight` (escalation cost → edge thickness; cheaper = thicker),
`source_kind` (so the `hybrid` bridge stands out).
**Dropped** — `collected_at` (snapshot telemetry) and `traversable` (uniformly
`true` in this fixture, so it carries no signal here; in real BloodHound data it
would gate config-only edges and belong in the keep list).

### Visual encoding

- **Node color** = object kind (see in-app legend). Domains/tenants are red and
  oversized (the roots); other crown jewels get a size boost.
- **Edge color** = privilege class: magenta `ADSyncedTo` (hybrid pivot), red AD
  ACL takeover (`GenericAll`/`WriteDacl`/…), amber Azure RBAC grants, violet
  `AZRunsAs`, green membership, cyan sessions, muted grey containment.
- **Interactions** — hover for the full node detail; click to isolate a node's
  neighborhood; "Spotlight hybrid bridges"; "Crown jewels only"; AD/Azure source
  filter; search by name / SID / UPN.

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `HOLISTIC_EDGES` / `HOLISTIC_NODES` | `…/ad_entra/bh_sf100/parquet/bh_{edges,nodes}.parquet` | input Parquet |
| `HOLISTIC_OUT_DIR` | `scripts/holistic/out` | output dir |
| `HOLISTIC_FULL_LAYOUT` | `1` | also run full-graph FA2 + cache positions |
| `HOLISTIC_REUSE` | `0` | reuse cached `out/positions_full.parquet` |
| `HOLISTIC_RENDER_COORDS` | `refined` | `refined` (FA2 on the slice) or `holistic` (true full-graph coords) |
| `HOLISTIC_STRATEGY` | `crown_ego` | or `danger_edges` (top danger-ranked edges) |
| `HOLISTIC_CROWN_BUDGET` | `350` | crown seeds **per principal kind** |
| `HOLISTIC_EGO_K` | `22` | cheapest in/out edges kept per crown seed |
| `HOLISTIC_BRIDGE_BUDGET` | `4000` | `ADSyncedTo` edges sampled in |
| `HOLISTIC_NODE_BUDGET` / `HOLISTIC_EDGE_BUDGET` | `32000` / `90000` | hard render caps |
| `HOLISTIC_FA2_ITERS` / `HOLISTIC_FULL_ITERS` | `350` / `120` | FA2 iterations |

## Notes

- **GPU is a shared resource.** `run.sh` wraps the layout in
  `flock /tmp/cudf-gpu.lock`, per `AGENTS.md`. Run it the same way if you invoke
  `holistic_layout.py` directly.
- Requires the repo `.venv` (cudf + cugraph 26.04) and a CUDA GPU. `sigma.min.js`
  + `graphology.umd.min.js` are vendored under `vendor/` — the viz is fully
  offline once `out/graph.json` exists.
- Generated outputs (`out/`) are git-ignored; they regenerate in seconds.
