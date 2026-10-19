# Decision models in the loop

Demo of one orchestration pattern over the existing Algeon server: a probe
query returns a bounded candidate set with its context, a decision model
answers a fixed question set for each candidate, the typed answers go back
into Algeon as a Parquet table, and a recompute query runs over the full data
with the corrected structure. Measured results are not part of the
repository: each case's `run.py --mode replay` reproduces them from a capture
file, and the capture files stay local (see "Layout").

## Loop contract

- Probe and recompute are separate Flight SQL queries. No GPU grant is held
  while the decision backend runs.
- The probe's row limit (`max_entities` in each case) is the only cost bound.
- Decisions are cached in a registry keyed by the SHA-256 of the entity state
  text and of the canonical question JSON. A second run of the same case makes
  zero backend calls; a new time window calls the backend only for entities
  the registry has not seen.
- The recompute joins the typed decision table (`CREATE EXTERNAL TABLE ...
  STORED AS PARQUET`) and records `host_datafusion_boundary_count` from the
  query stats together with the `EXPLAIN GPU` summary.
- Each case has a control that uses no model, and a `policy.json` with the
  accept, review and reject bands on the action `noul`.

## Layout

```
fixture/demo/decision_models/
  README.md
  scripts/client/      decision_client.py registry.py policy.py capture.py algeon.py case.py
  scripts/cases/       a_brand_resolution/ b_bridge_purification/ c_complaint_causes/
                       d_barbie_mechanisms/ e_pokemon_submarkets/ f_import_gate/
  scripts/tests/       unit tests (no server, no API)
  registry/            *.parquet registries and *.engine.parquet decision tables (local only)
  captures/            <case>[_<label>][.<backend>]_<mode>.json (local only)
  data/ember/          EMBER2024 challenge node tables written by f_import_gate/prepare_tables.py (local only)
  data/barbie/         bridge rows and exclusion tables written by d_barbie_mechanisms/run.py (local only)
```

The root `.gitignore` versions only `README.md`, `scripts/` and `*.sh` under a
demo directory. Registries and captures stay local: the MerRec-derived ones
carry CC BY-NC text and may be published only under `website/static/data`
(see `scripts/test-harness.md`, "Third-party data").

## Backends

| `--backend` | Environment variables | Notes |
| --- | --- | --- |
| `jev` (default) | `JEV_API_URL`, `JEV_API_KEY` | TypeSafe `/systemone`: `choice`, `noul` and `score` heads, each question evaluated independently, at most 12 questions per state. |
| `openai_compatible` | `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_MODEL` | Any chat-completions endpoint that honors `response_format` JSON schemas; used for the cross-model agreement check. |
| replay (`--mode replay`) | none | Serves recorded responses from the live capture; no network. |

The variables have no defaults. A missing variable stops the run with a
message that names it. No file under the repository may contain an endpoint
host, path or token; captures record only `backend` and `model_id`, and the
capture writer refuses content that contains a bearer token, a JWT-shaped
string or the configured endpoint host.

## Running a case

Start a server with the MerRec scale configuration (see
`fixture/demo/merrec/scripts/scale/README.md`) and register the source tables;
each `run.py` calls the idempotent `CREATE EXTERNAL TABLE IF NOT EXISTS`
statements for the tables it reads.

```bash
cd fixture/demo/decision_models/scripts
python3 cases/b_bridge_purification/run.py --mode live             # probe, decide, recompute, control
python3 cases/b_bridge_purification/run.py --mode replay           # same numbers, zero API calls
python3 cases/b_bridge_purification/run.py --mode live --refresh   # decide every entity again (once per run)
python3 -m unittest discover -s tests -t .
```

`--refresh` ignores registry rows that existed before the run; a state the
run has already decided is reused by its later batches. Each case directory
has a `compare_captures.py` that checks a replay capture against its live
capture.

Common flags: `--uri` (default `grpc://127.0.0.1:50064`), `--backend`,
`--model`, `--max-workers`, `--max-entities`, `--label` (capture suffix such
as a time window) and `--replay-from` (extra captures that serve replay
responses). Case-specific flags are documented in each `run.py`.

Run order: case B (`--label all`) produces
`registry/brand_ontology_all.engine.parquet`, which case A reads to exclude
non-brand entries before pairing; case A produces
`registry/brand_canonical.engine.parquet`, which case B applies when run with
`--with-canonical`. Cases C, D, E and F are independent; F needs `prepare_tables.py` once.

## Registry schema

| Column | Type | Meaning |
| --- | --- | --- |
| `entity_key` | Utf8 | Case-defined key (`brand_id`, `brand_id_a:brand_id_b`, `review_id`). |
| `content_hash` | Utf8 | SHA-256 of the state text. |
| `question_set_hash` | Utf8 | SHA-256 of the canonical question JSON. |
| `model_id` | Utf8 | Backend-reported model, for example `typesafe/jev-1.13-20260917`. |
| `decided_at` | Timestamp(us) | UTC. |
| `<q>`, `<q>_p` | Utf8, Float64 | Choice option and its probability. |
| `<q>` | Float64 | `noul` probability or `score` expected value. |
| `input_tokens`, `latency_ms` | Int64 | Per decision. |

The engine-facing `*.engine.parquet` tables carry `entity_key`, the case's
typed key columns, the answer columns and the policy band.
