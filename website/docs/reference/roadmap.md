---
sidebar_position: 9
title: Roadmap
description: What Algeon is working on next, what waits for an upstream release or an adopter, and what waits for hardware it does not have.
---

# Roadmap

Algeon is pre-release. The items below are the current order of work, not
dates. Public APIs will keep changing until the near-term list is done.
Near-term items depend on nothing outside the project. Mid-term items wait
for an upstream release, an accepted pull request or an adopter's
requirements. Longer-term items wait for hardware. Substrait input and adapters for
engines other than DataFusion are not planned; the
[overview's scope section](/#scope) lists what else is out of scope by
choice.

## Near term

- A better admission model. Grants are fixed at admission today, so a
  conservative cap leaves VRAM idle while a tight one fails the query. The
  goal is more queries per second and more concurrent requests on the same
  device, with fewer out-of-memory failures, by sizing grants closer to
  what a plan needs and recovering from a failed allocation instead of
  failing the query.
- Packaged releases that deploy without a native build.
- Public API documentation. The first goal was a working engine; rustdoc
  for the public surface follows, with the APIs still moving.
- A complete test harness, with tests that no longer protect a contract
  removed, to shorten the development loop.
- ClickBench alongside the TPC-H and TPC-DS results.
- More cuVS and cuML algorithms, driven by user requests.

## Mid term

- Lakehouse sources, Iceberg first, with the Glue and REST catalogs.
  `iceberg-rust` and `datafusion-iceberg` are maturing upstream, and Algeon
  will integrate them once a release can be pinned.
- Build against NVIDIA's own repositories instead of the current forks, so
  users do not compile the shared libraries themselves. The fork changes
  split into fixes to upstream as pull requests, Algeon-specific features,
  and changes to drop; each needs an issue to decide which.
- Governance controls in the Flight SQL server: authentication and
  authorization that follow the access policies already in place around the
  catalog, and audit logging of what each query read. The shape of both
  follows the first adopter's platform.
- End-to-end comparisons with GPU workloads in production use. Which
  systems to compare against, and how, is still open.
- Potential cuOpt support, with scope guided by adopter requirements.

## Longer term

- Spilling to disk through [cuCascade](https://github.com/NVIDIA/cuCascade),
  so a working set larger than the memory grant does not fail.
- Development and testing on hardware with real
  [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/); current
  runs go through KvikIO compatibility mode.
- Multi-GPU execution. Development runs on one consumer GPU today, so this
  waits for sponsored hardware.
- File formats other than Parquet on the GPU scan path. Non-Parquet
  sources run on the DataFusion CPU path today.
