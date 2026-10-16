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
engines other than DataFusion are not planned. The
[overview's scope section](/#scope) lists what else is out of scope by
choice.

## Near term

- Admission grants sized to the plan. Grants are fixed at admission today.
  A conservative cap leaves VRAM idle and a tight one fails the query. The
  planned model sizes grants closer to what a plan needs and recovers from a
  failed allocation inside the query, for more queries per second and more
  concurrent requests on one device with fewer out-of-memory failures.
- Packaged releases that deploy without a native build.
- Public API documentation. The first goal was a working engine; rustdoc for
  the public surface follows as the APIs settle.
- Remove test coverage for retired contracts to reduce test execution times.
- ClickBench alongside the TPC-H and TPC-DS results.
- More cuVS and cuML algorithms, driven by user requests.

## Mid term

- Lakehouse sources, Iceberg first, with the Glue and REST catalogs.
  `iceberg-rust` and `datafusion-iceberg` are maturing upstream, and Algeon
  will integrate them once a release can be pinned.
- Build against NVIDIA's own repositories instead of the current forks, so
  that users do not compile the shared libraries themselves. Each fork change
  either goes upstream as a pull request, stays as an Algeon-specific feature
  or is dropped, with an issue per change to record the decision.
- Governance controls in the Flight SQL server: authentication and
  authorization that follow the access policies already in place around the
  catalog, and audit logging of what each query read. The shape of both
  follows the first adopter's platform.
- End-to-end comparisons with GPU workloads in production use. Which systems
  to compare against, and how, is still open.
- CPU backends behind the same table functions, for inputs small enough
  that the GPU has no advantage.
  [NetworKit](https://github.com/networkit/networkit),
  [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS)
  and [FAISS](https://github.com/facebookresearch/faiss) are the candidates.
- Potential cuOpt support, with scope guided by adopter requirements.

## Longer term

- Spilling to disk through [cuCascade](https://github.com/NVIDIA/cuCascade),
  so a working set larger than the memory grant does not fail.
- Development and testing on hardware with
  [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/). Current
  runs go through KvikIO compatibility mode. The object-storage counterpart,
  [cuObject](https://docs.nvidia.com/gpudirect-storage/cuobject/), follows
  once KvikIO's remote reads support it and an RDMA-capable object store is
  available for testing.
- Multi-GPU execution. Development runs on one GPU today; this
  waits for sponsored hardware.
- File formats other than Parquet on the GPU scan path. Non-Parquet
  sources run on the DataFusion CPU path today.
