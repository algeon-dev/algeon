---
title: "Introducing Algeon"
description: Specialized GPU algorithms, composed through SQL.
slug: introducing-algeon
tags: [announcement]
authors: morris
---

Algeon brings specialized GPU algorithms into composable SQL workflows over
open data. It lets a query prepare relational inputs, run graph or vector work
on the GPU, and keep composing the result as rows instead of handing data to a
separate notebook or export pipeline.

Apache DataFusion is the first integration.

{/* truncate */}

## Why Algeon

GPU libraries already provide strong implementations of relational, graph, and
vector algorithms. Query engines already provide SQL, catalogs, projection and
predicate pushdown, and open table formats. The costly part is often the gap
between them: copying selected data into a one-off GPU job, materializing its
output, and wiring the result back into the original workflow.

Algeon treats specialized computation as a query operation with explicit input,
output, resource, error, and observation contracts. A graph algorithm can
consume a relation prepared by SQL and return a relation that the same query can
join, filter, aggregate, or return to the caller.

## Algeon for DataFusion

The `algeon-datafusion` adapter works from DataFusion physical plans. Supported
relational candidates lower to a native cuDF engine. Candidates that are not
selected retain their executable DataFusion implementation, and callers can
inspect structured planning evidence before execution.

The same session exposes cuGraph and cuVS operations through SQL table
functions. Native attempts share one admission path, one per-device ledger, and
one allocator-enforced memory cap. Parquet and Iceberg sources can feed the
workflow without introducing an Algeon-specific storage format.

The project is experimental. Its supported operators, data types, source
contracts, and operational limits are documented rather than implied by the
presence of a GPU kernel. Start with the [integration overview](/integration)
and [GPU function catalog](/guides/gpu-function-discovery).

## Where it goes next

Algeon's planning, runtime, and execution crates do not depend on DataFusion.
That boundary is intended to support additional adapters after the first
integration has real users and a stable portable contract. DuckDB is the next
narrow direction under consideration; Spark remains a longer-term direction.
Neither is supported today.

The immediate work is simpler: release the DataFusion integration, make one
graph workflow useful end to end, and learn from how people operate it.
