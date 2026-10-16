# algeon-interop

`algeon-interop` (library `rapids_interop`) defines the buffer,
execution-binding and completion contracts the cuDF, cuGraph, cuVS and cuML
components use to share GPU buffers. A producer keeps its storage alive
until the consumer's completion obligation is met, and the exported types
state those obligations. Applications do not depend on it directly; they
use [`algeon-datafusion`](https://docs.rs/algeon-datafusion).
