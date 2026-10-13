# cuDF component

`algeon-cudf` and `algeon-cudf-sys` are published implementation dependencies
of Algeon. Applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion); the APIs below serve
workspace component consumers. See the [API documentation
boundary](https://algeon.dev/design#api-documentation).

`components/cudf` wraps [`libcudf`](https://github.com/algeon-dev/cudf), the
RAPIDS C++ DataFrame library, for the native backend of Algeon. It is part of
the root Cargo workspace. This page describes how the binding is layered, where
each ownership and execution rule is enforced, and which costs the design
avoids. The normative rules for raw-handle code are in
[`algeon-cudf-sys/SAFETY.md`](https://github.com/algeon-dev/algeon/blob/main/components/cudf/algeon-cudf-sys/SAFETY.md).

## Layers

```
+--------------------------------------------------------------------+
| query engine, cuGraph / cuVS / cuML components                     |
+--------------------------------------------------------------------+
        |  safe Rust: DataFrame, Series, Runtime, Exec, prepared exports
        v
+--------------------------------------------------------------------+
| algeon-cudf  (import cudf)                                         |
| owned handles, retained provenance, memory domains, Error          |
+--------------------------------------------------------------------+
        |  unsafe extern "C" calls with an explicit device, stream and MR
        v
+--------------------------------------------------------------------+
| algeon-cudf-sys  (import cudf_sys)                                 |
| generated declarations, borrowed handle tokens, option builders    |
+--------------------------------------------------------------------+
        |  C ABI: status codes, opaque handles, size-tagged structs
        v
+--------------------------------------------------------------------+
| libcudf_rust  (C++ shim in algeon-cudf-sys/native)                 |
| exception boundary, device guard, handle provenance                |
+--------------------------------------------------------------------+
        |  C++ API
        v
+--------------------------------------------------------------------+
| libcudf and RMM  (pinned fork in components/cudf/cudf)             |
+--------------------------------------------------------------------+
```

The shim keeps libcudf objects behind opaque handles, turns C++ exceptions into
status codes, and switches to the requested CUDA device before it calls
libcudf. The sys crate declares that ABI to Rust; it has no owning types and no
`Drop` implementations. The safe crate decides who owns each handle, which
device, stream and allocator an operation uses, and how a failure is reported.
The query engine and the other GPU components depend on the safe crate only.

`algeon-interop` (import `rapids_interop`) sits beside these layers. It owns
the buffer, readiness and lease types that cuDF, cuGraph, cuVS and cuML share,
and `cudf::interop::contract` re-exports it for code that already depends on
`cudf`. Producers and consumers that do not use `cudf` depend on
`algeon-interop` directly.

## Workspace

- `components/cudf/algeon-cudf/` - the safe execution crate, packaged as `algeon-cudf` (import `cudf`).
- `components/cudf/algeon-cudf-sys/` - the raw FFI declarations and the native C shim around `libcudf` (import `cudf_sys`).
- `components/algeon-interop/` - shared RAPIDS interop contract types and traits (import `rapids_interop`).
- `components/cudf/cudf/` - the pinned **[forked cuDF](https://github.com/algeon-dev/cudf)** source submodule used to build `libcudf`.

## Fork changes

The pinned fork is upstream `v26.08.00` plus the changes below on branch
`algeon-26.08`. Upstream `v26.10.00` supersedes the memory-resource parameters
on the device-view factories, row preprocessing, `sizes_to_offsets` and gather
temporaries, and the pinned-pool stream change; every other change is still
absent upstream and is re-applied on the 26.10 HashCSR hash join, per-thread
stream pool and reworked Parquet pruning code. Five preflight commits that the
fork later reverted are not carried.

Features:

- Retained join state with exact output sizing and delayed materialization:
  `hash_join::pre_build_reservation_size`, `filtered_join` pre-build sizing
  with `begin_left_semi_probe` and `begin_left_anti_probe`,
  `conditional_full_join_size`, `mixed_left_semi_join_size`,
  `mixed_left_anti_join_size`, `mixed_full_join_size`, and
  `mixed_left_join_state` for a chunked right side. Admission reserves the
  exact bytes before a build and materializes at most the admitted size.
- Explicit memory resource through join temporaries, gather, lexicographic
  sort scratch and row-operator preprocessing, so scratch allocations are
  charged to the query reservation instead of the ambient resource.
- Prepared regex backreference replacement,
  `replace_with_backrefs_output_size` and `replace_with_backrefs_prepared`:
  size first, then execute from retained state.
- Stream routing: `cudf::detail::install_stream_provider` lets libcudf's
  internal `fork_streams` use attempt-owned streams, and the device-buffer
  datasource stops using a global pool stream.
- Worker future drain guard (`drain_futures`, `future_drain_guard`) in the
  Parquet, CSV and compression paths, so a failing worker never leaves
  futures unjoined.
- Coalesced remote Parquet byte ranges across gaps up to
  `LIBCUDF_PARQUET_REMOTE_COALESCE_GAP_BYTES` (64 KiB by default), which cuts
  the number of remote reads.

Bug fixes:

- Parquet predicate pushdown: a bare boolean column or literal used as the
  reader filter no longer crashes, the stats survivor count is reported only
  when the stats stage ran, the decimal literal type check honors scale, and
  decimal fixed-length dictionary values are decoded big-endian.
- cuco exception safety, applied as a build-time patch: temporary storage is
  neither leaked nor double-freed when an allocation throws.
- A GCC 14.2 internal compiler error in `make_datasources` is avoided by
  naming the lambda parameter type.

Build:

- `CUDF_RMM_BUILD_SHARED` builds and installs RMM as a shared library and
  installs nvtx3 and rapids_logger, so cuDF, cuGraph, cuVS and cuML bind one
  `librmm.so` in a process.

## One call across the boundary

A `DataFrame`, `Series` or `GroupBy` method that submits GPU work, allocates,
or copies to the host runs through `.on(exec)`. `Exec` names the stream and
memory resource for the outputs, and `Exec::RETAINED` keeps the handles of the
primary input. Metadata reads and `Expr` construction take no execution scope.
A sort and a Parquet read follow the same steps.

```
 cudf (Rust)                                libcudf_rust (C++)
 ---------------------------------------    ----------------------------------
 resolve the output domain
   input stream + MR, or the Exec
   override; another device is an error
 DeviceEntryGuard::enter(device)
   cudaSetDevice only when the thread
   is on a different device
 fill cudf_exec_options_t
   struct_size, device, flags,
   stream, mr, reserved = {0}
 cudf_sort_ex(view, ..., &opts, &out) -->   cudf_clear_error()
                                            validate struct_size, flags,
                                              reserved slots, stream, mr
                                            cudf_device_guard.activate(device)
                                            try {
                                              cudf::sort(view, stream, mr)
                                              out = handle + captured
                                                    provenance
                                              return CUDF_SUCCESS
                                            } catch (...) {
                                              status = translate_exception()
                                              write the thread-local detail
                                              return status
                                            }
 on error                             <--
   copy cudf_last_error_detail()
   build Error { code, operation,
                 facts, ffi_detail }
 on success
   wrap out in a Column that holds the
   output's Arc<ExecutionDomain>
 guard drop restores the prior device
```

`cudf_exec_options_t` carries its own `struct_size` and four reserved slots. A
field can be added to the struct without changing a function signature, and the
shim rejects flag bits outside the defined set and reserved slots that are not
zero. The production entry points, the `_ex` functions, refuse a null options
pointer, a missing device flag, a null stream or a null memory resource.
Permissive variants exist for narrow diagnostics and regression hooks.

The shim clears the thread's error state when a call starts. When the call
fails, it writes a fixed, copyable record: the status, the failure source of an
allocation, the query domain and member identity, the requested bytes and
alignment, the CUDA error code, and a message of at most
`CUDF_ERROR_MESSAGE_CAPACITY` bytes. The record stays valid until the next shim
call on the same thread. Rust copies it into `CudfErrorDetail` before it makes
another shim call.

## Ownership and lifetime

Each kind of libcudf object has its own opaque handle type in the C headers.
`cpp_header_smoke.cpp` asserts at build time that column, table and scalar
handles are distinct types, which keeps a column from being passed where a
table is expected. A handle returned through an owning output belongs to the
caller only after the call reports `CUDF_SUCCESS`. A shim function that fails
part way releases what it had built before it returns.

```
  create, status == CUDF_SUCCESS             borrow
 -------------------------------> OWNED -------------> ColumnView, TableView
                                    |                  (lifetime-bound; dropping
                                    | Drop              a view frees no memory)
                                    v
                      enter the owner device, cudf_*_destroy(handle)
                                    |
                       +------------+------------+
                       |                         |
                 CUDF_SUCCESS                 any error
                       |                         |
                       v                         v
                   released               quarantined
                                          the handle is not touched again;
                                          the device health registry records
                                          the failure, and FATAL_CUDA marks
                                          the device unhealthy for new work
```

Destruction runs under the provenance captured at creation. Because the shim
enters the owning device before it frees, a `Column` can drop on any thread,
and after its `Runtime` is gone. A destroy call consumes ownership even when it
fails: the safe crate records a quarantined teardown in the device health
registry and does not retry. A fatal CUDA status marks the device unhealthy for
the rest of the process, and new work bound to it is rejected. Teardown may
still try to enter the device, but native destruction is suppressed when that
entry cannot be proven.

Where native state is keyed by another handle, Rust field order carries the
invariant. `StreamReservationLease` declares its reservation before its stream,
because a recycled raw stream value could otherwise inherit the dying
reservation's accounting. `ExecutionDomain` drops its binding extensions before
its stream and memory resource, because cross-crate native owners may borrow
those handles during teardown.

Host memory that crosses the boundary is borrowed for the duration of the call.
Host-to-device copies are asynchronous on the stream, and
`cudf_host_source_copy_sync_guard` synchronizes that stream before the shim
returns and the Rust-owned Arrow buffer can go out of scope. A host-copy API
returns only after its output is readable. A device-resident operation may
return with GPU work pending; the owner's retained provenance carries the
ordering.

The main types carry the thread policy below; the contract tests assert the
owner and borrowed-token rows at compile time.

| Type | Send | Sync | Reason |
| --- | --- | --- | --- |
| `Column`, `Table`, `Scalar`, `DeviceBuffer` | yes | yes | immutable after construction; observers create their own view handles |
| `Series`, `DataFrame`, `Runtime` | yes | yes | reference-counted owners over the types above |
| `CudaStream`, `CudaEvent`, `DeviceMemoryResource` | yes | yes | CUDA handles and RMM resources accept concurrent use |
| `ColumnView`, `TableView` | no | no | hold a raw native view handle bound to the owner's lifetime |
| `DataFrameMixedLeftJoinState` | yes | no | mutation requires exclusive access |
| `DeviceEntryGuard` | no | no | the current CUDA device is host-thread state |
| `cudf_sys::BorrowedStream` | yes | no | moving is allowed; shared submission needs the owner's synchronization |
| `cudf_sys::BorrowedEvent`, `BorrowedMemoryResource` | yes | yes | carry no ownership or mutable Rust state |

## Execution provenance

Each owned GPU object carries retained provenance: the device it is pinned to,
the stream that last ordered work on it, and the memory resource for follow-on
allocations. Objects created in one `Runtime` share one bundle, and operations
among them need no synchronization. The bundle's `ExecutionBindingId` changes
whenever the stream or memory resource changes. Consumers compare that identity
without reading native state.

When a data operation such as `merge`, `concat` or a gather combines inputs
from two runtimes on the same device, the safe crate orders the streams with
CUDA events in both directions. The host thread does not block in either
direction.

```
  time -------------------------------------------------------------------->

  stream A    ==[producer of a]==*                             *==[a may be freed]==
  (input a)                       \ record e1         wait e2 /
                                   v                         ^
  stream B    ======================*==[op(a, b)]============*======================
  (output)                       wait e1                 record e2
```

Before the operation, the output stream waits for work already enqueued on each
input's stream. After the operation is enqueued, each foreign input's stream
waits for the operation's completion event. Dropping that input therefore
cannot release its buffers while the operation still reads them.

Structural operations that keep incoming columns unchanged, `with_column` and
`assign` among them, reject mixed provenance and require `copy_into_runtime`
first. Inputs on different devices fail with `RuntimeBindingMismatch`; the
binding does not move data between GPUs on its own.

`DeviceEntryGuard` is `!Send` and `!Sync` because the current CUDA device
belongs to the host thread. The guard enters the owner device; the shim's own
guard repeats the check and is a no-op when the device already matches. A
failed device transition poisons the thread's device authority, and later GPU
work on that thread fails with `FatalCuda`. The remedy in that error is to stop
using the thread for CUDA work.

## Memory domains

```
  device 0                                                   device 1
  +----------------------------------------------------+     +------------+
  | CUDA async pool                                    |     | same shape |
  |  reservation-aware adaptor                         |     +------------+
  |  (installed once as the current RMM resource)      |
  |                                                    |
  |  +--------------------------+                      |
  |  | query domain, cap = G1   |   stream s1 -> m1    |
  |  |  members m1, m2          |   stream s2 -> m2    |
  |  +--------------------------+                      |
  |  +--------------------------+                      |
  |  | query domain, cap = G2   |   stream s3 -> m3    |
  |  |  member m3               |                      |
  |  +--------------------------+                      |
  +----------------------------------------------------+
```

Three levels share one pool on each device. The device domain is created once,
when `DeviceMemoryManager::install` runs at service start or when the first
runtime on that device needs it, and its adaptor stays the current RMM resource
for the process lifetime. Operations do not swap it. A `QueryMemoryDomain` has
an immutable cap; each stream of a query attaches one `StreamReservationLease`,
and the final clone of that lease detaches the member.

The memory resource selected through `Exec` governs outputs only, as the `mr`
parameter of libcudf does. Temporaries that libcudf allocates through the
current resource reach the adaptor, which routes them by stream to the attached
member and charges them to the query cap. Budgeting therefore attaches a stream
reservation and does not rely on the output resource. An allocation above the
cap fails at its source with the failure source `RESERVATION_LIMIT`, a pool
failure reports `PHYSICAL`, and Rust keeps no second ledger.

Size facts are computed without allocating. The shim's `cudf_*_payload_size_ex`
entry points use checked arithmetic, and an overflow returns
`CUDF_ERROR_SIZE_OVERFLOW`. `ops::materialization::table_upper_bound_preflight`
bounds a materialized table from row counts and string payload evidence.
Admission in the query engine consumes these facts; see [Memory and query
lifetime](https://algeon.dev/design/admission#allocation-enforcement).

## Errors

The shim translates each C++ exception at the boundary:

| C++ exception | `cudf_status_t` |
| --- | --- |
| `attributed_out_of_memory`, `rmm::out_of_memory`, `rmm::bad_alloc`, `std::bad_alloc` | `OUT_OF_MEMORY`, with the failure source and request facts the thrower supplied |
| `cudf::rust::invariant_error` | `INVARIANT_VIOLATION`, a shim invariant failed |
| `cudf::fatal_cuda_error` | `FATAL_CUDA` |
| `cudf::cuda_error` | `CUDA` |
| `cudf::data_type_error` | `TYPE` |
| `cudf::logic_error` | `INVALID_OPERATION` |
| `std::out_of_range` | `INDEX` |
| `std::invalid_argument` | `INVALID_ARGUMENT` |
| `std::overflow_error` | `SIZE_OVERFLOW` |
| `std::ios_base::failure` | `IO` |
| other `std::runtime_error` | `IO` inside I/O entry points, `UNKNOWN` elsewhere |
| anything else | `UNKNOWN` |

On the Rust side, `Error` stores one `ErrorCode`; `ErrorKind` and `ErrorStatus`
are projections of it and cannot be set on their own. The error also records
the operation label, typed facts such as the type id or row counts, free-form
details with a `remedy`, the copied native detail, and the chain of cuDF
boundaries it passed. `classify_cudf_error` turns an allocation failure into
physical or reservation attribution for the engine. Ownership of errors across
crates is defined in [Error
ownership](https://algeon.dev/design/errors#ownership-by-layer).

## ABI discipline

The C headers under `algeon-cudf-sys/native/include/cudf_rust` are the source
of truth. `src/bindings.rs` is generated from them with the bindgen-cli version
pinned in `Cargo.toml`, is checked in, and is compared against a fresh run by
`scripts/dev/bindings.sh cudf --check` in the full gate. Consumer builds do not
run bindgen.

`native/src/abi_checks.cpp` holds the compile-time assertions behind that
contract. Each shared enum is 32 bits wide. Each `cudf_type_id_t` value equals
the `cudf::type_id` it mirrors, and a pin on `NUM_TYPE_IDS` fails the build
when a cuDF release adds a type id. Each size-tagged struct is standard layout
with the field offsets Rust expects, and exported function pointers match their
declared signatures. `c_header_smoke.c` compiles the umbrella header as C and
checks the declared function pointer types.

The shim links with a version script that exports `cudf_*` symbols only, and it
installs with `RUNPATH` set to `$ORIGIN`, so the dynamic loader resolves
libcudf and RMM from the same prefix. The production shim excludes the
`cudf_test_*` entry points; those live in a separate header behind the
`test-hooks` feature, and `native_abi_contract_cpu` inspects the production
shim built without hooks. `CUDF_RUST_CLANG_LINT=1` runs clang-format and
clang-tidy over the shim during the Cargo build.

## Costs the design avoids

No host lock surrounds GPU work. Device domains share no allocator state with
one another, and the only cross-device lock is the registry lookup that finds a
device's manager; it does not span GPU work.

`Series` holds an `Arc<Column>`. A clone, or a move between DataFrames, copies
nothing on the GPU. Length and null count are cached when the `Series` is built
and are read without a device call, and `Display` prints metadata without
materializing values.

Within one runtime, or with `Exec::RETAINED`, no fences are recorded. Both
device guards skip `cudaSetDevice` when the host thread already sits on the
target device.

A prepared device export describes the buffers where they are. The consumer
orders its reads after the producer's readiness event, and the export's
retained owner keeps the storage alive until the lease completes. Arrow
`RecordBatch` conversion goes through host memory on a separate path; the
binding does not promise zero-copy to Arrow.

The query cap is enforced inside the RMM adaptor, where allocations happen, and
each failure carries its own attribution. Preflight sizes are arithmetic over
metadata. Neither needs a trial allocation or a second accounting pass.

## Tests

The sys crate tests the ABI itself. `bindings_contract_cpu` checks the
checked-in bindgen output against the wrapper header and the pinned bindgen
version; `native_abi_contract_cpu` reads the production shim with `nm` and
`readelf` and checks its exported symbols and CMake contract;
`execution_context_contract_cpu` checks at source level that each production
entry point resolves its options through the strict resolver;
`current_resource_invariant_contract_cpu` checks that the shim contains no
current-resource swap and installs it only during device domain setup;
`thread_contract_tests_gpu` asserts the `Send`/`Sync` claims of the borrowed
tokens; `ffi_regressions_gpu` exercises owning handles through failure and
cleanup paths; and `ffi_multi_device_gpu_exclusive` needs more than one GPU.

The safe crate tests the owner contracts. `allocation_failure_sweep_gpu`,
`failure_callback_lifetime_contract_gpu` and the `device_memory_manager_*` and
`query_memory_domain_*` targets cover memory domains; `execution_contracts_gpu`
and `interop_contracts_gpu` cover provenance, teardown and leases;
`multi_gpu_contracts_gpu_exclusive` needs more than one GPU;
`semantic_contracts_cpu` and `semantic_contracts_gpu` hold the operation
tables.

The `test-hooks` feature builds the `cudf_test_*` ABI, which injects faults at
exact points: the next provenance destroy returns a chosen status, a strings
import fails after its offsets copy, an event synchronize blocks until
released. `cudf::testing::DeviceSafetyTestScope` injects `cudaGetDevice` and
`cudaSetDevice` results on the current thread with an isolated health registry.
Target suffixes `_cpu`, `_gpu` and `_gpu_exclusive` select the nextest lane;
see [Test
layout](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#test-layout).

## Build and validation

Native build defaults are repo-relative: `target/native/cudf-build` for the
cuDF CMake cache and `target/native/cudf-install` for the installed cuDF/RMM
prefix. From the monorepo root, use `scripts/build/build_libcudf.sh`; it drives
the pinned source and keeps the cache/install identity aligned. Override the
two locations only with matching `CUDF_CPP_BUILD_DIR` and `CUDF_INSTALL_DIR`
selectors. The root `Cargo.lock`, `.gitmodules`, and `scripts/check_all.sh`
remain canonical for every component.

```bash
cargo clippy -p cudf -p cudf-sys -p rapids-interop --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
cargo nt -E 'package(cudf) | package(cudf-sys) | package(rapids-interop)'
```

Use `scripts/dev/bindings.sh cudf [--check]` for the two checked-in binding
surfaces; consumer builds do not run bindgen.
