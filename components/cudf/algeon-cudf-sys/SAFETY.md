# Safety contract

`cudf_sys` exposes raw C ABI calls. The C headers and generated bindings
cannot prove CUDA lifetime, execution, or ownership rules; callers must do so.
Use the safe [`cudf`](../README.md) API unless a lower-level
integration needs this boundary.

## Handles and ownership

The ABI's nominal opaque handles are intentionally distinct. A handle returned
through a documented owning output becomes the caller's responsibility only on
success and must be released with its matching `cudf_*_destroy` operation.
Never destroy an unchecked output after a failed call, a borrowed view, or a
borrowed stream/event/memory-resource token. Do not cast handles between
domains or invent handles from native pointers.

Native functions clean up their own partially constructed state on failure.
Callers must still keep every owned input alive until the documented operation
and any submitted GPU work finish. Parent/child or retained-resource lifetime
rules in the relevant C header are part of the ABI contract.

## Execution and memory provenance

Execution-sensitive production `_ex` calls require an explicit device, stream,
and memory resource. Each must describe the same live CUDA execution domain;
do not substitute ambient current-device or default-resource state. A memory
resource must outlive every allocation and output created through it.

Runtime identity and allocation ownership are separate. The private RMM
identity boundary only verifies that RAPIDS consumers use one compatible RMM
runtime; it neither transfers allocation ownership nor permits arbitrary raw
RMM pointers as ABI inputs. The owner that created an allocation remains
responsible for its lifetime and device-aware release.

Query-memory-domain grants and member limits are immutable after creation.
`cudf_reservation_handle` owns one query-domain member attachment and must be
detached exactly once. Member resources retain their allocation owner until
all allocations are released. A detached routed member with pending allocation
submissions blocks stream reuse until those submissions finish. Raw sibling
stream attachments require the caller to retain the stream through detach and
asynchronous completion.

## Completion and callbacks

A host-copy/materialization API returns only after its host output is ready to
read. A device-resident or asynchronous operation may return while GPU work is
pending. Keep input/output device storage, views, execution context, stream,
memory resource, and callback state alive until that work is completed on the
correct stream; use the documented event or stream synchronization path.

Pinned host-copy sources require completion even when submission or event
recording returns an error. The staged Arrow importer retains each slab until
event completion, synchronizes the owning stream on early exit, and quarantines
the source and destination allocations with their execution domain when teardown
cannot prove completion. DeviceBuffer host
writes require exclusive Rust access.

Failure callbacks and their `callback_arg` must remain valid for the lifetime
specified by the creating API. They run on the allocating thread and must not
unwind across the C boundary, allocate through the same memory resource, or
re-enter a conflicting operation on the same resource. Return only the action
value documented by the callback's header.

`cudf_memory_manager_install` borrows its callback and argument for the
process lifetime after its installation record reports creation. An incomplete
current-resource switch still retains that binding. Identity-aware reuse must
supply the same callback function and argument; it cannot replace them. Read
the returned record before releasing callback state on an installation error.

## Destruction and threads

Destroy owning handles only once, after dependent asynchronous work completes,
and with the device/provenance required by their API. Do not use a copied raw
handle after destruction. Error paths must either transfer a documented owner
or leave no caller-owned native object to release.

Raw pointers and generated handle aliases carry no automatic safe concurrency
guarantee. `BorrowedStream` is `Send` but not `Sync`: moving it needs the
owner's synchronization and sharing it concurrently is not allowed.
`BorrowedEvent` and `BorrowedMemoryResource` are `Send + Sync` only because
they retain no ownership or mutable Rust state; their underlying CUDA/RMM
objects must still outlive all uses. Higher-level owner types define the rest
of the thread-safety policy.

## Cumulative mixed left membership

`cudf_mixed_left_join_state` retains the AST owner and borrowed logical-left
views; callers retain left device storage and AST literal scalar owners until
destruction. Right batches are borrowed only through completion of `consume`.
The safe DataFrame wrapper retains a frame clone and the complete AST, and
uses execution completion fences to order batch release across streams.
Mutation requires exclusive access; this state is `Send` but not `Sync`.

Construction, consumption, sealing, and materialization require the original
device, stream and MR. Native state buffers and preprocessing retain the
statistics resource reference, whose box outlives them and forwards allocation
to that MR. Temporary AST/preprocessing buffers release on the same stream.
Sealing is one way: native consumption and repeated sealing reject a sealed
state. The safe API consumes its mutable owner into a finalized owner; only
that owner exposes exact output rows and materialization. Gather-map outputs
retain the original execution domain independently of state destruction.
