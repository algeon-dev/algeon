# Native FFI safety contract

## Handles and ownership

`cuml_handle_t` is created and destroyed by this shim. It borrows the CUDA stream, cuDF memory-resource owner, and the resource pointer supplied at creation. All three owners must outlive the handle. The handle owns its RAFT wrapper, but not the stream or memory resource. Destruction accepts null and must not race an operation using the same handle.

## Execution and memory provenance

The caller must enter the handle's CUDA device before every operation and pass the exact stream and current cuDF-owned RMM resource used to create it. The shim verifies that the supplied resource belongs to the cuDF owner and is the active RMM resource. Device pointers passed to algorithm functions must belong to that device and remain valid through stream completion.

## Completion and callbacks

Native operations may enqueue work on the supplied stream. Callers must retain inputs and output buffers until the operation's documented completion point. The thread-local error detail is valid only until the next shim call on that thread, and must be copied immediately. No Rust callback may unwind across the C ABI.

## Destruction and threads

A handle is not safe for concurrent access or use from another CUDA device. Its owner must serialize operations and destruction. Cleanup functions catch native exceptions and return a status rather than unwinding across the C ABI.

## Random forest training streams

The ABI exposes a native RF stream pool. When the runtime stream has a query-domain reservation, Rust attaches a sibling reservation to every pool stream using the same allocation limit. Without a query domain, Rust routes the manager's backend resource to every pool stream. Keep the reservation and sibling leases or standalone routes alive through fit. Native fit waits for and synchronizes the pool on both success and exception paths. Rust clears leases or routes before destroying the pool, because the pool owns those CUDA streams. RF parameters use `n_streams=1`; that setting does not remove the pool or its allocation route.
