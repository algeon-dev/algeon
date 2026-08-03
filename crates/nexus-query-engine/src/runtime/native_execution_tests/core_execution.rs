use super::*;

include!("core_execution/support.rs");
include!("core_execution/memory.rs");
// P2 deliberately validates uncached source execution. Durable source-cache
// scenarios remain private P5 substrate and are not part of this target.
