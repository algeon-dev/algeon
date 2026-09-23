use std::error::Error as StdError;

use cuml::{Error, ErrorCode, ErrorKind, ErrorStatus, FailurePhase};
use static_assertions::assert_impl_all;

assert_impl_all!(Error: Send, Sync, StdError);

#[test]
fn structured_error_projections_are_available_at_the_public_boundary() {
    let _: fn(&Error) -> ErrorCode = Error::code;
    let _: fn(&Error) -> ErrorKind = Error::kind;
    let _: fn(&Error) -> ErrorStatus = Error::status;
    let _: fn(&Error) -> FailurePhase = Error::phase;
    let _: fn(&Error) -> bool = Error::is_temporary;
    let _: fn(&Error) -> bool = Error::is_oom;
    let _: fn(&Error) -> Option<&(dyn StdError + 'static)> = Error::source;
}
