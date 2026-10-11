use std::error::Error as StdError;

use cuml::Error;
use static_assertions::assert_impl_all;

assert_impl_all!(Error: Send, Sync, StdError);
