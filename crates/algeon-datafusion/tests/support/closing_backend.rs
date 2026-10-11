use std::{ops::Deref, thread, time::Duration};

use algeon_datafusion::backend::AlgeonGpuBackend;

/// Keeps the process owner claim until every case-local session has dropped,
/// then waits for backend cleanup before another table case constructs a backend.
pub(crate) struct ClosingBackend(AlgeonGpuBackend);

impl ClosingBackend {
    pub(crate) fn new(backend: AlgeonGpuBackend) -> Self {
        Self(backend)
    }
}

impl Deref for ClosingBackend {
    type Target = AlgeonGpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for ClosingBackend {
    fn drop(&mut self) {
        let result = self.0.close_and_wait(Duration::from_secs(10));
        if let Err(error) = result
            && !thread::panicking()
        {
            panic!("table case backend failed to close: {error}");
        }
    }
}
