use std::sync::{Arc, OnceLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static SERIAL: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn semaphore() -> Arc<Semaphore> {
    Arc::clone(SERIAL.get_or_init(|| Arc::new(Semaphore::new(1))))
}

pub(crate) async fn lock_async() -> OwnedSemaphorePermit {
    semaphore().acquire_owned().await.expect("GPU fixture semaphore remains open")
}
