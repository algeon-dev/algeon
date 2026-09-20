use std::sync::Arc;

static SERIAL: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();

fn semaphore() -> Arc<tokio::sync::Semaphore> {
    Arc::clone(SERIAL.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(1))))
}

pub(crate) async fn lock_async() -> tokio::sync::OwnedSemaphorePermit {
    semaphore().acquire_owned().await.expect("GPU fixture semaphore remains open")
}
