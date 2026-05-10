use std::{sync::Arc, time::Duration};

static SERIAL: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();

fn semaphore() -> Arc<tokio::sync::Semaphore> {
    SERIAL.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(1))).clone()
}

pub(crate) fn lock() -> tokio::sync::OwnedSemaphorePermit {
    loop {
        match semaphore().try_acquire_owned() {
            Ok(permit) => return permit,
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
}

pub(crate) async fn lock_async() -> tokio::sync::OwnedSemaphorePermit {
    semaphore().acquire_owned().await.expect("GPU fixture semaphore remains open")
}
