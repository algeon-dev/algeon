use std::{sync::Arc, time::Duration};

static SERIAL: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();

fn semaphore() -> Arc<tokio::sync::Semaphore> {
    Arc::clone(SERIAL.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(1))))
}

pub(crate) fn lock() -> tokio::sync::OwnedSemaphorePermit {
    loop {
        match semaphore().try_acquire_owned() {
            Ok(permit) => return permit,
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
}
