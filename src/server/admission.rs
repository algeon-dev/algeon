use std::time::Instant;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::error::{Result, ServerError};

#[derive(Debug)]
pub struct QueryAdmission {
    permits: std::sync::Arc<Semaphore>,
    max_in_flight: usize,
}

impl QueryAdmission {
    pub fn new(max_in_flight: usize) -> Result<Self> {
        if max_in_flight == 0 {
            return Err(ServerError::invalid_config("max_in_flight", "must be greater than zero"));
        }
        Ok(Self { permits: std::sync::Arc::new(Semaphore::new(max_in_flight)), max_in_flight })
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Close the admission queue. After close, any pending or future
    /// [`Self::acquire`] call returns a session error so graceful shutdown can
    /// drain in-flight work without admitting more.
    pub fn close(&self) {
        self.permits.close();
    }

    pub async fn acquire(&self) -> Result<QueryAdmissionPermit> {
        let queued_at = Instant::now();
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| ServerError::session("query admission queue closed", error))?;
        Ok(QueryAdmissionPermit { permit, queued_at })
    }
}

pub struct QueryAdmissionPermit {
    permit: OwnedSemaphorePermit,
    queued_at: Instant,
}

impl QueryAdmissionPermit {
    pub fn wait_millis(&self) -> u128 {
        self.queued_at.elapsed().as_millis()
    }

    pub fn into_inner(self) -> OwnedSemaphorePermit {
        self.permit
    }
}
