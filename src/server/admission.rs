use std::time::Instant;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::error::{Result, ServerError};

#[derive(Debug)]
pub struct QueryAdmission {
    permits: std::sync::Arc<Semaphore>,
}

impl QueryAdmission {
    pub fn new(max_in_flight: usize) -> Result<Self> {
        if max_in_flight == 0 {
            return Err(ServerError::invalid_config("max_in_flight", "must be greater than zero"));
        }
        Ok(Self { permits: std::sync::Arc::new(Semaphore::new(max_in_flight)) })
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
