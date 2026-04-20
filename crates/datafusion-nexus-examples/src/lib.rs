//! Shared types for the embedded backend example.
//!
//! This library holds the state passed between the example's HTTP API frontend,
//! Flight SQL frontend, pipeline registry, and shared GPU backend.

use std::{net::SocketAddr, sync::Arc};

use datafusion::prelude::SessionContext;
use datafusion_nexus::backend::NexusGpuBackend;

pub mod flight;
pub mod pipelines;
pub mod rest;

#[derive(Clone)]
pub struct AppState {
    pub ctx: Arc<SessionContext>,
    pub backend: NexusGpuBackend,
    pub pipelines: pipelines::PipelineRegistry,
}

impl AppState {
    pub fn new(ctx: Arc<SessionContext>, backend: NexusGpuBackend) -> Self {
        Self { ctx, backend, pipelines: pipelines::PipelineRegistry::default() }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BindAddrs {
    pub rest: SocketAddr,
    pub flight: Option<SocketAddr>,
}
