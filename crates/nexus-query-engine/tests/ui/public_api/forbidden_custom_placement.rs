use nexus_query_engine::{
    error::Result,
    runtime::{DeviceRegistry, PlacementPolicy, PlacementRequest},
};

struct CallerPolicy;

impl PlacementPolicy for CallerPolicy {
    fn candidates(&self, _registry: &DeviceRegistry, _request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>> {
        Ok(vec![])
    }
}

fn main() {}
