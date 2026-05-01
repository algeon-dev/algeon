use datafusion_nexus::backend::{GrantBand, MemoryShare};

fn main() {
    let _ = std::mem::size_of::<(GrantBand, MemoryShare)>();
}
