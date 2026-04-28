#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    GpuResident,
    MaterializeToHost,
}
