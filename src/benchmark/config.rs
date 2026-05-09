use datafusion::common::{DataFusionError, Result as DataFusionResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LargerIslandBenchmarkConfig {
    pub(super) input_rows: usize,
    pub(super) batch_size: usize,
    pub(super) group_count: usize,
    pub(super) limit: usize,
    pub(super) warmup_iterations: usize,
    pub(super) iterations: usize,
}

impl Default for LargerIslandBenchmarkConfig {
    fn default() -> Self {
        Self {
            input_rows: 1_000_000,
            batch_size: 65_536,
            group_count: 4_096,
            limit: 100,
            warmup_iterations: 1,
            iterations: 3,
        }
    }
}

impl LargerIslandBenchmarkConfig {
    pub fn input_rows(&self) -> usize {
        self.input_rows
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn group_count(&self) -> usize {
        self.group_count
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn warmup_iterations(&self) -> usize {
        self.warmup_iterations
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn with_input_rows(mut self, input_rows: usize) -> Self {
        self.input_rows = input_rows;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_group_count(mut self, group_count: usize) -> Self {
        self.group_count = group_count;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_warmup_iterations(mut self, warmup_iterations: usize) -> Self {
        self.warmup_iterations = warmup_iterations;
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero("input_rows", self.input_rows)?;
        validate_nonzero("batch_size", self.batch_size)?;
        validate_nonzero("group_count", self.group_count)?;
        validate_nonzero("limit", self.limit)?;
        validate_nonzero("iterations", self.iterations)?;
        if self.group_count > i32::MAX as usize {
            return Err(DataFusionError::Plan(format!(
                "group_count must fit in Int32 benchmark keys, got {}",
                self.group_count
            )));
        }
        Ok(())
    }
}

fn validate_nonzero(name: &'static str, value: usize) -> DataFusionResult<()> {
    if value == 0 {
        return Err(DataFusionError::Plan(format!("{name} must be greater than zero")));
    }
    Ok(())
}
