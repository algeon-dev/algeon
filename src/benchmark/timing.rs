use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedBenchmarkRun {
    pub iterations: usize,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
}

impl TimedBenchmarkRun {
    pub fn from_samples(samples: impl IntoIterator<Item = Duration>) -> Self {
        let samples = samples.into_iter().collect::<Vec<_>>();
        if samples.is_empty() {
            return Self {
                iterations: 0,
                total: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                mean: Duration::ZERO,
            };
        }

        let total = samples.iter().copied().sum();
        let min = samples.iter().copied().min().unwrap_or(Duration::ZERO);
        let max = samples.iter().copied().max().unwrap_or(Duration::ZERO);
        let mean = duration_div(total, samples.len());
        Self { iterations: samples.len(), total, min, max, mean }
    }
}

pub(super) fn duration_from_nanos(nanos: usize) -> Duration {
    Duration::from_nanos(nanos.min(u64::MAX as usize) as u64)
}

pub(super) fn duration_div(duration: Duration, divisor: usize) -> Duration {
    if divisor == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / divisor as u128;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

pub(super) fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub(super) fn speedup(cpu: Duration, gpu: Duration) -> f64 {
    if cpu.is_zero() {
        return 0.0;
    }
    if gpu.is_zero() {
        return f64::INFINITY;
    }
    cpu.as_secs_f64() / gpu.as_secs_f64()
}
