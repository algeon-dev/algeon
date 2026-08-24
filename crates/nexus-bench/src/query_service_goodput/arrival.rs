use std::time::Duration;

use super::config::{ArrivalProcess, GoodputConfig};

/// Segment of the run an arrival belongs to.
///
/// Only `Window` arrivals enter the goodput denominator. Warmup arrivals are
/// executed and reported so a cold-to-hot transition stays visible, but they
/// are never mixed into the measured window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Warmup,
    Window,
}

impl Phase {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Window => "window",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Arrival {
    pub(super) sequence: u64,
    pub(super) offset: Duration,
    pub(super) item_index: usize,
    pub(super) phase: Phase,
}

/// Deterministic open-loop arrival trace.
///
/// The trace is generated once from the seed and replayed identically by every
/// comparison group, so groups are paired rather than independently sampled.
pub(super) fn build_trace(config: &GoodputConfig, weights: &[u32], rate_qps: f64) -> Vec<Arrival> {
    let total_seconds = config.warmup_seconds + config.window_seconds;
    let mut rng = SplitMix64::new(config.seed);
    let mut arrivals = Vec::new();
    let mut sequence = 0_u64;
    let mut elapsed = 0.0_f64;
    let push = |arrivals: &mut Vec<Arrival>, sequence: &mut u64, elapsed: f64, item_index: usize| {
        let phase = if elapsed < config.warmup_seconds { Phase::Warmup } else { Phase::Window };
        arrivals.push(Arrival { sequence: *sequence, offset: Duration::from_secs_f64(elapsed), item_index, phase });
        *sequence += 1;
    };
    match config.arrival {
        ArrivalProcess::Poisson => loop {
            elapsed += rng.next_exponential(rate_qps);
            if elapsed >= total_seconds {
                break;
            }
            let item_index = rng.next_weighted_index(weights);
            push(&mut arrivals, &mut sequence, elapsed, item_index);
        },
        ArrivalProcess::Wave => {
            let interval = config.wave_size as f64 / rate_qps;
            loop {
                elapsed += interval;
                if elapsed >= total_seconds {
                    break;
                }
                for _ in 0..config.wave_size {
                    let item_index = rng.next_weighted_index(weights);
                    push(&mut arrivals, &mut sequence, elapsed, item_index);
                }
            }
        },
    }
    arrivals
}

/// SplitMix64. Inlined rather than taken as a dependency so a recorded seed
/// reproduces the exact trace regardless of the dependency graph.
#[derive(Debug)]
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform draw in the half-open interval (0, 1).
    ///
    /// Zero is excluded so the exponential transform below never sees a
    /// logarithm of zero.
    pub(super) fn next_unit(&mut self) -> f64 {
        let mantissa = (self.next_u64() >> 11) as f64;
        (mantissa + 0.5) / (1_u64 << 53) as f64
    }

    pub(super) fn next_exponential(&mut self, rate: f64) -> f64 {
        -self.next_unit().ln() / rate
    }

    pub(super) fn next_weighted_index(&mut self, weights: &[u32]) -> usize {
        let total = weights.iter().map(|weight| u64::from(*weight)).sum::<u64>();
        debug_assert!(total > 0, "mix weights are validated to be positive");
        let mut draw = self.next_u64() % total;
        for (index, weight) in weights.iter().enumerate() {
            let weight = u64::from(*weight);
            if draw < weight {
                return index;
            }
            draw -= weight;
        }
        weights.len() - 1
    }
}
