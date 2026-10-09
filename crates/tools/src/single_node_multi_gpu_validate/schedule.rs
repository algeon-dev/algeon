//! Deterministic arrival planning.
//!
//! The plan is a pure value: offsets from an unspecified `t0`, never a clock
//! read. That keeps the seed-to-schedule mapping reproducible and lets the
//! whole schedule be asserted without a GPU.
//!
//! Only the measured run drives a plan, so the CPU-only feature reaches these
//! values from tests alone.
#![cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]

use serde::Serialize;

use super::profile::{ArrivalMode, MAX_PLANNED_ATTEMPTS, RateOverrunPolicy, SchedulingProfile};
use crate::ToolResult;

/// Which measurement window an arrival belongs to. Warmup arrivals execute but
/// are excluded from reported rates and latency distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SchedulePhase {
    Warmup,
    Measurement,
}

/// One planned attempt: when to submit it, which query to run, and which
/// shared session to submit it through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct ScheduledArrival {
    pub(super) sequence: usize,
    /// Closed-loop round this arrival belongs to within its step. The driver
    /// drains a round before submitting the next, making `concurrency` an
    /// in-flight ceiling. Rounds restart at zero for each step.
    pub(super) round: usize,
    /// Multi-phase step this arrival belongs to. Single-phase profiles place every
    /// arrival in step zero.
    pub(super) step_index: usize,
    pub(super) offset_ns: u64,
    pub(super) phase: SchedulePhase,
    /// The overrun policy this arrival was planned under. Rate steps may mix
    /// `delay` and `drop` phases inside one run, so the driver reads the
    /// policy from the arrival rather than from one plan-level knob.
    pub(super) rate_overrun: RateOverrunPolicy,
    pub(super) query_index: usize,
    pub(super) session_index: usize,
}

/// One sequential phase of the plan. A single-phase profile produces exactly
/// one step spanning the whole measurement window, which keeps the driver on
/// one code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct StepPlan {
    pub(super) step_index: usize,
    /// Closed-loop in-flight ceiling for this step (concurrency mode).
    pub(super) concurrency: usize,
    /// Poisson offered rate for this step (rate mode).
    pub(super) arrival_rate_per_second: Option<u64>,
    pub(super) rate_overrun: &'static str,
    /// Offset of the step's start from the run's t0 (after warmup).
    pub(super) start_ns: u64,
    pub(super) measurement_ns: u64,
    /// Arrivals this step planned. Concurrency mode over-plans rounds and is
    /// paced by the step window, so its count is an upper bound there.
    pub(super) planned_arrivals: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct ArrivalPlan {
    pub(super) arrival_mode: &'static str,
    pub(super) seed: u64,
    pub(super) concurrency: usize,
    pub(super) arrival_rate_per_second: Option<u64>,
    /// Explicit closed-loop batch bound, when the profile chose one. A
    /// `None` value preserves duration-paced closed-loop scheduling.
    pub(super) max_rounds: Option<usize>,
    /// Duration-paced closed-loop profiles construct each round at the driver
    /// boundary instead of materializing a speculative wall-clock number of
    /// arrivals. This prevents the planning cap from omitting later steps
    /// while retaining a finite, reproducible plan for rate and fixed-cohort
    /// fixed-cohort validation scenarios.
    pub(super) duration_paced_concurrency: bool,
    pub(super) rate_overrun: &'static str,
    pub(super) warmup_ns: u64,
    pub(super) measurement_ns: u64,
    pub(super) warmup_arrivals: usize,
    pub(super) measurement_arrivals: usize,
    /// True when [`MAX_PLANNED_ATTEMPTS`] cut a rate-mode plan short, which
    /// means the artifact does not describe the offered process the profile
    /// asked for. Concurrency mode over-plans rounds on purpose and is paced by
    /// the measurement window, so the cap is never a truncation there.
    pub(super) truncated_by_attempt_cap: bool,
    /// Sequential phases of this plan. Always non-empty; a single-phase
    /// profile contributes exactly one step.
    pub(super) steps: Vec<StepPlan>,
    /// deadline/cancellation storm knobs, echoed from the profile so the driver can arm them
    /// per attempt.
    pub(super) attempt_deadline_ms: Option<u64>,
    pub(super) cancel_every_n: Option<usize>,
    pub(super) cancel_after_ms: Option<u64>,
    pub(super) arrivals: Vec<ScheduledArrival>,
}

impl ArrivalPlan {
    /// Builds the plan for `query_count` selected queries and
    /// `session_count` shared sessions.
    ///
    /// Concurrency mode issues `concurrency` simultaneous arrivals per round
    /// and needs no seed. Rate mode draws exponential interarrival gaps, so
    /// the offered process is Poisson at the configured rate. In steps mode
    /// the steps are planned sequentially against their own windows; one
    /// seeded stream spans every rate step so the recorded seed still maps to
    /// exactly one schedule.
    pub(super) fn plan(scheduling: &SchedulingProfile, query_count: usize, session_count: usize) -> ToolResult<Self> {
        if query_count == 0 {
            return Err("a measured run requires at least one selected workload query".into());
        }
        if session_count == 0 {
            return Err("a measured run requires at least one shared session".into());
        }
        let warmup_ns = seconds_to_ns(scheduling.warmup_seconds)?;
        let rate_overrun = scheduling.rate_overrun;
        let (step_shapes, measurement_ns) = if scheduling.steps.is_empty() {
            if scheduling.measurement_seconds == 0 {
                return Err("scheduling.measurement_seconds must be greater than zero for a measured run".into());
            }
            let shape = StepShape {
                concurrency: scheduling.concurrency,
                arrival_rate_per_second: scheduling.arrival_rate_per_second,
                rate_overrun,
                measurement_ns: seconds_to_ns(scheduling.measurement_seconds)?,
            };
            (vec![shape], shape.measurement_ns)
        } else {
            let mut shapes = Vec::with_capacity(scheduling.steps.len());
            let mut total_ns = 0_u64;
            for step in &scheduling.steps {
                let measurement_ns = seconds_to_ns(step.measurement_seconds)?;
                total_ns =
                    total_ns.checked_add(measurement_ns).ok_or("scheduling steps window overflows nanoseconds")?;
                shapes.push(StepShape {
                    concurrency: step.concurrency.unwrap_or(0),
                    arrival_rate_per_second: step.arrival_rate_per_second,
                    rate_overrun: step.rate_overrun.unwrap_or(RateOverrunPolicy::Delay),
                    measurement_ns,
                });
            }
            (shapes, total_ns)
        };
        // The total horizon itself must fit the clock even though each step
        // accumulates its own checked window.
        let _ = warmup_ns.checked_add(measurement_ns).ok_or("scheduling window overflows nanoseconds")?;

        let duration_paced_concurrency =
            scheduling.arrival_mode == ArrivalMode::Concurrency && scheduling.max_rounds.is_none();
        let mut rng = SplitMix64::new(scheduling.seed);
        let mut arrivals = Vec::new();
        let mut steps = Vec::with_capacity(step_shapes.len());
        let mut step_start_ns = warmup_ns;
        for (step_index, shape) in step_shapes.iter().enumerate() {
            let first_arrival = arrivals.len();
            // The step's measurement window ends where the next begins; the
            // final step ends at the run horizon. Step zero additionally owns
            // the warmup phase, so its rate arrivals draw from offset zero.
            let step_end_ns = step_start_ns
                .checked_add(shape.measurement_ns)
                .ok_or("scheduling steps window overflows nanoseconds")?;
            match scheduling.arrival_mode {
                ArrivalMode::Concurrency if !duration_paced_concurrency => plan_concurrency(
                    shape.concurrency,
                    scheduling.max_rounds,
                    shape.measurement_ns,
                    step_index,
                    shape.rate_overrun,
                    warmup_ns,
                    query_count,
                    session_count,
                    &mut arrivals,
                ),
                ArrivalMode::Concurrency => {},
                ArrivalMode::Rate => {
                    let rate = shape
                        .arrival_rate_per_second
                        .ok_or("rate scheduling requires scheduling.arrival_rate_per_second")?;
                    plan_rate(
                        rate,
                        &mut rng,
                        if step_index == 0 { 0 } else { step_start_ns },
                        step_end_ns,
                        step_index,
                        shape.rate_overrun,
                        warmup_ns,
                        query_count,
                        session_count,
                        &mut arrivals,
                    );
                },
            }
            steps.push(StepPlan {
                step_index,
                concurrency: shape.concurrency,
                arrival_rate_per_second: shape.arrival_rate_per_second,
                rate_overrun: rate_overrun_label(shape.rate_overrun),
                start_ns: step_start_ns,
                measurement_ns: shape.measurement_ns,
                planned_arrivals: arrivals.len() - first_arrival,
            });
            step_start_ns = step_end_ns;
            if !duration_paced_concurrency && arrivals.len() >= MAX_PLANNED_ATTEMPTS {
                break;
            }
        }
        if arrivals.is_empty() && !duration_paced_concurrency {
            return Err("the scheduling window planned zero arrivals".into());
        }

        let truncated_by_attempt_cap =
            scheduling.arrival_mode == ArrivalMode::Rate && arrivals.len() >= MAX_PLANNED_ATTEMPTS;
        let warmup_arrivals = arrivals.iter().filter(|arrival| arrival.phase == SchedulePhase::Warmup).count();
        Ok(Self {
            arrival_mode: match scheduling.arrival_mode {
                ArrivalMode::Concurrency => "concurrency",
                ArrivalMode::Rate => "rate",
            },
            seed: scheduling.seed,
            concurrency: scheduling.concurrency,
            arrival_rate_per_second: scheduling.arrival_rate_per_second,
            max_rounds: scheduling.max_rounds,
            duration_paced_concurrency,
            rate_overrun: rate_overrun_label(scheduling.rate_overrun),
            warmup_ns,
            measurement_ns,
            warmup_arrivals,
            measurement_arrivals: arrivals.len() - warmup_arrivals,
            truncated_by_attempt_cap,
            steps,
            attempt_deadline_ms: scheduling.attempt_deadline_ms,
            cancel_every_n: scheduling.cancel_every_n,
            cancel_after_ms: scheduling.cancel_after_ms,
            arrivals,
        })
    }

    pub(super) const fn is_duration_paced_concurrency(&self) -> bool {
        self.duration_paced_concurrency
    }
}

/// The resolved per-step planning inputs, shared by the single-phase and
/// steps-mode paths above.
#[derive(Clone, Copy)]
struct StepShape {
    concurrency: usize,
    arrival_rate_per_second: Option<u64>,
    rate_overrun: RateOverrunPolicy,
    measurement_ns: u64,
}

fn rate_overrun_label(policy: RateOverrunPolicy) -> &'static str {
    match policy {
        RateOverrunPolicy::Delay => "delay",
        RateOverrunPolicy::Drop => "drop",
    }
}

/// Closed-loop rounds of exactly `concurrency` arrivals.
///
/// Every arrival in a round shares offset zero and carries the round index; the
/// driver submits round `N + 1` only once round `N` has drained. `concurrency`
/// is therefore an in-flight ceiling, not an offered arrival rate, and the
/// step window bounds the step rather than pacing it.
///
/// Rounds are planned generously against the step horizon because completion
/// pacing is observed. The driver stops at the step deadline, and unsubmitted
/// arrivals never become attempt rows.
#[expect(clippy::too_many_arguments, reason = "arrival-planning matrix helper; each parameter is one planning axis")]
fn plan_concurrency(
    concurrency: usize,
    max_rounds: Option<usize>,
    step_window_ns: u64,
    step_index: usize,
    rate_overrun: RateOverrunPolicy,
    warmup_ns: u64,
    query_count: usize,
    session_count: usize,
    arrivals: &mut Vec<ScheduledArrival>,
) {
    // A generous round budget: the driver ends on the wall-clock step
    // deadline, and an attempt cannot be shorter than this floor in practice.
    const MIN_ROUND_NS: u64 = 1_000_000;
    let duration_paced_rounds = usize::try_from(step_window_ns / MIN_ROUND_NS).unwrap_or(usize::MAX).max(1);
    let rounds = max_rounds.unwrap_or(duration_paced_rounds).min(duration_paced_rounds);
    for round in 0..rounds {
        for _ in 0..concurrency {
            if arrivals.len() >= MAX_PLANNED_ATTEMPTS {
                return;
            }
            let sequence = arrivals.len();
            arrivals.push(arrival(sequence, round, step_index, 0, rate_overrun, warmup_ns, query_count, session_count));
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "arrival-planning matrix helper; each parameter is one planning axis")]
fn plan_rate(
    rate_per_second: u64,
    rng: &mut SplitMix64,
    step_start_ns: u64,
    horizon_ns: u64,
    step_index: usize,
    rate_overrun: RateOverrunPolicy,
    warmup_ns: u64,
    query_count: usize,
    session_count: usize,
    arrivals: &mut Vec<ScheduledArrival>,
) {
    let mean_gap_ns = 1_000_000_000_f64 / rate_per_second as f64;
    let mut offset_ns = step_start_ns;
    while arrivals.len() < MAX_PLANNED_ATTEMPTS {
        // Exponential interarrival: -mean * ln(U), U in (0, 1].
        let gap_ns = (-mean_gap_ns * rng.next_unit_interval().ln()).round();
        let gap_ns = if gap_ns.is_finite() && gap_ns >= 0.0 { gap_ns as u64 } else { 0 };
        offset_ns = offset_ns.saturating_add(gap_ns);
        if offset_ns >= horizon_ns {
            break;
        }
        let sequence = arrivals.len();
        arrivals.push(arrival(sequence, 0, step_index, offset_ns, rate_overrun, warmup_ns, query_count, session_count));
    }
}

#[expect(clippy::too_many_arguments, reason = "arrival-planning matrix helper; each parameter is one planning axis")]
fn arrival(
    sequence: usize,
    round: usize,
    step_index: usize,
    offset_ns: u64,
    rate_overrun: RateOverrunPolicy,
    warmup_ns: u64,
    query_count: usize,
    session_count: usize,
) -> ScheduledArrival {
    ScheduledArrival {
        sequence,
        round,
        step_index,
        offset_ns,
        // Concurrency mode has no planned offsets, so the driver reclassifies
        // its arrivals against the observed warmup deadline; rate mode's
        // offsets already place each arrival in a window.
        phase: if offset_ns < warmup_ns { SchedulePhase::Warmup } else { SchedulePhase::Measurement },
        rate_overrun,
        query_index: sequence % query_count,
        session_index: sequence % session_count,
    }
}

fn seconds_to_ns(seconds: u64) -> ToolResult<u64> {
    seconds.checked_mul(1_000_000_000).ok_or_else(|| "scheduling window overflows nanoseconds".into())
}

/// SplitMix64. Inlined rather than taken as a dependency so the recorded seed
/// maps to the same schedule across toolchain and crate-version changes.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A draw in (0, 1]; zero is excluded so the exponential transform stays
    /// finite.
    fn next_unit_interval(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64 + 1.0) / (1_u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_node_multi_gpu_validate::profile::ScheduleStepProfile;

    fn scheduling(mode: ArrivalMode, rate: Option<u64>, seed: u64) -> SchedulingProfile {
        SchedulingProfile {
            arrival_mode: mode,
            concurrency: 4,
            arrival_rate_per_second: rate,
            max_rounds: None,
            seed,
            warmup_seconds: 2,
            measurement_seconds: 8,
            rate_overrun: RateOverrunPolicy::Delay,
            steps: Vec::new(),
            attempt_deadline_ms: None,
            cancel_every_n: None,
            cancel_after_ms: None,
        }
    }

    /// Duration-paced concurrency profiles stay compact: the runner creates
    /// their closed-loop rounds at the step boundary instead of silently
    /// dropping later rounds or steps at the static planning cap.
    #[test]
    fn concurrency_mode_defers_duration_paced_rounds_to_the_driver() {
        let plan = ArrivalPlan::plan(&scheduling(ArrivalMode::Concurrency, None, 0), 3, 2).unwrap();
        assert_eq!(plan.arrival_mode, "concurrency");
        assert_eq!(plan.concurrency, 4);
        assert_eq!(plan.warmup_ns, 2_000_000_000);
        assert_eq!(plan.measurement_ns, 8_000_000_000);
        assert!(plan.is_duration_paced_concurrency());
        assert!(plan.arrivals.is_empty());
        assert_eq!(plan.steps[0].planned_arrivals, 0);

        {
            assert_eq!(plan.steps.len(), 1);
            assert_eq!(plan.steps[0].step_index, 0);
            assert_eq!(plan.steps[0].concurrency, plan.concurrency);
            assert_eq!(plan.steps[0].start_ns, plan.warmup_ns);
            assert_eq!(plan.steps[0].measurement_ns, plan.measurement_ns);
        }
    }

    #[test]
    fn configured_round_bound_preserves_one_complete_closed_loop_batch() {
        let mut profile = scheduling(ArrivalMode::Concurrency, None, 0);
        profile.max_rounds = Some(2);

        let plan = ArrivalPlan::plan(&profile, 3, 2).unwrap();

        assert_eq!(plan.max_rounds, Some(2));
        assert_eq!(plan.arrivals.len(), 8);
        assert!(plan.arrivals.iter().take(4).all(|arrival| arrival.round == 0));
        assert!(plan.arrivals.iter().skip(4).all(|arrival| arrival.round == 1));
    }

    /// Rate mode attributes phases from planned offsets. Concurrency mode has
    /// no offsets, so the driver assigns its phases against the observed
    /// warmup deadline instead.
    #[test]
    fn rate_mode_phases_partition_every_arrival_at_the_warmup_boundary() {
        let plan = ArrivalPlan::plan(&scheduling(ArrivalMode::Rate, Some(200), 7), 2, 1).unwrap();
        assert_eq!(plan.warmup_arrivals + plan.measurement_arrivals, plan.arrivals.len());
        for arrival in &plan.arrivals {
            let expected =
                if arrival.offset_ns < plan.warmup_ns { SchedulePhase::Warmup } else { SchedulePhase::Measurement };
            assert_eq!(arrival.phase, expected);
            assert!(arrival.offset_ns < plan.warmup_ns + plan.measurement_ns);
        }
    }

    #[test]
    fn a_seed_reproduces_a_byte_identical_plan_and_a_different_seed_does_not() {
        let first = ArrivalPlan::plan(&scheduling(ArrivalMode::Rate, Some(500), 42), 4, 2).unwrap();
        let repeat = ArrivalPlan::plan(&scheduling(ArrivalMode::Rate, Some(500), 42), 4, 2).unwrap();
        assert_eq!(serde_json::to_vec(&first).unwrap(), serde_json::to_vec(&repeat).unwrap());

        let other = ArrivalPlan::plan(&scheduling(ArrivalMode::Rate, Some(500), 43), 4, 2).unwrap();
        assert_ne!(first.arrivals, other.arrivals);
    }

    #[test]
    fn rate_mode_offsets_are_monotonic_and_average_near_the_configured_rate() {
        let plan = ArrivalPlan::plan(&scheduling(ArrivalMode::Rate, Some(1000), 11), 1, 1).unwrap();
        assert!(plan.arrivals.windows(2).all(|pair| pair[0].offset_ns <= pair[1].offset_ns));
        // 1000/s over a 10s horizon; exponential draws keep this near 10_000.
        let observed = plan.arrivals.len() as f64;
        assert!((8_000.0..12_000.0).contains(&observed), "{observed}");
    }

    #[test]
    fn queries_and_sessions_are_assigned_round_robin_over_the_selected_sets() {
        let mut profile = scheduling(ArrivalMode::Concurrency, None, 0);
        profile.max_rounds = Some(2);
        let plan = ArrivalPlan::plan(&profile, 3, 2).unwrap();
        for arrival in &plan.arrivals {
            assert_eq!(arrival.query_index, arrival.sequence % 3);
            assert_eq!(arrival.session_index, arrival.sequence % 2);
        }
        assert!(plan.arrivals.iter().any(|arrival| arrival.query_index == 2));
        assert!(plan.arrivals.iter().any(|arrival| arrival.session_index == 1));
    }

    #[test]
    fn planning_rejects_a_zero_measurement_window_and_an_empty_query_or_session_set() {
        let mut zero = scheduling(ArrivalMode::Concurrency, None, 0);
        zero.measurement_seconds = 0;
        assert!(ArrivalPlan::plan(&zero, 1, 1).unwrap_err().to_string().contains("measurement_seconds"));

        let valid = scheduling(ArrivalMode::Concurrency, None, 0);
        assert!(ArrivalPlan::plan(&valid, 0, 1).unwrap_err().to_string().contains("workload query"));
        assert!(ArrivalPlan::plan(&valid, 1, 0).unwrap_err().to_string().contains("shared session"));
    }

    #[test]
    fn planning_is_bounded_regardless_of_rate_and_window() {
        let mut huge = scheduling(ArrivalMode::Rate, Some(1_000_000), 1);
        huge.measurement_seconds = 3_600;
        let plan = ArrivalPlan::plan(&huge, 1, 1).unwrap();
        assert_eq!(plan.arrivals.len(), MAX_PLANNED_ATTEMPTS);
        // A silent cap would read as "this is the plan you asked for".
        assert!(plan.truncated_by_attempt_cap);

        let modest = scheduling(ArrivalMode::Rate, Some(10), 1);
        assert!(!ArrivalPlan::plan(&modest, 1, 1).unwrap().truncated_by_attempt_cap);

        // Duration-paced concurrency mode carries no speculative round vector,
        // so no later step can be silently cut off by the planning cap.
        let mut closed_loop = scheduling(ArrivalMode::Concurrency, None, 0);
        closed_loop.measurement_seconds = 3_600;
        let plan = ArrivalPlan::plan(&closed_loop, 1, 1).unwrap();
        assert!(plan.is_duration_paced_concurrency());
        assert!(plan.arrivals.is_empty());
        assert!(!plan.truncated_by_attempt_cap);
    }

    #[test]
    fn concurrency_steps_plan_sequential_phases_with_restarted_rounds() {
        let mut profile = scheduling(ArrivalMode::Concurrency, None, 0);
        profile.measurement_seconds = 0;
        profile.steps = vec![
            ScheduleStepProfile {
                concurrency: Some(2),
                arrival_rate_per_second: None,
                measurement_seconds: 5,
                rate_overrun: None,
            },
            ScheduleStepProfile {
                concurrency: Some(8),
                arrival_rate_per_second: None,
                measurement_seconds: 3,
                rate_overrun: None,
            },
        ];
        let plan = ArrivalPlan::plan(&profile, 3, 2).unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.measurement_ns, 8_000_000_000);
        assert_eq!(plan.steps[0].concurrency, 2);
        assert_eq!(plan.steps[1].concurrency, 8);
        assert_eq!(plan.steps[1].start_ns, plan.warmup_ns + 5_000_000_000);
        assert!(plan.is_duration_paced_concurrency());
        assert_eq!(plan.steps[0].planned_arrivals, 0);
        assert_eq!(plan.steps[1].planned_arrivals, 0);
        assert!(plan.arrivals.is_empty());
    }

    #[test]
    fn rate_steps_mix_overrun_policies_and_share_one_seeded_stream() {
        let mut profile = scheduling(ArrivalMode::Rate, None, 42);
        profile.measurement_seconds = 0;
        profile.steps = vec![
            ScheduleStepProfile {
                concurrency: None,
                arrival_rate_per_second: Some(1_000),
                measurement_seconds: 4,
                rate_overrun: Some(RateOverrunPolicy::Delay),
            },
            ScheduleStepProfile {
                concurrency: None,
                arrival_rate_per_second: Some(10_000),
                measurement_seconds: 4,
                rate_overrun: Some(RateOverrunPolicy::Drop),
            },
        ];
        let plan = ArrivalPlan::plan(&profile, 2, 1).unwrap();
        let repeat = ArrivalPlan::plan(&profile, 2, 1).unwrap();
        assert_eq!(serde_json::to_vec(&plan).unwrap(), serde_json::to_vec(&repeat).unwrap());
        assert_eq!(plan.steps.len(), 2);
        let step1 = plan.arrivals.iter().filter(|arrival| arrival.step_index == 1).collect::<Vec<_>>();
        assert!(!step1.is_empty());
        assert!(step1.iter().all(|arrival| arrival.rate_overrun == RateOverrunPolicy::Drop));
        assert!(
            plan.arrivals
                .iter()
                .filter(|arrival| arrival.step_index == 0)
                .all(|arrival| arrival.rate_overrun == RateOverrunPolicy::Delay)
        );
        // The denser second step offers meaningfully more arrivals.
        let step0 = plan.arrivals.iter().filter(|arrival| arrival.step_index == 0).count();
        assert!(step1.len() > step0 * 3, "step0 {step0} step1 {}", step1.len());
        // Offsets stay global: a later step never plans inside an earlier window.
        assert!(step1.iter().all(|arrival| arrival.offset_ns >= plan.steps[1].start_ns));
    }
}
