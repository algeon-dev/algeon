use std::{path::PathBuf, time::Duration};

use datafusion::prelude::SessionContext;
use serde_json::{from_value, json};

use super::{CloseUnderLoadAction, CloseUnderLoadSchedule, duration_paced_arrival, duration_paced_workload_arrival};
use crate::single_node_multi_gpu_validate::{
    RUN_PROFILE_SCHEMA,
    measure::{
        RunTiming, SessionPool,
        probes::{FeatureOffCugraphProbeSchedule, append_feature_off_cugraph_probe},
    },
    profile::{FixtureKind, FixtureWorkloadProfile, MAX_PLANNED_ATTEMPTS, RunProfile, SchedulingProfile},
    schedule::ArrivalPlan,
    workload::{SelectedQuery, select_queries},
};

fn workload(kind: FixtureKind, queries: &[&str]) -> FixtureWorkloadProfile {
    FixtureWorkloadProfile {
        kind,
        path: PathBuf::from("/fixture"),
        scale: "test".to_string(),
        preparation_command: String::new(),
        queries: queries.iter().map(|query| (*query).to_string()).collect(),
    }
}

fn sessions_for_queries(queries: &[SelectedQuery], session_count: usize) -> SessionPool {
    queries.iter().map(|query| (query.fixture_index, vec![SessionContext::new(); session_count])).collect()
}

#[test]
fn duration_paced_workload_rotation_excludes_only_the_dedicated_probe() {
    let profile: RunProfile = from_value(json!({
        "schema": RUN_PROFILE_SCHEMA,
        "name": "feature-off-rotation",
        "feature_off_cugraph_probe": {
            "max_rejection_latency_ms": 250,
            "warmup_probe_attempts": 2,
            "measurement_probe_attempts": 8,
            "min_relational_gpu_reaching_attempts": 1,
        },
    }))
    .unwrap();
    let mut queries =
        select_queries(&[workload(FixtureKind::Tpch, &["q01", "q06"]), workload(FixtureKind::Tpcds, &["q01"])])
            .unwrap();
    let workload_queries = queries.clone();
    let probe_profile = profile.feature_off_cugraph_probe.unwrap();
    let probe = FeatureOffCugraphProbeSchedule {
        query_index: append_feature_off_cugraph_probe(&profile, &mut queries).unwrap().unwrap(),
        warmup_probe_attempts: probe_profile.warmup_probe_attempts,
        measurement_probe_attempts: probe_profile.measurement_probe_attempts,
    };
    let sessions = sessions_for_queries(&queries, 3);
    let mut sequence = 0;
    let mut workload_sequence = 0;
    for (round, probe_attempts) in
        [probe.warmup_probe_attempts, probe.measurement_probe_attempts, 0].into_iter().enumerate()
    {
        for _ in 0..8 {
            let expected = duration_paced_arrival(workload_sequence, round, 0, &workload_queries, &sessions);
            let arrival = duration_paced_workload_arrival(
                sequence,
                &mut workload_sequence,
                round,
                0,
                &queries,
                &sessions,
                Some(probe),
            );
            assert_eq!(arrival.sequence, sequence);
            assert_ne!(arrival.query_index, probe.query_index);
            assert_eq!(arrival.query_index, expected.query_index);
            assert_eq!(arrival.session_index, expected.session_index);
            sequence += 1;
        }
        sequence += probe_attempts;
    }

    let graph_queries =
        select_queries(&[workload(FixtureKind::Tpch, &["q01"]), workload(FixtureKind::Dblp, &["edge_scan", "bfs"])])
            .unwrap();
    let sessions = sessions_for_queries(&graph_queries, 2);
    let mut workload_sequence = 0;
    let mut scheduled_graph_query = false;
    for sequence in 0..graph_queries.len() * 2 {
        let expected = duration_paced_arrival(sequence, 0, 0, &graph_queries, &sessions);
        let arrival =
            duration_paced_workload_arrival(sequence, &mut workload_sequence, 0, 0, &graph_queries, &sessions, None);
        assert_eq!(arrival, expected);
        scheduled_graph_query |= graph_queries[arrival.query_index].graph;
    }
    assert!(scheduled_graph_query);
}

#[test]
fn finite_close_under_load_submits_a_full_cohort_after_drain_or_round_exhaustion() {
    let queries = select_queries(&[workload(FixtureKind::Tpch, &["q01", "q03", "q06"])]).unwrap();
    let sessions = sessions_for_queries(&queries, 3);
    for (concurrency, max_rounds, drain_crosses_midpoint) in
        [(8, 3, true), (8, 2, false), (8, 1, false), (MAX_PLANNED_ATTEMPTS / 3 + 1, 3, false)]
    {
        let scheduling = SchedulingProfile {
            concurrency,
            max_rounds: Some(max_rounds),
            measurement_seconds: 2,
            ..SchedulingProfile::default()
        };
        let plan = ArrivalPlan::plan(&scheduling, queries.len(), 3).unwrap();
        assert!(!plan.is_duration_paced_concurrency());
        let timing = RunTiming::new(&plan, Duration::ZERO);
        let close_at = timing.started + Duration::from_secs(1);
        let mut schedule = CloseUnderLoadSchedule::new(&plan, timing);
        let mut sequence = 0;
        let mut now = timing.started;
        let mut ordinary = Vec::new();
        let mut waited_for_close = false;
        let close_cohort = loop {
            match schedule.next(now) {
                CloseUnderLoadAction::WaitUntil(deadline) => {
                    assert!(!drain_crosses_midpoint);
                    assert_eq!(deadline, close_at);
                    waited_for_close = true;
                    now = deadline;
                },
                CloseUnderLoadAction::Submit(cohort) => {
                    if cohort.submission_deadline.is_none() {
                        break cohort;
                    }
                    ordinary.extend(cohort.arrivals(&mut sequence, &queries, &sessions, || now));
                    if drain_crosses_midpoint {
                        now = close_at + Duration::from_nanos(1);
                    }
                },
            }
        };
        assert_eq!(waited_for_close, !drain_crosses_midpoint);
        if drain_crosses_midpoint {
            assert_eq!(ordinary.len(), concurrency);
        }
        let close_round = close_cohort.round;
        let close_cohort = close_cohort.arrivals(&mut sequence, &queries, &sessions, || now).collect::<Vec<_>>();
        assert_eq!(close_cohort.len(), concurrency);
        for (index, arrival) in close_cohort.into_iter().enumerate() {
            let expected = duration_paced_arrival(ordinary.len() + index, close_round, 0, &queries, &sessions);
            assert_eq!(arrival, expected);
        }
        assert_eq!(sequence, ordinary.len() + concurrency);
        assert!(sequence <= plan.arrivals.len());
        if !drain_crosses_midpoint {
            assert_eq!(sequence, plan.arrivals.len());
        }
    }
}
