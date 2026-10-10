use datafusion::prelude::SessionContext;

use super::{duration_paced_arrival, duration_paced_workload_arrival};
use crate::single_node_multi_gpu_validate::{
    measure::{
        SessionPool,
        probes::{FeatureOffCugraphProbeSchedule, append_feature_off_cugraph_probe},
    },
    profile::RunProfile,
    workload::{SelectedQuery, select_queries},
};

fn sessions_for_queries(queries: &[SelectedQuery], session_count: usize) -> SessionPool {
    queries.iter().map(|query| (query.fixture_index, vec![SessionContext::new(); session_count])).collect()
}

#[test]
fn duration_paced_workload_rotation_excludes_only_the_dedicated_probe() {
    let profile =
        RunProfile::parse_bytes(include_str!("../../../../fixtures/stress/stress-s1-feature-off-sf1.json").as_bytes())
            .unwrap();
    let mut queries = select_queries(&profile.workloads).unwrap();
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
        for _ in 0..profile.scheduling.concurrency {
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

    let graph_profile =
        RunProfile::parse_bytes(include_str!("../../../../fixtures/stress/stress-s1-embedded-sf1.json").as_bytes())
            .unwrap();
    let graph_queries = select_queries(&graph_profile.workloads).unwrap();
    let sessions = sessions_for_queries(&graph_queries, graph_profile.session.shared_session_count);
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
