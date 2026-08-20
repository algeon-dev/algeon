use super::*;

pub(super) fn is_bfs_path_output(spec: &CugraphAlgorithmSpec) -> bool {
    matches!(
        &spec.params,
        CugraphAlgorithmParams::Bfs(params) if params.output.mode == CugraphBfsOutputMode::Path
    )
}
