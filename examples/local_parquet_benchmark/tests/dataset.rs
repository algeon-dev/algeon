use super::*;

#[test]
fn benchmark_payload_pruning_pattern_is_file_local() {
    let config = test_config();
    let file_rows = config.row_group_size * 4;

    assert!(benchmark_payload_value(0, file_rows, &config) < 10);
    assert!(benchmark_payload_value(config.row_group_size - 1, file_rows, &config) < 10);
    assert!(benchmark_payload_value(config.row_group_size, file_rows, &config) < 10);
    assert!(benchmark_payload_value((config.row_group_size * 2) - 1, file_rows, &config) < 10);
    assert!(benchmark_payload_value(config.row_group_size * 2, file_rows, &config) >= 10);
    assert!(benchmark_payload_value((config.row_group_size * 4) - 1, file_rows, &config) >= 10);
}
