use super::*;

mod bounded_join;
mod partial_aggregate;

#[test]
fn owner_preflight_preserves_materialized_string_bound() {
    let rows = 109_914_910;
    let total_payload = 4_706_637_611;
    let bound = join_output_upper_bound_bytes(
        rows,
        &[JoinOutputColumnUpperBoundInput {
            data_type: ArrowDataType::Utf8,
            input_max_string_length_bytes: None,
            input_total_string_bytes: Some(total_payload),
        }],
    )
    .unwrap()
    .expect("a string column with payload evidence has a bound");
    let null_mask = rows.div_ceil(8).div_ceil(64) * 64;
    assert_eq!(
        bound,
        total_payload + (rows + 1) * STRING_OFFSET_BYTES_CONSERVATIVE + null_mask + PER_COLUMN_ALIGNMENT_PADDING_BYTES
    );
}
