use cudf::{DataType, TypeId};
use query_planning::plan::{EdgeNormalizeSpec, GraphConstructionOptions, GraphEdgeColumnBindings};

use super::{EdgeKeyPacking, EdgeNormalization, EndpointRanges};

fn choose(directed: bool, distinct_edge_pairs: bool, source: (i64, i64), destination: (i64, i64)) -> EdgeNormalization {
    let spec = EdgeNormalizeSpec::python_cugraph(
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        &GraphConstructionOptions { directed, distinct_edge_pairs, ..Default::default() },
    );
    let ranges = EndpointRanges { vertex_type: DataType::new(TypeId::Int64), source, destination };
    EdgeNormalization::choose(&spec, || Ok(Some(ranges))).expect("the ranges are supplied")
}

fn packing(source_low: i64, destination_low: i64, destination_bits: u32) -> Option<EdgeKeyPacking> {
    Some(EdgeKeyPacking { vertex_type: DataType::new(TypeId::Int64), source_low, destination_low, destination_bits })
}

/// MerRec's shape: user ids offset by `10^10` span 26 bits and item ids 28
/// bits. Their union spans 34 bits, but each column packs from its own
/// minimum when the rows hold no reversed rows: directed input, or disjoint
/// undirected input whose reversed rows are appended after deduplication.
#[test]
fn far_apart_ranges_pack_each_column_from_its_own_minimum() {
    let users = (10_000_000_000, 10_000_000_000 + (1 << 26) - 1);
    let items = (0, (1 << 28) - 1);
    let per_column = EdgeNormalization::DeduplicateInput(packing(users.0, items.0, 28));
    assert_eq!(choose(true, false, users, items), per_column);
    assert_eq!(choose(false, false, users, items), per_column);
    assert_eq!(choose(false, true, users, items), EdgeNormalization::AppendReverse);
}

/// Disjoint undirected ranges too wide for one key still deduplicate only the
/// input rows, on the endpoint columns.
#[test]
fn disjoint_undirected_ranges_deduplicate_the_input_rows_without_a_key() {
    let source = (i64::MIN, i64::MIN + (1 << 40));
    let destination = (i64::MAX - (1 << 31), i64::MAX);
    assert_eq!(choose(false, false, source, destination), EdgeNormalization::DeduplicateInput(None));
}

/// Reversed rows put destination values in the source column, so ranges that
/// overlap, even in one value, pack both fields over their union.
#[test]
fn overlapping_undirected_ranges_pack_both_fields_over_the_union() {
    let source = (0, 5);
    assert_eq!(
        choose(false, false, source, (5, (1 << 32) - 1)),
        EdgeNormalization::DeduplicateExpanded(packing(0, 0, 32))
    );
    assert_eq!(choose(false, false, source, (5, 1 << 32)), EdgeNormalization::DeduplicateExpanded(None));
    assert_eq!(choose(true, false, source, (5, 1 << 32)), EdgeNormalization::DeduplicateInput(packing(0, 5, 32)));
}

/// The field widths sum to at most 64 bits. A single value takes 0 bits. The
/// source field may take all 64 bits, while a 64-bit destination field would
/// shift by 64 and does not pack.
#[test]
fn field_widths_pack_up_to_64_bits_with_a_destination_below_64() {
    let forty_bits = (1 << 50, (1 << 50) + (1 << 40) - 1);
    assert_eq!(
        choose(true, false, forty_bits, (0, (1 << 24) - 1)),
        EdgeNormalization::DeduplicateInput(packing(1 << 50, 0, 24))
    );
    assert_eq!(choose(true, false, forty_bits, (0, 1 << 24)), EdgeNormalization::DeduplicateInput(None));
    assert_eq!(
        choose(true, false, (i64::MIN, i64::MAX), (3, 3)),
        EdgeNormalization::DeduplicateInput(packing(i64::MIN, 3, 0))
    );
    assert_eq!(
        choose(true, false, (7, 7), (i64::MIN, -1)),
        EdgeNormalization::DeduplicateInput(packing(7, i64::MIN, 63))
    );
    assert_eq!(choose(true, false, (7, 7), (i64::MIN, i64::MAX)), EdgeNormalization::DeduplicateInput(None));
}
