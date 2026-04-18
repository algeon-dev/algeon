use cudf::{LookupTable, ops::lookup};

use crate::{
    allocation::allocation_count_delta,
    common::{explicit_i32_column, explicit_i32_series},
};

#[test]
fn lookup_probe_allocates_no_more_than_explicit_lookup_table_build_and_probe() {
    let input = explicit_i32_column(&[2, 9, 1, 2], 930);
    let keys = explicit_i32_series("id", &[1, 2, 3], 931);
    let values = explicit_i32_series("label", &[10, 20, 30], 931);

    let one_shot_allocations = allocation_count_delta(|| {
        let result = lookup::probe(&input.view().unwrap(), &keys.view().unwrap(), &values.view().unwrap()).unwrap();
        assert_eq!(result.view().unwrap().to_optional_vec::<i32>().unwrap(), vec![Some(20), None, Some(10), Some(20)]);
        result
    });

    let explicit_build_probe_allocations = allocation_count_delta(|| {
        let table = LookupTable::new(keys.clone(), values.clone()).unwrap();
        let result = table.probe(&explicit_i32_series("vertex", &[2, 9, 1, 2], 930)).unwrap();
        assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(20), None, Some(10), Some(20)]);
        (table, result)
    });

    assert!(
        one_shot_allocations <= explicit_build_probe_allocations,
        "lookup::probe() should not allocate more device memory than explicitly building and probing a LookupTable",
    );
}
