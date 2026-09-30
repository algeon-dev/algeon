use std::collections::HashMap;

use cudf::{
    DataFrame, Runtime, Series, TypeId,
    core::{Column, Table},
    ops::dictionary,
    testing::{interop::column_export_with_lease_backend_for_test, prelude::*},
};
use rapids_interop::{
    BufferBinding, ColumnLayout, InteropError, InteropErrorKind, PhysicalValidity, PreparedAccess,
    PreparedColumnExport, PreparedExportProfile, SemanticColumn,
};

use crate::common::{self, run_named_cases};

fn binding_bytes(binding: &BufferBinding<'_>) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr() as *const u8, binding.bytes().byte_len()) }.to_vec()
}

fn nested_source_frame() -> DataFrame {
    let runtime = Runtime::builder().build().unwrap();
    common::nested_dataframe(
        &runtime,
        vec![
            ("category", common::utf8_array(&["beta", "alpha", "beta"])),
            (
                "nested",
                common::nested_int64_list_array(&[vec![vec![1, 2], vec![3]], vec![vec![], vec![4, 5]], vec![vec![6]]]),
            ),
            (
                "point",
                common::struct_array(vec![
                    ("coords", common::int64_list_array(&[vec![1, 2], vec![], vec![3]])),
                    ("label", common::utf8_array(&["one", "two", "three"])),
                ]),
            ),
            ("id", common::int64_array(&[10, 20, 30])),
        ],
    )
}

fn binding_map<'a>(bindings: &'a [BufferBinding<'a>]) -> HashMap<u32, &'a BufferBinding<'a>> {
    bindings.iter().map(|binding| (binding.id().get(), binding)).collect()
}

fn semantic_column_map(columns: &[SemanticColumn]) -> HashMap<u32, &SemanticColumn> {
    columns.iter().map(|column| (column.id().get(), column)).collect()
}

fn interop_error_context_value<'a>(err: &'a InteropError, key: &str) -> Option<&'a str> {
    err.context().iter().find_map(|(ctx_key, ctx_value)| (*ctx_key == key).then_some(ctx_value.as_str()))
}

fn reset_exec_observation<T>(f: impl FnOnce() -> T) -> T {
    common::reset_exec_observation();
    f()
}

fn assert_exec_observation_matches_runtime(runtime: &Runtime) {
    assert_eq!(common::last_exec_device(), runtime.device().ordinal() as i32);
    assert_eq!(common::last_exec_stream_value(), runtime.stream().testing_raw_handle_value());
    assert_eq!(common::last_exec_mr_value(), runtime.memory_resource().testing_raw_handle_value());
}

fn assert_column_bound_to_runtime(imported: &Column, runtime: &Runtime) {
    assert_eq!(imported.testing_execution_domain_stream_value(), runtime.stream().testing_raw_handle_value());
    assert_eq!(imported.testing_execution_domain_mr_value(), runtime.memory_resource().testing_raw_handle_value());
}

fn assert_column_imported_into_runtime(imported: &Column, runtime: &Runtime) {
    assert_column_bound_to_runtime(imported, runtime);
    assert_exec_observation_matches_runtime(runtime);
}

fn assert_table_bound_to_runtime(imported: &Table, runtime: &Runtime) {
    assert_eq!(imported.testing_execution_domain_stream_value(), runtime.stream().testing_raw_handle_value());
    assert_eq!(imported.testing_execution_domain_mr_value(), runtime.memory_resource().testing_raw_handle_value());
}

fn assert_table_imported_into_runtime(imported: &Table, runtime: &Runtime) {
    assert_table_bound_to_runtime(imported, runtime);
    assert_exec_observation_matches_runtime(runtime);
}

fn assert_binding_equivalent(source: &BufferBinding<'_>, imported: &BufferBinding<'_>) {
    assert_eq!(source.placement(), imported.placement());
    assert_eq!(binding_bytes(source), binding_bytes(imported));
}

fn validity_mask_is_elidable(validity: &PhysicalValidity, _bindings: &HashMap<u32, &BufferBinding<'_>>) -> bool {
    matches!(validity, PhysicalValidity::ValidityMask { null_count: Some(0), .. })
}

fn assert_validity_equivalent(
    source: &PhysicalValidity,
    imported: &PhysicalValidity,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    imported_bindings: &HashMap<u32, &BufferBinding<'_>>,
) {
    match (source, imported) {
        (PhysicalValidity::NoValidityBuffer, PhysicalValidity::NoValidityBuffer) => {},
        (
            PhysicalValidity::ValidityMask { mask: source_mask, null_count: source_null_count },
            PhysicalValidity::ValidityMask { mask: imported_mask, null_count: imported_null_count },
        ) => {
            assert_eq!(source_null_count, imported_null_count);
            assert_binding_equivalent(
                source_bindings.get(&source_mask.get()).expect("missing source validity binding"),
                imported_bindings.get(&imported_mask.get()).expect("missing imported validity binding"),
            );
        },
        (PhysicalValidity::ValidityMask { .. }, PhysicalValidity::NoValidityBuffer) => {
            assert!(validity_mask_is_elidable(source, source_bindings));
        },
        (PhysicalValidity::NoValidityBuffer, PhysicalValidity::ValidityMask { .. }) => {
            assert!(validity_mask_is_elidable(imported, imported_bindings));
        },
    }
}

fn assert_semantic_column_equivalent(
    source: &SemanticColumn,
    imported: &SemanticColumn,
    source_columns: &HashMap<u32, &SemanticColumn>,
    imported_columns: &HashMap<u32, &SemanticColumn>,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    imported_bindings: &HashMap<u32, &BufferBinding<'_>>,
) {
    assert_eq!(source.data_type(), imported.data_type());
    assert_eq!(source.len(), imported.len());
    assert_eq!(source.offset(), imported.offset());
    assert_validity_equivalent(source.validity(), imported.validity(), source_bindings, imported_bindings);

    match (source.layout(), imported.layout()) {
        (ColumnLayout::FixedWidth { values: source_values }, ColumnLayout::FixedWidth { values: imported_values }) => {
            assert_binding_equivalent(
                source_bindings.get(&source_values.get()).expect("missing source fixed-width binding"),
                imported_bindings.get(&imported_values.get()).expect("missing imported fixed-width binding"),
            );
        },
        (
            ColumnLayout::BinaryLike { offsets: source_offsets, data: source_data },
            ColumnLayout::BinaryLike { offsets: imported_offsets, data: imported_data },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_offsets.get()).expect("missing source binary offsets binding"),
                imported_bindings.get(&imported_offsets.get()).expect("missing imported binary offsets binding"),
            );
            assert_binding_equivalent(
                source_bindings.get(&source_data.get()).expect("missing source binary data binding"),
                imported_bindings.get(&imported_data.get()).expect("missing imported binary data binding"),
            );
        },
        (
            ColumnLayout::List { offsets: source_offsets, child: source_child },
            ColumnLayout::List { offsets: imported_offsets, child: imported_child },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_offsets.get()).expect("missing source list offsets binding"),
                imported_bindings.get(&imported_offsets.get()).expect("missing imported list offsets binding"),
            );
            assert_semantic_column_equivalent(
                source_columns.get(&source_child.get()).expect("missing source list child"),
                imported_columns.get(&imported_child.get()).expect("missing imported list child"),
                source_columns,
                imported_columns,
                source_bindings,
                imported_bindings,
            );
        },
        (ColumnLayout::Struct { children: source_children }, ColumnLayout::Struct { children: imported_children }) => {
            assert_eq!(source_children.len(), imported_children.len());
            for (source_child, imported_child) in source_children.iter().zip(imported_children.iter()) {
                assert_semantic_column_equivalent(
                    source_columns.get(&source_child.get()).expect("missing source struct child"),
                    imported_columns.get(&imported_child.get()).expect("missing imported struct child"),
                    source_columns,
                    imported_columns,
                    source_bindings,
                    imported_bindings,
                );
            }
        },
        (
            ColumnLayout::Dictionary { indices: source_indices, keys: source_keys },
            ColumnLayout::Dictionary { indices: imported_indices, keys: imported_keys },
        ) => {
            assert_binding_equivalent(
                source_bindings.get(&source_indices.get()).expect("missing source dictionary indices binding"),
                imported_bindings.get(&imported_indices.get()).expect("missing imported dictionary indices binding"),
            );
            assert_semantic_column_equivalent(
                source_columns.get(&source_keys.get()).expect("missing source dictionary keys"),
                imported_columns.get(&imported_keys.get()).expect("missing imported dictionary keys"),
                source_columns,
                imported_columns,
                source_bindings,
                imported_bindings,
            );
        },
        _ => panic!("layout mismatch: source={:?} imported={:?}", source.layout(), imported.layout()),
    }
}

fn assert_host_column_exports_structurally_equivalent(
    source_export: &PreparedColumnExport<'_>,
    imported_export: &PreparedColumnExport<'_>,
) {
    assert_eq!(source_export.profile(), PreparedExportProfile::Host);
    assert_eq!(imported_export.profile(), PreparedExportProfile::Host);

    let source_columns = semantic_column_map(source_export.semantic_columns());
    let imported_columns = semantic_column_map(imported_export.semantic_columns());
    let source_bindings = binding_map(source_export.bindings());
    let imported_bindings = binding_map(imported_export.bindings());

    assert_semantic_column_equivalent(
        source_export.column(),
        imported_export.column(),
        &source_columns,
        &imported_columns,
        &source_bindings,
        &imported_bindings,
    );
}

fn assert_host_column_exports_equivalent(source: &Column, imported: &Column) {
    let source_export = source.try_prepare_host_column_export().unwrap();
    let imported_export = imported.try_prepare_host_column_export().unwrap();
    assert_host_column_exports_structurally_equivalent(&source_export, &imported_export);
}

fn assert_host_table_exports_equivalent(source: &Table, imported: &Table) {
    let source_export = source.try_prepare_host_table_export().unwrap();
    let imported_export = imported.try_prepare_host_table_export().unwrap();

    assert_eq!(source_export.profile(), PreparedExportProfile::Host);
    assert_eq!(imported_export.profile(), PreparedExportProfile::Host);

    assert_eq!(source_export.table().num_rows(), imported_export.table().num_rows());
    assert_eq!(source_export.table().columns().len(), imported_export.table().columns().len());

    let source_columns = semantic_column_map(source_export.semantic_columns());
    let imported_columns = semantic_column_map(imported_export.semantic_columns());
    let source_bindings = binding_map(source_export.bindings());
    let imported_bindings = binding_map(imported_export.bindings());

    for (source_column, imported_column) in source_export.columns().iter().zip(imported_export.columns().iter()) {
        assert_semantic_column_equivalent(
            source_columns.get(&source_column.id().get()).expect("missing source top-level column"),
            imported_columns.get(&imported_column.id().get()).expect("missing imported top-level column"),
            &source_columns,
            &imported_columns,
            &source_bindings,
            &imported_bindings,
        );
    }
}

fn assert_nullable_i32_materialized(column: &Column, expected: Vec<Option<i32>>) {
    let view = column.testing_view().unwrap();
    let values = view.to_vec::<i32>().unwrap();
    let validity = view.to_host_validity().unwrap().unwrap();
    let materialized =
        values.into_iter().zip(validity).map(|(value, is_valid)| is_valid.then_some(value)).collect::<Vec<_>>();
    assert_eq!(materialized, expected);
    assert_eq!(view.offset().unwrap(), 0);
}

fn two_numeric_columns_table(runtime: &Runtime) -> Table {
    let first = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let second = runtime.column_from_slice(&[10i64, 20, 30]).unwrap();
    Table::from_columns(vec![first, second]).unwrap()
}

fn zero_column_table(runtime: &Runtime) -> Table {
    let frame = named_numeric_frame(runtime).select(std::iter::empty::<&str>()).unwrap();
    let (table, names) = frame.try_into_table().unwrap();
    assert!(names.is_empty());
    table
}

fn assert_two_numeric_columns_table(table: &Table) {
    let view = table.testing_view().unwrap();
    assert_eq!(view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(view.column(1).unwrap().to_vec::<i64>().unwrap(), vec![10, 20, 30]);
}

fn named_numeric_frame(runtime: &Runtime) -> DataFrame {
    let ids = runtime.series(&[1i32, 2, 3]).unwrap();
    let values = runtime.series(&[10i64, 20, 30]).unwrap();
    DataFrame::from_series([("id", ids), ("value", values)]).unwrap()
}

fn assert_named_numeric_dataframe(imported: &DataFrame, left_name: &str, right_name: &str) {
    assert_eq!(imported.column_names(), vec![left_name, right_name]);
    assert_eq!(imported.series(left_name).unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(imported.series(right_name).unwrap().to_vec_strict::<i64>().unwrap(), vec![10, 20, 30]);
}

fn dictionary_backed_nested_frame() -> DataFrame {
    let frame = nested_source_frame();
    let category = frame.series("category").unwrap();
    let nested = frame.series("nested").unwrap();
    let point = frame.series("point").unwrap();
    let id = frame.series("id").unwrap();
    let encoded = dictionary::encode(&category.testing_view().unwrap(), TypeId::Int32).unwrap();
    let category = Series::from_column(encoded).unwrap();
    DataFrame::from_series([("category", category), ("nested", nested), ("point", point), ("id", id)]).unwrap()
}

mod column_import_contracts {
    use super::*;

    fn prepared_device_column_export_import_copies_data_and_uses_destination_runtime() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = source_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
        let export = source.try_prepare_device_column_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        assert_eq!(imported.testing_view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
        assert_column_imported_into_runtime(&imported, &destination_runtime);
    }

    fn series_import_from_prepared_column_export_returns_unnamed_series() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = source_runtime.column_from_strs(&["one", "two", "three"]).unwrap();
        let export = source.try_prepare_device_column_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            Series::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap()
        });
        assert_eq!(imported.to_strings_strict().unwrap(), vec!["one", "two", "three"]);
        assert_column_imported_into_runtime(imported.as_column(), &destination_runtime);
    }

    fn prepared_host_column_export_import_copies_data_and_uses_destination_runtime() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source =
            source_runtime.column_from_slice_nullable(&[10i32, 20, 30, 40], &[true, false, true, true]).unwrap();
        let sliced = source.testing_slice_view(1, 2).unwrap();
        let export = sliced.try_prepare_host_column_export().unwrap();
        let destination_runtime = common::sample_runtime();

        let imported = reset_exec_observation(|| {
            Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        let normalized = source_runtime.column_from_slice_nullable(&[20i32, 30], &[false, true]).unwrap();
        assert_column_bound_to_runtime(&imported, &destination_runtime);
        assert_host_column_exports_equivalent(&normalized, &imported);
    }

    fn prepared_sliced_nullable_column_export_import_preserves_active_range() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source =
            source_runtime.column_from_slice_nullable(&[10i32, 20, 30, 40], &[true, false, true, true]).unwrap();
        let sliced = source.testing_slice_view(1, 2).unwrap();
        let export = sliced.try_prepare_device_column_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported =
            Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap();

        assert_nullable_i32_materialized(&imported, vec![None, Some(30)]);
    }

    fn prepared_dictionary_column_export_import_round_trips_string_keys() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = source_runtime.column_from_strs(&["beta", "alpha", "beta", "gamma"]).unwrap();
        let dictionary = dictionary::encode(&source.testing_view().unwrap(), TypeId::Int32).unwrap();
        let export = dictionary.try_prepare_device_column_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        assert_column_imported_into_runtime(&imported, &destination_runtime);
        assert_host_column_exports_equivalent(&dictionary, &imported);
    }

    fn prepared_list_column_export_import_round_trips_recursive_nested_children() {
        let runtime = Runtime::builder().build().unwrap();
        let frame = common::nested_dataframe(
            &runtime,
            vec![(
                "nested",
                common::nested_int64_list_array(&[vec![vec![1, 2], vec![3]], vec![vec![], vec![4, 5]], vec![vec![6]]]),
            )],
        );
        let nested = frame.series("nested").unwrap();
        let source = nested.as_column();
        let export = source.try_prepare_device_column_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        assert_column_imported_into_runtime(&imported, &destination_runtime);
        assert_host_column_exports_equivalent(source, &imported);
    }

    fn prepared_sliced_string_column_export_import_preserves_active_range() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = source_runtime
            .column_from_strs_nullable(&["zero", "one", "two", "three"], &[true, false, true, true])
            .unwrap();
        let sliced = source.testing_slice_view(1, 2).unwrap();
        let export = sliced.try_prepare_device_column_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        let normalized = source_runtime.column_from_strs_nullable(&["one", "two"], &[false, true]).unwrap();
        assert_column_bound_to_runtime(&imported, &destination_runtime);
        assert_host_column_exports_equivalent(&normalized, &imported);
    }

    fn prepared_host_sliced_string_column_export_import_preserves_active_range() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = source_runtime
            .column_from_strs_nullable(&["zero", "one", "two", "three"], &[true, false, true, true])
            .unwrap();
        let sliced = source.testing_slice_view(1, 2).unwrap();
        let export = sliced.try_prepare_host_column_export().unwrap();
        let destination_runtime = common::sample_runtime();

        let imported = reset_exec_observation(|| {
            Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        let normalized = source_runtime.column_from_strs_nullable(&["one", "two"], &[false, true]).unwrap();
        assert_column_bound_to_runtime(&imported, &destination_runtime);
        assert_host_column_exports_equivalent(&normalized, &imported);
    }

    #[test]
    fn import_tests_cases_column_import_contracts() {
        run_named_cases(&[
            (
                "prepared_device_column_export_import_copies_data_and_uses_destination_runtime",
                prepared_device_column_export_import_copies_data_and_uses_destination_runtime,
            ),
            (
                "series_import_from_prepared_column_export_returns_unnamed_series",
                series_import_from_prepared_column_export_returns_unnamed_series,
            ),
            (
                "prepared_host_column_export_import_copies_data_and_uses_destination_runtime",
                prepared_host_column_export_import_copies_data_and_uses_destination_runtime,
            ),
            (
                "prepared_sliced_nullable_column_export_import_preserves_active_range",
                prepared_sliced_nullable_column_export_import_preserves_active_range,
            ),
            (
                "prepared_dictionary_column_export_import_round_trips_string_keys",
                prepared_dictionary_column_export_import_round_trips_string_keys,
            ),
            (
                "prepared_list_column_export_import_round_trips_recursive_nested_children",
                prepared_list_column_export_import_round_trips_recursive_nested_children,
            ),
            (
                "prepared_sliced_string_column_export_import_preserves_active_range",
                prepared_sliced_string_column_export_import_preserves_active_range,
            ),
            (
                "prepared_host_sliced_string_column_export_import_preserves_active_range",
                prepared_host_sliced_string_column_export_import_preserves_active_range,
            ),
        ]);
    }
}

mod table_import_contracts {
    use super::*;

    fn prepared_device_table_export_import_copies_all_columns_and_uses_destination_runtime() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = two_numeric_columns_table(&source_runtime);
        let export = source.try_prepare_device_table_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            Table::try_import_from_prepared_table_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        assert_two_numeric_columns_table(&imported);
        assert_table_imported_into_runtime(&imported, &destination_runtime);
    }

    fn prepared_device_table_export_import_round_trips_zero_column_non_empty_shape() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = zero_column_table(&source_runtime);
        let export = source.try_prepare_device_table_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        assert_eq!(export.table().columns().len(), 0);
        assert_eq!(export.table().num_rows(), 3);
        let imported =
            Table::try_import_from_prepared_table_export_into_runtime(&export, &destination_runtime).unwrap();

        assert_eq!(imported.num_columns().unwrap(), 0);
        assert_eq!(imported.num_rows().unwrap(), 3);
        assert_table_bound_to_runtime(&imported, &destination_runtime);
    }

    fn prepared_host_table_export_import_round_trips_zero_column_non_empty_shape() {
        let source_runtime = Runtime::builder().build().unwrap();
        let source = zero_column_table(&source_runtime);
        let export = source.try_prepare_host_table_export().unwrap();
        let destination_runtime = common::sample_runtime();

        assert_eq!(export.table().columns().len(), 0);
        assert_eq!(export.table().num_rows(), 3);
        let imported =
            Table::try_import_from_prepared_table_export_into_runtime(&export, &destination_runtime).unwrap();

        assert_eq!(imported.num_columns().unwrap(), 0);
        assert_eq!(imported.num_rows().unwrap(), 3);
        assert_table_bound_to_runtime(&imported, &destination_runtime);
    }

    fn prepared_device_table_export_import_round_trips_recursive_nested_columns_into_destination_runtime() {
        let frame = dictionary_backed_nested_frame();
        let (source, _names) = frame.try_into_table().unwrap();
        let export = source.try_prepare_device_table_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            Table::try_import_from_prepared_table_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        assert_table_imported_into_runtime(&imported, &destination_runtime);
        assert_host_table_exports_equivalent(&source, &imported);
    }

    fn prepared_host_table_export_import_round_trips_recursive_nested_columns_into_destination_runtime() {
        let frame = dictionary_backed_nested_frame();
        let (source, _names) = frame.try_into_table().unwrap();
        let export = source.try_prepare_host_table_export().unwrap();
        let destination_runtime = common::sample_runtime();

        let imported = reset_exec_observation(|| {
            Table::try_import_from_prepared_table_export_into_runtime(&export, &destination_runtime).unwrap()
        });

        assert_table_imported_into_runtime(&imported, &destination_runtime);
        assert_host_table_exports_equivalent(&source, &imported);
    }

    #[test]
    fn import_tests_cases_table_import_contracts() {
        run_named_cases(&[
            (
                "prepared_device_table_export_import_copies_all_columns_and_uses_destination_runtime",
                prepared_device_table_export_import_copies_all_columns_and_uses_destination_runtime,
            ),
            (
                "prepared_device_table_export_import_round_trips_zero_column_non_empty_shape",
                prepared_device_table_export_import_round_trips_zero_column_non_empty_shape,
            ),
            (
                "prepared_host_table_export_import_round_trips_zero_column_non_empty_shape",
                prepared_host_table_export_import_round_trips_zero_column_non_empty_shape,
            ),
            (
                "prepared_device_table_export_import_round_trips_recursive_nested_columns_into_destination_runtime",
                prepared_device_table_export_import_round_trips_recursive_nested_columns_into_destination_runtime,
            ),
            (
                "prepared_host_table_export_import_round_trips_recursive_nested_columns_into_destination_runtime",
                prepared_host_table_export_import_round_trips_recursive_nested_columns_into_destination_runtime,
            ),
        ]);
    }
}

mod dataframe_import_contracts {
    use super::*;

    fn dataframe_import_from_prepared_table_export_uses_root_names_for_recursive_nested_layouts() {
        let frame = nested_source_frame();

        let export_frame = frame.clone();
        let export = export_frame.try_prepare_device_table_export().unwrap();
        let (source_table, source_names) = frame.try_into_table().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = reset_exec_observation(|| {
            DataFrame::try_import_from_prepared_table_export_into_runtime(&export, None, &destination_runtime).unwrap()
        });
        let (imported_table, imported_names) = imported.try_into_table().unwrap();

        assert_eq!(imported_names, source_names);
        assert_host_table_exports_equivalent(&source_table, &imported_table);
    }

    fn dataframe_import_from_prepared_table_export_with_explicit_names_uses_caller_names_even_when_root_has_names() {
        let runtime = Runtime::builder().build().unwrap();
        let frame = named_numeric_frame(&runtime);
        let export = frame.try_prepare_device_table_export().unwrap();
        let destination_runtime = common::sample_runtime();
        let _event_lock = common::lock_event_observation();

        let imported = DataFrame::try_import_from_prepared_table_export_into_runtime(
            &export,
            Some(vec!["left".to_string(), "right".to_string()]),
            &destination_runtime,
        )
        .unwrap();

        assert_named_numeric_dataframe(&imported, "left", "right");
    }

    fn dataframe_import_from_prepared_table_export_without_root_names_fails() {
        let runtime = Runtime::builder().build().unwrap();
        let table = two_numeric_columns_table(&runtime);
        let export = table.try_prepare_device_table_export().unwrap();
        let destination_runtime = common::sample_runtime();

        let err = DataFrame::try_import_from_prepared_table_export_into_runtime(&export, None, &destination_runtime)
            .unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("interop::DataFrame::try_import_from_prepared_table_export_into_runtime"),
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("schema_names"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("missing_schema_names"));
        assert_eq!(
            interop_error_context_value(&err, "remedy"),
            Some("pass Some(names) when the root does not carry schema names"),
        );
    }

    #[test]
    fn import_tests_cases_dataframe_import_contracts() {
        run_named_cases(&[
            ("dataframe_import_from_prepared_table_export_uses_root_names_for_recursive_nested_layouts", dataframe_import_from_prepared_table_export_uses_root_names_for_recursive_nested_layouts),
            ("dataframe_import_from_prepared_table_export_with_explicit_names_uses_caller_names_even_when_root_has_names", dataframe_import_from_prepared_table_export_with_explicit_names_uses_caller_names_even_when_root_has_names),
            ("dataframe_import_from_prepared_table_export_without_root_names_fails", dataframe_import_from_prepared_table_export_without_root_names_fails),
        ]);
    }
}

mod error_mapping_contracts {
    use super::*;

    fn prepared_host_column_export_constructor_rejects_device_bindings() {
        let runtime = Runtime::builder().build().unwrap();
        let source = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
        let device_export = source.try_prepare_device_column_export().unwrap();

        let err = PreparedColumnExport::try_new(
            device_export.column().id(),
            device_export.semantic_columns().to_vec(),
            device_export.bindings().to_vec(),
            PreparedAccess::Host,
        )
        .unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::PlacementUnsupported);
    }

    fn prepared_device_column_import_maps_shared_completion_sync_failures_to_readiness_unavailable() {
        use std::sync::Arc;

        use rapids_interop::{
            ActiveLeaseBackend, CompletionAuthority, CompletionWitness, CompletionWitnessBackend, InteropError,
            LeaseBackend, ProducerReadiness,
        };

        struct SyncFailureWitness;

        impl CompletionWitnessBackend for SyncFailureWitness {
            fn is_ready(&self) -> std::result::Result<bool, InteropError> {
                Ok(false)
            }

            fn synchronize(&self) -> std::result::Result<(), InteropError> {
                Err(InteropError::new(InteropErrorKind::InvalidView, "test shared completion synchronize failure")
                    .with_context("phase", "synchronize"))
            }
        }

        struct SyncFailureBackend;

        impl ActiveLeaseBackend for SyncFailureBackend {
            fn complete(
                &self,
                _readiness: ProducerReadiness<'_>,
                _completion: CompletionAuthority<'_>,
            ) -> std::result::Result<Option<CompletionWitness>, InteropError> {
                Ok(Some(CompletionWitness::new(Arc::new(SyncFailureWitness) as Arc<dyn CompletionWitnessBackend>)))
            }

            fn drop_incomplete(
                &self,
                _readiness: ProducerReadiness<'_>,
                _completion: CompletionAuthority<'_>,
            ) -> std::result::Result<(), InteropError> {
                Ok(())
            }
        }

        let source_runtime = Runtime::builder().build().unwrap();
        let source = source_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
        let export = column_export_with_lease_backend_for_test(
            &source.try_prepare_device_column_export().unwrap(),
            LeaseBackend::new(SyncFailureBackend),
        )
        .unwrap();
        let destination_runtime = Runtime::builder().with_device(source_runtime.device()).build().unwrap();

        let err = match Column::try_import_from_prepared_column_export_into_runtime(&export, &destination_runtime) {
            Ok(_) => panic!("expected device import completion synchronization failure"),
            Err(err) => err,
        };

        assert_eq!(err.kind(), InteropErrorKind::ReadinessUnavailable);
        assert_eq!(err.to_string(), "test shared completion synchronize failure");
        assert!(!err.to_string().contains("InvalidView"));
        assert!(!err.to_string().contains("context:"));
        assert!(!err.to_string().contains("phase:"));
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("interop::Column::try_import_from_prepared_column_export_into_runtime")
        );
        assert_eq!(interop_error_context_value(&err, "phase"), Some("synchronize"));
    }

    #[test]
    fn import_tests_cases_error_mapping_contracts() {
        run_named_cases(&[
            (
                "prepared_host_column_export_constructor_rejects_device_bindings",
                prepared_host_column_export_constructor_rejects_device_bindings,
            ),
            (
                "prepared_device_column_import_maps_shared_completion_sync_failures_to_readiness_unavailable",
                prepared_device_column_import_maps_shared_completion_sync_failures_to_readiness_unavailable,
            ),
        ]);
    }
}
