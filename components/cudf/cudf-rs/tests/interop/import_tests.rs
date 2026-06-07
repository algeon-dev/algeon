use std::collections::HashMap;

use cudf::{
    Column, DataFrame, Runtime, Table, TypeId,
    interop::{
        BufferBinding, ColumnLayout, ExportProfile, InteropErrorKind, PhysicalValidity, PreparedColumnExport,
        SemanticColumn,
    },
    io::json::JsonReaderOptions,
    ops::dictionary,
};
use tempfile::TempDir;

use crate::common;

fn binding_bytes(binding: &BufferBinding<'_>) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr().as_ptr() as *const u8, binding.bytes().byte_len()) }
        .to_vec()
}

fn read_json_dataframe(filename: &str, rows: &str) -> DataFrame {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(filename);
    std::fs::write(&path, rows).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::read_json_with_options(JsonReaderOptions::builder().filepath(&path).with_runtime(&runtime).build())
        .unwrap()
}

fn binding_map<'a>(bindings: &'a [BufferBinding<'a>]) -> HashMap<u32, &'a BufferBinding<'a>> {
    bindings.iter().map(|binding| (binding.id().get(), binding)).collect()
}

fn semantic_column_map<'a>(columns: &'a [SemanticColumn]) -> HashMap<u32, &'a SemanticColumn> {
    columns.iter().map(|column| (column.id().get(), column)).collect()
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

fn schema_nullability_is_elidable(
    source: &SemanticColumn,
    imported: &SemanticColumn,
    source_bindings: &HashMap<u32, &BufferBinding<'_>>,
    imported_bindings: &HashMap<u32, &BufferBinding<'_>>,
) -> bool {
    match (source.schema_nullability(), imported.schema_nullability()) {
        (cudf::interop::SchemaNullability::Nullable, cudf::interop::SchemaNullability::NonNullable) => {
            validity_mask_is_elidable(source.validity(), source_bindings)
                && matches!(imported.validity(), PhysicalValidity::NoValidityBuffer)
        },
        (cudf::interop::SchemaNullability::NonNullable, cudf::interop::SchemaNullability::Nullable) => {
            matches!(source.validity(), PhysicalValidity::NoValidityBuffer)
                && validity_mask_is_elidable(imported.validity(), imported_bindings)
        },
        _ => false,
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
    if source.schema_nullability() != imported.schema_nullability() {
        assert!(
            schema_nullability_is_elidable(source, imported, source_bindings, imported_bindings),
            "schema nullability mismatch: source={source:?} imported={imported:?}"
        );
    }
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
    assert_eq!(source_export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(imported_export.diagnostics().profile(), ExportProfile::PreparedHostExport);

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

    assert_eq!(source_export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(imported_export.diagnostics().profile(), ExportProfile::PreparedHostExport);

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

#[test]
fn borrowed_column_view_import_copies_data_and_uses_destination_runtime() {
    let source_runtime = Runtime::builder().build().unwrap();
    let source = Column::from_slice_with_runtime(&source_runtime, &[1i32, 2, 3]).unwrap();
    let source_view = source.view().unwrap();
    let borrowed = source_view.try_borrow_column_view().unwrap();
    let destination_runtime = common::sample_runtime();

    common::reset_exec_observation();
    let imported = Column::try_from_borrowed_column_view_with_runtime(&borrowed, &destination_runtime).unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(
        imported.__debug_execution_domain_stream_value(),
        destination_runtime.stream().__debug_raw_handle_value()
    );
    assert_eq!(
        imported.__debug_execution_domain_mr_value(),
        destination_runtime.memory_resource().__debug_raw_handle_value()
    );
    assert_eq!(common::last_exec_device(), destination_runtime.device().ordinal() as i32);
    assert_eq!(common::last_exec_stream_value(), destination_runtime.stream().__debug_raw_handle_value());
    assert_eq!(common::last_exec_mr_value(), destination_runtime.memory_resource().__debug_raw_handle_value());
}

#[test]
fn prepared_device_column_export_import_copies_data_and_uses_destination_runtime() {
    let source_runtime = Runtime::builder().build().unwrap();
    let source = Column::from_slice_with_runtime(&source_runtime, &[1i32, 2, 3]).unwrap();
    let export = source.try_prepare_device_column_export().unwrap();
    let destination_runtime = common::sample_runtime();

    common::reset_exec_observation();
    let imported = Column::try_from_prepared_column_export_with_runtime(&export, &destination_runtime).unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(
        imported.__debug_execution_domain_stream_value(),
        destination_runtime.stream().__debug_raw_handle_value()
    );
    assert_eq!(
        imported.__debug_execution_domain_mr_value(),
        destination_runtime.memory_resource().__debug_raw_handle_value()
    );
    assert_eq!(common::last_exec_device(), destination_runtime.device().ordinal() as i32);
    assert_eq!(common::last_exec_stream_value(), destination_runtime.stream().__debug_raw_handle_value());
    assert_eq!(common::last_exec_mr_value(), destination_runtime.memory_resource().__debug_raw_handle_value());
}

#[test]
fn borrowed_sliced_nullable_column_import_preserves_active_range() {
    let source_runtime = Runtime::builder().build().unwrap();
    let source =
        Column::from_slice_nullable_with_runtime(&source_runtime, &[10i32, 20, 30, 40], &[true, false, true, true])
            .unwrap();
    let sliced = source.slice_view(1, 2).unwrap();
    let borrowed = sliced.try_borrow_column_view().unwrap();
    let destination_runtime = common::sample_runtime();

    let imported = Column::try_from_borrowed_column_view_with_runtime(&borrowed, &destination_runtime).unwrap();

    let imported_view = imported.view().unwrap();
    let values = imported_view.to_vec::<i32>().unwrap();
    let validity = imported_view.validity().unwrap().unwrap();
    let materialized =
        values.into_iter().zip(validity).map(|(value, is_valid)| is_valid.then_some(value)).collect::<Vec<_>>();
    assert_eq!(materialized, vec![None, Some(30)]);
    assert_eq!(imported_view.offset().unwrap(), 0);
}

#[test]
fn prepared_sliced_string_column_export_import_preserves_active_range() {
    let source_runtime = Runtime::builder().build().unwrap();
    let source = Column::from_strings_nullable_with_runtime(
        &source_runtime,
        &["zero", "one", "two", "three"],
        &[true, false, true, true],
    )
    .unwrap();
    let sliced = source.slice_view(1, 2).unwrap();
    let export = sliced.try_prepare_device_column_export().unwrap();
    let destination_runtime = common::sample_runtime();

    common::reset_exec_observation();
    let imported = Column::try_from_prepared_column_export_with_runtime(&export, &destination_runtime).unwrap();

    assert_eq!(
        imported.__debug_execution_domain_stream_value(),
        destination_runtime.stream().__debug_raw_handle_value()
    );
    assert_eq!(
        imported.__debug_execution_domain_mr_value(),
        destination_runtime.memory_resource().__debug_raw_handle_value()
    );
    assert_eq!(common::last_exec_device(), destination_runtime.device().ordinal() as i32);
    assert_eq!(common::last_exec_stream_value(), destination_runtime.stream().__debug_raw_handle_value());
    assert_eq!(common::last_exec_mr_value(), destination_runtime.memory_resource().__debug_raw_handle_value());

    let normalized =
        Column::from_strings_nullable_with_runtime(&source_runtime, &["one", "two"], &[false, true]).unwrap();
    assert_host_column_exports_equivalent(&normalized, &imported);
}

#[test]
fn borrowed_table_view_import_copies_all_columns_and_uses_destination_runtime() {
    let source_runtime = Runtime::builder().build().unwrap();
    let first = Column::from_slice_with_runtime(&source_runtime, &[1i32, 2, 3]).unwrap();
    let second = Column::from_slice_with_runtime(&source_runtime, &[10i64, 20, 30]).unwrap();
    let source = Table::from_columns(vec![first, second]).unwrap();
    let source_view = source.view().unwrap();
    let borrowed = source_view.try_borrow_table_view().unwrap();
    let destination_runtime = common::sample_runtime();

    common::reset_exec_observation();
    let imported = Table::try_from_borrowed_table_view_with_runtime(&borrowed, &destination_runtime).unwrap();

    let imported_view = imported.view().unwrap();
    let imported_first = imported_view.column(0).unwrap();
    let imported_second = imported_view.column(1).unwrap();
    assert_eq!(imported_first.to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(imported_second.to_vec::<i64>().unwrap(), vec![10, 20, 30]);
    assert_eq!(
        imported.__debug_execution_domain_stream_value(),
        destination_runtime.stream().__debug_raw_handle_value()
    );
    assert_eq!(
        imported.__debug_execution_domain_mr_value(),
        destination_runtime.memory_resource().__debug_raw_handle_value()
    );
    assert_eq!(common::last_exec_device(), destination_runtime.device().ordinal() as i32);
    assert_eq!(common::last_exec_stream_value(), destination_runtime.stream().__debug_raw_handle_value());
    assert_eq!(common::last_exec_mr_value(), destination_runtime.memory_resource().__debug_raw_handle_value());
}

#[test]
fn prepared_device_table_export_import_round_trips_recursive_nested_columns_into_destination_runtime() {
    let frame = read_json_dataframe(
        "prepared_import_nested.jsonl",
        "{\"category\":\"beta\",\"nested\":[[1,2],[3]],\"point\":{\"coords\":[1,2],\"label\":\"one\"},\"id\":10}\n\
         {\"category\":\"alpha\",\"nested\":[[],[4,5]],\"point\":{\"coords\":[],\"label\":\"two\"},\"id\":20}\n\
         {\"category\":\"beta\",\"nested\":[[6]],\"point\":{\"coords\":[3],\"label\":\"three\"},\"id\":30}\n",
    );

    let category = frame.series("category").unwrap();
    let nested = frame.series("nested").unwrap();
    let point = frame.series("point").unwrap();
    let id = frame.series("id").unwrap();
    let encoded = dictionary::encode(&category.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let category = cudf::Series::from_column(encoded).unwrap().rename("category");
    let frame =
        DataFrame::from_columns([("category", category), ("nested", nested), ("point", point), ("id", id)]).unwrap();
    let (source, _names) = frame.try_into_table().unwrap();
    let export = source.try_prepare_device_table_export().unwrap();
    let destination_runtime = common::sample_runtime();

    common::reset_exec_observation();
    let imported = Table::try_from_prepared_table_export_with_runtime(&export, &destination_runtime).unwrap();

    assert_eq!(
        imported.__debug_execution_domain_stream_value(),
        destination_runtime.stream().__debug_raw_handle_value()
    );
    assert_eq!(
        imported.__debug_execution_domain_mr_value(),
        destination_runtime.memory_resource().__debug_raw_handle_value()
    );
    assert_eq!(common::last_exec_device(), destination_runtime.device().ordinal() as i32);
    assert_eq!(common::last_exec_stream_value(), destination_runtime.stream().__debug_raw_handle_value());
    assert_eq!(common::last_exec_mr_value(), destination_runtime.memory_resource().__debug_raw_handle_value());
    assert_host_table_exports_equivalent(&source, &imported);
}

#[test]
fn prepared_host_column_export_import_is_rejected_as_out_of_scope() {
    let runtime = Runtime::builder().build().unwrap();
    let source = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let export = source.try_prepare_host_column_export().unwrap();
    let destination_runtime = common::sample_runtime();

    let err = match Column::try_from_prepared_column_export_with_runtime(&export, &destination_runtime) {
        Ok(_) => panic!("expected prepared host export import to be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}
