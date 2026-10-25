//! Checks that reading a data file by column name returns the Iceberg columns.
//!
//! Iceberg binds file columns by field id, falling back to the table's name
//! mapping for files written without ids (PyIceberg `add_files` registers
//! such files). DataFusion's Parquet reader and the native reader bind by
//! name. A file is readable here only when both rules select the same file
//! column for every projected field, level by level through structs, lists
//! and maps. A field absent under both rules is read as null, which is
//! correct only for an optional field whose value is not kept in partition
//! data.

use std::collections::HashSet;

use datafusion::{
    arrow::datatypes::{DataType, Field, Fields},
    parquet::arrow::PARQUET_FIELD_ID_META_KEY,
};
use iceberg::spec::{MappedField, NestedField, PartitionSpec, Schema as IcebergSchema, Transform, Type};

use crate::error::UnsupportedReason;

/// Why one projected field of one file is unreadable, with the Iceberg
/// field it concerns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BindingRejection {
    pub(crate) reason: UnsupportedReason,
    pub(crate) field: String,
}

/// The binding facts of one data file.
pub(crate) struct FileBinding<'a> {
    /// Field ids of identity partition sources, whose values Iceberg keeps
    /// in partition data when the file lacks the column.
    pub(crate) identity_partition_sources: HashSet<i32>,
    /// The table's name mapping, consulted for files without field ids.
    pub(crate) name_mapping: Option<&'a [MappedField]>,
}

impl FileBinding<'_> {
    /// Checks every field in `projected_field_ids` of `schema` against the
    /// columns of the file's Arrow schema `file_fields`, as Parquet's Arrow
    /// conversion produced it (with `PARQUET:field_id` metadata).
    pub(crate) fn check(
        &self,
        schema: &IcebergSchema,
        projected_field_ids: &[i32],
        file_fields: &Fields,
    ) -> Result<(), BindingRejection> {
        let has_field_ids = fields_have_ids(file_fields);
        let checker = Checker { identity_partition_sources: &self.identity_partition_sources, has_field_ids };
        for field_id in projected_field_ids {
            let Some(field) = schema.as_struct().field_by_id(*field_id) else {
                continue;
            };
            let mapping = match (has_field_ids, self.name_mapping) {
                (true, _) => Vec::new(),
                (false, Some(mapping)) => mapping.iter().collect(),
                (false, None) => return Err(checker.rejection(UnsupportedReason::MissingFieldIds, &field.name)),
            };
            checker.check_struct_member(field, file_fields, &mapping, &field.name)?;
        }
        Ok(())
    }
}

/// Name-mapping entries of one level. Empty for files with field ids.
type Mapping<'a> = [&'a MappedField];

struct Checker<'a> {
    identity_partition_sources: &'a HashSet<i32>,
    has_field_ids: bool,
}

impl Checker<'_> {
    /// Checks one struct member, found by name and by id among `file_fields`.
    fn check_struct_member(
        &self,
        field: &NestedField,
        file_fields: &Fields,
        mapping: &Mapping<'_>,
        path: &str,
    ) -> Result<(), BindingRejection> {
        let by_name = file_fields.iter().position(|file_field| file_field.name() == &field.name);
        let mapped = mapping.iter().copied().find(|entry| entry.field_id() == Some(field.id));
        let by_id = if self.has_field_ids {
            file_fields.iter().position(|file_field| field_id(file_field) == Some(field.id))
        } else {
            mapped.and_then(|entry| {
                file_fields.iter().position(|file_field| entry.names().iter().any(|name| name == file_field.name()))
            })
        };
        let mismatch =
            if self.has_field_ids { UnsupportedReason::FieldIdMismatch } else { UnsupportedReason::MissingFieldIds };
        match (by_name, by_id) {
            (None, None) => {
                if self.identity_partition_sources.contains(&field.id) {
                    Err(self.rejection(UnsupportedReason::PartitionColumnAbsent, path))
                } else if field.required {
                    Err(self.rejection(UnsupportedReason::RequiredColumnAbsent, path))
                } else {
                    Ok(())
                }
            },
            (Some(by_name), Some(by_id)) if by_name == by_id => {
                self.check_nested(&field.field_type, &file_fields[by_name], &children(mapped), path)
            },
            _ => Err(self.rejection(mismatch, path)),
        }
    }

    /// Checks the children of a column both rules selected.
    fn check_nested(
        &self,
        field_type: &Type,
        file_field: &Field,
        mapping: &Mapping<'_>,
        path: &str,
    ) -> Result<(), BindingRejection> {
        match (field_type, file_field.data_type()) {
            (Type::Struct(struct_type), DataType::Struct(file_children)) => {
                for child in struct_type.fields() {
                    let child_path = format!("{path}.{}", child.name);
                    self.check_struct_member(child, file_children, mapping, &child_path)?;
                }
                Ok(())
            },
            (Type::List(list), DataType::List(element) | DataType::LargeList(element)) => {
                self.check_positional(&list.element_field, element, mapping, path)
            },
            (Type::Map(map), DataType::Map(entries, _)) => {
                let DataType::Struct(entry_fields) = entries.data_type() else {
                    return Ok(());
                };
                let [key, value] = entry_fields.iter().collect::<Vec<_>>()[..] else {
                    return Ok(());
                };
                self.check_positional(&map.key_field, key, mapping, path)?;
                self.check_positional(&map.value_field, value, mapping, path)
            },
            _ => Ok(()),
        }
    }

    /// List elements and map keys and values bind by position, so only
    /// their field ids, when the file has them, and their children are
    /// compared.
    fn check_positional(
        &self,
        field: &NestedField,
        file_field: &Field,
        mapping: &Mapping<'_>,
        path: &str,
    ) -> Result<(), BindingRejection> {
        if self.has_field_ids && field_id(file_field).is_some_and(|id| id != field.id) {
            return Err(self.rejection(UnsupportedReason::FieldIdMismatch, path));
        }
        let mapped = mapping.iter().copied().find(|entry| entry.field_id() == Some(field.id));
        self.check_nested(&field.field_type, file_field, &children(mapped), path)
    }

    fn rejection(&self, reason: UnsupportedReason, path: &str) -> BindingRejection {
        BindingRejection { reason, field: path.to_string() }
    }
}

fn children(entry: Option<&MappedField>) -> Vec<&MappedField> {
    entry.map(|entry| entry.fields().iter().map(AsRef::as_ref).collect()).unwrap_or_default()
}

/// Field ids of the identity partition sources of a partition spec.
pub(crate) fn identity_partition_sources(spec: &PartitionSpec) -> HashSet<i32> {
    spec.fields().iter().filter(|field| field.transform == Transform::Identity).map(|field| field.source_id).collect()
}

fn field_id(field: &Field) -> Option<i32> {
    field.metadata().get(PARQUET_FIELD_ID_META_KEY).and_then(|id| id.parse().ok())
}

/// Whether any column of the file, at any level, carries a field id.
fn fields_have_ids(fields: &Fields) -> bool {
    fields.iter().any(|field| field_id(field).is_some() || data_type_has_ids(field.data_type()))
}

fn data_type_has_ids(data_type: &DataType) -> bool {
    match data_type {
        DataType::Struct(children) => fields_have_ids(children),
        DataType::List(child) | DataType::LargeList(child) | DataType::Map(child, _) => {
            field_id(child).is_some() || data_type_has_ids(child.data_type())
        },
        _ => false,
    }
}

/// The first projected column that the file stores with another type than
/// the table schema `table_fields` declares, as after Iceberg promotes `int`
/// to `long` or widens a decimal. DataFusion's reader casts such a column,
/// the native reader decodes the stored type, so the scan rejects the file.
/// String and binary types compare by family, because Iceberg, Parquet
/// writers and DataFusion choose different widths and view types for them;
/// a dictionary compares by its value type. Columns absent from the file are
/// the binding check's concern.
pub(crate) fn column_with_stored_type_mismatch(
    schema: &IcebergSchema,
    projected_field_ids: &[i32],
    table_fields: &Fields,
    file_fields: &Fields,
) -> Option<String> {
    projected_field_ids.iter().find_map(|field_id| {
        let name = &schema.as_struct().field_by_id(*field_id)?.name;
        let table_field = table_fields.iter().find(|field| field.name() == name)?;
        let file_field = file_fields.iter().find(|field| field.name() == name)?;
        (!same_stored_type(table_field.data_type(), file_field.data_type())).then(|| name.clone())
    })
}

fn same_stored_type(table: &DataType, file: &DataType) -> bool {
    match (table, file) {
        (_, DataType::Dictionary(_, value)) => same_stored_type(table, value),
        (
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
        )
        | (
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
        ) => true,
        (
            DataType::List(table)
            | DataType::LargeList(table)
            | DataType::ListView(table)
            | DataType::LargeListView(table),
            DataType::List(file) | DataType::LargeList(file) | DataType::ListView(file) | DataType::LargeListView(file),
        ) => same_stored_type(table.data_type(), file.data_type()),
        (DataType::FixedSizeList(table, table_len), DataType::FixedSizeList(file, file_len)) => {
            table_len == file_len && same_stored_type(table.data_type(), file.data_type())
        },
        (DataType::Map(table, _), DataType::Map(file, _)) => same_stored_type(table.data_type(), file.data_type()),
        (DataType::Struct(table), DataType::Struct(file)) => table.iter().all(|table_child| {
            file.iter()
                .find(|file_child| file_child.name() == table_child.name())
                .is_none_or(|file_child| same_stored_type(table_child.data_type(), file_child.data_type()))
        }),
        // Iceberg names UTC `+00:00`; writers also record `UTC`.
        (DataType::Timestamp(table_unit, table_zone), DataType::Timestamp(file_unit, file_zone)) => {
            table_unit == file_unit && table_zone.is_some() == file_zone.is_some()
        },
        (table, file) => table == file,
    }
}
