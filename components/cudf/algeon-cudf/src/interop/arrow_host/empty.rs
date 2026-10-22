//! Zero-row device frames typed by an Arrow schema, built without host data.

use std::sync::Arc;

use arrow::datatypes::{DataType as ArrowDataType, Schema};

use super::schema::cudf_dtype_from_arrow_type_in;
use crate::{
    core::{
        column::Column,
        error::{self, Error, ErrorCode, Result},
        types::{DataType, TypeId},
    },
    frame::dataframe::DataFrame,
    runtime::Runtime,
};

const OPERATION: &str = "arrow_host::empty_dataframe_from_arrow_schema";

impl Runtime {
    /// Create a zero-row `DataFrame` typed by an Arrow schema inside this
    /// runtime, without a host copy.
    ///
    /// Field names become column names in schema order. Each field gets the
    /// cudf type that [`DataFrame::from_record_batch`] would create for it,
    /// including the element type of a list, so the frame concatenates with
    /// non-empty frames of the same schema. A `Struct` field becomes a cudf
    /// `STRUCT` column whose children follow the same rules. Columns carry
    /// this runtime's retained provenance and allocate no device memory. Arrow
    /// nullability is not encoded in cudf types and is ignored. A schema with
    /// no fields gives the same frame as [`Runtime::empty_dataframe`].
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::UnsupportedDtype`] for a field type outside that
    /// mapping, such as `Dictionary` or a `Struct` without fields, and a
    /// duplicate column name error when two fields share a name.
    pub fn empty_dataframe_from_arrow_schema(&self, schema: &Schema) -> Result<DataFrame> {
        if schema.fields().is_empty() {
            return Ok(self.empty_dataframe());
        }
        let mut names = Vec::with_capacity(schema.fields().len());
        let mut columns = Vec::with_capacity(schema.fields().len());
        let mut nodes = Vec::new();
        for field in schema.fields() {
            nodes.clear();
            push_cudf_type_tree(field.data_type(), &mut nodes)
                .map_err(|error| error.with_detail("field_name", field.name().clone()))?;
            columns.push(Column::empty_from_type_tree_in_domain(
                &nodes,
                Arc::clone(self.execution_domain()),
                OPERATION,
            )?);
            names.push(field.name().clone());
        }
        DataFrame::new(names, columns).map_err(|error| error.reanchor(OPERATION))
    }
}

/// Appends the cudf type tree of `data_type` in pre-order. Leaf and list types
/// come from the host import mapping, which also decides the element types a
/// list may carry; only `Struct` is expanded here.
fn push_cudf_type_tree(data_type: &ArrowDataType, nodes: &mut Vec<cudf_sys::cudf_column_type_node_t>) -> Result<()> {
    if let ArrowDataType::Struct(fields) = data_type {
        if fields.is_empty() {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OPERATION,
                "a cudf STRUCT column requires at least one child field",
            )
            .with_detail("arrow_type", format!("{data_type:?}"))
            .with_detail("reason", "struct_without_fields"));
        }
        nodes.push(type_node(DataType::new(TypeId::Struct), fields.len())?);
        for field in fields {
            push_cudf_type_tree(field.data_type(), nodes)
                .map_err(|error| error.with_detail("child_field_name", field.name().clone()))?;
        }
        return Ok(());
    }

    let dtype = cudf_dtype_from_arrow_type_in(data_type, OPERATION)?;
    let element = match data_type {
        ArrowDataType::List(child) | ArrowDataType::LargeList(child) | ArrowDataType::FixedSizeList(child, _) => {
            Some(child)
        },
        _ => None,
    };
    nodes.push(type_node(dtype, usize::from(element.is_some()))?);
    match element {
        Some(child) => push_cudf_type_tree(child.data_type(), nodes),
        None => Ok(()),
    }
}

fn type_node(dtype: DataType, num_children: usize) -> Result<cudf_sys::cudf_column_type_node_t> {
    let num_children = i32::try_from(num_children)
        .map_err(|_| error::range_limit_error(OPERATION, "num_children", "i32::MAX", num_children))?;
    Ok(cudf_sys::cudf_column_type_node_t { type_id: dtype.id().to_raw(), scale: dtype.scale(), num_children })
}
