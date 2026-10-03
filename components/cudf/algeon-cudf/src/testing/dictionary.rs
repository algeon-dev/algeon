use crate::{
    core::{
        Result,
        column::{Column, ColumnView},
        types::TypeId,
    },
    ops::dictionary,
    runtime::{Exec, execution_resolver::ResolvedExecution},
};

/// Dictionary-encode `input` with `indices_type` codes, so interop and
/// metadata tests can build `DICTIONARY32` columns.
pub fn encode(input: &ColumnView<'_>, indices_type: TypeId) -> Result<Column> {
    let operation = "testing::dictionary::encode";
    let execution = ResolvedExecution::unary(input.execution_domain(), Exec::RETAINED, operation)?;
    dictionary::encode_impl(input, indices_type, execution.output_domain(), execution.exec_options(), operation)
}

/// Borrow the keys child of a dictionary column.
pub fn keys<'a>(dict_column: &ColumnView<'a>) -> Result<ColumnView<'a>> {
    dictionary::keys(dict_column)
}
