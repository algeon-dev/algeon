use crate::{
    core::{error::Result, table::Table},
    frame::dataframe::DataFrame,
};

/// Trigger the internal DataFrame reassembly helper directly for contract
/// tests.
pub fn reassemble_dataframe_from_table_and_names_for_test(table: Table, names: Vec<String>) -> Result<DataFrame> {
    super::reassemble::dataframe_from_table_and_names(table, names)
}
