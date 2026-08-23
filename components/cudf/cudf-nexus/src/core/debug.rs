use std::fmt;

use super::{Column, ColumnView, Scalar, Table, TableView};

#[expect(dead_code, reason = "debug info fields formatted via Debug")]
#[derive(Debug)]
struct ExecutionDomainDebugInfo {
    kind: &'static str,
    id: u64,
    stream: usize,
    memory_resource: Option<usize>,
}

impl ExecutionDomainDebugInfo {
    fn for_column(column: &Column) -> Self {
        Self {
            kind: column.__debug_execution_domain_kind(),
            id: column.__debug_execution_domain_id(),
            stream: column.__debug_execution_domain_stream_value(),
            memory_resource: Some(column.__debug_execution_domain_mr_value()),
        }
    }

    fn for_column_view(view: &ColumnView<'_>) -> Self {
        Self {
            kind: view.__debug_execution_domain_kind(),
            id: view.__debug_execution_domain_id(),
            stream: view.__debug_execution_domain_stream_value(),
            memory_resource: Some(view.__debug_execution_domain_mr_value()),
        }
    }

    fn for_table(table: &Table) -> Self {
        Self {
            kind: table.__debug_execution_domain_kind(),
            id: table.__debug_execution_domain_id(),
            stream: table.__debug_execution_domain_stream_value(),
            memory_resource: Some(table.__debug_execution_domain_mr_value()),
        }
    }

    fn for_table_view(view: &TableView<'_>) -> Self {
        Self {
            kind: view.__debug_execution_domain_kind(),
            id: view.__debug_execution_domain_id(),
            stream: view.__debug_execution_domain_stream_value(),
            memory_resource: Some(view.__debug_execution_domain_mr_value()),
        }
    }

    fn for_scalar(scalar: &Scalar) -> Self {
        Self {
            kind: scalar.__debug_execution_domain_kind(),
            id: scalar.__debug_execution_domain_id(),
            stream: scalar.__debug_execution_domain_stream_value(),
            memory_resource: None,
        }
    }
}

#[derive(Debug)]
struct ColumnDebugMetadata {
    dtype: String,
    len: Option<usize>,
    nullable: Option<bool>,
    null_count: Option<usize>,
    execution_domain: ExecutionDomainDebugInfo,
}

impl ColumnDebugMetadata {
    fn unknown(execution_domain: ExecutionDomainDebugInfo) -> Self {
        Self { dtype: "?".to_string(), len: None, nullable: None, null_count: None, execution_domain }
    }
}

#[expect(dead_code, reason = "debug info fields formatted via Debug")]
#[derive(Debug)]
struct TableColumnDebugSummary {
    index: usize,
    dtype: String,
    nullable: Option<bool>,
    null_count: Option<usize>,
}

#[derive(Debug)]
struct TableDebugMetadata {
    num_columns: Option<usize>,
    num_rows: Option<usize>,
    columns: Vec<TableColumnDebugSummary>,
    execution_domain: ExecutionDomainDebugInfo,
}

impl TableDebugMetadata {
    fn unknown(execution_domain: ExecutionDomainDebugInfo) -> Self {
        Self { num_columns: None, num_rows: None, columns: Vec::new(), execution_domain }
    }
}

fn dtype_name<T>(dtype: Option<T>) -> String
where T: Into<super::types::DataType> {
    dtype.map(|dtype| format!("{:?}", dtype.into().id())).unwrap_or_else(|| "?".to_string())
}

fn build_column_debug_metadata(
    view: Option<&ColumnView<'_>>,
    execution_domain: ExecutionDomainDebugInfo,
) -> ColumnDebugMetadata {
    let Some(view) = view else {
        return ColumnDebugMetadata::unknown(execution_domain);
    };

    ColumnDebugMetadata {
        dtype: dtype_name(view.data_type().ok()),
        len: view.len().ok(),
        nullable: view.nullable().ok(),
        null_count: view.null_count().ok(),
        execution_domain,
    }
}

fn build_table_column_debug_summary(view: Option<&ColumnView<'_>>, index: usize) -> TableColumnDebugSummary {
    let Some(view) = view else {
        return TableColumnDebugSummary { index, dtype: "?".to_string(), nullable: None, null_count: None };
    };

    TableColumnDebugSummary {
        index,
        dtype: dtype_name(view.data_type().ok()),
        nullable: view.nullable().ok(),
        null_count: view.null_count().ok(),
    }
}

fn build_table_debug_metadata(
    view: Option<&TableView<'_>>,
    execution_domain: ExecutionDomainDebugInfo,
) -> TableDebugMetadata {
    let Some(view) = view else {
        return TableDebugMetadata::unknown(execution_domain);
    };

    let num_columns = view.num_columns().ok();
    let num_rows = view.num_rows().ok();
    let columns = num_columns
        .map(|num_columns| {
            (0..num_columns)
                .map(|index| {
                    let column = view.column(index).ok();
                    build_table_column_debug_summary(column.as_ref(), index)
                })
                .collect()
        })
        .unwrap_or_default();

    TableDebugMetadata { num_columns, num_rows, columns, execution_domain }
}

impl fmt::Debug for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let view = self.view().ok();
        let metadata = build_column_debug_metadata(view.as_ref(), ExecutionDomainDebugInfo::for_column(self));

        f.debug_struct("Column")
            .field("dtype", &metadata.dtype)
            .field("len", &metadata.len)
            .field("nullable", &metadata.nullable)
            .field("null_count", &metadata.null_count)
            .field("execution_domain", &metadata.execution_domain)
            .finish()
    }
}

impl fmt::Debug for ColumnView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metadata = build_column_debug_metadata(Some(self), ExecutionDomainDebugInfo::for_column_view(self));

        f.debug_struct("ColumnView")
            .field("dtype", &metadata.dtype)
            .field("len", &metadata.len)
            .field("nullable", &metadata.nullable)
            .field("null_count", &metadata.null_count)
            .field("execution_domain", &metadata.execution_domain)
            .finish()
    }
}

impl fmt::Debug for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let view = self.view().ok();
        let metadata = build_table_debug_metadata(view.as_ref(), ExecutionDomainDebugInfo::for_table(self));

        f.debug_struct("Table")
            .field("num_columns", &metadata.num_columns)
            .field("num_rows", &metadata.num_rows)
            .field("columns", &metadata.columns)
            .field("execution_domain", &metadata.execution_domain)
            .finish()
    }
}

impl fmt::Debug for TableView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metadata = build_table_debug_metadata(Some(self), ExecutionDomainDebugInfo::for_table_view(self));

        f.debug_struct("TableView")
            .field("num_columns", &metadata.num_columns)
            .field("num_rows", &metadata.num_rows)
            .field("columns", &metadata.columns)
            .field("execution_domain", &metadata.execution_domain)
            .finish()
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dtype = dtype_name(self.data_type().ok());
        let valid = self.is_valid().ok();
        let execution_domain = ExecutionDomainDebugInfo::for_scalar(self);

        f.debug_struct("Scalar")
            .field("dtype", &dtype)
            .field("valid", &valid)
            .field("execution_domain", &execution_domain)
            .finish()
    }
}
