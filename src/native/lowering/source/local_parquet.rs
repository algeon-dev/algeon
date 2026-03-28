use std::sync::Arc;

use datafusion::{
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSourceExec,
    },
    physical_plan::ExecutionPlan,
};
use nexus_query_engine::{
    LocalParquetSource as NativeLocalParquetSource, QueryPlanBuilder as NativeQueryPlanBuilder,
    SourceNode as NativeSourceNode,
};

use super::{
    SourceUnprojectedFilterOutput, combine_scan_limits, lower_required_file_source_projection_exprs,
    lower_source_filter_and_projection, reader_predicate_proves_local_parquet_at_most_one_row,
    required_scan_projection, required_scan_projection_needs_native_projection,
    required_scan_projection_requires_native_projection, required_scan_projection_schema,
    required_source_scan_output_indices, source_bindings_for_schema, source_reader_pushdown_predicate,
};
use crate::{
    native::{
        lowering::{LoweredNativeInput, RequiredFields, expr::lower_physical_expr_with_bindings},
        pruning::scan_predicate_without_dynamic_filter_hints,
        source::{
            local_parquet_files, local_parquet_source_diagnostics, native_source_output_schema,
            required_parquet_column_requirements_for_scan_columns, validate_slice1_parquet_scan,
        },
        validate::{validate_physical_expr, validate_schema},
    },
    policy::FallbackReason,
};

pub(super) fn lower_local_parquet_source_input(
    source_exec: &DataSourceExec,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    pushed_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, FallbackReason> {
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;
    let required_scan_output_indices =
        required_source_scan_output_indices(source_exec.schema(), filter, required_output_fields)?;
    let column_requirements = required_parquet_column_requirements_for_scan_columns(
        file_scan,
        parquet_source,
        filter,
        &required_scan_output_indices,
    )?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let source_output_schema = native_source_output_schema(file_scan, &column_requirements)?;

    validate_schema(Arc::clone(&source_output_schema))?;
    if let Some(scan_projection) = scan_projection {
        for projection_expr in
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns)
        {
            validate_physical_expr(projection_expr.expr, &table_schema)?;
        }
    }
    if let Some(filter) = filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }
    validate_slice1_parquet_scan(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;

    let native_files =
        local_parquet_files(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    let native_file_paths = native_files.iter().map(|file| file.path.clone()).collect::<Vec<_>>();
    let scan_projection_requires_native = scan_projection.is_some_and(|scan_projection| {
        let required_scan_projection =
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns);
        required_scan_projection_requires_native_projection(&required_scan_projection, &table_schema)
    });
    let mut native_source = NativeLocalParquetSource::from_files(
        native_files,
        Arc::clone(&source_output_schema),
        column_requirements.read_columns.clone(),
    )
    .with_column_roles(
        column_requirements.projected_output_columns.clone(),
        column_requirements.pure_filter_columns.clone(),
    )
    .with_column_source_refs(
        column_requirements.read_column_file_indices.clone(),
        column_requirements.projected_output_column_source_refs.clone(),
        column_requirements.pure_filter_column_source_refs.clone(),
    )
    .with_file_column_count(file_scan.file_schema().fields().len())
    .with_partition_columns(column_requirements.partition_columns.clone())
    .with_missing_columns(column_requirements.missing_columns.clone())
    .with_scan_limit(if filter.is_none() {
        combine_scan_limits(file_scan.limit, pushed_scan_limit)
    } else {
        file_scan.limit
    });
    let mut reader_pushdown_covers_filter = false;
    if let Some(predicate) = source_reader_pushdown_predicate(
        parquet_source,
        filter,
        file_scan.limit,
        scan_projection_requires_native,
        &source_output_schema,
        if filter.is_none() { Arc::clone(&table_schema) } else { source_exec.schema() },
        &native_source,
    )? {
        reader_pushdown_covers_filter = predicate.covers_filter;
        let max_rows = if reader_predicate_proves_local_parquet_at_most_one_row(
            table_schema.as_ref(),
            file_scan.file_schema().as_ref(),
            &native_file_paths.iter().map(|path| path.as_path()).collect::<Vec<_>>(),
            &predicate.expr,
        ) {
            Some(1)
        } else {
            None
        };
        native_source = native_source.with_reader_predicate(predicate.expr).with_max_rows(max_rows);
    }
    let reader_pushdown = native_source.reader_predicate.is_some();
    let selected_row_group_stats = native_source.diagnostics.selected_row_group_stats.clone();
    native_source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        Arc::clone(&table_schema),
        Some(&column_requirements),
        reader_pushdown,
        false,
    );
    native_source.diagnostics.selected_row_group_stats = selected_row_group_stats;

    let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
    let scan_only_post_scan_predicate = if filter.is_none() && !reader_pushdown_covers_filter {
        parquet_source.filter().and_then(|predicate| scan_predicate_without_dynamic_filter_hints(&predicate))
    } else {
        None
    };
    let (scan_input_node, scan_input_schema) = if let Some(predicate) = scan_only_post_scan_predicate.as_ref() {
        let bindings = source_bindings_for_schema(&source_output_schema, Arc::clone(&table_schema));
        (
            builder.add_filter(
                source_node,
                lower_physical_expr_with_bindings(predicate, &table_schema, Some(&bindings))?,
                Arc::clone(&source_output_schema),
            ),
            Arc::clone(&source_output_schema),
        )
    } else {
        (source_node, Arc::clone(&source_output_schema))
    };
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection {
        let required_scan_projection =
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns);
        if filter.is_none()
            && required_scan_projection_needs_native_projection(
                &required_scan_projection,
                &table_schema,
                &source_output_schema,
                source_exec.schema(),
            )?
        {
            let expressions = lower_required_file_source_projection_exprs(&required_scan_projection, &table_schema)?;
            let schema = required_scan_projection_schema(&required_scan_projection, source_exec.schema())?;
            (
                builder.add_projection(
                    scan_input_node,
                    crate::native::lowering::projection_outputs(expressions, &schema)?,
                ),
                schema,
            )
        } else {
            (scan_input_node, scan_input_schema)
        }
    } else {
        (scan_input_node, scan_input_schema)
    };
    let scan_output_bindings = source_bindings_for_schema(&scan_output_schema, source_exec.schema());
    let scan_output = LoweredNativeInput::with_bindings(scan_output_node, scan_output_schema, scan_output_bindings);
    if let Some(filter) = filter {
        lower_source_filter_and_projection(
            builder,
            scan_output,
            filter,
            reader_pushdown_covers_filter,
            required_output_fields,
            SourceUnprojectedFilterOutput::ProjectRequiredOutput,
        )
    } else {
        super::project_scan_output_columns(
            builder,
            scan_output.node,
            scan_output.schema,
            scan_output.bindings,
            required_output_fields,
        )
    }
}
