use cudf::{
    DuplicateKeepOption, Runtime, SortOptions,
    core::{Table, table::TableView},
    frame::dataframe::DropDuplicatesOptions,
};
use rapids_interop::{BorrowedColumnView, BorrowedTableView, PreparedColumnExport, PreparedTableExport};

use super::{
    borrowed::{
        BorrowedEdgeList, borrowed_table_to_edge_list, build_graph_from_borrowed_edge_list_raw,
        order_borrowed_inputs_for_handle, validate_edge_list_policy,
    },
    egress::dataframe_from_named_series,
    errors::{
        map_borrowed_projection_interop_error, map_cudf_error_with, map_owned_import_interop_error,
        map_prepared_lease_activation_interop_error, map_prepared_projection_interop_error,
    },
    options::{DataFrameGraphOptions, GraphConstructionPolicy, SeriesGraphOptions, TableGraphOptions},
    prepared::{
        PreparedEdgeColumn, PreparedEdgeList, build_graph_from_prepared_edge_list_raw, complete_prepared_lease,
        finish_prepared_graph_result, prepared_lease_authorities, prepared_profile_supported,
    },
};
use crate::{
    error::{Error, ErrorKind, Result},
    graph::{Graph, Transposition},
    handle::EnteredContext,
    types::GraphProperties,
};

const NORMALIZED_SRC_COL: &str = "src";
const NORMALIZED_DST_COL: &str = "dst";
const NORMALIZED_WEIGHT_COL: &str = "weight";

pub(super) fn runtime_for_handle(handle: &EnteredContext<'_>, operation: &'static str) -> Result<Runtime> {
    Runtime::builder()
        .with_device(cudf::DeviceId::new(handle.device() as usize).map_err(|error| {
            Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("invalid handle device {}: {}", handle.device(), error))
                .set_source(error)
        })?)
        .build()
        .map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                format!("failed to create cudf runtime for device {}", handle.device()),
                [("device", handle.device().to_string())],
            )
        })
}

fn canonical_edge_dataframe_from_owned_series(
    src: &cudf::Series,
    dst: &cudf::Series,
    weights: Option<&cudf::Series>,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let mut pairs = vec![(NORMALIZED_SRC_COL.to_string(), src.clone()), (NORMALIZED_DST_COL.to_string(), dst.clone())];
    if let Some(weights) = weights {
        pairs.push((NORMALIZED_WEIGHT_COL.to_string(), weights.clone()));
    }

    dataframe_from_named_series(pairs, operation)
        .map_err(|error| error.with_message("failed to assemble canonical cudf edge DataFrame for graph normalization"))
}

fn validate_table_column_selection(
    num_cols: usize,
    options: &TableGraphOptions,
    table_kind: &str,
    operation: &'static str,
) -> Result<()> {
    let validate_col_index = |idx: usize, name: &str| -> Result<()> {
        if idx >= num_cols {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} column index {} is out of bounds ({} has {} columns)",
                    name, idx, table_kind, num_cols
                ))
                .with_context("column_name", name.to_string())
                .with_context("column_index", idx.to_string())
                .with_context("num_columns", num_cols.to_string()));
        }
        Ok(())
    };

    validate_col_index(options.src_col, "src")?;
    validate_col_index(options.dst_col, "dst")?;
    if let Some(weight_col) = options.weight_col {
        validate_col_index(weight_col, "weight")?;
    }

    Ok(())
}

pub(super) fn canonical_edge_dataframe_from_imported_table(
    table: Table,
    options: &TableGraphOptions,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let num_columns = table
        .num_columns()
        .map_err(|error| map_cudf_error_with(error, operation, "failed to inspect imported cudf table schema", []))?;
    validate_table_column_selection(num_columns, options, "imported table", operation)?;

    let names = (0..num_columns).map(|index| format!("column_{index}")).collect::<Vec<_>>();
    let imported = cudf::DataFrame::from_table(table, names.clone()).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to wrap imported cudf table for graph normalization",
            std::iter::empty(),
        )
    })?;

    let src = imported.series(&names[options.src_col]).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to select source column from imported cudf table", [])
    })?;
    let dst = imported.series(&names[options.dst_col]).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to select destination column from imported cudf table", [])
    })?;
    let weights = options
        .weight_col
        .map(|index| {
            imported.series(&names[index]).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to select weight column from imported cudf table", [])
            })
        })
        .transpose()?;

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)
}

fn normalize_python_cugraph_dataframe(
    df: &cudf::DataFrame,
    has_weights: bool,
    properties: GraphProperties,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let base = if properties.is_symmetric {
        let reverse_weights = if has_weights {
            Some(df.series(NORMALIZED_WEIGHT_COL).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to access weight column for reverse-edge expansion", [])
            })?)
        } else {
            None
        };
        let reverse = canonical_edge_dataframe_from_owned_series(
            &df.series(NORMALIZED_DST_COL).map_err(|error| {
                map_cudf_error_with(
                    error,
                    operation,
                    "failed to access destination column for reverse-edge expansion",
                    [],
                )
            })?,
            &df.series(NORMALIZED_SRC_COL).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to access source column for reverse-edge expansion", [])
            })?,
            reverse_weights.as_ref(),
            operation,
        )?;

        df.concat(&reverse).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to append reverse edges for Python-style graph normalization",
                [],
            )
        })?
    } else {
        df.clone()
    };

    if properties.is_multigraph {
        return Ok(base);
    }

    let mut sort_by = vec![NORMALIZED_SRC_COL, NORMALIZED_DST_COL];
    if has_weights {
        sort_by.push(NORMALIZED_WEIGHT_COL);
    }

    let sorted = base.sort_values(sort_by, SortOptions::new().with_stable(true)).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to sort edges for Python-style graph normalization", [])
    })?;
    let deduped = sorted
        .drop_duplicates(
            DropDuplicatesOptions::builder()
                .subset(&[NORMALIZED_SRC_COL, NORMALIZED_DST_COL][..])
                .keep(DuplicateKeepOption::KeepFirst)
                .build(),
        )
        .map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to collapse duplicate edges for Python-style graph normalization",
                [],
            )
        })?;

    if has_weights {
        deduped.select([NORMALIZED_SRC_COL, NORMALIZED_DST_COL, NORMALIZED_WEIGHT_COL]).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to project normalized weighted edge columns into canonical order",
                [],
            )
        })
    } else {
        deduped.select([NORMALIZED_SRC_COL, NORMALIZED_DST_COL]).map_err(|error| {
            map_cudf_error_with(error, operation, "failed to project normalized edge columns into canonical order", [])
        })
    }
}

fn build_graph_from_owned_dataframe_raw<T: Transposition>(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
    properties: GraphProperties,
    renumber: bool,
    has_weights: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    let export = df
        .try_prepare_device_table_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
    let retained_owner = export.clone_retained_owner();
    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let lease = export
        .try_activate_read_lease(read_start, completion)
        .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;

    let columns = lease.export().columns();
    let bindings = lease.export().bindings();
    let edge_list = PreparedEdgeList {
        src: PreparedEdgeColumn { column: &columns[0], bindings },
        dst: PreparedEdgeColumn { column: &columns[1], bindings },
        weights: if has_weights { Some(PreparedEdgeColumn { column: &columns[2], bindings }) } else { None },
    };

    let graph_result = build_graph_from_prepared_edge_list_raw(handle, &edge_list, properties, renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), retained_owner, operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}

pub(super) fn build_python_normalized_graph_from_dataframe<T: Transposition>(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
    properties: GraphProperties,
    renumber: bool,
    has_weights: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    let normalized = normalize_python_cugraph_dataframe(df, has_weights, properties, operation)?;
    build_graph_from_owned_dataframe_raw(handle, &normalized, properties, renumber, has_weights, operation)
}

fn canonical_edge_dataframe_from_borrowed_edge_list(
    handle: &EnteredContext<'_>,
    edge_list: &BorrowedEdgeList<'_>,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    validate_edge_list_policy(handle, edge_list, operation)?;
    let _barriers = order_borrowed_inputs_for_handle(handle, edge_list, operation)?;

    let owned_runtime;
    let runtime = if let Some(runtime) = handle.runtime() {
        runtime
    } else {
        owned_runtime = runtime_for_handle(handle, operation)?;
        &owned_runtime
    };
    let src = cudf::Series::try_import_from_borrowed_column_view_into_runtime(&edge_list.src, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let dst = cudf::Series::try_import_from_borrowed_column_view_into_runtime(&edge_list.dst, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let weights = edge_list
        .weights
        .as_ref()
        .map(|column| {
            cudf::Series::try_import_from_borrowed_column_view_into_runtime(column, runtime)
                .map_err(|error| map_owned_import_interop_error(error, operation))
        })
        .transpose()?;

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)
}

fn canonical_edge_dataframe_from_prepared_exports(
    handle: &EnteredContext<'_>,
    src: &PreparedColumnExport<'_>,
    dst: &PreparedColumnExport<'_>,
    weights: Option<&PreparedColumnExport<'_>>,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let owned_runtime;
    let runtime = if let Some(runtime) = handle.runtime() {
        runtime
    } else {
        owned_runtime = runtime_for_handle(handle, operation)?;
        &owned_runtime
    };
    let src = cudf::Series::try_import_from_prepared_column_export_into_runtime(src, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let dst = cudf::Series::try_import_from_prepared_column_export_into_runtime(dst, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let weights = weights
        .map(|column| {
            cudf::Series::try_import_from_prepared_column_export_into_runtime(column, runtime)
                .map_err(|error| map_owned_import_interop_error(error, operation))
        })
        .transpose()?;

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction functions
// ═══════════════════════════════════════════════════════════════════════════

/// Build a cuGraph [`Graph`] directly from borrowed RAPIDS column descriptors.
pub fn graph_from_borrowed_columns<T: Transposition>(
    handle: &EnteredContext<'_>,
    src: &BorrowedColumnView<'_>,
    dst: &BorrowedColumnView<'_>,
    weights: Option<&BorrowedColumnView<'_>>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let edge_list = BorrowedEdgeList { src: src.clone(), dst: dst.clone(), weights: weights.cloned() };
        let canonical =
            canonical_edge_dataframe_from_borrowed_edge_list(handle, &edge_list, "graph_from_borrowed_columns")?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            weights.is_some(),
            "graph_from_borrowed_columns",
        );
    }

    let edge_list = BorrowedEdgeList { src: src.clone(), dst: dst.clone(), weights: weights.cloned() };
    build_graph_from_borrowed_edge_list_raw(
        handle,
        &edge_list,
        options.properties,
        options.renumber,
        "graph_from_borrowed_columns",
    )
}

/// Build a cuGraph [`Graph`] directly from a borrowed RAPIDS table descriptor.
pub fn graph_from_borrowed_table_view<T: Transposition>(
    handle: &EnteredContext<'_>,
    view: &BorrowedTableView<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_borrowed_table_view";
    let edge_list = borrowed_table_to_edge_list(view, options, operation)?;
    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_borrowed_edge_list(handle, &edge_list, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    build_graph_from_borrowed_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation)
}

/// Build a cuGraph [`Graph`] directly from prepared-device RAPIDS column exports.
///
/// This is the primary single-GPU shared-contract path. Runtime-bound handles
/// activate one read lease per input export on the consumer stream and retain
/// same-device completion ordering explicitly. Non-runtime-bound handles fall
/// back to synchronous-by-return lease authorities.
///
/// When [`GraphConstructionPolicy::PythonCugraph`] is selected, this function
/// may first import the edge columns into owned cudf objects on the consumer
/// device so Python-style normalization can run before raw graph construction.
pub fn graph_from_prepared_columns<T: Transposition>(
    handle: &EnteredContext<'_>,
    src: &PreparedColumnExport<'_>,
    dst: &PreparedColumnExport<'_>,
    weights: Option<&PreparedColumnExport<'_>>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_prepared_columns";
    prepared_profile_supported(src.diagnostics().profile(), operation)?;
    prepared_profile_supported(dst.diagnostics().profile(), operation)?;
    if let Some(weights) = weights {
        prepared_profile_supported(weights.diagnostics().profile(), operation)?;
    }

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_prepared_exports(handle, src, dst, weights, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            weights.is_some(),
            operation,
        );
    }

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let src_retained_owner = src.clone_retained_owner();
    let dst_retained_owner = dst.clone_retained_owner();
    let wgt_retained_owner = weights.and_then(|weights| weights.clone_retained_owner());
    let mut src_lease = Some(
        src.try_activate_read_lease(read_start, completion)
            .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?,
    );
    let mut dst_lease = None;
    let mut wgt_lease = None;

    let graph_result = (|| {
        dst_lease = Some(
            dst.try_activate_read_lease(read_start, completion)
                .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?,
        );
        wgt_lease = weights
            .map(|weights| {
                weights
                    .try_activate_read_lease(read_start, completion)
                    .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))
            })
            .transpose()?;

        let edge_list = PreparedEdgeList {
            src: PreparedEdgeColumn {
                column: src_lease.as_ref().expect("src lease should be active").export().column(),
                bindings: src_lease.as_ref().expect("src lease should be active").export().bindings(),
            },
            dst: PreparedEdgeColumn {
                column: dst_lease.as_ref().expect("dst lease should be active").export().column(),
                bindings: dst_lease.as_ref().expect("dst lease should be active").export().bindings(),
            },
            weights: wgt_lease.as_ref().map(|lease| PreparedEdgeColumn {
                column: lease.export().column(),
                bindings: lease.export().bindings(),
            }),
        };

        build_graph_from_prepared_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation)
    })();

    let src_complete = src_lease
        .take()
        .map(|lease| complete_prepared_lease(lease.complete(), src_retained_owner, operation))
        .unwrap_or(Ok(None));
    let dst_complete = dst_lease
        .take()
        .map(|lease| complete_prepared_lease(lease.complete(), dst_retained_owner, operation))
        .unwrap_or(Ok(None));
    let wgt_complete = wgt_lease
        .take()
        .map(|lease| complete_prepared_lease(lease.complete(), wgt_retained_owner, operation))
        .unwrap_or(Ok(None));

    finish_prepared_graph_result(graph_result, [src_complete, dst_complete, wgt_complete])
}

fn validate_prepared_table_column_selection(
    num_cols: usize,
    options: &TableGraphOptions,
    operation: &'static str,
) -> Result<()> {
    validate_table_column_selection(num_cols, options, "prepared table export", operation)
}

/// Build a cuGraph [`Graph`] directly from a prepared-device RAPIDS table export.
///
/// When [`GraphConstructionPolicy::PythonCugraph`] is selected, this function
/// may first import the selected edge columns into owned cudf objects on the
/// consumer device so Python-style normalization can run before raw graph
/// construction.
pub fn graph_from_prepared_table_export<T: Transposition>(
    handle: &EnteredContext<'_>,
    export: &PreparedTableExport<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_prepared_table_export";
    prepared_profile_supported(export.diagnostics().profile(), operation)?;
    validate_prepared_table_column_selection(export.columns().len(), options, operation)?;

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let owned_runtime;
        let runtime = if let Some(runtime) = handle.runtime() {
            runtime
        } else {
            owned_runtime = runtime_for_handle(handle, operation)?;
            &owned_runtime
        };
        let imported = Table::try_import_from_prepared_table_export_into_runtime(export, runtime)
            .map_err(|error| map_owned_import_interop_error(error, operation))?;
        let canonical = canonical_edge_dataframe_from_imported_table(imported, options, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let retained_owner = export.clone_retained_owner();
    let lease = export
        .try_activate_read_lease(read_start, completion)
        .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;

    let columns = lease.export().columns();
    let bindings = lease.export().bindings();
    let edge_list = PreparedEdgeList {
        src: PreparedEdgeColumn { column: &columns[options.src_col], bindings },
        dst: PreparedEdgeColumn { column: &columns[options.dst_col], bindings },
        weights: options.weight_col.map(|index| PreparedEdgeColumn { column: &columns[index], bindings }),
    };

    let graph_result =
        build_graph_from_prepared_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), retained_owner, operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}

/// Build a cuGraph [`Graph`] from a cudf low-level [`TableView`].
///
/// **Borrowed compatibility path**: this entry point projects the cudf table
/// view into the shared RAPIDS borrowed ABI. Runtime-bound handles order the
/// consumer stream after same-device producer streams with device-side CUDA
/// dependencies; non-runtime-bound handles may still perform synchronous
/// host-side fencing before cuGraph reads the borrowed device buffers.
///
/// For new single-GPU integrations, prefer
/// [`graph_from_prepared_table_export()`], which keeps producer read-start,
/// completion, and liveness explicit at the shared-contract layer. This
/// function remains the supported compatibility path for callers that already
/// operate on low-level cudf table views and want an immediate bridge into the
/// explicit RAPIDS borrowed ABI.
///
/// **Raw-policy input path**: under [`GraphConstructionPolicy::RawLibcugraph`],
/// cuGraph reads the edge data directly from the table's column device pointers
/// and builds its own internal CSR/CSC representation. Under
/// [`GraphConstructionPolicy::PythonCugraph`], this function may first import
/// the selected edge columns into owned cudf objects on the consumer device so
/// Python-style normalization can run before raw graph construction. The
/// returned [`Graph`] is independent of the input table.
///
/// # Errors
///
/// Returns an error if:
/// - any column index is out of bounds
/// - selected columns contain null values
/// - column types are incompatible with cugraph
/// - graph construction fails
///
/// [`TableView`]: cudf::core::table::TableView
pub fn graph_from_table_view<T: Transposition>(
    handle: &EnteredContext<'_>,
    view: &TableView<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_table_view";
    let borrowed =
        view.try_borrow_table_view().map_err(|error| map_borrowed_projection_interop_error(error, operation))?;
    let edge_list = borrowed_table_to_edge_list(&borrowed, options, operation)?;
    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_borrowed_edge_list(handle, &edge_list, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    build_graph_from_borrowed_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation)
}

/// Build a cuGraph [`Graph`] from a cudf [`DataFrame`] using column names.
///
/// **Prepared-export-first input**: resolves column names, prepares a
/// producer-rooted device table export, then activates read access explicitly
/// during graph construction. No intermediate table copy is performed.
///
/// # Errors
///
/// Returns an error if any column name is not found in the DataFrame, or if
/// graph construction fails.
///
/// [`DataFrame`]: cudf::DataFrame
pub fn graph_from_dataframe<T: Transposition>(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
    options: &DataFrameGraphOptions<'_>,
) -> Result<Graph<T>> {
    let operation = "graph_from_dataframe";
    let names = df.column_names();
    let resolve_index = |name: &str, role: &str| -> Result<usize> {
        names.iter().position(|candidate| *candidate == name).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("{} column '{}' not found in DataFrame schema", role, name))
                .with_context("column_label", role.to_string())
                .with_context("column_name", name.to_string())
        })
    };

    let table_options = TableGraphOptions {
        src_col: resolve_index(options.src_col, "src")?,
        dst_col: resolve_index(options.dst_col, "dst")?,
        weight_col: options.weight_col.map(|name| resolve_index(name, "weight")).transpose()?,
        properties: options.properties,
        renumber: options.renumber,
        construction_policy: options.construction_policy,
    };

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let src = df.series(options.src_col).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to project source column for Python-style graph normalization",
                [],
            )
        })?;
        let dst = df.series(options.dst_col).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to project destination column for Python-style graph normalization",
                [],
            )
        })?;
        let weights = options
            .weight_col
            .map(|name| {
                df.series(name).map_err(|error| {
                    map_cudf_error_with(
                        error,
                        operation,
                        "failed to project weight column for Python-style graph normalization",
                        [],
                    )
                })
            })
            .transpose()?;

        let canonical = canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    let export = df
        .try_prepare_device_table_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
    graph_from_prepared_table_export(handle, &export, &table_options)
}

/// Build a cuGraph [`Graph`] from cudf [`Series`] (src vertices, dst vertices,
/// optional weights).
///
/// **Prepared-export-first input**: each Series is projected into a
/// producer-rooted device export and consumed through explicit read-lease
/// activation during graph construction. The Graph's internal CSR/CSC is
/// independent after construction.
///
/// # Example
///
/// ```no_run
/// use cugraph::{
///     Forward, ResourceHandle,
///     cudf_interop::{SeriesGraphOptions, graph_from_series},
/// };
///
/// let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
/// let handle =
///     ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
/// let entered = handle.enter("graph_from_series docs").unwrap();
/// let src = runtime.series(&[0i32, 1, 2]).unwrap();
/// let dst = runtime.series(&[1i32, 2, 0]).unwrap();
///
/// let opts = SeriesGraphOptions::default().renumber();
/// let graph = graph_from_series::<Forward>(&entered, &src, &dst, None, &opts)
///     .unwrap();
/// ```
///
/// [`Series`]: cudf::Series
pub fn graph_from_series<T: Transposition>(
    handle: &EnteredContext<'_>,
    src: &cudf::Series,
    dst: &cudf::Series,
    weights: Option<&cudf::Series>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_series";
    let src_export = src
        .try_prepare_device_column_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
    let dst_export = dst
        .try_prepare_device_column_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
    let wgt_export = weights
        .map(|column| {
            column
                .try_prepare_device_column_export()
                .map_err(|error| map_prepared_projection_interop_error(error, operation))
        })
        .transpose()?;

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_prepared_exports(
            handle,
            &src_export,
            &dst_export,
            wgt_export.as_ref(),
            operation,
        )?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            weights.is_some(),
            operation,
        );
    }

    graph_from_prepared_columns(handle, &src_export, &dst_export, wgt_export.as_ref(), options)
}
