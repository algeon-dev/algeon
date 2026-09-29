use cudf::{
    DuplicateKeepOption, Runtime, SortOptions,
    core::Table,
    frame::dataframe::DropDuplicatesOptions,
    interop::contract::{PreparedColumnExport, PreparedTableExport},
};

use super::{
    egress::dataframe_from_named_series,
    errors::{
        map_cudf_error_with, map_owned_import_interop_error, map_prepared_lease_activation_interop_error,
        map_prepared_projection_interop_error,
    },
    options::{GraphConstructionPolicy, PreparedGraphOptions, TableGraphOptions, TableVertexPairOptions},
    prepared::{
        PreparedEdgeColumn, PreparedEdgeList, PreparedVertexPairList, build_graph_from_prepared_edge_list_raw,
        complete_prepared_lease, finish_prepared_graph_result, finish_prepared_vertex_pairs_result,
        prepared_lease_authorities, prepared_profile_supported,
        vertex_pairs_from_prepared_columns as build_vertex_pairs_from_prepared_columns,
    },
};
use crate::{
    error::{Error, ErrorKind, Result},
    graph::{Graph, Transposition},
    handle::EnteredContext,
    similarity::VertexPairs,
    types::GraphProperties,
};

const NORMALIZED_SRC_COL: &str = "src";
const NORMALIZED_DST_COL: &str = "dst";
const NORMALIZED_WEIGHT_COL: &str = "weight";
const NORMALIZED_EDGE_ID_COL: &str = "edge_id";

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
    edge_ids: Option<&cudf::Series>,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let mut pairs = vec![(NORMALIZED_SRC_COL.to_string(), src.clone()), (NORMALIZED_DST_COL.to_string(), dst.clone())];
    if let Some(weights) = weights {
        pairs.push((NORMALIZED_WEIGHT_COL.to_string(), weights.clone()));
    }
    if let Some(edge_ids) = edge_ids {
        pairs.push((NORMALIZED_EDGE_ID_COL.to_string(), edge_ids.clone()));
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
    if let Some(edge_id_col) = options.edge_id_col {
        validate_col_index(edge_id_col, "edge_id")?;
    }

    let selected = [
        ("src", Some(options.src_col)),
        ("dst", Some(options.dst_col)),
        ("weight", options.weight_col),
        ("edge_id", options.edge_id_col),
    ];
    for (index, (role, column)) in selected.iter().enumerate() {
        let Some(column) = column else { continue };
        if let Some((duplicate_role, _)) = selected[..index].iter().find(|(_, other)| other == &Some(*column)) {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("{role} and {duplicate_role} must select distinct columns"))
                .with_context("column_role", role.to_string())
                .with_context("duplicate_role", duplicate_role.to_string())
                .with_context("column_index", column.to_string()));
        }
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
    let edge_ids = options
        .edge_id_col
        .map(|index| {
            imported.series(&names[index]).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to select edge-ID column from imported cudf table", [])
            })
        })
        .transpose()?;

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), edge_ids.as_ref(), operation)
}

fn normalize_python_cugraph_dataframe(
    df: &cudf::DataFrame,
    has_weights: bool,
    has_edge_ids: bool,
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
        let reverse_edge_ids = if has_edge_ids {
            Some(df.series(NORMALIZED_EDGE_ID_COL).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to access edge-ID column for reverse-edge expansion", [])
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
            reverse_edge_ids.as_ref(),
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

    let mut selected = vec![NORMALIZED_SRC_COL, NORMALIZED_DST_COL];
    if has_weights {
        selected.push(NORMALIZED_WEIGHT_COL);
    }
    if has_edge_ids {
        selected.push(NORMALIZED_EDGE_ID_COL);
    }
    deduped.select(selected).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to project normalized edge columns into canonical order", [])
    })
}

fn build_graph_from_owned_dataframe_raw<T: Transposition>(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
    properties: GraphProperties,
    renumber: bool,
    has_weights: bool,
    has_edge_ids: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    let export = df
        .try_prepare_device_table_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
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
        edge_ids: if has_edge_ids {
            Some(PreparedEdgeColumn { column: &columns[2 + usize::from(has_weights)], bindings })
        } else {
            None
        },
    };

    let graph_result = build_graph_from_prepared_edge_list_raw(handle, &edge_list, properties, renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}

pub(super) fn build_python_normalized_graph_from_prepared_frame<T: Transposition>(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
    properties: GraphProperties,
    renumber: bool,
    has_weights: bool,
    has_edge_ids: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    let normalized = normalize_python_cugraph_dataframe(df, has_weights, has_edge_ids, properties, operation)?;
    build_graph_from_owned_dataframe_raw(
        handle,
        &normalized,
        properties,
        renumber,
        has_weights,
        has_edge_ids,
        operation,
    )
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

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), None, operation)
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction functions
// ═══════════════════════════════════════════════════════════════════════════

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
    options: &PreparedGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_prepared_columns";
    prepared_profile_supported(src.profile(), operation)?;
    prepared_profile_supported(dst.profile(), operation)?;
    if let Some(weights) = weights {
        prepared_profile_supported(weights.profile(), operation)?;
    }

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_prepared_exports(handle, src, dst, weights, operation)?;
        return build_python_normalized_graph_from_prepared_frame(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            weights.is_some(),
            false,
            operation,
        );
    }

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
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
            edge_ids: None,
        };

        build_graph_from_prepared_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation)
    })();

    let src_complete = src_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation));
    let dst_complete = dst_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation));
    let wgt_complete = wgt_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation));

    finish_prepared_graph_result(graph_result, [src_complete, dst_complete, wgt_complete].into_iter().flatten())
}

/// Create explicit cuGraph [`VertexPairs`] directly from two prepared-device
/// cudf column exports.
///
/// This is the zero-copy pairwise-similarity ingress path.  Both columns must
/// have equal active row counts, the same `Int32` or `Int64` physical dtype,
/// no nulls in their active ranges, and device placement matching `handle`.
/// The graph vertex domain is then validated through [`Graph::vertex_pairs`].
///
/// Runtime-bound handles activate one read lease per export on the consumer
/// stream.  On successful asynchronous completion, the returned
/// [`VertexPairs`] retains both producer owners and completion witnesses until
/// it is dropped (or consumed by a similarity result).  No host round trip is
/// used to materialize or order the pair columns.
pub fn vertex_pairs_from_prepared_columns<T: Transposition>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    first: &PreparedColumnExport<'_>,
    second: &PreparedColumnExport<'_>,
) -> Result<VertexPairs> {
    let operation = "vertex_pairs_from_prepared_columns";
    prepared_profile_supported(first.profile(), operation)?;
    prepared_profile_supported(second.profile(), operation)?;

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let mut first_lease = Some(
        first
            .try_activate_read_lease(read_start, completion)
            .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?,
    );
    let mut second_lease = None;

    let pairs_result = (|| {
        second_lease = Some(
            second
                .try_activate_read_lease(read_start, completion)
                .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?,
        );

        let pairs = PreparedVertexPairList {
            first: PreparedEdgeColumn {
                column: first_lease.as_ref().expect("first lease should be active").export().column(),
                bindings: first_lease.as_ref().expect("first lease should be active").export().bindings(),
            },
            second: PreparedEdgeColumn {
                column: second_lease.as_ref().expect("second lease should be active").export().column(),
                bindings: second_lease.as_ref().expect("second lease should be active").export().bindings(),
            },
        };

        build_vertex_pairs_from_prepared_columns(handle, graph, &pairs, operation)
            .map_err(|error| error.with_operation(operation))
    })();

    let first_complete = first_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation));
    let second_complete = second_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation));

    finish_prepared_vertex_pairs_result(pairs_result, [first_complete, second_complete].into_iter().flatten())
}

fn validate_prepared_table_column_selection(
    num_cols: usize,
    options: &TableGraphOptions,
    operation: &'static str,
) -> Result<()> {
    validate_table_column_selection(num_cols, options, "prepared table export", operation)
}

fn validate_prepared_vertex_pair_table_column_selection(
    num_cols: usize,
    options: &TableVertexPairOptions,
    operation: &'static str,
) -> Result<()> {
    let validate_col_index = |idx: usize, role: &str| -> Result<()> {
        if idx >= num_cols {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} vertex-pair column index {} is out of bounds (prepared table export has {} columns)",
                    role, idx, num_cols
                ))
                .with_context("column_role", role.to_string())
                .with_context("column_index", idx.to_string())
                .with_context("num_columns", num_cols.to_string()));
        }
        Ok(())
    };

    validate_col_index(options.first_col, "first")?;
    validate_col_index(options.second_col, "second")?;
    Ok(())
}

/// Create explicit cuGraph [`VertexPairs`] from two selected columns of a
/// prepared-device cudf table export.
///
/// `options` selects the first and second vertex columns without copying or
/// reordering the table.  The selected columns obey the same equal-row-count,
/// equal-`Int32`/`Int64`, null-free, device-placement, and lease-lifetime
/// contract as [`vertex_pairs_from_prepared_columns`].
pub fn vertex_pairs_from_prepared_table_export<T: Transposition>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    export: &PreparedTableExport<'_>,
    options: &TableVertexPairOptions,
) -> Result<VertexPairs> {
    let operation = "vertex_pairs_from_prepared_table_export";
    prepared_profile_supported(export.profile(), operation)?;
    validate_prepared_vertex_pair_table_column_selection(export.columns().len(), options, operation)?;

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let lease = export
        .try_activate_read_lease(read_start, completion)
        .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;

    let columns = lease.export().columns();
    let bindings = lease.export().bindings();
    let pairs = PreparedVertexPairList {
        first: PreparedEdgeColumn { column: &columns[options.first_col], bindings },
        second: PreparedEdgeColumn { column: &columns[options.second_col], bindings },
    };
    let pairs_result = build_vertex_pairs_from_prepared_columns(handle, graph, &pairs, operation)
        .map_err(|error| error.with_operation(operation));
    let lease_complete = complete_prepared_lease(lease.complete(), operation);

    finish_prepared_vertex_pairs_result(pairs_result, [lease_complete])
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
    prepared_profile_supported(export.profile(), operation)?;
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
        return build_python_normalized_graph_from_prepared_frame(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            options.edge_id_col.is_some(),
            operation,
        );
    }

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let lease = export
        .try_activate_read_lease(read_start, completion)
        .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;

    let columns = lease.export().columns();
    let bindings = lease.export().bindings();
    let edge_list = PreparedEdgeList {
        src: PreparedEdgeColumn { column: &columns[options.src_col], bindings },
        dst: PreparedEdgeColumn { column: &columns[options.dst_col], bindings },
        weights: options.weight_col.map(|index| PreparedEdgeColumn { column: &columns[index], bindings }),
        edge_ids: options.edge_id_col.map(|index| PreparedEdgeColumn { column: &columns[index], bindings }),
    };

    let graph_result =
        build_graph_from_prepared_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}
