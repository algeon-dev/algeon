use cudf::{core::Table, io::parquet};

use super::{
    errors::{map_cudf_error_with, map_prepared_projection_interop_error},
    ingress::{
        build_python_normalized_graph_from_dataframe, canonical_edge_dataframe_from_imported_table,
        graph_from_prepared_table_export, runtime_for_handle,
    },
    options::{GraphConstructionPolicy, ParquetGraphOptions, TableGraphOptions},
};
use crate::{
    error::Result,
    graph::{Graph, Transposition},
    handle::EnteredContext,
};

/// A graph loaded from an external data source along with its backing table.
///
/// The [`table`](LoadedGraph::table) field holds the source edge-list data in
/// GPU memory. The [`graph`](LoadedGraph::graph) field holds the constructed
/// graph with its own internal CSR/CSC representation.
///
/// **Ownership**: the graph's internal representation is independent of the
/// table — it was built by copying data from the table's columns during
/// construction. The table is retained so callers can reuse the loaded
/// edge-list data for additional queries or inspection without re-reading
/// the source file.
pub struct LoadedGraph<T: Transposition> {
    /// The constructed graph.
    pub graph: Graph<T>,
    /// The source edge-list table in GPU memory.
    pub table: Table,
}

impl<T: Transposition> std::fmt::Debug for LoadedGraph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedGraph").field("graph", &self.graph).finish_non_exhaustive()
    }
}

/// Read a Parquet edge-list and build a cuGraph [`Graph`] on the GPU.
///
/// This is the file-I/O entry point for graph construction. It creates an
/// internal cudf [`cudf::Runtime`] pinned to `handle.device()`, reads the Parquet file
/// into a cudf [`Table`] on that device, prepares a producer-rooted table
/// export, then delegates graph construction to
/// [`graph_from_prepared_table_export()`].
///
/// **Copy semantics**: this function performs two copies:
/// 1. **Disk → GPU**: the Parquet file is read into a cudf [`Table`] in GPU
///    memory (managed by cudf's Parquet reader).
/// 2. **Table → Graph**: cuGraph reads the table's column data and builds its
///    own internal CSR/CSC representation.
///
/// The returned [`LoadedGraph`] bundles both the graph and the table. The
/// graph is independent of the table — the table is retained so callers can
/// reuse the loaded edge-list data without re-reading the Parquet file.
///
/// [`Table`]: cudf::core::Table
///
/// # Errors
///
/// Returns an error if:
/// - the Parquet file cannot be read
/// - any column index is out of bounds
/// - selected columns contain null values
/// - column types are incompatible with cugraph
/// - graph construction fails
pub fn graph_from_parquet<T: Transposition>(
    handle: &EnteredContext<'_>,
    options: &ParquetGraphOptions<'_>,
) -> Result<LoadedGraph<T>> {
    let operation = "graph_from_parquet";
    let owned_runtime;
    let runtime = if let Some(runtime) = handle.runtime() {
        runtime
    } else {
        owned_runtime = runtime_for_handle(handle, operation)?;
        &owned_runtime
    };

    let reader_opts = parquet::ParquetTableReadOptions::builder().filepath(options.path).with_runtime(runtime).build();

    let result = parquet::read_parquet_table(&reader_opts).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to read Parquet file into cudf table",
            [("path", options.path.to_string())],
        )
    })?;
    let table = result.table;

    let graph = {
        let table_opts = TableGraphOptions {
            src_col: options.src_col,
            dst_col: options.dst_col,
            weight_col: options.weight_col,
            properties: options.properties,
            renumber: options.renumber,
            construction_policy: options.construction_policy,
        };
        if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
            let canonical = canonical_edge_dataframe_from_imported_table(
                table.deep_copy().map_err(|error| {
                    map_cudf_error_with(
                        error,
                        operation,
                        "failed to deep-copy loaded cudf table for Python-style graph normalization",
                        [],
                    )
                })?,
                &table_opts,
                operation,
            )?;
            build_python_normalized_graph_from_dataframe(
                handle,
                &canonical,
                options.properties,
                options.renumber,
                options.weight_col.is_some(),
                operation,
            )?
        } else {
            let export = table
                .try_prepare_device_table_export()
                .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
            graph_from_prepared_table_export::<T>(handle, &export, &table_opts)?
        }
    };

    Ok(LoadedGraph { graph, table })
}
