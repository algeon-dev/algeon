//! Parquet table readers.

use std::{ffi::CString, os::raw::c_char, ptr, sync::Arc};

use super::{
    super::{collect_column_names, path_to_cstring, validate_input_path},
    evidence::{
        ParquetReadEvidence, empty_pruning_evidence_raw, empty_remote_read_evidence_raw, multi_table_read_evidence,
        pruning_evidence_from_raw, remote_read_evidence_from_raw, validate_required_remote_evidence,
    },
    options::{
        ParquetMultiTableInput, ParquetMultiTableReadOptions, ParquetTableReadOptions, checked_i64_from_usize,
        row_group_combination_error,
    },
    remote::{RemoteParquetSource, RemoteProfileFfi, RemoteReadMode},
};
use crate::{
    core::{
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, Result},
        table::Table,
    },
    io::TableReadOutput,
    runtime::execution_domain::ExecutionDomain,
};

/// Internal Parquet reader output with execution evidence.
pub(crate) struct ParquetTableReadOutput {
    output: TableReadOutput,
    read_evidence: Option<ParquetReadEvidence>,
}

impl ParquetTableReadOutput {
    pub(crate) fn read_evidence(&self) -> Option<&ParquetReadEvidence> {
        self.read_evidence.as_ref()
    }

    pub(crate) fn into_table_read_output(self) -> TableReadOutput {
        self.output
    }
}

fn column_filter_args(
    columns: Option<&Vec<String>>,
    operation: &'static str,
) -> Result<(Option<Vec<CString>>, Vec<*const c_char>)> {
    let cstring_columns: Option<Vec<CString>> = columns
        .map(|cols| {
            cols.iter()
                .enumerate()
                .map(|(index, s)| {
                    CString::new(s.as_str()).map_err(|_| {
                        error::interior_null_byte_error(
                            operation,
                            "column_name",
                            "column name contains interior null byte",
                        )
                        .with_fact(FACT_COLUMN_INDEX, index)
                        .with_column(s.to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let col_ptrs_vec: Vec<*const c_char> =
        cstring_columns.as_ref().map(|cols| cols.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    Ok((cstring_columns, col_ptrs_vec))
}

fn assemble_table_output(
    table_handle: cudf_sys::cudf_table_handle,
    names_ptr: *mut *mut c_char,
    num_names: i32,
    output_domain: Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<(Table, Vec<String>)> {
    // SAFETY: table_handle was returned by a successful FFI call, transferring ownership.
    // Wrap early so table is freed via RAII if collect_column_names() fails.
    let table = unsafe { Table::from_raw_with_domain(table_handle, output_domain) }?;

    // SAFETY: names_ptr and num_names were returned by a successful FFI call.
    // cudf_io_free_column_names frees the C-allocated name strings and array.
    let column_names = unsafe {
        let names = collect_column_names(names_ptr, num_names, operation);
        cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        names?
    };
    Ok((table, column_names))
}

struct SingleSourceSelection {
    filepath: Option<CString>,
    remote_reader_path: Option<CString>,
    evidence: ParquetReadEvidence,
    label: &'static str,
}

fn prepare_single_source(options: &ParquetTableReadOptions<'_>) -> Result<SingleSourceSelection> {
    const OPERATION: &str = "io::parquet::read_parquet_table";
    let filepath = options.filepath.as_ref().map(|path| path_to_cstring(path, OPERATION)).transpose()?;
    if let Some(path) = options.filepath.as_ref() {
        validate_input_path(path, OPERATION)?;
    }
    let remote_reader_path = options
        .remote_source
        .as_ref()
        .map(|source| {
            source.validate_contract_for_current_reader(OPERATION)?;
            CString::new(source.reader_path()).map_err(|_| {
                error::interior_null_byte_error(
                    OPERATION,
                    "remote_parquet_uri",
                    "remote Parquet URI contains interior null byte",
                )
                .with_detail("resource", "remote_parquet_source")
            })
        })
        .transpose()?;
    let selected_source_count = usize::from(options.filepath.is_some())
        + usize::from(options.remote_source.is_some())
        + usize::from(options.filebytes.is_some());
    match selected_source_count {
        0 => {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                OPERATION,
                "Parquet read options must select a source",
            )
            .with_detail("resource", "parquet_source")
            .with_detail("reason", "missing_source")
            .with_detail("remedy", "set filepath, remote_source, or filebytes"));
        },
        1 => {},
        _ => {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                OPERATION,
                "Parquet read options must select exactly one source",
            )
            .with_detail("resource", "parquet_source")
            .with_detail("reason", "ambiguous_source")
            .with_detail("remedy", "set only one of filepath, remote_source, or filebytes"));
        },
    }
    if matches!(options.filebytes, Some([])) {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            OPERATION,
            "host-buffer Parquet reads require a non-empty buffer",
        )
        .with_detail("resource", "filebytes")
        .with_detail("reason", "empty_host_buffer"));
    }
    let (evidence, label) = if filepath.is_some() {
        (ParquetReadEvidence::local_path_reader(), "filepath")
    } else if remote_reader_path.is_some() {
        let source =
            options.remote_source.as_ref().expect("remote source validation requires source when remote path exists");
        (ParquetReadEvidence::remote_path_reader(source), "remote_path")
    } else {
        (ParquetReadEvidence::host_buffer(), "host_buffer")
    };
    Ok(SingleSourceSelection { filepath, remote_reader_path, evidence, label })
}

fn prepare_single_range(options: &ParquetTableReadOptions<'_>) -> Result<(i64, i64, String)> {
    const OPERATION: &str = "io::parquet::read_parquet_table";
    let num_rows_context = options.num_rows.map(|num_rows| num_rows.to_string()).unwrap_or_else(|| "all".to_string());

    if options.row_groups.is_some() && (options.skip_rows > 0 || options.num_rows.is_some()) {
        return Err(row_group_combination_error(
            options.row_groups.as_ref().map_or(0, Vec::len),
            options.skip_rows,
            num_rows_context,
            OPERATION,
        ));
    }
    let skip_rows = checked_i64_from_usize(options.skip_rows, "skip_rows", OPERATION)?;
    let num_rows = match options.num_rows {
        Some(num_rows) => checked_i64_from_usize(num_rows, "num_rows", OPERATION)?,
        None => -1,
    };
    Ok((skip_rows, num_rows, num_rows_context))
}

struct SingleReadInputs<'a> {
    path_input: Option<&'a CString>,
    use_explicit_remote_datasource: bool,
    remote_profile: Option<&'a cudf_sys::cudf_io_remote_parquet_profile_t>,
    filebytes: Option<&'a [u8]>,
    col_ptrs: *mut *const c_char,
    num_cols: i32,
    skip_rows: i64,
    num_rows: i64,
    row_groups: Option<&'a Vec<i32>>,
    predicate: cudf_sys::cudf_ast_expression_handle,
    exec_options: &'a cudf_sys::cudf_exec_options_t,
}

struct SingleReadOutputs {
    table_handle: cudf_sys::cudf_table_handle,
    names_ptr: *mut *mut c_char,
    num_names: i32,
    remote_raw_evidence: cudf_sys::cudf_io_remote_parquet_read_evidence_t,
    pruning_raw_evidence: cudf_sys::cudf_io_parquet_pruning_evidence_t,
}

fn invoke_single_read(inputs: &SingleReadInputs<'_>, outputs: &mut SingleReadOutputs) -> cudf_sys::cudf_status_t {
    // `rg_ptr` borrows from `inputs.row_groups`, which must outlive the FFI call.
    let (rg_ptr, num_rg) = match inputs.row_groups {
        Some(rg) if !rg.is_empty() => (rg.as_ptr(), rg.len() as i32),
        _ => (ptr::null(), 0),
    };
    if inputs.use_explicit_remote_datasource {
        let filepath = inputs.path_input.expect("remote source validation requires path input");
        let profile_raw = inputs.remote_profile.expect("remote profile exists for remote source");
        // SAFETY: filepath is a valid CString. Profile pointers refer to
        // CStrings owned by remote_profile_ffi. Other pointers follow the same
        // contracts as the path reader.
        unsafe {
            cudf_sys::cudf_io_read_remote_parquet_with_metadata_filter_ex(
                filepath.as_ptr(),
                profile_raw,
                inputs.col_ptrs,
                inputs.num_cols,
                inputs.skip_rows,
                inputs.num_rows,
                rg_ptr,
                num_rg,
                inputs.predicate,
                &mut outputs.table_handle,
                &mut outputs.names_ptr,
                &mut outputs.num_names,
                &mut outputs.remote_raw_evidence,
                &mut outputs.pruning_raw_evidence,
                inputs.exec_options,
            )
        }
    } else if let Some(filepath) = inputs.path_input {
        // SAFETY: filepath is a valid CString. col_ptrs is either null or a
        // valid pointer to num_cols CString pointers. rg_ptr is either null
        // or a valid pointer to num_rg i32 values. predicate is either null
        // or an owned AST handle retained by options for the duration of this call.
        unsafe {
            cudf_sys::cudf_io_read_parquet_with_metadata_filter_ex(
                filepath.as_ptr(),
                inputs.col_ptrs,
                inputs.num_cols,
                inputs.skip_rows,
                inputs.num_rows,
                rg_ptr,
                num_rg,
                inputs.predicate,
                &mut outputs.table_handle,
                &mut outputs.names_ptr,
                &mut outputs.num_names,
                &mut outputs.pruning_raw_evidence,
                inputs.exec_options,
            )
        }
    } else {
        let bytes = inputs.filebytes.expect("source validation requires filebytes when filepath is absent");
        // SAFETY: bytes is a non-empty borrowed host buffer that outlives this
        // FFI call. Other pointers follow the same contracts as the path reader.
        unsafe {
            cudf_sys::cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
                bytes.as_ptr().cast(),
                bytes.len(),
                inputs.col_ptrs,
                inputs.num_cols,
                inputs.skip_rows,
                inputs.num_rows,
                rg_ptr,
                num_rg,
                inputs.predicate,
                &mut outputs.table_handle,
                &mut outputs.names_ptr,
                &mut outputs.num_names,
                &mut outputs.pruning_raw_evidence,
                inputs.exec_options,
            )
        }
    }
}

/// Read a Parquet file into a low-level table result.
///
/// Returns the table data and the column names stored in the Parquet file's
/// schema. This avoids having to hardcode column indices when working with
/// Parquet files.
pub fn read_parquet_table(options: &ParquetTableReadOptions<'_>) -> Result<TableReadOutput> {
    read_parquet_table_with_evidence(options).map(ParquetTableReadOutput::into_table_read_output)
}

pub(crate) fn read_parquet_table_with_evidence(
    options: &ParquetTableReadOptions<'_>,
) -> Result<ParquetTableReadOutput> {
    const OPERATION: &str = "io::parquet::read_parquet_table";
    let source = prepare_single_source(options)?;
    let SingleSourceSelection { filepath, remote_reader_path, evidence: mut read_evidence, label: source_label } =
        source;
    let path_input = filepath.as_ref().or(remote_reader_path.as_ref());
    let (skip_rows, num_rows, num_rows_context) = prepare_single_range(options)?;

    let output_domain = Arc::clone(options.runtime.execution_domain());
    let exec_options = output_domain.enter_execution()?.with_parquet_pass_read_limit(options.pass_read_limit);

    let (_cstring_columns, mut col_ptrs_vec) = column_filter_args(options.columns.as_ref(), OPERATION)?;
    let (col_ptrs, num_cols) = if col_ptrs_vec.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        (col_ptrs_vec.as_mut_ptr(), col_ptrs_vec.len() as i32)
    };

    let predicate = options.predicate.as_ref().map_or(ptr::null_mut(), |predicate| predicate.raw_handle());

    let remote_profile_ffi =
        options.remote_source.as_ref().map(|source| RemoteProfileFfi::new(source.profile(), OPERATION)).transpose()?;
    let remote_profile_raw = options
        .remote_source
        .as_ref()
        .zip(remote_profile_ffi.as_ref())
        .map(|(source, profile_ffi)| profile_ffi.as_raw(source.profile(), source.known_size_bytes()));

    let use_explicit_remote_datasource =
        options.remote_source.as_ref().is_some_and(RemoteParquetSource::should_use_explicit_remote_datasource);
    let inputs = SingleReadInputs {
        path_input,
        use_explicit_remote_datasource,
        remote_profile: remote_profile_raw.as_ref(),
        filebytes: options.filebytes,
        col_ptrs,
        num_cols,
        skip_rows,
        num_rows,
        row_groups: options.row_groups.as_ref(),
        predicate,
        exec_options: exec_options.options(),
    };
    let mut outputs = SingleReadOutputs {
        table_handle: ptr::null_mut(),
        names_ptr: ptr::null_mut(),
        num_names: 0,
        remote_raw_evidence: empty_remote_read_evidence_raw(),
        pruning_raw_evidence: empty_pruning_evidence_raw(),
    };
    let status = invoke_single_read(&inputs, &mut outputs);
    let SingleReadOutputs { table_handle, names_ptr, num_names, remote_raw_evidence, pruning_raw_evidence } = outputs;
    if use_explicit_remote_datasource {
        let source =
            options.remote_source.as_ref().expect("remote source exists when explicit remote datasource is used");
        read_evidence = remote_read_evidence_from_raw(Some(source), "remote_path", remote_raw_evidence);
    }
    read_evidence = read_evidence.with_pruning(pruning_evidence_from_raw(pruning_raw_evidence));

    error::check_status_in_with_details(
        status,
        OPERATION,
        [
            ("column_name_count", num_cols.to_string()),
            ("skip_rows", options.skip_rows.to_string()),
            ("num_rows", num_rows_context),
            ("parquet_source", source_label.to_string()),
        ],
    )?;

    let (table, column_names) = assemble_table_output(table_handle, names_ptr, num_names, output_domain, OPERATION)?;
    validate_required_remote_evidence(
        &read_evidence,
        options.remote_source.as_ref().map(RemoteParquetSource::read_mode),
        OPERATION,
    )?;

    Ok(ParquetTableReadOutput {
        output: TableReadOutput { table, column_names: Some(column_names) },
        read_evidence: Some(read_evidence),
    })
}

fn validate_multi_inputs(options: &ParquetMultiTableReadOptions) -> Result<()> {
    const OPERATION: &str = "io::parquet::read_parquet_multi_table";
    if options.inputs.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, OPERATION, "multi-file Parquet read requires files")
            .with_detail("resource", "filepaths")
            .with_detail("reason", "empty_input"));
    }
    for (file_index, input) in options.inputs.iter().enumerate() {
        match input {
            ParquetMultiTableInput::LocalPath(path) => {
                validate_input_path(path, OPERATION).map_err(|error| {
                    error
                        .with_detail("file_index", file_index.to_string())
                        .with_detail("path", path.display().to_string())
                })?;
            },
            ParquetMultiTableInput::RemoteSource(source) => {
                source.validate_contract_for_current_reader(OPERATION).map_err(|error| {
                    error
                        .with_detail("file_index", file_index.to_string())
                        .with_detail("path", input.display_path())
                        .with_detail("remote_read_mode", source.read_mode().as_str())
                })?;
            },
        }
    }
    if let Some(row_groups_per_file) = &options.row_groups_per_file
        && row_groups_per_file.len() != options.inputs.len()
    {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            OPERATION,
            "row-group selection count must match file count",
        )
        .with_detail("reason", "row_group_file_count_mismatch")
        .with_detail("file_count", options.inputs.len().to_string())
        .with_detail("row_group_file_count", row_groups_per_file.len().to_string()));
    }
    Ok(())
}

fn multi_filepath_args(options: &ParquetMultiTableReadOptions) -> Result<(Vec<CString>, Vec<*const c_char>)> {
    const OPERATION: &str = "io::parquet::read_parquet_multi_table";
    let filepath_cstrings = options
        .inputs
        .iter()
        .enumerate()
        .map(|(file_index, input)| match input {
            ParquetMultiTableInput::LocalPath(path) => path_to_cstring(path, OPERATION).map_err(|error| {
                error.with_detail("file_index", file_index.to_string()).with_detail("path", path.display().to_string())
            }),
            ParquetMultiTableInput::RemoteSource(source) => CString::new(source.reader_path()).map_err(|_| {
                error::interior_null_byte_error(
                    OPERATION,
                    "remote_parquet_uri",
                    "remote Parquet URI contains interior null byte",
                )
                .with_detail("resource", "remote_parquet_source")
                .with_detail("file_index", file_index.to_string())
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let filepath_ptrs = filepath_cstrings.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();
    Ok((filepath_cstrings, filepath_ptrs))
}

fn flatten_multi_row_groups(options: &ParquetMultiTableReadOptions) -> Result<(Vec<i32>, Vec<i32>)> {
    const OPERATION: &str = "io::parquet::read_parquet_multi_table";
    let mut row_group_offsets = Vec::new();
    let mut row_groups = Vec::new();
    if let Some(row_groups_per_file) = &options.row_groups_per_file {
        row_group_offsets.reserve(row_groups_per_file.len() + 1);
        row_group_offsets.push(0_i32);
        for groups in row_groups_per_file {
            row_groups.extend(groups.iter().copied());
            let offset = i32::try_from(row_groups.len())
                .map_err(|_| error::range_limit_error(OPERATION, "row_groups", "i32::MAX", row_groups.len()))?;
            row_group_offsets.push(offset);
        }
    }
    Ok((row_group_offsets, row_groups))
}

struct MultiRemoteArgs {
    profiles: Option<Vec<RemoteProfileFfi>>,
    source_flags: Option<Vec<i32>>,
}

impl MultiRemoteArgs {
    fn has_explicit_remote_sources(&self) -> bool {
        self.source_flags.is_some()
    }

    fn source_flags(&self) -> Option<&[i32]> {
        self.source_flags.as_deref()
    }

    fn raw_profiles(&self, inputs: &[ParquetMultiTableInput]) -> Vec<cudf_sys::cudf_io_remote_parquet_profile_t> {
        let Some(profiles) = self.profiles.as_ref() else {
            return Vec::new();
        };
        inputs
            .iter()
            .zip(profiles)
            .map(|(input, profile_ffi)| match input.remote_source() {
                Some(source) => profile_ffi.as_raw(source.profile(), source.known_size_bytes()),
                None => cudf_sys::cudf_io_remote_parquet_profile_t {
                    aws_region: ptr::null(),
                    aws_access_key_id: ptr::null(),
                    aws_secret_access_key: ptr::null(),
                    aws_session_token: ptr::null(),
                    aws_endpoint_url: ptr::null(),
                    path_style_access: -1,
                    unsigned_request: 0,
                    known_size_bytes: -1,
                },
            })
            .collect()
    }
}

fn prepare_multi_remote(options: &ParquetMultiTableReadOptions) -> Result<MultiRemoteArgs> {
    const OPERATION: &str = "io::parquet::read_parquet_multi_table";
    let explicit_remote_source_flags = options
        .inputs
        .iter()
        .map(|input| input.remote_source().is_some_and(RemoteParquetSource::should_use_explicit_remote_datasource))
        .collect::<Vec<_>>();
    if !explicit_remote_source_flags.iter().any(|flag| *flag) {
        return Ok(MultiRemoteArgs { profiles: None, source_flags: None });
    }
    let profiles = options
        .inputs
        .iter()
        .map(|input| match input.remote_source() {
            Some(source) => RemoteProfileFfi::new(source.profile(), OPERATION),
            None => Ok(RemoteProfileFfi::empty()),
        })
        .collect::<Result<Vec<_>>>()?;
    let source_flags = Some(explicit_remote_source_flags.iter().map(|flag| i32::from(*flag)).collect::<Vec<_>>());
    Ok(MultiRemoteArgs { profiles: Some(profiles), source_flags })
}

fn required_multi_read_mode(options: &ParquetMultiTableReadOptions) -> Option<RemoteReadMode> {
    options
        .inputs
        .iter()
        .filter_map(ParquetMultiTableInput::remote_source)
        .map(RemoteParquetSource::read_mode)
        .find(|mode| *mode == RemoteReadMode::RequireDirect)
        .or_else(|| {
            options
                .inputs
                .iter()
                .filter_map(ParquetMultiTableInput::remote_source)
                .map(RemoteParquetSource::read_mode)
                .find(|mode| *mode == RemoteReadMode::RequireKvikio)
        })
}

fn assemble_multi_evidence(
    options: &ParquetMultiTableReadOptions,
    has_explicit_remote_sources: bool,
    remote_raw_evidence: cudf_sys::cudf_io_remote_parquet_read_evidence_t,
    pruning_raw_evidence: cudf_sys::cudf_io_parquet_pruning_evidence_t,
) -> Option<ParquetReadEvidence> {
    if has_explicit_remote_sources {
        let first_remote = options.inputs.iter().find_map(ParquetMultiTableInput::remote_source);
        Some(remote_read_evidence_from_raw(
            first_remote,
            if options.inputs.iter().all(|input| input.remote_source().is_some()) {
                "remote_path_multi"
            } else {
                "mixed_path_multi"
            },
            remote_raw_evidence,
        ))
    } else {
        multi_table_read_evidence(&options.inputs)
    }
    .map(|evidence| evidence.with_pruning(pruning_evidence_from_raw(pruning_raw_evidence)))
}

/// Read multiple Parquet files into a low-level table result.
///
/// This uses libcudf's multi-source Parquet reader. It does not materialize
/// partition columns or perform DataFrame-level source projection.
pub(crate) fn read_parquet_multi_table(options: &ParquetMultiTableReadOptions) -> Result<ParquetTableReadOutput> {
    const OPERATION: &str = "io::parquet::read_parquet_multi_table";
    validate_multi_inputs(options)?;

    let file_count = i32::try_from(options.inputs.len())
        .map_err(|_| error::range_limit_error(OPERATION, "filepaths", "i32::MAX", options.inputs.len()))?;
    let (_filepath_cstrings, mut filepath_ptrs) = multi_filepath_args(options)?;

    let (_cstring_columns, mut col_ptrs_vec) = column_filter_args(options.columns.as_ref(), OPERATION)?;
    let (col_ptrs, num_cols) = if col_ptrs_vec.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        (col_ptrs_vec.as_mut_ptr(), col_ptrs_vec.len() as i32)
    };

    let (row_group_offsets, row_groups) = flatten_multi_row_groups(options)?;
    let (row_group_offsets_ptr, num_row_group_offsets) = if row_group_offsets.is_empty() {
        (ptr::null(), 0)
    } else {
        (row_group_offsets.as_ptr(), row_group_offsets.len() as i32)
    };
    let (row_groups_ptr, num_row_groups) =
        if row_groups.is_empty() { (ptr::null(), 0) } else { (row_groups.as_ptr(), row_groups.len() as i32) };

    let output_domain = Arc::clone(options.runtime.execution_domain());
    let exec_options = output_domain.enter_execution()?.with_parquet_pass_read_limit(options.pass_read_limit);
    let predicate = options.predicate.as_ref().map_or(ptr::null_mut(), |predicate| predicate.raw_handle());
    let remote = prepare_multi_remote(options)?;

    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut c_char = ptr::null_mut();
    let mut num_names: i32 = 0;
    let mut remote_raw_evidence = empty_remote_read_evidence_raw();
    let mut pruning_raw_evidence = empty_pruning_evidence_raw();

    // SAFETY: filepath_ptrs and column/row-group buffers are valid for this
    // call. The predicate handle, when present, is retained by options.
    let status = if remote.has_explicit_remote_sources() {
        let remote_source_flags = remote.source_flags().expect("remote flags exist when remote sources exist");
        let remote_profile_raw = remote.raw_profiles(&options.inputs);
        unsafe {
            cudf_sys::cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex(
                filepath_ptrs.as_mut_ptr(),
                remote_source_flags.as_ptr(),
                remote_profile_raw.as_ptr(),
                file_count,
                col_ptrs,
                num_cols,
                row_group_offsets_ptr,
                num_row_group_offsets,
                row_groups_ptr,
                num_row_groups,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &mut remote_raw_evidence,
                &mut pruning_raw_evidence,
                exec_options.options(),
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_io_read_parquet_multi_with_metadata_filter_ex(
                filepath_ptrs.as_mut_ptr(),
                file_count,
                col_ptrs,
                num_cols,
                row_group_offsets_ptr,
                num_row_group_offsets,
                row_groups_ptr,
                num_row_groups,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &mut pruning_raw_evidence,
                exec_options.options(),
            )
        }
    };

    error::check_status_in_with_details(
        status,
        OPERATION,
        [
            ("file_count", options.inputs.len().to_string()),
            ("column_name_count", num_cols.to_string()),
            ("row_group_count", row_groups.len().to_string()),
        ],
    )?;

    let (table, column_names) = assemble_table_output(table_handle, names_ptr, num_names, output_domain, OPERATION)?;

    let read_evidence = assemble_multi_evidence(
        options,
        remote.has_explicit_remote_sources(),
        remote_raw_evidence,
        pruning_raw_evidence,
    );
    if let Some(evidence) = read_evidence.as_ref() {
        validate_required_remote_evidence(evidence, required_multi_read_mode(options), OPERATION)?;
    }
    Ok(ParquetTableReadOutput { output: TableReadOutput { table, column_names: Some(column_names) }, read_evidence })
}
