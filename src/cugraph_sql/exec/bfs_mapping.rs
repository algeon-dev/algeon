use std::{mem::size_of, sync::Arc};

use super::*;

pub(super) struct PreparedCugraphInputSchemas {
    pub(super) execution_spec: CugraphAlgorithmSpec,
    pub(super) edge_schema: SchemaRef,
    pub(super) predicate_schemas: Vec<SchemaRef>,
    pub(super) logical_mapping: bool,
}

#[derive(Clone, Debug)]
pub(super) struct HostBfsCodebook {
    values: Arc<Vec<String>>,
}

impl HostBfsCodebook {
    fn from_sources(sources: &[(&StagedCugraphSource, Vec<String>)]) -> Result<Self> {
        Self::from_sources_with_limit(sources, i32::MAX as usize)
    }

    pub(super) fn from_sources_with_limit(
        sources: &[(&StagedCugraphSource, Vec<String>)],
        max_distinct: usize,
    ) -> Result<Self> {
        let bound = Self::allocation_upper_bound(sources)?;
        let mut values = Vec::with_capacity(bound.value_count);
        for_each_codebook_array(sources, |array, column| collect_string_values(array, &mut values, column))?;
        values.sort_unstable();
        values.dedup();
        if values.len() > max_distinct {
            return Err(Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host codebook exceeds its checked Int32 domain",
            )
            .with_fact("reason", "logical_bfs_codebook_overflow")
            .with_fact("distinct_values", values.len())
            .with_fact("max_distinct_values", max_distinct)
            .into());
        }
        values.shrink_to_fit();
        let values = Arc::new(values);
        for (index, _) in values.iter().enumerate() {
            i32::try_from(index).map_err(|_| {
                Error::new(
                    ErrorCode::SizeOverflow,
                    "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                    "logical BFS host codebook index does not fit Int32",
                )
                .with_fact("reason", "logical_bfs_codebook_overflow")
                .with_fact("index", index)
            })?;
        }
        Ok(Self { values })
    }

    pub(super) fn allocation_upper_bound(
        sources: &[(&StagedCugraphSource, Vec<String>)],
    ) -> Result<HostBfsCodebookAllocationBound> {
        let mut value_count = 0usize;
        let mut string_bytes = 0usize;
        for_each_codebook_array(sources, |array, column| {
            let (batch_values, batch_bytes) = string_array_bound(array, column)?;
            value_count = value_count.checked_add(batch_values).ok_or_else(|| {
                Error::new(
                    ErrorCode::SizeOverflow,
                    "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                    "logical BFS host codebook value count overflowed usize",
                )
                .with_fact("reason", "logical_bfs_codebook_size_overflow")
            })?;
            string_bytes = string_bytes.checked_add(batch_bytes).ok_or_else(|| {
                Error::new(
                    ErrorCode::SizeOverflow,
                    "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                    "logical BFS host codebook string bytes overflowed usize",
                )
                .with_fact("reason", "logical_bfs_codebook_size_overflow")
            })?;
            Ok(())
        })?;
        let vector_bytes = value_count.checked_mul(size_of::<String>()).ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host codebook vector bytes overflowed usize",
            )
            .with_fact("reason", "logical_bfs_codebook_size_overflow")
        })?;
        let vector_overlap = vector_bytes.checked_mul(2).ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host codebook vector overlap overflowed usize",
            )
            .with_fact("reason", "logical_bfs_codebook_size_overflow")
        })?;
        let bytes = vector_overlap.checked_add(string_bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host codebook allocation bound overflowed usize",
            )
            .with_fact("reason", "logical_bfs_codebook_size_overflow")
        })?;
        Ok(HostBfsCodebookAllocationBound { bytes, value_count })
    }

    pub(super) fn code(&self, value: &str) -> Option<i32> {
        self.values
            .binary_search_by(|candidate| candidate.as_str().cmp(value))
            .ok()
            .and_then(|index| i32::try_from(index).ok())
    }

    pub(super) fn decode_code(&self, code: i64) -> Result<String> {
        Ok(self.value_for_code(code)?.to_owned())
    }

    fn value_for_code(&self, code: i64) -> Result<&str> {
        let index = usize::try_from(code).map_err(|_| logical_bfs_unknown_code_error(code))?;
        self.values.get(index).map(String::as_str).ok_or_else(|| logical_bfs_unknown_code_error(code))
    }

    pub(super) fn memory_size(&self) -> Result<usize> {
        let vector_bytes = self.values.capacity().checked_mul(size_of::<String>());
        let string_bytes = self.values.iter().try_fold(0usize, |total, value| total.checked_add(value.capacity()));
        vector_bytes.and_then(|vector| string_bytes.and_then(|strings| vector.checked_add(strings))).ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host codebook size overflowed usize",
            )
            .with_fact("reason", "logical_bfs_codebook_size_overflow")
            .into()
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct HostBfsCodebookAllocationBound {
    pub(super) bytes: usize,
    value_count: usize,
}

#[derive(Debug)]
pub(super) struct PreparedBfsLogicalHostInputs {
    pub(super) edge: StagedCugraphSource,
    pub(super) logical_literal_source: Option<StagedCugraphSource>,
    pub(super) predicates: Vec<StagedCugraphSource>,
    pub(super) logical_codebook: Option<HostBfsCodebook>,
    pub(super) execution_spec: CugraphAlgorithmSpec,
}

pub(super) fn prepare_bfs_logical_host_inputs(
    spec: &CugraphAlgorithmSpec,
    edge: StagedCugraphSource,
    logical_literal_source: Option<StagedCugraphSource>,
    predicates: Vec<StagedCugraphSource>,
    staged_bytes: &mut usize,
    max_staging_bytes: usize,
    reservation: &mut MemoryReservation,
    host_evidence: &mut HostReservationEvidence,
) -> Result<PreparedBfsLogicalHostInputs> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(PreparedBfsLogicalHostInputs {
            edge,
            logical_literal_source,
            predicates,
            logical_codebook: None,
            execution_spec: spec.clone(),
        });
    };
    if !bfs_uses_logical_mapping(spec, params, &edge.schema())? {
        return Ok(PreparedBfsLogicalHostInputs {
            edge,
            logical_literal_source,
            predicates,
            logical_codebook: None,
            execution_spec: spec.clone(),
        });
    }

    let StagedCugraphSource::Host { .. } = &edge else {
        return Err(logical_bfs_native_source_error("edge", "native"));
    };
    if let Some(source) = &logical_literal_source
        && !matches!(source, StagedCugraphSource::Host { .. })
    {
        return Err(logical_bfs_native_source_error("source_vertices", "native"));
    }
    if let Some((index, _)) =
        predicates.iter().enumerate().find(|(_, source)| matches!(source, StagedCugraphSource::Native { .. }))
    {
        return Err(logical_bfs_native_source_error("side_input", format!("native_index_{index}")));
    }

    validate_bfs_logical_edge_schema(spec, &edge.schema())?;
    let edge_columns = vec![spec.src_col.clone(), spec.dst_col.clone()];
    let literal_columns = logical_literal_source.as_ref().map(|_| vec!["__source_vertex".to_string()]);
    let predicate_columns = predicates
        .iter()
        .enumerate()
        .map(|(index, _)| bfs_logical_predicate_columns(params, index))
        .collect::<Vec<_>>();

    let mut domain_sources = Vec::with_capacity(2 + predicates.len() + usize::from(logical_literal_source.is_some()));
    domain_sources.push((&edge, edge_columns));
    if let (Some(source), Some(columns)) = (&logical_literal_source, literal_columns.as_ref()) {
        domain_sources.push((source, columns.clone()));
    }
    for (source, columns) in predicates.iter().zip(predicate_columns.iter()) {
        domain_sources.push((source, columns.clone()));
    }
    let codebook_bound = HostBfsCodebook::allocation_upper_bound(&domain_sources)?;
    reserve_logical_mapping_bytes("codebook", codebook_bound.bytes, staged_bytes, max_staging_bytes, reservation)?;
    if let Err(error) = host_evidence.record_input_peak(reservation) {
        release_logical_mapping_bytes(codebook_bound.bytes, staged_bytes, reservation)?;
        return Err(error);
    }
    let codebook_result = (|| {
        let codebook = HostBfsCodebook::from_sources(&domain_sources)?;
        let codebook_bytes = codebook.memory_size()?;
        if codebook_bytes > codebook_bound.bytes {
            return Err(Error::new(
                ErrorCode::CudfInvariantViolation,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS codebook exceeded its checked pre-reserved allocation bound",
            )
            .with_fact("reserved_bytes", codebook_bound.bytes)
            .with_fact("actual_bytes", codebook_bytes)
            .with_fact("reason", "logical_bfs_codebook_bound_mismatch")
            .into());
        }
        Ok((codebook, codebook_bytes))
    })();
    let (codebook, codebook_bytes) = match codebook_result {
        Ok(result) => result,
        Err(error) => {
            release_logical_mapping_bytes(codebook_bound.bytes, staged_bytes, reservation)?;
            return Err(error);
        },
    };
    let codebook_excess = codebook_bound.bytes - codebook_bytes;
    release_logical_mapping_bytes(codebook_excess, staged_bytes, reservation)?;
    host_evidence.record_codebook(codebook_bytes);
    host_evidence.record_input_peak(reservation)?;

    let edge = encode_staged_source(
        edge,
        &[spec.src_col.clone(), spec.dst_col.clone()],
        &codebook,
        staged_bytes,
        max_staging_bytes,
        reservation,
        host_evidence,
    )?;
    let logical_literal_source = logical_literal_source
        .map(|source| {
            encode_staged_source(
                source,
                &["__source_vertex".to_string()],
                &codebook,
                staged_bytes,
                max_staging_bytes,
                reservation,
                host_evidence,
            )
        })
        .transpose()?;
    let predicates = predicates
        .into_iter()
        .enumerate()
        .map(|(index, source)| {
            let columns = bfs_logical_predicate_columns(params, index);
            encode_staged_source(
                source,
                &columns,
                &codebook,
                staged_bytes,
                max_staging_bytes,
                reservation,
                host_evidence,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let mut execution_spec = spec.clone();
    execution_spec.output_schema = bfs_numeric_execution_schema(params)?;
    Ok(PreparedBfsLogicalHostInputs {
        edge,
        logical_literal_source,
        predicates,
        logical_codebook: Some(codebook),
        execution_spec,
    })
}

fn encode_staged_source(
    source: StagedCugraphSource,
    columns: &[String],
    codebook: &HostBfsCodebook,
    staged_bytes: &mut usize,
    max_staging_bytes: usize,
    reservation: &mut MemoryReservation,
    host_evidence: &mut HostReservationEvidence,
) -> Result<StagedCugraphSource> {
    let StagedCugraphSource::Host { schema, batches } = source else {
        return Err(logical_bfs_native_source_error("logical_input", "native"));
    };
    let mapped_schema = remap_bfs_logical_schema(&schema, &columns.iter().map(String::as_str).collect::<Vec<_>>())?;
    let mut mapped_batches = Vec::with_capacity(batches.len());
    for batch in batches {
        let original_bytes = get_record_batch_memory_size(&batch);
        let encoded_upper_bound = original_bytes.checked_add(size_of::<RecordBatch>()).ok_or_else(|| {
            host_staging_limit_error("logical_bfs", "encoded_input", *staged_bytes, usize::MAX, max_staging_bytes)
        })?;
        let peak_bytes = staged_bytes.checked_add(encoded_upper_bound).ok_or_else(|| {
            host_staging_limit_error(
                "logical_bfs",
                "encoded_input",
                *staged_bytes,
                encoded_upper_bound,
                max_staging_bytes,
            )
        })?;
        if peak_bytes > max_staging_bytes {
            return Err(host_staging_limit_error(
                "logical_bfs",
                "encoded_input",
                *staged_bytes,
                encoded_upper_bound,
                max_staging_bytes,
            ));
        }
        reservation.try_grow(encoded_upper_bound).map_err(|source| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::MemoryReservationFailed,
                    "datafusion_nexus.cugraph_sql.host_staging_reservation",
                    "DataFusion host memory pool denied encoded logical BFS staging",
                )
                .with_fact("boundary", "encoded_input")
                .with_fact("requested_bytes", encoded_upper_bound)
                .with_fact("superseded_bytes", original_bytes)
                .with_source(source),
            )
        })?;
        host_evidence.record_input_peak(reservation)?;
        let mut arrays = batch.columns().to_vec();
        for column in columns {
            let index = batch.schema().index_of(column).map_err(DataFusionError::from)?;
            arrays[index] = encode_logical_array(batch.column(index).as_ref(), codebook, column)?;
        }
        let mapped = RecordBatch::try_new(Arc::clone(&mapped_schema), arrays).map_err(DataFusionError::from)?;
        let mapped_bytes = get_record_batch_memory_size(&mapped);
        if mapped_bytes > encoded_upper_bound {
            return Err(Error::new(
                ErrorCode::CudfInvariantViolation,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS encoded batch exceeded its reserved upper bound",
            )
            .with_fact("reserved_bytes", encoded_upper_bound)
            .with_fact("actual_bytes", mapped_bytes)
            .into());
        }
        let next_bytes = staged_bytes
            .checked_sub(original_bytes)
            .and_then(|bytes| bytes.checked_add(mapped_bytes))
            .ok_or_else(|| {
                host_staging_limit_error("logical_bfs", "encoded_input", *staged_bytes, mapped_bytes, max_staging_bytes)
            })?;
        if next_bytes > max_staging_bytes {
            return Err(host_staging_limit_error(
                "logical_bfs",
                "encoded_input",
                *staged_bytes,
                mapped_bytes,
                max_staging_bytes,
            ));
        }
        // End the old batch ownership before lowering its charged staging bytes.
        drop(batch);
        reservation.shrink(original_bytes);
        reservation.shrink(encoded_upper_bound - mapped_bytes);
        *staged_bytes = next_bytes;
        host_evidence.record_input_peak(reservation)?;
        mapped_batches.push(mapped);
    }
    Ok(StagedCugraphSource::Host { schema: mapped_schema, batches: mapped_batches })
}

fn reserve_logical_mapping_bytes(
    boundary: &'static str,
    requested_bytes: usize,
    staged_bytes: &mut usize,
    max_staging_bytes: usize,
    reservation: &mut MemoryReservation,
) -> Result<()> {
    let next = staged_bytes.checked_add(requested_bytes).ok_or_else(|| {
        host_staging_limit_error("logical_bfs", boundary, *staged_bytes, requested_bytes, max_staging_bytes)
    })?;
    if next > max_staging_bytes {
        return Err(host_staging_limit_error(
            "logical_bfs",
            boundary,
            *staged_bytes,
            requested_bytes,
            max_staging_bytes,
        ));
    }
    reservation.try_grow(requested_bytes).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied logical BFS mapping storage",
            )
            .with_fact("boundary", boundary)
            .with_fact("requested_bytes", requested_bytes)
            .with_fact("staged_bytes", *staged_bytes)
            .with_source(source),
        )
    })?;
    *staged_bytes = next;
    Ok(())
}

fn release_logical_mapping_bytes(
    released_bytes: usize,
    staged_bytes: &mut usize,
    reservation: &mut MemoryReservation,
) -> Result<()> {
    let next = staged_bytes
        .checked_sub(released_bytes)
        .ok_or_else(|| plan_error("logical BFS host mapping reservation accounting underflowed"))?;
    reservation.shrink(released_bytes);
    *staged_bytes = next;
    Ok(())
}

fn for_each_codebook_array<F>(sources: &[(&StagedCugraphSource, Vec<String>)], mut visit: F) -> Result<()>
where F: FnMut(&dyn Array, &str) -> Result<()> {
    for (source, columns) in sources {
        let StagedCugraphSource::Host { schema, batches } = source else {
            return Err(logical_bfs_native_source_error("codebook", "native"));
        };
        for column in columns {
            let field = schema.field_with_name(column).map_err(|error| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                        "logical BFS host mapping could not find a required vertex column",
                    )
                    .with_fact("column", column)
                    .with_source(error),
                )
            })?;
            if !is_bfs_logical_vertex_type(field.data_type()) {
                return Err(Error::new(
                    ErrorCode::UnsupportedType,
                    "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                    "logical BFS host mapping requires UTF8 vertex columns",
                )
                .with_fact("column", column)
                .with_fact("data_type", format!("{:?}", field.data_type()))
                .with_fact("reason", "logical_bfs_vertex_dtype_unsupported")
                .into());
            }
            for batch in batches {
                let index = batch.schema().index_of(column).map_err(DataFusionError::from)?;
                visit(batch.column(index).as_ref(), column)?;
            }
        }
    }
    Ok(())
}

fn string_array_bound(array: &dyn Array, column: &str) -> Result<(usize, usize)> {
    let mut value_count = Some(0usize);
    let mut string_bytes = Some(0usize);
    let mut record = |value: &str| {
        value_count = value_count.and_then(|count| count.checked_add(1));
        string_bytes = string_bytes.and_then(|bytes| bytes.checked_add(value.len()));
    };
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                record(array.value(row));
            }
        }
    } else if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                record(array.value(row));
            }
        }
    } else if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                record(array.value(row));
            }
        }
    } else {
        return Err(Error::new(
            ErrorCode::UnsupportedType,
            "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
            "logical BFS host mapping received a non-string vertex array",
        )
        .with_fact("column", column)
        .with_fact("data_type", format!("{:?}", array.data_type()))
        .with_fact("reason", "logical_bfs_vertex_dtype_unsupported")
        .into());
    }
    let value_count = value_count.ok_or_else(|| {
        Error::new(
            ErrorCode::SizeOverflow,
            "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
            "logical BFS host codebook value count overflowed usize",
        )
        .with_fact("reason", "logical_bfs_codebook_size_overflow")
    })?;
    let string_bytes = string_bytes.ok_or_else(|| {
        Error::new(
            ErrorCode::SizeOverflow,
            "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
            "logical BFS host codebook string bytes overflowed usize",
        )
        .with_fact("reason", "logical_bfs_codebook_size_overflow")
    })?;
    Ok((value_count, string_bytes))
}

fn collect_string_values(array: &dyn Array, values: &mut Vec<String>, column: &str) -> Result<()> {
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                let value = array.value(row);
                let mut owned = String::with_capacity(value.len());
                owned.push_str(value);
                values.push(owned);
            }
        }
        return Ok(());
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                let value = array.value(row);
                let mut owned = String::with_capacity(value.len());
                owned.push_str(value);
                values.push(owned);
            }
        }
        return Ok(());
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                let value = array.value(row);
                let mut owned = String::with_capacity(value.len());
                owned.push_str(value);
                values.push(owned);
            }
        }
        return Ok(());
    }
    Err(Error::new(
        ErrorCode::UnsupportedType,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
        "logical BFS host mapping received a non-string vertex array",
    )
    .with_fact("column", column)
    .with_fact("data_type", format!("{:?}", array.data_type()))
    .with_fact("reason", "logical_bfs_vertex_dtype_unsupported")
    .into())
}

pub(super) fn encode_logical_array(array: &dyn Array, codebook: &HostBfsCodebook, column: &str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(None);
            continue;
        }
        let value = if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
            array.value(row)
        } else if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
            array.value(row)
        } else if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
            array.value(row)
        } else {
            return Err(Error::new(
                ErrorCode::UnsupportedType,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host mapping received a non-string vertex array",
            )
            .with_fact("column", column)
            .with_fact("data_type", format!("{:?}", array.data_type()))
            .with_fact("reason", "logical_bfs_vertex_dtype_unsupported")
            .into());
        };
        let code = codebook.code(value).ok_or_else(|| {
            Error::new(
                ErrorCode::CudfInvariantViolation,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host mapping value was absent from its deterministic codebook",
            )
            .with_fact("column", column)
            .with_fact("reason", "logical_bfs_codebook_value_missing")
        })?;
        values.push(Some(code));
    }
    Ok(Arc::new(Int32Array::from(values)) as ArrayRef)
}

fn logical_bfs_native_source_error(source_kind: &'static str, source_state: impl Into<String>) -> DataFusionError {
    Error::new(
        ErrorCode::RequiredGpuOperationUnsupported,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
        "logical BFS requires host Arrow sources before admission",
    )
    .with_fact("reason", "logical_bfs_native_source_unsupported")
    .with_fact("source_kind", source_kind)
    .with_fact("source_state", source_state.into())
    .into()
}

fn logical_bfs_unknown_code_error(code: i64) -> DataFusionError {
    Error::new(
        ErrorCode::CudfInvariantViolation,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
        "logical BFS output contained a code outside its host codebook",
    )
    .with_fact("reason", "logical_bfs_output_code_unknown")
    .with_fact("code", code)
    .into()
}

pub(super) fn prepare_bfs_logical_input_schemas(
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    predicate_schemas: Vec<SchemaRef>,
) -> Result<PreparedCugraphInputSchemas> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(PreparedCugraphInputSchemas {
            execution_spec: spec.clone(),
            edge_schema,
            predicate_schemas,
            logical_mapping: false,
        });
    };
    if !validate_bfs_logical_mapping_capability(spec, params, &edge_schema)? {
        return Ok(PreparedCugraphInputSchemas {
            execution_spec: spec.clone(),
            edge_schema,
            predicate_schemas,
            logical_mapping: false,
        });
    }
    validate_bfs_logical_edge_schema(spec, &edge_schema)?;
    let edge_schema = remap_bfs_logical_schema(&edge_schema, &[&spec.src_col, &spec.dst_col])?;
    let mut mapped_predicates = Vec::with_capacity(predicate_schemas.len() + usize::from(params.source.is_utf8()));
    if params.source.is_utf8() {
        mapped_predicates.push(Arc::new(Schema::new(vec![arrow::datatypes::Field::new(
            "__source_vertex",
            DataType::Int32,
            false,
        )])));
    }
    for (index, schema) in predicate_schemas.into_iter().enumerate() {
        let columns = bfs_logical_predicate_columns(params, index);
        let columns = columns.iter().map(String::as_str).collect::<Vec<_>>();
        mapped_predicates.push(remap_bfs_logical_schema(&schema, &columns)?);
    }
    let mut execution_spec = spec.clone();
    execution_spec.output_schema = bfs_numeric_execution_schema(params)?;
    Ok(PreparedCugraphInputSchemas {
        execution_spec,
        edge_schema,
        predicate_schemas: mapped_predicates,
        logical_mapping: true,
    })
}

fn validate_bfs_logical_mapping_capability(
    spec: &CugraphAlgorithmSpec,
    params: &super::CugraphBfsParams,
    edge_schema: &SchemaRef,
) -> Result<bool> {
    let logical_mapping = bfs_uses_logical_mapping(spec, params, edge_schema)?;
    super::super::validate_bfs_logical_edge_id_capability(spec, logical_mapping)?;
    Ok(logical_mapping)
}

pub(super) fn validate_bfs_logical_edge_schema(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<()> {
    for column in [&spec.src_col, &spec.dst_col] {
        let field = edge_schema.field_with_name(column).map_err(|_| {
            plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{column}'"))
        })?;
        if !is_bfs_logical_vertex_type(field.data_type()) {
            return Err(plan_error(format!(
                "cuGraph BFS logical vertex mapping requires a string edge column '{column}', found {:?}",
                field.data_type()
            )));
        }
    }
    Ok(())
}

pub(super) fn bfs_uses_logical_mapping(
    spec: &CugraphAlgorithmSpec,
    params: &super::CugraphBfsParams,
    edge_schema: &SchemaRef,
) -> Result<bool> {
    if params.source.is_utf8() {
        return Ok(true);
    }
    if !matches!(params.source, CugraphBfsSource::Relation(_)) {
        return Ok(false);
    }
    let source_field = edge_schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{}'", spec.src_col))
    })?;
    let destination_field = edge_schema.field_with_name(&spec.dst_col).map_err(|_| {
        plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{}'", spec.dst_col))
    })?;
    Ok(is_bfs_logical_vertex_type(source_field.data_type())
        && is_bfs_logical_vertex_type(destination_field.data_type()))
}

pub(super) fn bfs_literal_string_source_batch(spec: &CugraphAlgorithmSpec) -> Result<Option<RecordBatch>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    let Some(values) = params.source.literal_values() else {
        return Ok(None);
    };
    if !values.first().is_some_and(CugraphBfsSourceVertex::is_utf8) {
        return Ok(None);
    }
    let values = values
        .iter()
        .map(|value| match value {
            CugraphBfsSourceVertex::Utf8(value) => Ok(value.as_str()),
            CugraphBfsSourceVertex::Int64(_) => Err(plan_error("string BFS source list contained a numeric value")),
        })
        .collect::<Result<Vec<_>>>()?;
    let schema = Arc::new(Schema::new(vec![arrow::datatypes::Field::new("__source_vertex", DataType::Utf8, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values)) as ArrayRef])
        .map(Some)
        .map_err(DataFusionError::from)
}

pub(super) fn is_bfs_logical_vertex_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

pub(super) fn bfs_logical_predicate_columns(params: &super::CugraphBfsParams, index: usize) -> Vec<String> {
    let source_offset = bfs_planned_source_input_offset(params);
    if index < source_offset {
        return match &params.source {
            CugraphBfsSource::Relation(input) => vec![input.column.clone()],
            _ => Vec::new(),
        };
    }
    let index = index - source_offset;
    let mut expected_index = 0;
    for input in [
        params.predicates.include_vertices.as_ref(),
        params.predicates.exclude_vertices.as_ref(),
        params.predicates.target_vertices.as_ref(),
        params.predicates.include_edge_ids.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if index == expected_index {
            return vec![input.column.clone()];
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        return vec![input.source_col.clone(), input.destination_col.clone()];
    }
    Vec::new()
}

pub(super) fn bfs_planned_source_input_offset(params: &super::CugraphBfsParams) -> usize {
    usize::from(params.source.planned_source_vertices_input())
}

pub(super) fn remap_bfs_logical_schema(schema: &SchemaRef, columns: &[&str]) -> Result<SchemaRef> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if columns.iter().any(|column| *column == field.name()) {
                arrow::datatypes::Field::new(field.name(), DataType::Int32, field.is_nullable())
            } else {
                field.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn bfs_numeric_execution_schema(params: &super::CugraphBfsParams) -> Result<SchemaRef> {
    let native_schema = bfs_native_result_schema(params.output.return_target_info);
    let fields: Vec<_> = match params.output.mode {
        CugraphBfsOutputMode::Raw => native_schema.fields().iter().map(|field| field.as_ref().clone()).collect(),
        CugraphBfsOutputMode::Normalized => Arc::new(Schema::new(vec![
            normalized_bfs_base_field(&native_schema, "vertex", false)?,
            normalized_bfs_base_field(&native_schema, "distance", true)?,
            normalized_bfs_base_field(&native_schema, "predecessor", true)?,
            arrow::datatypes::Field::new("reachable", DataType::Boolean, false),
        ]))
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .chain(native_schema.fields().iter().skip(3).map(|field| field.as_ref().clone()))
        .collect(),
        CugraphBfsOutputMode::Path => native_schema.fields().iter().map(|field| field.as_ref().clone()).collect(),
    };
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn bfs_native_result_schema(return_target_info: bool) -> SchemaRef {
    let mut native_params = GraphBfsParams::new(0, None);
    native_params.return_target_info = return_target_info;
    GraphAlgorithm::Bfs.definition().result_schema_for_params(&GraphAlgorithmParams::Bfs(Box::new(native_params)))
}

pub(super) fn normalized_bfs_base_field(
    schema: &SchemaRef,
    name: &'static str,
    nullable: bool,
) -> Result<arrow::datatypes::Field> {
    let field = schema.field_with_name(name).map_err(|error| {
        DataFusionError::from(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.bfs_result_schema",
                "native BFS descriptor is missing a raw BFS base field",
            )
            .with_fact("column", name)
            .with_source(error),
        )
    })?;
    Ok(arrow::datatypes::Field::new(field.name(), field.data_type().clone(), nullable))
}

pub(super) fn bfs_decoded_output_schema_for_host(spec: &CugraphAlgorithmSpec) -> SchemaRef {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Arc::clone(&spec.output_schema);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Arc::clone(&spec.output_schema);
    }
    Arc::new(Schema::new(vec![
        arrow::datatypes::Field::new("vertex", DataType::Utf8, false),
        arrow::datatypes::Field::new("distance", DataType::Int64, false),
        arrow::datatypes::Field::new("predecessor", DataType::Utf8, true),
    ]))
}

pub(super) fn decode_bfs_logical_host_output(
    output: EagerCugraphHostOutput,
    codebook: &HostBfsCodebook,
    output_schema: SchemaRef,
) -> Result<EagerCugraphHostOutput> {
    let (batches, reservation, host_evidence) = output.into_parts_with_evidence();
    let old_bytes =
        batches.iter().map(get_record_batch_memory_size).try_fold(0usize, |sum, bytes| sum.checked_add(bytes));
    let old_bytes = old_bytes.ok_or_else(|| plan_error("logical BFS host output size overflowed usize"))?;
    let decoded_upper_bound = batches.iter().try_fold(0usize, |sum, batch| {
        decoded_bfs_logical_host_batch_upper_bound(batch, codebook).and_then(|bytes| {
            sum.checked_add(bytes).ok_or_else(|| {
                Error::new(
                    ErrorCode::SizeOverflow,
                    "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
                    "decoded logical BFS host output bound overflowed usize",
                )
                .with_fact("boundary", "decoded_output")
                .with_fact("reason", "logical_bfs_decoded_output_bound_overflow")
                .into()
            })
        })
    })?;
    if let Some(evidence) = host_evidence.as_ref() {
        evidence.check_output_growth("decoded_output", reservation.size(), decoded_upper_bound)?;
    }
    reservation.try_grow(decoded_upper_bound).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied decoded logical BFS output staging",
            )
            .with_fact("boundary", "decoded_output")
            .with_fact("requested_bytes", decoded_upper_bound)
            .with_source(source),
        )
    })?;
    let decoded_result = (|| {
        let mut decoded = Vec::with_capacity(batches.len());
        for batch in batches.iter() {
            decoded.push(decode_bfs_logical_host_batch(batch, codebook, Arc::clone(&output_schema))?);
        }
        let new_bytes =
            decoded.iter().map(get_record_batch_memory_size).try_fold(0usize, |sum, bytes| sum.checked_add(bytes));
        let new_bytes = new_bytes.ok_or_else(|| plan_error("decoded logical BFS host output size overflowed usize"))?;
        if new_bytes > decoded_upper_bound {
            return Err(Error::new(
                ErrorCode::CudfInvariantViolation,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
                "decoded logical BFS host output exceeded its checked pre-reserved bound",
            )
            .with_fact("boundary", "decoded_output")
            .with_fact("reserved_bytes", decoded_upper_bound)
            .with_fact("actual_bytes", new_bytes)
            .with_fact("reason", "logical_bfs_decoded_output_bound_mismatch")
            .into());
        }
        Ok((decoded, new_bytes))
    })();
    let (decoded, new_bytes) = match decoded_result {
        Ok(result) => result,
        Err(error) => {
            reservation.shrink(decoded_upper_bound);
            return Err(error);
        },
    };
    reservation.shrink(decoded_upper_bound - new_bytes);
    // End encoded batch ownership before lowering its charged output bytes.
    drop(batches);
    reservation.shrink(old_bytes);
    Ok(EagerCugraphHostOutput::from_shared_parts_with_evidence(
        output_schema,
        Arc::new(decoded),
        reservation,
        host_evidence,
    ))
}

pub(super) fn decode_bfs_logical_host_batch(
    batch: &RecordBatch,
    codebook: &HostBfsCodebook,
    output_schema: SchemaRef,
) -> Result<RecordBatch> {
    let mut columns = batch.columns().to_vec();
    if let Ok(index) = batch.schema().index_of("vertex") {
        columns[index] = decode_vertex_array(batch.column(index).as_ref(), codebook, "vertex")?;
    }
    if let Ok(index) = batch.schema().index_of("predecessor") {
        columns[index] = decode_predecessor_array(batch.column(index).as_ref(), codebook)?;
    }
    RecordBatch::try_new(output_schema, columns).map_err(DataFusionError::from)
}

pub(super) fn decoded_bfs_logical_host_batch_upper_bound(
    batch: &RecordBatch,
    codebook: &HostBfsCodebook,
) -> Result<usize> {
    batch.columns().iter().enumerate().try_fold(0usize, |total, (index, array)| {
        let schema = batch.schema();
        let column = schema.field(index).name();
        let bytes = match column.as_str() {
            "vertex" => decoded_string_array_upper_bound(array.as_ref(), codebook, "vertex", false)?,
            "predecessor" => decoded_string_array_upper_bound(array.as_ref(), codebook, "predecessor", true)?,
            _ => array.get_array_memory_size(),
        };
        total.checked_add(bytes).ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::SizeOverflow,
                    "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
                    "decoded logical BFS host output batch bound overflowed usize",
                )
                .with_fact("boundary", "decoded_output")
                .with_fact("reason", "logical_bfs_decoded_output_bound_overflow"),
            )
        })
    })
}

fn decode_vertex_array(array: &dyn Array, codebook: &HostBfsCodebook, column: &'static str) -> Result<ArrayRef> {
    let data_bytes = decoded_string_array_data_bytes(array, codebook, column, false)?;
    let mut builder = arrow::array::StringBuilder::with_capacity(array.len(), data_bytes);
    for row in 0..array.len() {
        if array.is_null(row) {
            builder.append_null();
        } else {
            let value = codebook.value_for_code(numeric_array_value(array, row, column)?)?;
            builder.append_value(value);
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn decode_predecessor_array(array: &dyn Array, codebook: &HostBfsCodebook) -> Result<ArrayRef> {
    let data_bytes = decoded_string_array_data_bytes(array, codebook, "predecessor", true)?;
    let mut builder = arrow::array::StringBuilder::with_capacity(array.len(), data_bytes);
    for row in 0..array.len() {
        if array.is_null(row) {
            builder.append_null();
            continue;
        }
        let code = numeric_array_value(array, row, "predecessor")?;
        if code < 0 {
            builder.append_null();
        } else {
            builder.append_value(codebook.value_for_code(code)?);
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn decoded_string_array_upper_bound(
    array: &dyn Array,
    codebook: &HostBfsCodebook,
    column: &'static str,
    negative_is_null: bool,
) -> Result<usize> {
    let data_bytes = decoded_string_array_data_bytes(array, codebook, column, negative_is_null)?;
    let offset_bytes =
        array.len().checked_add(1).and_then(|offsets| offsets.checked_mul(size_of::<i32>())).ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
                "decoded logical BFS Arrow offsets overflowed usize",
            )
            .with_fact("column", column)
            .with_fact("reason", "logical_bfs_decoded_output_bound_overflow")
        })?;
    let validity_bytes = decoded_string_array_validity_bytes(array, column, negative_is_null)?;
    arrow_buffer_capacity(offset_bytes)
        .and_then(|offsets| {
            arrow_buffer_capacity(data_bytes).and_then(|data| {
                arrow_buffer_capacity(validity_bytes).and_then(|validity| {
                    size_of::<StringArray>()
                        .checked_add(offsets)
                        .and_then(|bytes| bytes.checked_add(data))
                        .and_then(|bytes| bytes.checked_add(validity))
                })
            })
        })
        .ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
                "decoded logical BFS Arrow buffer bound overflowed usize",
            )
            .with_fact("column", column)
            .with_fact("reason", "logical_bfs_decoded_output_bound_overflow")
            .into()
        })
}

fn decoded_string_array_data_bytes(
    array: &dyn Array,
    codebook: &HostBfsCodebook,
    column: &'static str,
    negative_is_null: bool,
) -> Result<usize> {
    let mut bytes = 0usize;
    for row in 0..array.len() {
        if array.is_null(row) {
            continue;
        }
        let code = numeric_array_value(array, row, column)?;
        if negative_is_null && code < 0 {
            continue;
        }
        bytes = bytes.checked_add(codebook.value_for_code(code)?.len()).ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
                "decoded logical BFS string bytes overflowed usize",
            )
            .with_fact("column", column)
            .with_fact("reason", "logical_bfs_decoded_output_bound_overflow")
        })?;
    }
    Ok(bytes)
}

fn decoded_string_array_validity_bytes(
    array: &dyn Array,
    column: &'static str,
    negative_is_null: bool,
) -> Result<usize> {
    if array.null_count() > 0 {
        return decoded_validity_bitmap_bytes(array.len(), column);
    }
    if negative_is_null {
        for row in 0..array.len() {
            if !array.is_null(row) && numeric_array_value(array, row, column)? < 0 {
                return decoded_validity_bitmap_bytes(array.len(), column);
            }
        }
    }
    Ok(0)
}

fn decoded_validity_bitmap_bytes(len: usize, column: &'static str) -> Result<usize> {
    len.checked_add(7).map(|bits| bits / 8).ok_or_else(|| {
        Error::new(
            ErrorCode::SizeOverflow,
            "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
            "decoded logical BFS null bitmap bound overflowed usize",
        )
        .with_fact("column", column)
        .with_fact("reason", "logical_bfs_decoded_output_bound_overflow")
        .into()
    })
}

fn arrow_buffer_capacity(bytes: usize) -> Option<usize> {
    bytes.checked_add(63).map(|rounded| rounded / 64 * 64)
}

fn numeric_array_value(array: &dyn Array, row: usize, column: &'static str) -> Result<i64> {
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(i64::from(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(array.value(row));
    }
    Err(Error::new(
        ErrorCode::UnsupportedType,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
        "logical BFS output vertex column was not an integral array",
    )
    .with_fact("column", column)
    .with_fact("data_type", format!("{:?}", array.data_type()))
    .into())
}
