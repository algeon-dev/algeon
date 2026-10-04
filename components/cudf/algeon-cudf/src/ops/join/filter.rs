use std::sync::Arc;

use super::*;

fn filter_join_kind_raw(join_type: JoinType) -> cudf_sys::cudf_join_kind_t {
    match join_type {
        JoinType::Inner => cudf_sys::CUDF_JOIN_KIND_INNER_JOIN,
        JoinType::Left => cudf_sys::CUDF_JOIN_KIND_LEFT_JOIN,
        JoinType::Full => cudf_sys::CUDF_JOIN_KIND_FULL_JOIN,
    }
}

fn validate_filter_join_index_column(
    indices: &ColumnView<'_>,
    resource: &'static str,
    operation: &'static str,
) -> Result<usize> {
    let dtype = indices.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
    if dtype.id() != TypeId::Int32 {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            format!("{resource} must be Int32, got {:?}", dtype.id()),
        )
        .with_fact(FACT_LEFT_TYPE_ID, dtype.id())
        .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Int32)
        .with_detail("resource", resource)
        .with_detail("reason", "join_index_type_mismatch")
        .with_detail("remedy", "provide INT32 join-index columns produced by cudf join-index APIs"));
    }

    let nullable = indices.nullable().map_err(|e| e.reanchor_if_needed(operation))?;
    if nullable {
        return Err(Error::new_with(ErrorCode::NullUnexpected, operation, format!("{resource} must not be nullable"))
            .with_detail("resource", resource)
            .with_detail("reason", "nullable_join_indices")
            .with_detail("remedy", "use non-nullable INT32 join-index columns produced by cudf join-index APIs"));
    }

    indices.len().map_err(|e| e.reanchor_if_needed(operation))
}

fn validate_filter_join_indices(
    left_indices: &ColumnView<'_>,
    right_indices: &ColumnView<'_>,
    operation: &'static str,
) -> Result<()> {
    let left_len = validate_filter_join_index_column(left_indices, "left_indices", operation)?;
    let right_len = validate_filter_join_index_column(right_indices, "right_indices", operation)?;
    if left_len != right_len {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            format!("left_indices length ({left_len}) does not match right_indices length ({right_len})"),
        )
        .with_fact(FACT_ACTUAL_ROWS, right_len)
        .with_fact(FACT_EXPECTED_ROWS, left_len)
        .with_detail("resource", "join_indices")
        .with_detail("reason", "left_right_index_length_mismatch")
        .with_detail("remedy", "filter index pairs produced by the same equality join result"));
    }
    Ok(())
}

fn filter_join_indices_impl(
    inputs: FilterJoinInputs<'_, '_>,
    join_type: JoinType,
    context: JoinCallContext<'_>,
) -> Result<JoinIndices> {
    validate_filter_join_indices(inputs.left_indices, inputs.right_indices, context.operation)?;

    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_filter_join_indices_ex(
            inputs.left.raw_handle(),
            inputs.right.raw_handle(),
            inputs.left_indices.raw_handle(),
            inputs.right_indices.raw_handle(),
            inputs.predicate.raw_handle(),
            filter_join_kind_raw(join_type),
            &mut out_left,
            &mut out_right,
            context.exec_options,
        )
    };

    error::check_status_in_with_details(
        status,
        context.operation,
        [
            ("join_type", format!("{:?}", join_type).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;

    Ok(JoinIndices {
        left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(context.output_domain)) }?,
        right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(context.output_domain)) }?,
    })
}

fn filter_join_indices_with_context(
    inputs: FilterJoinInputs<'_, '_>,
    join_type: JoinType,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let execution = compatible_filter_join_indices_domain(inputs, ctx, operation)?;
    filter_join_indices_impl(
        inputs,
        join_type,
        JoinCallContext::new(execution.output_domain(), execution.exec_options(), operation),
    )
}

/// Filter existing join-index pairs using a borrowed execution override.
///
/// Same result semantics as [`filter_join_indices`]. See the
/// [join module](crate::ops::join)'s shared `_in` execution-context contract.
pub(crate) fn filter_join_indices_in(
    left: &TableView<'_>,
    right: &TableView<'_>,
    left_indices: &ColumnView<'_>,
    right_indices: &ColumnView<'_>,
    predicate: &AstExpression,
    join_type: JoinType,
    ctx: Exec<'_>,
) -> Result<JoinIndices> {
    filter_join_indices_with_context(
        FilterJoinInputs { left, right, left_indices, right_indices, predicate },
        join_type,
        ctx,
        "join::filter_join_indices_in",
    )
}

impl FilteredJoin {
    /// Build a reusable filtered-join artifact using a borrowed execution override.
    pub fn new_in(build: Table, options: FilteredJoinOptions, ctx: Exec<'_>) -> Result<Self> {
        Self::new_with_context(build, options, ctx, "filtered_join::new_in")
    }

    pub(crate) fn pre_build_reservation_size_bytes_for_view(
        build_view: &TableView<'_>,
        options: FilteredJoinOptions,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let execution = ResolvedExecution::unary(build_view.execution_domain(), ctx, operation)?;
        let mut out = 0usize;
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_filtered_join_pre_build_reservation_size_ex(
                    build_view.raw_handle(),
                    options.null_eq().to_join_raw(),
                    options.load_factor_value(),
                    &mut out,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("null_equality", format!("{:?}", options.null_eq()).to_lowercase()),
                ("load_factor", options.load_factor_value().to_string()),
                ("execution_authority", "resolved_domain".to_string()),
                ("reservation_contract", "filtered_join_pre_build_reservation_size_bytes".to_string()),
            ],
        )?;
        Ok(out)
    }

    fn new_with_context(
        build: Table,
        options: FilteredJoinOptions,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let handle = {
            let build_view = build.view().map_err(|e| e.reanchor_if_needed(operation))?;
            let execution = ResolvedExecution::unary(build.execution_domain(), ctx, operation)?;
            let mut handle = ptr::null_mut();
            error::check_status_in_with_details(
                unsafe {
                    cudf_sys::cudf_filtered_join_create_ex(
                        build_view.raw_handle(),
                        options.null_eq().to_join_raw(),
                        options.load_factor_value(),
                        &mut handle,
                        execution.exec_options(),
                    )
                },
                operation,
                [
                    ("null_equality", format!("{:?}", options.null_eq()).to_lowercase()),
                    ("load_factor", options.load_factor_value().to_string()),
                    ("execution_authority", "resolved_domain".to_string()),
                ],
            )?;
            FilteredJoinHandle {
                handle: error::require_non_null(handle, operation, "filtered join")?,
                _domain: Arc::clone(execution.output_domain()),
            }
        };

        Ok(Self { handle: Arc::new(handle), build: Arc::new(build), options })
    }

    /// Returns bytes retained by this reusable filtered-join build artifact.
    ///
    /// This excludes the retained build key table and any per-probe output
    /// index allocations.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_filtered_join_device_allocated_size(self.handle.handle.as_ptr(), &mut out) },
            "filtered_join::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Begin a retained left-semi probe using a borrowed execution override.
    pub fn begin_left_semi_probe_in(
        &self,
        probe_keys: &TableView<'_>,
        ctx: Exec<'_>,
    ) -> Result<FilteredJoinProbeState> {
        self.begin_probe_with_context(probe_keys, true, ctx, "filtered_join::begin_left_semi_probe_in")
    }

    /// Begin a retained left-anti probe using a borrowed execution override.
    pub fn begin_left_anti_probe_in(
        &self,
        probe_keys: &TableView<'_>,
        ctx: Exec<'_>,
    ) -> Result<FilteredJoinProbeState> {
        self.begin_probe_with_context(probe_keys, false, ctx, "filtered_join::begin_left_anti_probe_in")
    }

    /// Return the exact output row count for a left semi probe using a borrowed execution override.
    ///
    /// This legacy sizing API does not retain probe membership. Bounded
    /// callers that need to reserve state and then materialize indices from
    /// the same probe must use [`Self::begin_left_semi_probe_in`] instead.
    pub fn left_semi_join_size_in(&self, probe_keys: &TableView<'_>, ctx: Exec<'_>) -> Result<usize> {
        self.probe_size_with_context(probe_keys, true, ctx, "filtered_join::left_semi_join_size_in")
    }

    /// Return the exact output row count for a left anti probe using a borrowed execution override.
    ///
    /// This legacy sizing API does not retain probe membership. Bounded
    /// callers that need to reserve state and then materialize indices from
    /// the same probe must use [`Self::begin_left_anti_probe_in`] instead.
    pub fn left_anti_join_size_in(&self, probe_keys: &TableView<'_>, ctx: Exec<'_>) -> Result<usize> {
        self.probe_size_with_context(probe_keys, false, ctx, "filtered_join::left_anti_join_size_in")
    }

    /// Probe with left semi semantics using a borrowed execution override and
    /// an exact caller-supplied output row count.
    ///
    /// The supplied size is a validation hint, not a retained-probe bounded
    /// memory contract. Bounded callers must use
    /// [`Self::begin_left_semi_probe_in`] and materialize indices from the
    /// returned probe state.
    pub fn left_semi_join_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
        ctx: Exec<'_>,
    ) -> Result<SingleSideJoinIndices> {
        self.probe_indices_with_output_size_in(
            probe_keys,
            true,
            output_size,
            ctx,
            "filtered_join::left_semi_join_indices_with_output_size_in",
        )
    }

    /// Probe with left anti semantics using a borrowed execution override and
    /// an exact caller-supplied output row count.
    ///
    /// The supplied size is a validation hint, not a retained-probe bounded
    /// memory contract. Bounded callers must use
    /// [`Self::begin_left_anti_probe_in`] and materialize indices from the
    /// returned probe state.
    pub fn left_anti_join_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        output_size: usize,
        ctx: Exec<'_>,
    ) -> Result<SingleSideJoinIndices> {
        self.probe_indices_with_output_size_in(
            probe_keys,
            false,
            output_size,
            ctx,
            "filtered_join::left_anti_join_indices_with_output_size_in",
        )
    }

    fn probe_size_with_context(
        &self,
        probe_keys: &TableView<'_>,
        semi_join: bool,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        let mut out = 0usize;
        let status = if semi_join {
            unsafe {
                cudf_sys::cudf_filtered_join_left_semi_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    execution.exec_options(),
                )
            }
        } else {
            unsafe {
                cudf_sys::cudf_filtered_join_left_anti_join_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out,
                    execution.exec_options(),
                )
            }
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_authority", "resolved_domain".to_string()),
                ("reservation_contract", "filtered_join_exact_output_size".to_string()),
            ],
        )?;
        Ok(out)
    }

    fn probe_indices_with_output_size_in(
        &self,
        probe_keys: &TableView<'_>,
        semi_join: bool,
        output_size: usize,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<SingleSideJoinIndices> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        let mut out = ptr::null_mut();
        let status = if semi_join {
            unsafe {
                cudf_sys::cudf_filtered_join_left_semi_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out,
                    execution.exec_options(),
                )
            }
        } else {
            unsafe {
                cudf_sys::cudf_filtered_join_left_anti_join_with_output_size_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    output_size,
                    &mut out,
                    execution.exec_options(),
                )
            }
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_authority", "resolved_domain".to_string()),
                ("output_size_provided", true.to_string()),
                ("output_size", output_size.to_string()),
            ],
        )?;

        Ok(SingleSideJoinIndices {
            indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }?,
        })
    }

    fn begin_probe_with_context(
        &self,
        probe_keys: &TableView<'_>,
        semi_join: bool,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<FilteredJoinProbeState> {
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], ctx, operation)?;
        let mut handle = ptr::null_mut();
        let status = if semi_join {
            unsafe {
                cudf_sys::cudf_filtered_join_begin_left_semi_probe_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut handle,
                    execution.exec_options(),
                )
            }
        } else {
            unsafe {
                cudf_sys::cudf_filtered_join_begin_left_anti_probe_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut handle,
                    execution.exec_options(),
                )
            }
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("join_type", if semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_authority", "resolved_domain".to_string()),
                ("reservation_contract", "filtered_join_retained_probe_state".to_string()),
            ],
        )?;
        let handle = FilteredJoinProbeHandle {
            handle: error::require_non_null(handle, operation, "filtered join probe state")?,
            _domain: Arc::clone(execution.output_domain()),
        };
        let mut output_size = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_filtered_join_probe_output_size(handle.handle.as_ptr(), &mut output_size) },
            operation,
        )?;
        Ok(FilteredJoinProbeState {
            handle,
            join: self.clone(),
            output_domain: Arc::clone(execution.output_domain()),
            output_size,
            semi_join,
        })
    }
}

impl FilteredJoinProbeState {
    /// Returns the exact selected row count for this probe.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns bytes retained by this probe state.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_filtered_join_probe_device_allocated_size(self.handle.handle.as_ptr(), &mut out) },
            "filtered_join_probe_state::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    /// Materialize selected probe indices from retained membership state using
    /// a borrowed execution override.
    pub fn materialize_indices_in(self, ctx: Exec<'_>) -> Result<SingleSideJoinIndices> {
        let operation = "filtered_join_probe_state::materialize_indices_in";
        let execution =
            ResolvedExecution::many(&self.output_domain, [self.join.build.execution_domain()], ctx, operation)?;
        let mut out = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_filtered_join_probe_materialize_indices_ex(
                    self.handle.handle.as_ptr(),
                    &mut out,
                    execution.exec_options(),
                )
            },
            operation,
            [
                ("join_type", if self.semi_join { "left_semi" } else { "left_anti" }.to_string()),
                ("output_size", self.output_size.to_string()),
                ("execution_authority", "resolved_domain".to_string()),
                ("reservation_contract", "filtered_join_retained_probe_state_materialize_indices".to_string()),
            ],
        )?;
        Ok(SingleSideJoinIndices {
            indices: unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }?,
        })
    }
}
