use std::{collections::BTreeSet, sync::Arc, time::Instant};

use arrow::datatypes::{DataType as ArrowDataType, Schema, SchemaRef as ArrowSchemaRef};
use cudf::{DataFrame, Exec};
use query_planning::{
    dynamic_filter::{
        DynamicFilter, DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterProducer,
        DynamicFilterRoute, LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON,
        dynamic_filter_consumer_source_feeds_downstream_work_fanout, dynamic_filter_i64_values,
        dynamic_filter_predicate_for_schema,
    },
    error::{QueryError, Result},
    expr::Expr,
    pipeline::ResidentResourceId,
    plan::{FilterNode, JoinNode, JoinNullEquality, JoinSide, NodeId, PlanNode},
    source::SourceNode,
};
use query_runtime::{
    metrics::DynamicFilterSourceSnapshot,
    runtime::resident::{GpuDataFrame, GpuResultHandle},
};

use super::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState, CompiledTaskExecutor, cudf_query_error,
};
use crate::exec::{
    bounded::{
        BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame,
        compiled::{
            ResidentFrameStreamPayload, frame_stream::map_dynamic_filter_resident_frame_stream, helpers::invalid,
            resource::CompiledGpuResourcePayload,
        },
        run_budgeted,
    },
    operators::execute_filter_with_budget,
    util::record_batch_memory_size,
};

impl CompiledGpuTaskExecutor<'_> {
    // `build_frames` is the complete build side, either one frame or every
    // partition of a streamed build; a filter is published only over all of it.
    pub(super) fn produce_hash_join_build_dynamic_filters(
        &mut self,
        join_node_id: NodeId,
        join: &JoinNode,
        build_frames: &[&DataFrame],
    ) -> Result<()> {
        let routes = self
            .plan
            .dynamic_filter_routes()
            .iter()
            .filter(|route| {
                matches!(
                    &route.producer,
                    DynamicFilterProducer::HashJoinBuild {
                        join_node_id: route_join_node_id,
                        ..
                    } if *route_join_node_id == join_node_id
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        if routes.is_empty() {
            return Ok(());
        }

        if join.spec.keys.len() != 1 || join.spec.null_equality != JoinNullEquality::NullsDoNotMatch {
            for route in &routes {
                self.dynamic_filters.insert_skipped(route.id, "unsupported_join_key_contract");
                self.execution.metrics().record_dynamic_filter_skipped(route, "unsupported_join_key_contract");
            }
            return Ok(());
        }

        let build_input = match join.spec.build_side {
            JoinSide::Left => join.inputs[0],
            JoinSide::Right => join.inputs[1],
        };
        let build_key = match join.spec.build_side {
            JoinSide::Left => join.spec.keys[0].left.name(),
            JoinSide::Right => join.spec.keys[0].right.name(),
        };
        let build_schema = Arc::clone(
            self.plan
                .node(build_input)
                .ok_or_else(|| {
                    invalid("compiled dynamic-filter producer references a missing build input")
                        .with_fact("plan_node_id", build_input)
                        .with_fact("join_node_id", join_node_id)
                })?
                .output_schema(),
        );
        if build_schema.index_of(build_key).is_err() {
            for route in &routes {
                self.dynamic_filters.insert_skipped(route.id, "producer_key_not_in_build_input");
                self.execution.metrics().record_dynamic_filter_skipped(route, "producer_key_not_in_build_input");
            }
            return Ok(());
        };

        self.produce_dynamic_filters_from_frames(
            &routes,
            &build_schema,
            build_frames,
            "producer_key_not_in_build_input",
        )
    }

    /// `group_key_state` is the aggregate's complete distinct group-key set,
    /// which route discovery guarantees is what an `AggregateGroupKeys` route
    /// reads. It may be the finalized output or the merged partition states a
    /// lazily finalized output stream still retains; `None` means the output
    /// cannot answer for its whole scope, so every route completes unpublished.
    pub(super) fn produce_aggregate_group_key_dynamic_filters(
        &mut self,
        aggregate_node_id: NodeId,
        aggregate_schema: &ArrowSchemaRef,
        group_key_state: Option<&[GpuDataFrame]>,
    ) -> Result<()> {
        let routes = self
            .plan
            .dynamic_filter_routes()
            .iter()
            .filter(|route| {
                matches!(
                    &route.producer,
                    DynamicFilterProducer::AggregateGroupKeys {
                        aggregate_node_id: route_aggregate_node_id
                    } if *route_aggregate_node_id == aggregate_node_id
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        if routes.is_empty() {
            return Ok(());
        }
        let Some(group_key_state) = group_key_state else {
            for route in &routes {
                self.dynamic_filters.insert_skipped(route.id, "producer_group_key_state_unavailable");
                self.execution.metrics().record_dynamic_filter_skipped(route, "producer_group_key_state_unavailable");
            }
            return Ok(());
        };
        let frames = group_key_state.iter().map(GpuDataFrame::frame).collect::<Vec<_>>();
        self.produce_dynamic_filters_from_frames(
            &routes,
            aggregate_schema,
            &frames,
            "producer_key_not_in_aggregate_output",
        )
    }

    /// Builds the filters for `routes` from the producer's key columns.
    ///
    /// Key values are copied device to host (`to_host_record_batch`), so the
    /// `max_build_rows` guard bounds that transfer and the host memory the
    /// value list takes. Above the large-producer threshold, routes lowered
    /// into reader predicates receive a min/max range computed on the device,
    /// and no value list is copied. Each route ends either produced or skipped
    /// with a reason, so consumers waiting on it can proceed.
    fn produce_dynamic_filters_from_frames(
        &mut self,
        routes: &[DynamicFilterRoute],
        producer_schema: &ArrowSchemaRef,
        producer_frames: &[&DataFrame],
        missing_key_reason: &'static str,
    ) -> Result<()> {
        if routes.is_empty() {
            return Ok(());
        }

        let options = self.policy.dynamic_filter;
        let producer_rows = producer_frames.iter().map(|frame| frame.num_rows()).sum::<usize>();
        if producer_rows > options.max_build_rows {
            for route in routes {
                self.dynamic_filters.insert_skipped(route.id, "producer_key_row_count_exceeds_runtime_limit");
                self.execution
                    .metrics()
                    .record_dynamic_filter_skipped(route, "producer_key_row_count_exceeds_runtime_limit");
            }
            return Ok(());
        }
        if producer_rows > options.large_local_parquet_range_producer_rows
            && routes.iter().all(route_targets_reader_lowered_local_parquet_source)
        {
            let producer_frame_bytes = producer_frames
                .iter()
                .map(|frame| GpuDataFrame::new((*frame).clone()).device_allocated_size_bytes().ok())
                .try_fold(0u64, |total, bytes| bytes.map(|bytes| total.saturating_add(bytes)));
            let range_routes = routes
                .iter()
                .filter_map(|route| {
                    if self
                        .large_keyset_dimension_sized_local_parquet_route_is_not_promising(route, producer_frame_bytes)
                    {
                        self.dynamic_filters.insert_skipped(route.id, LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON);
                        self.execution
                            .metrics()
                            .record_dynamic_filter_skipped(route, LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON);
                        None
                    } else {
                        Some(route.clone())
                    }
                })
                .collect::<Vec<_>>();
            if !range_routes.is_empty() {
                self.produce_large_local_parquet_range_dynamic_filters(
                    &range_routes,
                    producer_schema,
                    producer_frames,
                    producer_rows,
                    missing_key_reason,
                )?;
            }
            return Ok(());
        }

        let mut selected_columns = Vec::new();
        for route in routes {
            let Some(producer_column) = route.producer_columns.first() else {
                continue;
            };
            if producer_schema.index_of(producer_column).is_ok() && !selected_columns.contains(producer_column) {
                selected_columns.push(producer_column.clone());
            }
        }
        if selected_columns.is_empty() {
            for route in routes {
                let Some(producer_column) = route.producer_columns.first() else {
                    self.dynamic_filters.insert_skipped(route.id, "unsupported_producer_key_contract");
                    self.execution.metrics().record_dynamic_filter_skipped(route, "unsupported_producer_key_contract");
                    continue;
                };
                if producer_schema.index_of(producer_column).is_err() {
                    self.dynamic_filters.insert_skipped(route.id, missing_key_reason);
                    self.execution.metrics().record_dynamic_filter_skipped(route, missing_key_reason);
                }
            }
            return Ok(());
        }

        let selected_fields = selected_columns
            .iter()
            .map(|column| {
                producer_schema.field_with_name(column).cloned().map_err(|_error| {
                    invalid("compiled dynamic-filter producer schema is missing a selected key column")
                        .with_fact("producer_column", column.as_str())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let selected_schema = Arc::new(Schema::new(selected_fields));
        let mut batches = Vec::with_capacity(producer_frames.len());
        for producer_frame in producer_frames {
            let selected_frame = producer_frame.select(&selected_columns).map_err(|error| {
                cudf_query_error(
                    "query_engine.dynamic_filter.producer_projection",
                    "failed to select dynamic-filter producer key columns",
                    error,
                )
                .with_fact("producer_columns", selected_columns.join(","))
            })?;

            let timer = Instant::now();
            let handle = GpuResultHandle::new(
                self.execution.context(),
                GpuDataFrame::new(selected_frame),
                Arc::clone(&selected_schema),
            )?;
            let batch = handle.to_host_record_batch()?;
            self.execution.metrics().add_materialization_time(timer.elapsed());
            self.execution.metrics().record_device_to_host_materialization(
                batch.num_rows() as u64,
                record_batch_memory_size(&batch) as u64,
            );
            batches.push(batch);
        }

        for route in routes {
            let Some(producer_column) = route.producer_columns.first() else {
                self.dynamic_filters.insert_skipped(route.id, "unsupported_producer_key_contract");
                self.execution.metrics().record_dynamic_filter_skipped(route, "unsupported_producer_key_contract");
                continue;
            };
            let Ok(key_index) = selected_schema.index_of(producer_column) else {
                self.dynamic_filters.insert_skipped(route.id, missing_key_reason);
                self.execution.metrics().record_dynamic_filter_skipped(route, missing_key_reason);
                continue;
            };
            let mut values = Vec::with_capacity(producer_rows);
            let mut unsupported = false;
            for batch in &batches {
                match dynamic_filter_i64_values(batch.column(key_index).as_ref()) {
                    Some(batch_values) => values.extend(batch_values),
                    None => {
                        unsupported = true;
                        break;
                    },
                }
            }
            if unsupported {
                self.dynamic_filters.insert_skipped(route.id, "unsupported_producer_key_type");
                self.execution.metrics().record_dynamic_filter_skipped(route, "unsupported_producer_key_type");
                continue;
            }
            let filter = DynamicFilter::from_i64_values(values, options.small_in_list_limit);
            self.dynamic_filters.insert(route.id, filter.clone());
            self.execution.metrics().record_dynamic_filter_produced(route, &filter, producer_rows as u64);
        }
        Ok(())
    }

    fn produce_large_local_parquet_range_dynamic_filters(
        &mut self,
        routes: &[DynamicFilterRoute],
        producer_schema: &ArrowSchemaRef,
        producer_frames: &[&DataFrame],
        producer_rows: usize,
        missing_key_reason: &'static str,
    ) -> Result<()> {
        for route in routes {
            let Some(producer_column) = route.producer_columns.first() else {
                self.dynamic_filters.insert_skipped(route.id, "unsupported_producer_key_contract");
                self.execution.metrics().record_dynamic_filter_skipped(route, "unsupported_producer_key_contract");
                continue;
            };
            let Ok(producer_field) = producer_schema.field_with_name(producer_column) else {
                self.dynamic_filters.insert_skipped(route.id, missing_key_reason);
                self.execution.metrics().record_dynamic_filter_skipped(route, missing_key_reason);
                continue;
            };
            let mut filter = Some(DynamicFilter::Empty);
            for producer_frame in producer_frames {
                let series = producer_frame.series(producer_column).map_err(|error| {
                    cudf_query_error(
                        "query_engine.dynamic_filter.producer_range",
                        "failed to access dynamic-filter producer key column for range payload",
                        error,
                    )
                    .with_fact("producer_column", producer_column.as_str())
                })?;
                let Some(frame_filter) = dynamic_filter_i64_range_from_series(&series, producer_field.data_type())?
                else {
                    filter = None;
                    break;
                };
                filter = filter.map(|filter| merge_dynamic_filter_ranges(filter, frame_filter));
            }
            let Some(filter) = filter else {
                self.dynamic_filters.insert_skipped(route.id, "unsupported_producer_key_type");
                self.execution.metrics().record_dynamic_filter_skipped(route, "unsupported_producer_key_type");
                continue;
            };
            self.dynamic_filters.insert(route.id, filter.clone());
            self.execution.metrics().record_dynamic_filter_produced(route, &filter, producer_rows as u64);
        }
        Ok(())
    }

    /// A range filter from a large keyset is skipped for a consumer small
    /// enough that reading it costs less than the filter: a dimension-sized
    /// source with few row groups, or one whose selected bytes are below the
    /// producer's. A source that feeds further fan-out work keeps the filter
    /// because the saving applies downstream too.
    fn large_keyset_dimension_sized_local_parquet_route_is_not_promising(
        &self,
        route: &DynamicFilterRoute,
        producer_frame_bytes: Option<u64>,
    ) -> bool {
        let DynamicFilterConsumer::LocalParquetSource { source_node_id } = &route.consumer else {
            return false;
        };
        let Some(source) = self.plan.node(*source_node_id).and_then(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        }) else {
            return false;
        };
        let Some(stats) = source.diagnostics.selected_row_group_stats.as_ref() else {
            return false;
        };
        let Some(consumer_bytes) = stats.selected_source_bytes() else {
            return false;
        };
        if dynamic_filter_consumer_source_feeds_downstream_work_fanout(self.plan, *source_node_id) {
            return false;
        }
        let options = self.policy.dynamic_filter;
        if consumer_bytes > options.dimension_sized_consumer_bytes {
            return false;
        }
        stats.selected_row_groups <= options.tiny_local_parquet_consumer_row_groups
            || producer_frame_bytes.is_some_and(|producer_bytes| producer_bytes > consumer_bytes)
    }

    pub(super) fn apply_dynamic_filters_to_join_leaf_stream(
        &mut self,
        join_node_id: NodeId,
        stream: BoundedJoinLeafLocalParquetSourceStream,
    ) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
        self.apply_dynamic_filters_to_source_stream(stream)
            .map_err(|error| error.with_fact_if_absent("join_node_id", join_node_id))
    }

    /// Attaches every ready filter routed to the stream's source before the
    /// read starts, as reader predicates or post-decode stages per the route's
    /// lowering. A route that is not ready is a contract error. Compiled
    /// control edges order producers ahead of consumers and the scheduler
    /// runs one task at a time, so a producer cannot still be in progress;
    /// an unready route points at a missing edge.
    pub(super) fn apply_dynamic_filters_to_source_stream(
        &mut self,
        mut stream: BoundedJoinLeafLocalParquetSourceStream,
    ) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
        let source_node_id = stream.source_node_id();
        let routes = self
            .plan
            .dynamic_filter_routes()
            .iter()
            .filter(|route| {
                matches!(
                    &route.consumer,
                    DynamicFilterConsumer::LocalParquetSource {
                        source_node_id: route_source_node_id
                    } if *route_source_node_id == source_node_id
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        for route in &routes {
            if !self.dynamic_filters.is_ready(route.id) {
                return Err(invalid("compiled dynamic-filter consumer ran before its producer completed")
                    .with_fact("source_node_id", source_node_id)
                    .with_fact("dynamic_filter_id", route.id.to_string()));
            }
        }
        let options = self.policy.dynamic_filter;
        let deferred_reader_predicate_routes =
            stream.deferred_reader_predicate_routes(&routes, &self.dynamic_filters, options);
        for route in &routes {
            let Some(filter) = self.dynamic_filters.produced(route.id).cloned() else {
                continue;
            };
            let before = stream.dynamic_filter_source_snapshot();
            stream = stream.with_dynamic_filters_and_deferred_reader_predicates(
                self.execution.context(),
                &[route],
                &self.dynamic_filters,
                &deferred_reader_predicate_routes,
                options,
            )?;
            let after = stream.dynamic_filter_source_snapshot();
            self.execution.metrics().record_dynamic_filter_applied(route, &filter, before, after);
        }
        Ok(stream)
    }

    /// Applies the post-decode source routes ready since the last consumer to
    /// a source frame several consumers share. The planner routes a filter to
    /// a shared source only when every consumer accepts it, so the filtered
    /// frame replaces the stored one.
    pub(super) fn apply_ready_dynamic_filters_to_shared_source_resource(
        &mut self,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
        resource: ResidentResourceId,
    ) -> Result<()> {
        let Some(source_node_id) = self.local_parquet_source_node_for_resource(resource) else {
            return Ok(());
        };
        let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = &state.resource_payload(resource)?.payload else {
            return Ok(());
        };
        let input = frame.frame.clone();
        let already_applied = self.source_post_decode_applied_routes.get(&resource).cloned().unwrap_or_default();
        let Some((frame, applied_routes)) =
            self.apply_ready_dynamic_filters_to_shared_source_frame(source_node_id, input, &already_applied)?
        else {
            return Ok(());
        };
        let old = state.replace_resource_payload(resource, CompiledGpuResource::new(frame))?;
        <Self as CompiledTaskExecutor>::release_resource(self, resource, old)?;
        self.source_post_decode_applied_routes.entry(resource).or_default().extend(applied_routes);
        Ok(())
    }

    fn apply_ready_dynamic_filters_to_shared_source_frame(
        &mut self,
        source_node_id: NodeId,
        input: GpuDataFrame,
        already_applied: &BTreeSet<DynamicFilterId>,
    ) -> Result<Option<(ResidentGpuFrame, BTreeSet<DynamicFilterId>)>> {
        let routes = self
            .source_post_decode_dynamic_filter_routes(source_node_id)
            .into_iter()
            .filter(|route| !already_applied.contains(&route.id))
            .collect::<Vec<_>>();
        if routes.is_empty() {
            return Ok(None);
        }
        let Some(source_schema) = self.plan.node(source_node_id).and_then(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(Arc::clone(&source.schema)),
                _ => None,
            },
            _ => None,
        }) else {
            return Ok(None);
        };
        let source_snapshot = self.dynamic_filter_source_snapshot(source_node_id)?;
        let mut input = Some(input);
        let mut frame: Option<ResidentGpuFrame> = None;
        let mut applied_routes = BTreeSet::new();
        for route in &routes {
            let Some((filter, predicate)) = self.resolve_dynamic_filter_predicate(route, source_schema.as_ref()) else {
                continue;
            };
            let filter_node = FilterNode { input: source_node_id, predicate };
            let rows_before = frame.as_ref().map_or_else(
                || input.as_ref().map(|input| input.num_rows() as u64),
                |frame| Some(frame.frame.num_rows() as u64),
            );
            self.execution.metrics().record_dynamic_filter_applied(route, &filter, source_snapshot, source_snapshot);
            frame = Some(match frame.take() {
                Some(frame) => run_budgeted(
                    &mut self.execution,
                    source_node_id,
                    "compiled_dynamic_filter_source_post_decode",
                    frame.frame,
                    |context, input, token, metrics| {
                        execute_filter_with_budget(context, &filter_node, input, token, metrics)
                    },
                )?,
                None => run_budgeted(
                    &mut self.execution,
                    source_node_id,
                    "compiled_dynamic_filter_source_post_decode",
                    input.take().expect("first source dynamic filter consumes the cloned shared input"),
                    |context, input, token, metrics| {
                        execute_filter_with_budget(context, &filter_node, input, token, metrics)
                    },
                )?,
            });
            self.execution.metrics().record_dynamic_filter_source_post_decode_stage(
                route,
                &filter,
                rows_before,
                frame.as_ref().map(|frame| frame.frame.num_rows() as u64),
            );
            applied_routes.insert(route.id);
        }
        Ok(frame.map(|frame| (frame, applied_routes)))
    }

    pub(super) fn apply_dynamic_filters_to_join_probe_frame(
        &mut self,
        join_node_id: NodeId,
        join: &JoinNode,
        mut frame: ResidentGpuFrame,
    ) -> Result<ResidentGpuFrame> {
        let routes = self.join_probe_dynamic_filter_routes(join_node_id);
        if routes.is_empty() {
            return Ok(frame);
        }
        self.ensure_dynamic_filter_routes_ready(join_node_id, &routes)?;
        let probe_input = join_probe_input_node(join);
        let probe_schema = Arc::clone(
            self.plan
                .node(probe_input)
                .ok_or_else(|| {
                    invalid("compiled dynamic-filter probe consumer references a missing probe input")
                        .with_fact("join_node_id", join_node_id)
                        .with_fact("probe_input_node_id", probe_input)
                })?
                .output_schema(),
        );
        for route in &routes {
            let Some((filter, predicate)) = self.resolve_dynamic_filter_predicate(route, probe_schema.as_ref()) else {
                continue;
            };
            let filter_node = FilterNode { input: probe_input, predicate };
            let rows_before = frame.frame.num_rows() as u64;
            frame = run_budgeted(
                &mut self.execution,
                join_node_id,
                "compiled_dynamic_filter_join_probe",
                frame.frame,
                |context, input, token, metrics| {
                    execute_filter_with_budget(context, &filter_node, input, token, metrics)
                },
            )?;
            self.execution.metrics().record_dynamic_filter_applied_post_decode(
                route,
                &filter,
                Some(rows_before),
                Some(frame.frame.num_rows() as u64),
            );
        }
        Ok(frame)
    }

    pub(super) fn apply_dynamic_filters_to_filter_frame(
        &mut self,
        filter_node_id: NodeId,
        filter: &FilterNode,
        mut frame: ResidentGpuFrame,
    ) -> Result<ResidentGpuFrame> {
        let routes = self.filter_dynamic_filter_routes(filter_node_id);
        if routes.is_empty() {
            return Ok(frame);
        }
        self.ensure_dynamic_filter_routes_ready(filter_node_id, &routes)?;
        let input_schema = self.dynamic_filter_consumer_input_schema(filter_node_id, filter.input)?;
        for route in &routes {
            let Some((dynamic_filter, predicate)) = self.resolve_dynamic_filter_predicate(route, input_schema.as_ref())
            else {
                continue;
            };
            let dynamic_filter_node = FilterNode { input: filter.input, predicate };
            let rows_before = frame.frame.num_rows() as u64;
            frame = run_budgeted(
                &mut self.execution,
                filter_node_id,
                "compiled_dynamic_filter_filter",
                frame.frame,
                |context, input, token, metrics| {
                    execute_filter_with_budget(context, &dynamic_filter_node, input, token, metrics)
                },
            )?;
            self.execution.metrics().record_dynamic_filter_applied_post_decode(
                route,
                &dynamic_filter,
                Some(rows_before),
                Some(frame.frame.num_rows() as u64),
            );
        }
        Ok(frame)
    }

    pub(super) fn apply_dynamic_filters_to_filter_stream(
        &mut self,
        filter_node_id: NodeId,
        filter: &FilterNode,
        mut stream: ResidentFrameStreamPayload,
    ) -> Result<ResidentFrameStreamPayload> {
        let routes = self.filter_dynamic_filter_routes(filter_node_id);
        if routes.is_empty() {
            return Ok(stream);
        }
        self.ensure_dynamic_filter_routes_ready(filter_node_id, &routes)?;
        let input_schema = self.dynamic_filter_consumer_input_schema(filter_node_id, filter.input)?;
        for route in &routes {
            let Some((dynamic_filter, predicate)) = self.resolve_dynamic_filter_predicate(route, input_schema.as_ref())
            else {
                continue;
            };
            stream = map_dynamic_filter_resident_frame_stream(
                filter_node_id,
                stream,
                FilterNode { input: filter.input, predicate },
                route.clone(),
                dynamic_filter.clone(),
            );
            self.execution.metrics().record_dynamic_filter_applied_post_decode(route, &dynamic_filter, None, None);
        }
        Ok(stream)
    }

    /// Appends the routes whose consumer is this filter to a deferred source
    /// read, ahead of the filter's own stage, where a frame input would apply
    /// them before the filter.
    pub(super) fn apply_dynamic_filters_to_filter_source_stream(
        &mut self,
        filter_node_id: NodeId,
        filter: &FilterNode,
        mut stream: BoundedJoinLeafLocalParquetSourceStream,
    ) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
        let routes = self.filter_dynamic_filter_routes(filter_node_id);
        if routes.is_empty() {
            return Ok(stream);
        }
        self.ensure_dynamic_filter_routes_ready(filter_node_id, &routes)?;
        let input_schema = self.dynamic_filter_consumer_input_schema(filter_node_id, filter.input)?;
        for route in &routes {
            let Some((dynamic_filter, predicate)) = self.resolve_dynamic_filter_predicate(route, input_schema.as_ref())
            else {
                continue;
            };
            stream = stream.with_filter_dynamic_filter_stage(
                filter_node_id,
                FilterNode { input: filter.input, predicate },
                route.clone(),
                dynamic_filter.clone(),
            );
            self.execution.metrics().record_dynamic_filter_applied_post_decode(route, &dynamic_filter, None, None);
        }
        Ok(stream)
    }

    pub(super) fn apply_dynamic_filters_to_join_probe_stream(
        &mut self,
        join_node_id: NodeId,
        join: &JoinNode,
        mut stream: ResidentFrameStreamPayload,
    ) -> Result<ResidentFrameStreamPayload> {
        let routes = self.join_probe_dynamic_filter_routes(join_node_id);
        if routes.is_empty() {
            return Ok(stream);
        }
        self.ensure_dynamic_filter_routes_ready(join_node_id, &routes)?;
        let probe_input = join_probe_input_node(join);
        let probe_schema = Arc::clone(
            self.plan
                .node(probe_input)
                .ok_or_else(|| {
                    invalid("compiled dynamic-filter probe consumer references a missing probe input")
                        .with_fact("join_node_id", join_node_id)
                        .with_fact("probe_input_node_id", probe_input)
                })?
                .output_schema(),
        );
        for route in &routes {
            let Some((filter, predicate)) = self.resolve_dynamic_filter_predicate(route, probe_schema.as_ref()) else {
                continue;
            };
            stream = map_dynamic_filter_resident_frame_stream(
                join_node_id,
                stream,
                FilterNode { input: probe_input, predicate },
                route.clone(),
                filter.clone(),
            );
            self.execution.metrics().record_dynamic_filter_applied_post_decode(route, &filter, None, None);
        }
        Ok(stream)
    }

    /// The produced filter and its predicate over `schema`, or `None` with a
    /// skip recorded when the route was never produced, names no consumer
    /// column, or has no post-decode predicate form for this column type.
    fn resolve_dynamic_filter_predicate(
        &mut self,
        route: &DynamicFilterRoute,
        schema: &Schema,
    ) -> Option<(DynamicFilter, Expr)> {
        let dynamic_filter = self.dynamic_filters.produced(route.id).cloned()?;
        let Some(column) = route.consumer_columns.first() else {
            self.execution.metrics().record_dynamic_filter_skipped(route, "unsupported_consumer_key_contract");
            return None;
        };
        let Some(predicate) = dynamic_filter_predicate_for_schema(schema, column, &dynamic_filter) else {
            self.execution.metrics().record_dynamic_filter_skipped(route, "post_decode_predicate_unsupported");
            return None;
        };
        Some((dynamic_filter, predicate))
    }

    fn join_probe_dynamic_filter_routes(&self, join_node_id: NodeId) -> Vec<DynamicFilterRoute> {
        self.plan
            .dynamic_filter_routes()
            .iter()
            .filter(|route| {
                matches!(
                    &route.consumer,
                    DynamicFilterConsumer::HashJoinProbe {
                        join_node_id: route_join_node_id,
                        ..
                    } if *route_join_node_id == join_node_id
                )
            })
            .cloned()
            .collect()
    }

    fn filter_dynamic_filter_routes(&self, filter_node_id: NodeId) -> Vec<DynamicFilterRoute> {
        self.plan
            .dynamic_filter_routes()
            .iter()
            .filter(|route| {
                matches!(
                    &route.consumer,
                    DynamicFilterConsumer::Filter {
                        filter_node_id: route_filter_node_id
                    } if *route_filter_node_id == filter_node_id
                )
            })
            .cloned()
            .collect()
    }

    fn source_post_decode_dynamic_filter_routes(&self, source_node_id: NodeId) -> Vec<DynamicFilterRoute> {
        self.plan
            .dynamic_filter_routes()
            .iter()
            .filter(|route| {
                matches!(
                    &route.consumer,
                    DynamicFilterConsumer::LocalParquetSource {
                        source_node_id: route_source_node_id
                    } if *route_source_node_id == source_node_id
                ) && matches!(
                    route.consumer_lowering,
                    DynamicFilterConsumerLowering::PostDecodeOnly
                        | DynamicFilterConsumerLowering::SkippedUnsupportedStaticPredicateMix
                ) && self.dynamic_filters.produced(route.id).is_some()
            })
            .cloned()
            .collect()
    }

    fn dynamic_filter_source_snapshot(&self, source_node_id: NodeId) -> Result<DynamicFilterSourceSnapshot> {
        let Some(source) = self.plan.node(source_node_id).and_then(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        }) else {
            return Err(invalid("compiled dynamic-filter source consumer references a missing local parquet source")
                .with_fact("source_node_id", source_node_id));
        };
        let row_groups = source
            .diagnostics
            .selected_row_group_stats
            .as_ref()
            .map(|stats| stats.selected_row_groups as u64)
            .or_else(|| source.diagnostics.selected_row_groups.map(|row_groups| row_groups as u64))
            .unwrap_or(0);
        let source_bytes =
            source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_source_bytes());
        Ok(DynamicFilterSourceSnapshot { row_groups, source_bytes })
    }

    fn dynamic_filter_consumer_input_schema(
        &self,
        consumer_node_id: NodeId,
        input_node_id: NodeId,
    ) -> Result<ArrowSchemaRef> {
        self.plan.node(input_node_id).map(|node| Arc::clone(node.output_schema())).ok_or_else(|| {
            invalid("compiled dynamic-filter consumer references a missing input")
                .with_fact("consumer_node_id", consumer_node_id)
                .with_fact("consumer_input_node_id", input_node_id)
        })
    }

    fn ensure_dynamic_filter_routes_ready(&self, join_node_id: NodeId, routes: &[DynamicFilterRoute]) -> Result<()> {
        for route in routes {
            if !self.dynamic_filters.is_ready(route.id) {
                return Err(invalid("compiled dynamic-filter probe consumer ran before its producer completed")
                    .with_fact("join_node_id", join_node_id)
                    .with_fact("dynamic_filter_id", route.id.to_string()));
            }
        }
        Ok(())
    }
}

fn join_probe_input_node(join: &JoinNode) -> NodeId {
    match join.spec.build_side {
        JoinSide::Left => join.inputs[1],
        JoinSide::Right => join.inputs[0],
    }
}

fn route_targets_local_parquet_source(route: &DynamicFilterRoute) -> bool {
    matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
}

fn route_targets_reader_lowered_local_parquet_source(route: &DynamicFilterRoute) -> bool {
    route_targets_local_parquet_source(route)
        && matches!(
            route.consumer_lowering,
            DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate
        )
}

// Ranges over partitions of one producer combine to the range over the whole
// producer; an empty partition contributes nothing.
fn merge_dynamic_filter_ranges(left: DynamicFilter, right: DynamicFilter) -> DynamicFilter {
    match (left, right) {
        (DynamicFilter::Empty, other) | (other, DynamicFilter::Empty) => other,
        (
            DynamicFilter::RangeI64 { min: left_min, max: left_max },
            DynamicFilter::RangeI64 { min: right_min, max: right_max },
        ) => DynamicFilter::RangeI64 { min: left_min.min(right_min), max: left_max.max(right_max) },
        (left, _) => left,
    }
}

/// Min/max of one producer key column as an i64 range, reduced on the device
/// with retained execution so the reductions run on the producer's stream.
/// `Ok(None)` means the type has no range form here or a `u64` value exceeds
/// `i64::MAX`; `Empty` means the column has no non-null values.
fn dynamic_filter_i64_range_from_series(
    series: &cudf::Series,
    data_type: &ArrowDataType,
) -> Result<Option<DynamicFilter>> {
    let range = match data_type {
        ArrowDataType::Int32 | ArrowDataType::Date32 => {
            let min = series
                .on(Exec::RETAINED)
                .min_value::<i32>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "min", data_type))?
                .map(i64::from);
            let max = series
                .on(Exec::RETAINED)
                .max_value::<i32>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "max", data_type))?
                .map(i64::from);
            Some((min, max))
        },
        ArrowDataType::Int64 => {
            let min = series
                .on(Exec::RETAINED)
                .min_value::<i64>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "min", data_type))?;
            let max = series
                .on(Exec::RETAINED)
                .max_value::<i64>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "max", data_type))?;
            Some((min, max))
        },
        ArrowDataType::UInt32 => {
            let min = series
                .on(Exec::RETAINED)
                .min_value::<u32>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "min", data_type))?
                .map(i64::from);
            let max = series
                .on(Exec::RETAINED)
                .max_value::<u32>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "max", data_type))?
                .map(i64::from);
            Some((min, max))
        },
        ArrowDataType::UInt64 => {
            let min = series
                .on(Exec::RETAINED)
                .min_value::<u64>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "min", data_type))?;
            let max = series
                .on(Exec::RETAINED)
                .max_value::<u64>()
                .map_err(|error| dynamic_filter_range_reduction_error(error, "max", data_type))?;
            match (min, max) {
                // No non-null values, so the producer keyset is empty.
                (None, None) => Some((None, None)),
                (Some(min), Some(max)) => {
                    let (Ok(min), Ok(max)) = (i64::try_from(min), i64::try_from(max)) else {
                        // A value above `i64::MAX` cannot be represented in the i64 range
                        // payload; skip the filter instead of misreporting it as empty.
                        return Ok(None);
                    };
                    Some((Some(min), Some(max)))
                },
                (None, Some(_)) | (Some(_), None) => None,
            }
        },
        _ => None,
    };
    let Some((min, max)) = range else {
        return Ok(None);
    };
    Ok(Some(match (min, max) {
        (Some(min), Some(max)) => DynamicFilter::RangeI64 { min, max },
        _ => DynamicFilter::Empty,
    }))
}

fn dynamic_filter_range_reduction_error(
    error: cudf::Error,
    reduction: &'static str,
    data_type: &ArrowDataType,
) -> QueryError {
    cudf_query_error(
        "query_engine.dynamic_filter.producer_range",
        "failed to reduce dynamic-filter producer key column to a range payload",
        error,
    )
    .with_fact("reduction", reduction)
    .with_fact("producer_key_arrow_type", format!("{data_type:?}"))
}
