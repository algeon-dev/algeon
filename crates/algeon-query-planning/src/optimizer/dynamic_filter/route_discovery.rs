//! The route discovery pass body.
//!
//! For each supported join and key orientation the pass resolves the producer
//! lineage, then each consumer lineage, and picks the consumer that can reduce
//! work earliest: the local Parquet source when its key column is reader
//! visible and no other consumer depends on the unfiltered rows, otherwise the
//! first filter above it, otherwise the join's probe. A consumer key with
//! several source lineages, or a source with several parents, takes a
//! source-level route only under the `keyset_constraints` proof; otherwise one
//! join-probe route stands in for all of them. Every orientation is recorded
//! once in the diagnostics with its reason, so `considered_count` counts
//! orientations and residual candidates. A join with an unsupported shape counts
//! once. Route ids are positions in the returned list.

use super::{
    AggregateGroupKeyLineage, AnalysisContext, DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DynamicFilterConsumer,
    DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind, DynamicFilterProducer, DynamicFilterRoute,
    DynamicFilterRouteCandidateDiagnostics, DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions,
    DynamicFilterRouteRejectedReason, PlanNode, QueryPlan, consumer_column_type_is_supported, consumer_lowering,
    consumer_reader_column_is_visible, dedup_lineages, dynamic_filter_consumer_source_feeds_downstream_work_fanout,
    dynamic_filter_route_equivalent, dynamic_filter_route_would_create_node_cycle, input_reference_count,
    join_shape_supported, large_keyset_dimension_sized_local_parquet_consumer_is_not_promising, lineages_for_column,
    local_parquet_source, populate_selected_consumer_diagnostics, post_decode_consumer_feeds_scalar_decimal_sum,
    post_decode_consumer_for_orientation, producer_input_is_direct_aggregate_keyset_alias,
    producer_input_is_literal_batch_exact_keyset, producer_keyset_exceeds_runtime_limit, rejected_candidate,
    residual_equality_route_candidates, rewrite, route_orientations, shared_source_filter_is_keyset_constrained,
    subtree_contains_node, unique_lineage,
};
use crate::optimizer::NativeOptimizerChange;

/// Outcome of discovery. `change` is `Changed` when at least one route was
/// attached; the plan's nodes are the input's nodes.
pub(crate) struct DynamicFilterRouteDiscoveryResult {
    pub(crate) plan: QueryPlan,
    pub(crate) change: NativeOptimizerChange,
    pub(crate) diagnostics: DynamicFilterRouteDiscoveryDiagnostics,
}

pub(crate) fn optimize_dynamic_filter_route_discovery(
    plan: QueryPlan,
    options: &DynamicFilterRouteDiscoveryOptions,
    ctx: AnalysisContext<'_>,
) -> DynamicFilterRouteDiscoveryResult {
    if !options.enabled {
        return DynamicFilterRouteDiscoveryResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: DynamicFilterRouteDiscoveryDiagnostics::default(),
        };
    }

    let mut diagnostics = DynamicFilterRouteDiscoveryDiagnostics::default();
    let mut routes = Vec::new();
    let parent_counts = rewrite::parent_counts(&plan);

    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };

        if !join_shape_supported(join) {
            diagnostics.considered_count += 1;
            diagnostics.candidates.push(rejected_candidate(
                node.id,
                join,
                None,
                DynamicFilterRouteRejectedReason::UnsupportedJoinShape,
            ));
            continue;
        }

        for orientation in route_orientations(join) {
            diagnostics.considered_count += 1;
            let mut candidate = DynamicFilterRouteCandidateDiagnostics {
                rule_name: DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
                candidate_node_id: node.id,
                dynamic_filter_id: None,
                producer_build_side: Some(join.spec.build_side),
                producer_input_node_id: Some(orientation.producer_input),
                consumer_input_node_id: Some(orientation.consumer_input),
                consumer_kind: None,
                consumer_node_id: None,
                consumer_source_node_id: None,
                producer_key_column: Some(orientation.producer_key.name().to_string()),
                consumer_key_column: None,
                filter_kinds: Vec::new(),
                consumer_lowering: None,
                consumer_reduction_stage: None,
                selected_source_bytes: None,
                total_source_bytes: None,
                routed: false,
                rejected_reason: None,
            };

            let Some(producer_lineage) = unique_lineage(
                &plan,
                &parent_counts,
                orientation.producer_input,
                orientation.producer_key.name(),
                AggregateGroupKeyLineage::Producer,
            ) else {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ProducerKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            // When the build input is a direct alias of a grouping-only
            // aggregate's key set, that aggregate is the producer: its keys are
            // the build keys, and the payload exists once it finishes, so the
            // dependency root moves from the build input to the aggregate.
            let aggregate_group_key_producer =
                producer_lineage.aggregate_group_key_node_id.filter(|aggregate_node_id| {
                    producer_input_is_direct_aggregate_keyset_alias(
                        &plan,
                        orientation.producer_input,
                        orientation.producer_key.name(),
                        *aggregate_node_id,
                    )
                });
            let producer_dependency_root = aggregate_group_key_producer.unwrap_or(orientation.producer_input);
            if producer_keyset_exceeds_runtime_limit(&plan, producer_dependency_root, ctx.policy().dynamic_filter) {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ProducerKeysetExceedsRuntimeLimit);
                diagnostics.candidates.push(candidate);
                continue;
            }
            let consumer_lineages = dedup_lineages(lineages_for_column(
                &plan,
                &parent_counts,
                orientation.consumer_input,
                orientation.consumer_key.name(),
                AggregateGroupKeyLineage::Consumer,
            ));
            // A probe key with no source lineage (a join output, union, or an
            // expression) can still be filtered after decode at a filter or at
            // the probe itself.
            if consumer_lineages.is_empty() {
                let Some((consumer, consumer_column)) =
                    post_decode_consumer_for_orientation(&plan, &parent_counts, node.id, &orientation, true, true, ctx)
                else {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerKeyLineageUnavailable);
                    diagnostics.candidates.push(candidate);
                    continue;
                };
                if post_decode_consumer_feeds_scalar_decimal_sum(&plan, &parent_counts, &consumer) {
                    populate_selected_consumer_diagnostics(
                        &mut candidate,
                        &consumer,
                        &consumer_column,
                        DynamicFilterConsumerLowering::PostDecodeOnly,
                    );
                    candidate.rejected_reason =
                        Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                candidate.consumer_key_column = Some(consumer_column.clone());
                let filter_kinds =
                    vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains];
                let id = DynamicFilterId::new(routes.len());
                let producer_column = if aggregate_group_key_producer.is_some() {
                    producer_lineage.source_column.clone()
                } else {
                    orientation.producer_key.name().to_string()
                };
                let producer = aggregate_group_key_producer
                    .map(|aggregate_node_id| DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id })
                    .unwrap_or(DynamicFilterProducer::HashJoinBuild {
                        join_node_id: node.id,
                        build_side: join.spec.build_side,
                        build_input: orientation.producer_input,
                    });
                let route = DynamicFilterRoute::new(
                    id,
                    producer,
                    consumer,
                    vec![producer_column],
                    vec![consumer_column],
                    filter_kinds.clone(),
                    DynamicFilterConsumerLowering::PostDecodeOnly,
                );
                if dynamic_filter_route_would_create_node_cycle(&plan, &routes, &route) {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::DynamicFilterControlCycle);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                candidate.consumer_kind = Some(route.consumer.kind().to_string());
                candidate.consumer_node_id = Some(route.consumer.node_id());
                candidate.consumer_reduction_stage = Some(route.consumer_reduction_stage());
                routes.push(route);

                candidate.dynamic_filter_id = Some(id);
                candidate.filter_kinds = filter_kinds;
                candidate.consumer_lowering = Some(DynamicFilterConsumerLowering::PostDecodeOnly);
                candidate.routed = true;
                diagnostics.route_count += 1;
                diagnostics.candidates.push(candidate);
                continue;
            };

            // Several lineages mean the probe key is assembled from more than one
            // source (a union or a projection over a join). Each source is still
            // reported, but when one of them is shared, a single join-probe route
            // replaces per-source routes so no shared source is filtered for one
            // consumer only.
            let multiple_consumer_lineages = consumer_lineages.len() > 1;
            let mixed_lineage_join_probe_route_available = multiple_consumer_lineages
                && consumer_lineages.iter().any(|consumer_lineage| {
                    local_parquet_source(&plan, consumer_lineage.source_node_id).is_some()
                        && (input_reference_count(&plan, consumer_lineage.source_node_id) > 1
                            || consumer_lineage.has_shared_path)
                })
                && post_decode_consumer_for_orientation(&plan, &parent_counts, node.id, &orientation, true, true, ctx)
                    .is_some_and(|(consumer, _)| {
                        !post_decode_consumer_feeds_scalar_decimal_sum(&plan, &parent_counts, &consumer)
                    });
            let mut needs_mixed_lineage_join_probe_route = false;
            for consumer_lineage in consumer_lineages {
                let mut candidate = candidate.clone();
                candidate.consumer_source_node_id = Some(consumer_lineage.source_node_id);
                candidate.consumer_key_column = Some(consumer_lineage.source_column.clone());

                if producer_lineage.source_node_id == consumer_lineage.source_node_id {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::SameSource);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                if subtree_contains_node(&plan, producer_dependency_root, consumer_lineage.source_node_id) {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerUpstreamOfProducer);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                let consumer_source = local_parquet_source(&plan, consumer_lineage.source_node_id);
                let (consumer, consumer_column, consumer_lowering) = if let Some(consumer_source) = consumer_source {
                    let stats = consumer_source.diagnostics.selected_row_group_stats.as_ref();
                    candidate.selected_source_bytes = stats.and_then(|stats| stats.selected_source_bytes());
                    candidate.total_source_bytes = stats.and_then(|stats| stats.total_source_bytes());

                    if input_reference_count(&plan, consumer_lineage.source_node_id) > 1
                        || consumer_lineage.has_shared_path
                    {
                        if !multiple_consumer_lineages
                            && shared_source_filter_is_keyset_constrained(
                                &plan,
                                &parent_counts,
                                consumer_lineage.source_node_id,
                                consumer_lineage.source_column.as_str(),
                                producer_dependency_root,
                                producer_lineage.source_node_id,
                                producer_lineage.source_column.as_str(),
                            )
                        {
                            if !consumer_reader_column_is_visible(consumer_source, &consumer_lineage.source_column) {
                                candidate.rejected_reason =
                                    Some(DynamicFilterRouteRejectedReason::ConsumerColumnNotReaderVisible);
                                diagnostics.candidates.push(candidate);
                                continue;
                            }
                            if !consumer_column_type_is_supported(consumer_source, &consumer_lineage.source_column, ctx)
                            {
                                candidate.rejected_reason =
                                    Some(DynamicFilterRouteRejectedReason::UnsupportedConsumerColumnType);
                                diagnostics.candidates.push(candidate);
                                continue;
                            }
                            (
                                DynamicFilterConsumer::LocalParquetSource {
                                    source_node_id: consumer_lineage.source_node_id,
                                },
                                consumer_lineage.source_column.clone(),
                                DynamicFilterConsumerLowering::PostDecodeOnly,
                            )
                        } else {
                            if multiple_consumer_lineages {
                                needs_mixed_lineage_join_probe_route = true;
                                candidate.rejected_reason =
                                    Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan);
                                diagnostics.candidates.push(candidate);
                                continue;
                            }
                            let Some((consumer, consumer_column)) = post_decode_consumer_for_orientation(
                                &plan,
                                &parent_counts,
                                node.id,
                                &orientation,
                                true,
                                false,
                                ctx,
                            ) else {
                                candidate.rejected_reason =
                                    Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan);
                                diagnostics.candidates.push(candidate);
                                continue;
                            };
                            if post_decode_consumer_feeds_scalar_decimal_sum(&plan, &parent_counts, &consumer) {
                                populate_selected_consumer_diagnostics(
                                    &mut candidate,
                                    &consumer,
                                    &consumer_column,
                                    DynamicFilterConsumerLowering::PostDecodeOnly,
                                );
                                candidate.rejected_reason =
                                    Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum);
                                diagnostics.candidates.push(candidate);
                                continue;
                            }
                            candidate.consumer_key_column = Some(consumer_column.clone());
                            (consumer, consumer_column, DynamicFilterConsumerLowering::PostDecodeOnly)
                        }
                    } else {
                        if !consumer_reader_column_is_visible(consumer_source, &consumer_lineage.source_column) {
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::ConsumerColumnNotReaderVisible);
                            diagnostics.candidates.push(candidate);
                            continue;
                        }
                        if !consumer_column_type_is_supported(consumer_source, &consumer_lineage.source_column, ctx) {
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::UnsupportedConsumerColumnType);
                            diagnostics.candidates.push(candidate);
                            continue;
                        }
                        // The size heuristic is skipped when a join-probe route
                        // already covers this source, when the producer is an
                        // exact literal key set, or when the source feeds work
                        // that fans out, since fewer source rows then save more
                        // than the source read itself.
                        if !mixed_lineage_join_probe_route_available
                            && !producer_input_is_literal_batch_exact_keyset(&plan, orientation.producer_input)
                            && !dynamic_filter_consumer_source_feeds_downstream_work_fanout(
                                &plan,
                                consumer_lineage.source_node_id,
                            )
                            && large_keyset_dimension_sized_local_parquet_consumer_is_not_promising(
                                &plan,
                                orientation.producer_input,
                                consumer_source,
                                ctx.policy().dynamic_filter,
                            )
                        {
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::LargeKeysetDimensionSizedLocalParquetConsumer);
                            diagnostics.candidates.push(candidate);
                            continue;
                        }

                        (
                            DynamicFilterConsumer::LocalParquetSource {
                                source_node_id: consumer_lineage.source_node_id,
                            },
                            consumer_lineage.source_column.clone(),
                            consumer_lowering(consumer_source, ctx),
                        )
                    }
                } else if multiple_consumer_lineages {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerNotLocalParquet);
                    diagnostics.candidates.push(candidate);
                    continue;
                } else {
                    let Some((consumer, consumer_column)) = post_decode_consumer_for_orientation(
                        &plan,
                        &parent_counts,
                        node.id,
                        &orientation,
                        true,
                        true,
                        ctx,
                    ) else {
                        candidate.rejected_reason =
                            Some(DynamicFilterRouteRejectedReason::UnsupportedConsumerColumnType);
                        diagnostics.candidates.push(candidate);
                        continue;
                    };
                    if post_decode_consumer_feeds_scalar_decimal_sum(&plan, &parent_counts, &consumer) {
                        populate_selected_consumer_diagnostics(
                            &mut candidate,
                            &consumer,
                            &consumer_column,
                            DynamicFilterConsumerLowering::PostDecodeOnly,
                        );
                        candidate.rejected_reason =
                            Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum);
                        diagnostics.candidates.push(candidate);
                        continue;
                    }
                    candidate.consumer_key_column = Some(consumer_column.clone());
                    (consumer, consumer_column, DynamicFilterConsumerLowering::PostDecodeOnly)
                };
                let filter_kinds =
                    vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains];
                let id = DynamicFilterId::new(routes.len());
                let producer_column = if aggregate_group_key_producer.is_some() {
                    producer_lineage.source_column.clone()
                } else {
                    orientation.producer_key.name().to_string()
                };
                let producer = aggregate_group_key_producer
                    .map(|aggregate_node_id| DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id })
                    .unwrap_or(DynamicFilterProducer::HashJoinBuild {
                        join_node_id: node.id,
                        build_side: join.spec.build_side,
                        build_input: orientation.producer_input,
                    });
                let route = DynamicFilterRoute::new(
                    id,
                    producer,
                    consumer,
                    vec![producer_column],
                    vec![consumer_column],
                    filter_kinds.clone(),
                    consumer_lowering,
                );
                if dynamic_filter_route_would_create_node_cycle(&plan, &routes, &route) {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::DynamicFilterControlCycle);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                candidate.consumer_kind = Some(route.consumer.kind().to_string());
                candidate.consumer_node_id = Some(route.consumer.node_id());
                candidate.consumer_reduction_stage = Some(route.consumer_reduction_stage());
                routes.push(route);

                candidate.dynamic_filter_id = Some(id);
                candidate.filter_kinds = filter_kinds;
                candidate.consumer_lowering = Some(consumer_lowering);
                candidate.routed = true;
                diagnostics.route_count += 1;
                diagnostics.candidates.push(candidate);
            }
            if needs_mixed_lineage_join_probe_route {
                let Some((consumer, consumer_column)) =
                    post_decode_consumer_for_orientation(&plan, &parent_counts, node.id, &orientation, true, true, ctx)
                else {
                    continue;
                };
                let mut candidate = candidate.clone();
                if post_decode_consumer_feeds_scalar_decimal_sum(&plan, &parent_counts, &consumer) {
                    populate_selected_consumer_diagnostics(
                        &mut candidate,
                        &consumer,
                        &consumer_column,
                        DynamicFilterConsumerLowering::PostDecodeOnly,
                    );
                    candidate.rejected_reason =
                        Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                let filter_kinds =
                    vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains];
                let id = DynamicFilterId::new(routes.len());
                let producer_column = if aggregate_group_key_producer.is_some() {
                    producer_lineage.source_column.clone()
                } else {
                    orientation.producer_key.name().to_string()
                };
                let producer = aggregate_group_key_producer
                    .map(|aggregate_node_id| DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id })
                    .unwrap_or(DynamicFilterProducer::HashJoinBuild {
                        join_node_id: node.id,
                        build_side: join.spec.build_side,
                        build_input: orientation.producer_input,
                    });
                let route = DynamicFilterRoute::new(
                    id,
                    producer,
                    consumer,
                    vec![producer_column],
                    vec![consumer_column.clone()],
                    filter_kinds.clone(),
                    DynamicFilterConsumerLowering::PostDecodeOnly,
                );
                if routes.iter().any(|existing| dynamic_filter_route_equivalent(existing, &route)) {
                    continue;
                }
                if dynamic_filter_route_would_create_node_cycle(&plan, &routes, &route) {
                    populate_selected_consumer_diagnostics(
                        &mut candidate,
                        &route.consumer,
                        &consumer_column,
                        DynamicFilterConsumerLowering::PostDecodeOnly,
                    );
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::DynamicFilterControlCycle);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                candidate.consumer_kind = Some(route.consumer.kind().to_string());
                candidate.consumer_node_id = Some(route.consumer.node_id());
                candidate.consumer_key_column = Some(consumer_column);
                candidate.consumer_reduction_stage = Some(route.consumer_reduction_stage());
                routes.push(route);

                candidate.dynamic_filter_id = Some(id);
                candidate.filter_kinds = filter_kinds;
                candidate.consumer_lowering = Some(DynamicFilterConsumerLowering::PostDecodeOnly);
                candidate.routed = true;
                diagnostics.route_count += 1;
                diagnostics.candidates.push(candidate);
            }
        }

        for residual_candidate in residual_equality_route_candidates(&plan, &parent_counts, node.id, join) {
            diagnostics.considered_count += 1;
            let mut candidate = DynamicFilterRouteCandidateDiagnostics {
                rule_name: DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
                candidate_node_id: node.id,
                dynamic_filter_id: None,
                producer_build_side: Some(residual_candidate.producer_build_side),
                producer_input_node_id: Some(residual_candidate.producer_input),
                consumer_input_node_id: Some(residual_candidate.consumer_input),
                consumer_kind: None,
                consumer_node_id: None,
                consumer_source_node_id: Some(residual_candidate.consumer_lineage.source_node_id),
                producer_key_column: Some(residual_candidate.producer_column.clone()),
                consumer_key_column: Some(residual_candidate.consumer_lineage.source_column.clone()),
                filter_kinds: Vec::new(),
                consumer_lowering: None,
                consumer_reduction_stage: None,
                selected_source_bytes: None,
                total_source_bytes: None,
                routed: false,
                rejected_reason: None,
            };

            let consumer_source = local_parquet_source(&plan, residual_candidate.consumer_lineage.source_node_id);
            let Some(consumer_source) = consumer_source else {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerNotLocalParquet);
                diagnostics.candidates.push(candidate);
                continue;
            };
            let stats = consumer_source.diagnostics.selected_row_group_stats.as_ref();
            candidate.selected_source_bytes = stats.and_then(|stats| stats.selected_source_bytes());
            candidate.total_source_bytes = stats.and_then(|stats| stats.total_source_bytes());

            if producer_keyset_exceeds_runtime_limit(
                &plan,
                residual_candidate.producer_input,
                ctx.policy().dynamic_filter,
            ) {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ProducerKeysetExceedsRuntimeLimit);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if input_reference_count(&plan, residual_candidate.consumer_lineage.source_node_id) > 1
                || residual_candidate.consumer_lineage.has_shared_path
            {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if residual_candidate.producer_lineage_has_shared_path {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ProducerLineageSharedPath);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if subtree_contains_node(
                &plan,
                residual_candidate.producer_input,
                residual_candidate.consumer_lineage.source_node_id,
            ) {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerUpstreamOfProducer);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if !consumer_reader_column_is_visible(consumer_source, &residual_candidate.consumer_lineage.source_column) {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerColumnNotReaderVisible);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if !consumer_column_type_is_supported(
                consumer_source,
                &residual_candidate.consumer_lineage.source_column,
                ctx,
            ) {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::UnsupportedConsumerColumnType);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if !producer_input_is_literal_batch_exact_keyset(&plan, residual_candidate.producer_input)
                && !dynamic_filter_consumer_source_feeds_downstream_work_fanout(
                    &plan,
                    residual_candidate.consumer_lineage.source_node_id,
                )
                && large_keyset_dimension_sized_local_parquet_consumer_is_not_promising(
                    &plan,
                    residual_candidate.producer_input,
                    consumer_source,
                    ctx.policy().dynamic_filter,
                )
            {
                candidate.rejected_reason =
                    Some(DynamicFilterRouteRejectedReason::LargeKeysetDimensionSizedLocalParquetConsumer);
                diagnostics.candidates.push(candidate);
                continue;
            }

            let filter_kinds =
                vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains];
            let consumer_lowering = consumer_lowering(consumer_source, ctx);
            let id = DynamicFilterId::new(routes.len());
            let route = DynamicFilterRoute::new(
                id,
                residual_candidate.producer,
                DynamicFilterConsumer::LocalParquetSource {
                    source_node_id: residual_candidate.consumer_lineage.source_node_id,
                },
                vec![residual_candidate.producer_column],
                vec![residual_candidate.consumer_lineage.source_column],
                filter_kinds.clone(),
                consumer_lowering,
            );
            if routes.iter().any(|existing| dynamic_filter_route_equivalent(existing, &route)) {
                continue;
            }
            if dynamic_filter_route_would_create_node_cycle(&plan, &routes, &route) {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::DynamicFilterControlCycle);
                diagnostics.candidates.push(candidate);
                continue;
            }
            candidate.consumer_kind = Some(route.consumer.kind().to_string());
            candidate.consumer_node_id = Some(route.consumer.node_id());
            candidate.consumer_reduction_stage = Some(route.consumer_reduction_stage());
            candidate.dynamic_filter_id = Some(id);
            candidate.filter_kinds = filter_kinds;
            candidate.consumer_lowering = Some(consumer_lowering);
            candidate.routed = true;
            diagnostics.route_count += 1;
            routes.push(route);
            diagnostics.candidates.push(candidate);
        }
    }

    let change = if routes.is_empty() { NativeOptimizerChange::Unchanged } else { NativeOptimizerChange::Changed };
    let plan = plan.with_dynamic_filter_routes(routes);
    DynamicFilterRouteDiscoveryResult { plan, change, diagnostics }
}
