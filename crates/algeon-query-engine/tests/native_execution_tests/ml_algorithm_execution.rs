#[cfg(feature = "cuml")]
mod enabled {
    use std::{fmt::Debug, fs, iter, str::FromStr, sync::Arc};

    use arrow::{
        array::{
            ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, ListArray, StringArray, UInt64Array,
        },
        buffer::OffsetBuffer,
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    };
    use cudf::{DataFrame, DenseMatrixLayout};
    use cuml::{
        CumlResources, DeviceMatrix, ForestModelFormat, ForestOutput, ForestOutputKind, ForestPrecision,
        LogisticRegressionModel, LogisticRegressionParams, MatrixLayout, OwnedFloat32Matrix, forest_inference_version,
        inspect_forest_model,
    };
    use query_engine::{
        error::{QueryError, QueryErrorCode},
        exec::{
            QueryExecutionRequest, admit_query_plan, output::QueryOutput, set_forest_after_import_hook_for_tests,
            set_forest_batch_element_limit_for_tests,
        },
        metrics::QueryMetrics,
        native_error::{cuml_query_error, query_error_code_for_cuml_error},
        observability::{
            LiveRuntimeObservationSnapshot, RuntimeObservationConfig, RuntimeObservationPayload,
            RuntimeOperatorStageKind, RuntimeOperatorStageState,
        },
        plan::{
            MlAlgorithm, MlAlgorithmParams, MlAlgorithmSpec, MlForestModelFacts, MlForestModelFormat,
            MlForestModelSnapshot, MlForestOutputContract, MlForestPrecision, MlForestPredictParams, MlHdbscanParams,
            MlLogisticRegressionParams, MlRandomForestParams, MlRelationBinding, MlTsvdParams, MlUmapInit,
            MlUmapParams, QueryPlan, QueryPlanBuilder, VectorRowIdBinding, ml_embedding_output_schema,
            ml_forest_output_schema, ml_hdbscan_output_schema, ml_predictions_output_schema, vector_pca_output_schema,
        },
        runtime::{
            ExecutionContext, MemoryControlPolicy, StandaloneConfig,
            resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
        },
        source::{GpuResidentSource, LocalParquetSource, SourceNode},
    };

    use super::super::{
        assert_attempt_domain_released, backend_context, backend_context_with_grant,
        capability_evidence_for_fixture_plan, execute_query_plan_direct,
    };

    #[derive(Clone, Copy)]
    enum Shape {
        Wide,
        Fixed,
        List,
    }

    fn input_schema(shape: Shape) -> SchemaRef {
        let mut fields = vec![Field::new("id", DataType::Int64, false)];
        match shape {
            Shape::Wide => fields.extend((0..3).map(|i| Field::new(format!("x{i}"), DataType::Float32, false))),
            Shape::Fixed | Shape::List => {
                let element = Arc::new(Field::new("item", DataType::Float32, false));
                let dtype = match shape {
                    Shape::Fixed => DataType::FixedSizeList(element, 3),
                    Shape::List => DataType::List(element),
                    Shape::Wide => unreachable!(),
                };
                fields.push(Field::new("vector", dtype, false));
            },
        }
        Arc::new(Schema::new(fields))
    }

    fn plan(schema: SchemaRef, algorithm: MlAlgorithm, shape: Shape, components: usize) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let binding = match shape {
            Shape::Wide => MlRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x0", "x1", "x2"]),
            Shape::Fixed | Shape::List => {
                MlRelationBinding::new_vector_column(input, VectorRowIdBinding::column("id"), "vector")
            },
        };
        let params = match algorithm {
            MlAlgorithm::Tsvd => MlAlgorithmParams::Tsvd(MlTsvdParams::new(binding, components)),
            MlAlgorithm::Hdbscan => {
                let mut params = MlHdbscanParams::new(binding);
                params.min_samples = 5;
                params.min_cluster_size = 5;
                MlAlgorithmParams::Hdbscan(params)
            },
            MlAlgorithm::Umap => {
                let mut params = MlUmapParams::new(binding);
                params.n_components = components;
                params.n_neighbors = 3;
                params.init = MlUmapInit::Random;
                MlAlgorithmParams::Umap(params)
            },
            MlAlgorithm::LogisticRegression
            | MlAlgorithm::RandomForestClassifier
            | MlAlgorithm::RandomForestRegressor
            | MlAlgorithm::ForestPredict => unreachable!("unsupervised plan helper"),
        };
        let output_schema = match algorithm {
            MlAlgorithm::Tsvd => vector_pca_output_schema(DataType::Int64, components),
            MlAlgorithm::Hdbscan => ml_hdbscan_output_schema(DataType::Int64),
            MlAlgorithm::Umap => ml_embedding_output_schema(DataType::Int64, components),
            MlAlgorithm::LogisticRegression
            | MlAlgorithm::RandomForestClassifier
            | MlAlgorithm::RandomForestRegressor
            | MlAlgorithm::ForestPredict => unreachable!("unsupervised plan helper"),
        };
        let root = builder.add_ml_algorithm(MlAlgorithmSpec::new(algorithm, params, output_schema));
        builder.finish_with_host_sink(root)
    }

    fn batch(shape: Shape, rows: usize) -> RecordBatch {
        let x: Vec<f32> = (0..rows).map(|i| i as f32 - 3.0).collect();
        let ids: ArrayRef =
            Arc::new(Int64Array::from((0..rows).map(|i| if i == 1 { 0 } else { i as i64 }).collect::<Vec<_>>()));
        let a: ArrayRef = Arc::new(Float32Array::from(x.clone()));
        let b: ArrayRef = Arc::new(Float32Array::from(x.iter().map(|v| 2.0 * v).collect::<Vec<_>>()));
        let c: ArrayRef = Arc::new(Float32Array::from(x.iter().map(|v| -v).collect::<Vec<_>>()));
        let columns = match shape {
            Shape::Wide => vec![ids, a, b, c],
            Shape::Fixed | Shape::List => {
                let values: ArrayRef =
                    Arc::new(Float32Array::from(x.into_iter().flat_map(|v| [v, 2.0 * v, -v]).collect::<Vec<_>>()));
                let element = Arc::new(Field::new("item", DataType::Float32, false));
                let vector: ArrayRef = match shape {
                    Shape::Fixed => Arc::new(FixedSizeListArray::try_new(element, 3, values, None).unwrap()),
                    Shape::List => Arc::new(
                        ListArray::try_new(
                            element,
                            OffsetBuffer::new((0..=rows as i32).map(|i| i * 3).collect::<Vec<_>>().into()),
                            values,
                            None,
                        )
                        .unwrap(),
                    ),
                    Shape::Wide => unreachable!(),
                };
                vec![ids, vector]
            },
        };
        RecordBatch::try_new(input_schema(shape), columns).unwrap()
    }

    fn execute(
        algorithm: MlAlgorithm,
        shape: Shape,
        rows: usize,
        components: usize,
    ) -> (Result<QueryOutput, QueryError>, QueryMetrics, Option<LiveRuntimeObservationSnapshot>) {
        execute_batch(algorithm, shape, components, batch(shape, rows))
    }

    fn execute_batch(
        algorithm: MlAlgorithm,
        shape: Shape,
        components: usize,
        batch: RecordBatch,
    ) -> (Result<QueryOutput, QueryError>, QueryMetrics, Option<LiveRuntimeObservationSnapshot>) {
        let schema = input_schema(shape);
        let plan = plan(Arc::clone(&schema), algorithm, shape, components);
        let context = backend_context();
        let attempt =
            context.begin_native_attempt("ml-execution", RuntimeObservationConfig::bounded(4096, 16 * 1024 * 1024));
        let binding = attempt.admission_binding();
        let evidence = capability_evidence_for_fixture_plan(&plan).expect("ML capability evidence");
        let admitted =
            context.admit_evidence_with_observation(evidence, Some(binding.clone())).expect("ML plan admission");
        let runtime = Arc::new(admitted.runtime().expect("execution runtime"));
        let mut inputs = GpuResidentInputMap::new();
        inputs.insert(
            "input".into(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).expect("upload")),
                schema,
            )
            .unwrap(),
        );
        let admitted_plan =
            admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
                .unwrap();
        let outcome = execute_query_plan_direct(
            admitted.context.context(),
            QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&inputs),
        );
        let (result, metrics, _) = outcome.into_parts();
        let observation = binding.observation_handle().live_snapshot();
        drop(inputs);
        drop(runtime);
        admitted.finish();
        (result, metrics, observation)
    }

    fn execute_supervised(
        algorithm: MlAlgorithm,
        classes: usize,
        labels_override: Option<Vec<i32>>,
        predict_rows: usize,
    ) -> (Result<QueryOutput, QueryError>, LiveRuntimeObservationSnapshot) {
        let training_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x0", DataType::Float32, false),
            Field::new("x1", DataType::Float32, false),
            Field::new(
                "label",
                if algorithm == MlAlgorithm::RandomForestRegressor { DataType::Float32 } else { DataType::Int32 },
                false,
            ),
        ]));
        let predict_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("x0", DataType::Float32, false),
            Field::new("x1", DataType::Float32, false),
        ]));
        let centers = if algorithm == MlAlgorithm::RandomForestRegressor {
            vec![-4.0_f32, 4.0]
        } else if classes == 2 {
            vec![-8.0_f32, 8.0]
        } else {
            vec![-8.0_f32, 0.0, 8.0]
        };
        let training_x =
            centers.iter().flat_map(|center| (0..6).map(move |i| center + (i as f32 - 2.5) * 0.1)).collect::<Vec<_>>();
        let labels = labels_override.unwrap_or_else(|| (0..classes).flat_map(|class| [class as i32; 6]).collect());
        let training_batch = RecordBatch::try_new(
            Arc::clone(&training_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..training_x.len() as i64)) as ArrayRef,
                Arc::new(Float32Array::from(training_x.clone())),
                Arc::new(Float32Array::from(training_x.iter().map(|value| value * 0.25).collect::<Vec<_>>())),
                if algorithm == MlAlgorithm::RandomForestRegressor {
                    Arc::new(Float32Array::from(training_x.iter().map(|x| 2.0 * x + 1.0).collect::<Vec<_>>()))
                } else {
                    Arc::new(Int32Array::from(labels))
                },
            ],
        )
        .unwrap();
        let predict_x = if predict_rows <= classes {
            centers[..predict_rows].to_vec()
        } else {
            training_x[..predict_rows].to_vec()
        };
        let predict_batch = RecordBatch::try_new(
            Arc::clone(&predict_schema),
            vec![
                Arc::new(StringArray::from((0..predict_rows).map(|i| format!("p{i}")).collect::<Vec<_>>())) as ArrayRef,
                Arc::new(Float32Array::from(predict_x.clone())),
                Arc::new(Float32Array::from(predict_x.iter().map(|value| value * 0.25).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let mut builder = QueryPlanBuilder::new();
        let training = builder
            .add_source(SourceNode::GpuResident(GpuResidentSource::new("training", Arc::clone(&training_schema))));
        let predict =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("predict", Arc::clone(&predict_schema))));
        let mut training_binding = MlRelationBinding::new(training, VectorRowIdBinding::column("id"), ["x0", "x1"]);
        training_binding.label = Some("label".to_owned());
        let predict_binding = MlRelationBinding::new(predict, VectorRowIdBinding::column("id"), ["x0", "x1"]);
        let params = match algorithm {
            MlAlgorithm::LogisticRegression => MlAlgorithmParams::LogisticRegression(MlLogisticRegressionParams::new(
                training_binding,
                predict_binding,
            )),
            MlAlgorithm::RandomForestClassifier => {
                let mut p = MlRandomForestParams::new(training_binding, predict_binding);
                p.n_trees = 100;
                MlAlgorithmParams::RandomForestClassifier(p)
            },
            MlAlgorithm::RandomForestRegressor => {
                let mut p = MlRandomForestParams::new(training_binding, predict_binding);
                p.n_trees = 24;
                p.criterion = None;
                MlAlgorithmParams::RandomForestRegressor(p)
            },
            _ => unreachable!("supervised fixture"),
        };
        let output_type =
            if algorithm == MlAlgorithm::RandomForestRegressor { DataType::Float32 } else { DataType::Int32 };
        let root = builder.add_ml_algorithm(MlAlgorithmSpec::new(
            algorithm,
            params,
            ml_predictions_output_schema(DataType::Utf8, output_type),
        ));
        let plan = builder.finish_with_host_sink(root);
        let context = backend_context();
        let attempt =
            context.begin_native_attempt("ml-supervised", RuntimeObservationConfig::bounded(4096, 16 * 1024 * 1024));
        let binding = attempt.admission_binding();
        let evidence = capability_evidence_for_fixture_plan(&plan).unwrap();
        let admitted = context.admit_evidence_with_observation(evidence, Some(binding.clone())).unwrap();
        let runtime = Arc::new(admitted.runtime().unwrap());
        let mut inputs = GpuResidentInputMap::new();
        for (name, batch, schema) in
            [("training", training_batch, training_schema), ("predict", predict_batch, predict_schema)]
        {
            inputs.insert(
                name.into(),
                GpuResultHandle::new(
                    &admitted,
                    GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
                    schema,
                )
                .unwrap(),
            );
        }
        let admitted_plan =
            admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
                .unwrap();
        let (result, _, _) = execute_query_plan_direct(
            admitted.context.context(),
            QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&inputs),
        )
        .into_parts();
        let observation = binding.observation_handle().live_snapshot().unwrap();
        drop(inputs);
        drop(runtime);
        admitted.finish();
        (result, observation)
    }

    #[test]
    fn cuml_query_error_projects_native_facts_without_message_based_classification() {
        let runtime = Arc::new(cudf::Runtime::new().expect("GPU runtime"));
        let resources = CumlResources::for_runtime(runtime).expect("cuML resources");
        let source = OwnedFloat32Matrix::new(&resources, 0, 2, MatrixLayout::ColumnMajor)
            .err()
            .expect("empty matrix is invalid");
        assert_eq!(query_error_code_for_cuml_error(&source), QueryErrorCode::CudfInvalidInput);
        let error = cuml_query_error("query_engine.test.from_cuml", "invalid matrix", source);
        assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
        assert_eq!(error.fact_value("cuml_code"), Some("ForeignInvalidArgument"));
        assert_eq!(error.fact_value("cuml_kind"), Some("InvalidInput"));
        assert_eq!(error.fact_value("cuml_status"), Some("permanent"));
        assert_eq!(error.fact_value("cuml_phase"), Some("preflight"));
        assert_eq!(error.fact_value("cuml_operation"), Some("OwnedFloat32Matrix::new"));
        assert!(error.fact_value("cuml_message").is_some());
        assert!(error.fact_value("cuml_call_chain").is_some());
    }

    #[test]
    fn unsupported_label_dtype_retains_reason_without_changing_typed_classification() {
        let runtime = Arc::new(cudf::Runtime::new().expect("GPU runtime"));
        let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
        let column = runtime.column_from_slice(&[-1.0_f32, 1.0]).expect("features");
        let packed = runtime.pack_columns(&[&column], DenseMatrixLayout::ColumnMajor).expect("matrix");
        let packed = packed.validate().expect("finite matrix");
        let input = DeviceMatrix::from_validated(&packed).expect("matrix view");
        let labels = runtime.series(&[0_i64, 1]).expect("unsupported labels");
        let source = LogisticRegressionModel::fit(&resources, &input, &labels, LogisticRegressionParams::default())
            .err()
            .expect("unsupported dtype");
        assert_eq!(source.detail_value("reason"), Some("unsupported_label_type"));
        assert_eq!(query_error_code_for_cuml_error(&source), QueryErrorCode::CudfUnsupported);
        let error = cuml_query_error("query_engine.test.unsupported_labels", "invalid label dtype", source);
        assert_eq!(error.code(), QueryErrorCode::CudfUnsupported);
        assert_eq!(error.fact_value("reason"), Some("unsupported_label_type"));
    }

    #[test]
    fn supervised_prediction_uses_predict_ids_and_records_fit_and_predict_stages() {
        for classes in [2, 3] {
            let (result, observation) = execute_supervised(MlAlgorithm::LogisticRegression, classes, None, classes);
            let QueryOutput::HostArrow(output) = result.expect("fit and classify") else { panic!("host output") };
            assert_eq!(
                output.schema().as_ref(),
                ml_predictions_output_schema(DataType::Utf8, DataType::Int32).as_ref()
            );
            let ids = output.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let predictions = output.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..classes {
                assert_eq!(ids.value(i), format!("p{i}"));
                assert_eq!(predictions.value(i), i as i32);
            }
            let stages = observation
                .events()
                .iter()
                .filter_map(|event| match &event.payload {
                    RuntimeObservationPayload::OperatorStage(stage)
                        if matches!(stage.lifecycle, RuntimeOperatorStageState::Finished) =>
                    {
                        Some(stage.as_ref())
                    },
                    _ => None,
                })
                .filter(|stage| {
                    matches!(
                        stage.stage_kind,
                        RuntimeOperatorStageKind::MlPack
                            | RuntimeOperatorStageKind::MlNative
                            | RuntimeOperatorStageKind::MlAssemble
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                stages.iter().map(|stage| stage.stage_kind).collect::<Vec<_>>(),
                [
                    RuntimeOperatorStageKind::MlPack,
                    RuntimeOperatorStageKind::MlNative,
                    RuntimeOperatorStageKind::MlPack,
                    RuntimeOperatorStageKind::MlNative,
                    RuntimeOperatorStageKind::MlAssemble
                ]
            );
            assert_eq!(stages[0].vector_relation_role.as_deref(), Some("training"));
            assert_eq!(stages[2].vector_relation_role.as_deref(), Some("predict"));
            assert_eq!(
                stages[1].strategy_label.as_deref(),
                Some(if classes == 2 { "qn_logistic:converged" } else { "qn_softmax:converged" })
            );
            assert_eq!(stages[3].strategy_label.as_deref(), Some("predict"));
        }
    }

    #[test]
    fn random_forest_classifier_and_regressor_score_independent_rows() {
        for algorithm in [MlAlgorithm::RandomForestClassifier, MlAlgorithm::RandomForestRegressor] {
            let (result, observation) = execute_supervised(algorithm, 2, None, 12);
            let QueryOutput::HostArrow(output) = result.expect("RF prediction") else { panic!("host output") };
            let ids = output.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(
                (0..12).map(|row| ids.value(row)).collect::<Vec<_>>(),
                (0..12).map(|row| format!("p{row}")).collect::<Vec<_>>()
            );
            if algorithm == MlAlgorithm::RandomForestClassifier {
                let predictions = output.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
                let accuracy = (0..12).filter(|&row| predictions.value(row) == (row / 6) as i32).count() as f32 / 12.0;
                assert!(accuracy >= 0.95, "RF training-set accuracy {accuracy}");
            } else {
                let predictions = output.column(2).as_any().downcast_ref::<Float32Array>().unwrap();
                let mse = (0..12)
                    .map(|i| {
                        let x = if i < 6 { -4.0 } else { 4.0 } + ((i % 6) as f32 - 2.5) * 0.1;
                        (predictions.value(i) - (2.0 * x + 1.0)).powi(2)
                    })
                    .sum::<f32>()
                    / 12.0;
                assert!(mse < 12.0, "RF regression MSE {mse}");
            }
            let stages = observation
                .events()
                .iter()
                .filter_map(|event| match &event.payload {
                    RuntimeObservationPayload::OperatorStage(stage)
                        if stage.lifecycle == RuntimeOperatorStageState::Finished
                            && matches!(
                                stage.stage_kind,
                                RuntimeOperatorStageKind::MlPack
                                    | RuntimeOperatorStageKind::MlNative
                                    | RuntimeOperatorStageKind::MlAssemble
                            ) =>
                    {
                        Some(stage.as_ref())
                    },
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                stages.iter().map(|stage| stage.stage_kind).collect::<Vec<_>>(),
                [
                    RuntimeOperatorStageKind::MlPack,
                    RuntimeOperatorStageKind::MlNative,
                    RuntimeOperatorStageKind::MlPack,
                    RuntimeOperatorStageKind::MlNative,
                    RuntimeOperatorStageKind::MlAssemble,
                ]
            );
            assert_eq!(
                stages[1].strategy_label.as_deref(),
                Some(if algorithm == MlAlgorithm::RandomForestClassifier {
                    "rf_classifier:gini"
                } else {
                    "rf_regressor:mse"
                })
            );
            assert_eq!(stages[3].strategy_label.as_deref(), Some("predict_nvforest"));
            let (empty, observation) = execute_supervised(algorithm, 2, None, 0);
            let QueryOutput::HostArrow(empty) = empty.expect("empty predict") else { panic!("host output") };
            assert_eq!(empty.num_rows(), 0);
            assert!(!observation.events().iter().any(|event| matches!(&event.payload,
                RuntimeObservationPayload::OperatorStage(stage) if stage.stage_kind == RuntimeOperatorStageKind::MlNative)));
        }
    }

    #[test]
    fn supervised_label_errors_and_empty_predict_are_structured() {
        for (labels, reason) in [
            (vec![0; 12], "single_class"),
            ((0..12).map(|i| if i < 6 { 0 } else { 2 }).collect(), "labels_not_contiguous"),
        ] {
            let (result, _) = execute_supervised(MlAlgorithm::LogisticRegression, 2, Some(labels), 2);
            let error = result.err().expect("invalid labels");
            assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
            assert_eq!(error.fact_value("reason"), Some(reason));
            assert_eq!(error.fact_value("cuml_code"), Some("LabelOutOfRange"));
        }
        let (result, observation) = execute_supervised(MlAlgorithm::LogisticRegression, 2, None, 0);
        let QueryOutput::HostArrow(output) = result.expect("empty predict") else { panic!("host output") };
        assert_eq!(output.num_rows(), 0);
        assert_eq!(output.schema().as_ref(), ml_predictions_output_schema(DataType::Utf8, DataType::Int32).as_ref());
        assert!(
            !observation.events().iter().any(|event| matches!(&event.payload,
            RuntimeObservationPayload::OperatorStage(stage) if stage.stage_kind == RuntimeOperatorStageKind::MlNative))
        );
    }

    #[test]
    fn tsvd_matches_rank_one_host_oracle_and_preserves_row_identity() {
        let algorithm = MlAlgorithm::Tsvd;
        for shape in [Shape::Wide, Shape::Fixed] {
            let (result, metrics, observation) = execute(algorithm, shape, 8, 1);
            let QueryOutput::HostArrow(output) = result.expect("cuML fit_transform") else { panic!("host output") };
            let ids = output.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            let ordinals = output.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            let scores = output.column(2).as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(ordinals.values().as_ref(), &(0..8_u64).collect::<Vec<_>>());
            assert_eq!(ids.values().as_ref(), &[0, 0, 2, 3, 4, 5, 6, 7]);
            for i in 0..8 {
                let x = i as f32 - 3.0;
                let expected = x.abs() * 6.0_f32.sqrt();
                assert!(
                    (scores.value(i).abs() - expected).abs() < 1e-2,
                    "{algorithm:?} {i}: {} vs {expected}",
                    scores.value(i)
                );
            }
            assert!(
                metrics
                    .plan_task_mappings()
                    .iter()
                    .any(|mapping| mapping.operator_kind.as_deref() == Some("ml_algorithm"))
            );
            let observation = observation.expect("ML observation sealed");
            {
                let stages = observation
                    .events()
                    .iter()
                    .filter_map(|event| match &event.payload {
                        RuntimeObservationPayload::OperatorStage(stage)
                            if matches!(stage.lifecycle, RuntimeOperatorStageState::Finished)
                                && matches!(
                                    stage.stage_kind,
                                    RuntimeOperatorStageKind::MlPack
                                        | RuntimeOperatorStageKind::MlNative
                                        | RuntimeOperatorStageKind::MlAssemble
                                ) =>
                        {
                            Some(stage.as_ref())
                        },
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    stages.iter().map(|stage| stage.stage_kind).collect::<Vec<_>>(),
                    vec![
                        RuntimeOperatorStageKind::MlPack,
                        RuntimeOperatorStageKind::MlNative,
                        RuntimeOperatorStageKind::MlAssemble,
                    ]
                );
                for stage in stages {
                    assert_eq!(stage.ml_algorithm.as_deref(), Some(algorithm.as_str()));
                    assert_eq!(
                        (stage.native_version_major, stage.native_version_minor, stage.native_version_patch),
                        (Some(26), Some(8), Some(0))
                    );
                    assert_eq!(stage.input_rows, Some(8));
                    assert_eq!(stage.vector_dimension, Some(3));
                    assert_eq!(stage.matrix_layout.as_deref(), Some("column_major"));
                    if stage.stage_kind == RuntimeOperatorStageKind::MlNative {
                        assert_eq!(stage.strategy_label.as_deref(), Some("cov_eig_dq"));
                    }
                }
            }
        }
    }

    #[test]
    fn two_components_split_into_distinct_device_columns() {
        let algorithm = MlAlgorithm::Tsvd;
        let (result, _, _) = execute(algorithm, Shape::Wide, 8, 2);
        let QueryOutput::HostArrow(output) = result.expect("two-component transform") else { panic!("host output") };
        assert_eq!(output.schema().field(2).name(), "pc_0");
        assert_eq!(output.schema().field(3).name(), "pc_1");
        let first = output.column(2).as_any().downcast_ref::<Float32Array>().unwrap();
        let second = output.column(3).as_any().downcast_ref::<Float32Array>().unwrap();
        assert!(first.values().iter().any(|value| value.abs() > 1.0));
        assert!(second.values().iter().all(|value| value.abs() < 1e-2), "{algorithm:?}: {second:?}");
    }

    #[test]
    fn runtime_row_count_and_uniform_list_dimension_errors_are_structured() {
        let (result, _, _) = execute(MlAlgorithm::Tsvd, Shape::Wide, 1, 1);
        assert_eq!(result.err().expect("single row").fact_value("reason"), Some("insufficient_input_rows"));
        let (result, _, _) = execute(MlAlgorithm::Tsvd, Shape::List, 8, 4);
        assert_eq!(
            result.err().expect("runtime component bound").fact_value("reason"),
            Some("component_count_exceeds_feature_dimension")
        );
    }

    #[test]
    fn empty_ml_input_returns_exact_schema_without_native_stage() {
        for algorithm in [MlAlgorithm::Tsvd, MlAlgorithm::Hdbscan, MlAlgorithm::Umap] {
            let expected = match algorithm {
                MlAlgorithm::Tsvd => vector_pca_output_schema(DataType::Int64, 2),
                MlAlgorithm::Hdbscan => ml_hdbscan_output_schema(DataType::Int64),
                MlAlgorithm::Umap => ml_embedding_output_schema(DataType::Int64, 2),
                MlAlgorithm::LogisticRegression
                | MlAlgorithm::RandomForestClassifier
                | MlAlgorithm::RandomForestRegressor
                | MlAlgorithm::ForestPredict => unreachable!("unsupervised empty result"),
            };
            let (result, _, observation) = execute(algorithm, Shape::Wide, 0, 2);
            let QueryOutput::HostArrow(output) = result.expect("empty ML result") else { panic!("host output") };
            assert_eq!(output.num_rows(), 0);
            assert_eq!(output.schema().as_ref(), expected.as_ref());
            assert!(
                !observation
                    .unwrap()
                    .events()
                    .iter()
                    .any(|event| matches!(&event.payload, RuntimeObservationPayload::OperatorStage(stage)
                    if stage.stage_kind == RuntimeOperatorStageKind::MlNative))
            );
        }
    }

    #[test]
    fn row_dependent_hdbscan_and_umap_rejections_are_structured() {
        for (algorithm, rows, reason) in [
            (MlAlgorithm::Hdbscan, 2, "min_samples_exceeds_input_rows"),
            (MlAlgorithm::Umap, 3, "n_neighbors_exceeds_input_rows"),
        ] {
            let (result, _, _) = execute(algorithm, Shape::Wide, rows, 2);
            assert_eq!(result.err().expect("row-bound rejection").fact_value("reason"), Some(reason));
        }
    }

    #[test]
    fn hdbscan_finds_three_blobs_and_a_noise_row() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for index in 0..30 {
            let center = (index / 10) as f32 * 50.0;
            let offset = index % 10;
            a.push(center + (offset % 3) as f32 * 0.03);
            b.push(center + (offset / 3) as f32 * 0.03);
            c.push(center + (offset % 2) as f32 * 0.02);
        }
        a.push(500.0);
        b.push(-500.0);
        c.push(900.0);
        let batch = RecordBatch::try_new(
            input_schema(Shape::Wide),
            vec![
                Arc::new(Int64Array::from_iter_values(100..131)) as ArrayRef,
                Arc::new(Float32Array::from(a)),
                Arc::new(Float32Array::from(b)),
                Arc::new(Float32Array::from(c)),
            ],
        )
        .unwrap();
        let (result, _, observation) = execute_batch(MlAlgorithm::Hdbscan, Shape::Wide, 2, batch);
        let QueryOutput::HostArrow(output) = result.expect("HDBSCAN fit") else { panic!("host output") };
        let ordinals = output.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let ids = output.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let labels = output.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let probabilities = output.column(3).as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(output.num_rows(), 31);
        assert_eq!(ordinals.values().as_ref(), &(0..31_u64).collect::<Vec<_>>());
        assert_eq!(ids.values().as_ref(), &(100..131_i64).collect::<Vec<_>>());
        assert_eq!(labels.value(30), -1);
        let clusters =
            (0..30).map(|i| labels.value(i)).filter(|id| *id >= 0).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(clusters.len(), 3);
        assert!(probabilities.values().iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
        let stages = observation
            .unwrap()
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage)
                    if matches!(stage.lifecycle, RuntimeOperatorStageState::Finished)
                        && stage.stage_kind == RuntimeOperatorStageKind::MlNative =>
                {
                    Some((
                        stage.ml_algorithm.clone(),
                        stage.matrix_layout.clone(),
                        stage.strategy_label.clone(),
                        stage.output_rows,
                    ))
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            stages,
            [(Some("hdbscan".to_string()), Some("row_major".to_string()), Some("eom".to_string()), Some(31))]
        );
    }

    #[test]
    fn umap_embedding_is_finite_and_observation_is_row_major() {
        let (result, _, observation) = execute(MlAlgorithm::Umap, Shape::Fixed, 12, 2);
        let QueryOutput::HostArrow(output) = result.expect("UMAP embedding") else { panic!("host output") };
        assert_eq!(output.num_rows(), 12);
        assert_eq!(
            output.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
            ["row_ordinal", "id", "dim_0", "dim_1"]
        );
        for index in 2..4 {
            let column = output.column(index).as_any().downcast_ref::<Float32Array>().unwrap();
            assert!(column.values().iter().all(|value| value.is_finite()));
        }
        let stages = observation
            .unwrap()
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage)
                    if matches!(stage.lifecycle, RuntimeOperatorStageState::Finished)
                        && matches!(
                            stage.stage_kind,
                            RuntimeOperatorStageKind::MlPack
                                | RuntimeOperatorStageKind::MlNative
                                | RuntimeOperatorStageKind::MlAssemble
                        ) =>
                {
                    Some((
                        stage.stage_kind,
                        stage.matrix_layout.clone(),
                        stage.strategy_label.clone(),
                        stage.output_rows,
                    ))
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[1].0, RuntimeOperatorStageKind::MlNative);
        assert_eq!(stages[1].1.as_deref(), Some("row_major"));
        assert_eq!(stages[1].2.as_deref(), Some("random"));
        assert_eq!(stages[1].3, Some(12));
    }

    #[test]
    fn random_forest_training_allocation_failure_records_task_and_releases_attempt() {
        let training_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x0", DataType::Float32, false),
            Field::new("label", DataType::Int32, false),
        ]));
        let predict_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x0", DataType::Float32, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let training = builder
            .add_source(SourceNode::GpuResident(GpuResidentSource::new("training", Arc::clone(&training_schema))));
        let predict =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("predict", Arc::clone(&predict_schema))));
        let mut training_binding = MlRelationBinding::new(training, VectorRowIdBinding::column("id"), ["x0"]);
        training_binding.label = Some("label".to_owned());
        let predict_binding = MlRelationBinding::new(predict, VectorRowIdBinding::column("id"), ["x0"]);
        let mut params = MlRandomForestParams::new(training_binding, predict_binding);
        params.n_trees = 16;
        let root = builder.add_ml_algorithm(MlAlgorithmSpec::new(
            MlAlgorithm::RandomForestClassifier,
            MlAlgorithmParams::RandomForestClassifier(params),
            ml_predictions_output_schema(DataType::Int64, DataType::Int32),
        ));
        let plan = builder.finish_with_host_sink(root);
        let training_batch = RecordBatch::try_new(
            Arc::clone(&training_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..128)) as ArrayRef,
                Arc::new(Float32Array::from((0..128).map(|i| i as f32 - 64.0).collect::<Vec<_>>())),
                Arc::new(Int32Array::from((0..128).map(|i| i32::from(i >= 64)).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let predict_batch = RecordBatch::try_new(
            Arc::clone(&predict_schema),
            vec![Arc::new(Int64Array::from(vec![128])), Arc::new(Float32Array::from(vec![0.0]))],
        )
        .unwrap();
        let context = backend_context_with_grant(8 * 1024);
        let admitted = context.admit_plan(&plan).expect("RF plan admission");
        let runtime = Arc::new(admitted.runtime().unwrap());
        let mut inputs = GpuResidentInputMap::new();
        for (name, batch, schema) in
            [("training", training_batch, training_schema), ("predict", predict_batch, predict_schema)]
        {
            inputs.insert(
                name.into(),
                GpuResultHandle::new(
                    &admitted,
                    GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
                    schema,
                )
                .unwrap(),
            );
        }
        let admitted_plan =
            admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
                .unwrap();
        let (result, metrics, _) = execute_query_plan_direct(
            admitted.context.context(),
            QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&inputs),
        )
        .into_parts();
        let error = result.err().expect("RF training must exceed tiny grant");
        assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded, "{error:?}");
        assert_eq!(error.fact_value("failure_source"), Some("RESERVATION_LIMIT"), "{error:?}");
        assert!(
            metrics.plan_task_mappings().iter().any(|mapping| mapping.operator_kind.as_deref() == Some("ml_algorithm"))
        );
        assert_eq!(metrics.compiled_pipeline_attempts, 1);
        drop(inputs);
        drop(runtime);
        assert_attempt_domain_released(&admitted);
        admitted.finish();
    }

    #[test]
    fn tiny_query_grant_reports_allocation_failure_and_releases_attempt() {
        let schema = input_schema(Shape::Wide);
        let plan = plan(Arc::clone(&schema), MlAlgorithm::Tsvd, Shape::Wide, 2);
        let context = backend_context_with_grant(8 * 1024_u64);
        let admitted = context.admit_plan(&plan).expect("ML plan admission");
        let runtime = Arc::new(admitted.runtime().unwrap());
        let mut inputs = GpuResidentInputMap::new();
        inputs.insert(
            "input".into(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch(Shape::Wide, 256)).unwrap()),
                schema,
            )
            .unwrap(),
        );
        let admitted_plan =
            admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
                .unwrap();
        let (result, metrics, _) = execute_query_plan_direct(
            admitted.context.context(),
            QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&inputs),
        )
        .into_parts();
        let error = result.err().expect("device grant too small for ML fit");
        assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded, "{error:?}");
        assert_eq!(error.fact_value("failure_source"), Some("RESERVATION_LIMIT"), "{error:?}");
        assert!(
            metrics.plan_task_mappings().iter().any(|mapping| mapping.operator_kind.as_deref() == Some("ml_algorithm"))
        );
        drop(inputs);
        drop(runtime);
        assert_attempt_domain_released(&admitted);
        admitted.finish();
    }

    #[test]
    fn unbounded_ml_request_rejects_before_source_work() {
        let schema = input_schema(Shape::Wide);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/algeon-unbounded-ml-source-must-not-open.parquet",
            Arc::clone(&schema),
            vec!["id".into(), "x0".into(), "x1".into(), "x2".into()],
        )));
        let root = builder.add_ml_algorithm(MlAlgorithmSpec::new(
            MlAlgorithm::Tsvd,
            MlAlgorithmParams::Tsvd(MlTsvdParams::new(
                MlRelationBinding::new(source, VectorRowIdBinding::column("id"), ["x0", "x1", "x2"]),
                1,
            )),
            vector_pca_output_schema(DataType::Int64, 1),
        ));
        let plan = builder.finish_with_host_sink(root);
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::new(cudf::Runtime::new().unwrap())),
            memory_control_policy: MemoryControlPolicy::Unbounded,
            ..StandaloneConfig::default()
        })
        .unwrap();
        let admitted_plan =
            admit_query_plan(&plan, context.statistical_aggregate_execution_mode(), context.planning_policy()).unwrap();
        let (result, metrics, _) =
            execute_query_plan_direct(&context, QueryExecutionRequest::admitted(&admitted_plan)).into_parts();
        let error = result.err().expect("unbounded ML request");
        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("compiled_ml_pipeline_required"));
        assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
        assert_eq!(metrics.source_read_time, std::time::Duration::ZERO);
    }

    const FOREST_FIXTURES: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../components/cuml/algeon-cuml/tests/fixtures/forest");

    fn forest_fixture(name: &str) -> Vec<u8> {
        fs::read(format!("{FOREST_FIXTURES}/{name}")).unwrap()
    }

    fn forest_csv<T: FromStr<Err: Debug>>(name: &str) -> Vec<Vec<T>> {
        String::from_utf8(forest_fixture(name))
            .unwrap()
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| line.split(',').map(|value| value.parse().unwrap()).collect())
            .collect()
    }

    fn forest_snapshot(file: &str, format: MlForestModelFormat, output: ForestOutput) -> MlForestModelSnapshot {
        let bytes = forest_fixture(file);
        let native_format = match format {
            MlForestModelFormat::XgboostJson => ForestModelFormat::XgboostJson,
            MlForestModelFormat::XgboostUbjson => ForestModelFormat::XgboostUbjson,
            MlForestModelFormat::XgboostLegacy => ForestModelFormat::XgboostLegacy,
            MlForestModelFormat::LightGbm => ForestModelFormat::LightGbm,
            MlForestModelFormat::TreeliteCheckpoint => ForestModelFormat::TreeliteCheckpoint,
        };
        let (info, checkpoint) = inspect_forest_model(&bytes, native_format).unwrap();
        let contract = match info.output_kind(output).unwrap() {
            ForestOutputKind::Score => MlForestOutputContract::Score,
            ForestOutputKind::ClassIndex { classes } => MlForestOutputContract::ClassIndex { classes },
            ForestOutputKind::Vector(width) => MlForestOutputContract::Vector(width),
        };
        let facts = MlForestModelFacts {
            num_features: info.num_features,
            num_trees: info.num_trees,
            native_output_stride: info.native_output_stride,
            precision: match info.precision {
                ForestPrecision::Single => MlForestPrecision::Single,
                ForestPrecision::Double => MlForestPrecision::Double,
            },
        };
        MlForestModelSnapshot::new(&bytes, checkpoint, format, facts, contract).unwrap()
    }

    /// Scores `rows` with `model` and asserts the attempt domain holds no
    /// allocation once the inputs are dropped, whatever the outcome.
    fn execute_forest(
        model: MlForestModelSnapshot,
        rows: &[Vec<f32>],
    ) -> (Result<QueryOutput, QueryError>, LiveRuntimeObservationSnapshot) {
        let features = model.num_features();
        let schema = Arc::new(Schema::new(
            iter::once(Field::new("id", DataType::Int64, false))
                .chain((0..features).map(|index| Field::new(format!("x{index}"), DataType::Float32, false)))
                .collect::<Vec<_>>(),
        ));
        let columns = iter::once(Arc::new(Int64Array::from_iter_values(100..100 + rows.len() as i64)) as ArrayRef)
            .chain((0..features).map(|index| {
                Arc::new(Float32Array::from(rows.iter().map(|row| row[index]).collect::<Vec<_>>())) as ArrayRef
            }))
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        let output_schema = ml_forest_output_schema(DataType::Int64, model.output());
        let binding =
            MlRelationBinding::new(input, VectorRowIdBinding::column("id"), (0..features).map(|i| format!("x{i}")));
        let root = builder.add_ml_algorithm(MlAlgorithmSpec::new(
            MlAlgorithm::ForestPredict,
            MlAlgorithmParams::ForestPredict(MlForestPredictParams { input: binding, model }),
            output_schema,
        ));
        let plan = builder.finish_with_host_sink(root);
        let context = backend_context();
        let attempt =
            context.begin_native_attempt("ml-forest", RuntimeObservationConfig::bounded(4096, 16 * 1024 * 1024));
        let binding = attempt.admission_binding();
        let evidence = capability_evidence_for_fixture_plan(&plan).unwrap();
        let admitted = context.admit_evidence_with_observation(evidence, Some(binding.clone())).unwrap();
        let runtime = Arc::new(admitted.runtime().unwrap());
        let mut inputs = GpuResidentInputMap::new();
        inputs.insert(
            "input".into(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
                schema,
            )
            .unwrap(),
        );
        let admitted_plan =
            admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
                .unwrap();
        let (result, _, _) = execute_query_plan_direct(
            admitted.context.context(),
            QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&inputs),
        )
        .into_parts();
        let observation = binding.observation_handle().live_snapshot().unwrap();
        drop(inputs);
        drop(runtime);
        assert_attempt_domain_released(&admitted);
        admitted.finish();
        (result, observation)
    }

    /// Finished ML stages as (kind, native strategy label, chunk index).
    fn forest_stages(
        observation: &LiveRuntimeObservationSnapshot,
    ) -> Vec<(RuntimeOperatorStageKind, Option<String>, Option<usize>)> {
        observation
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage)
                    if stage.lifecycle == RuntimeOperatorStageState::Finished
                        && matches!(
                            stage.stage_kind,
                            RuntimeOperatorStageKind::MlPack
                                | RuntimeOperatorStageKind::MlNative
                                | RuntimeOperatorStageKind::MlAssemble
                        ) =>
                {
                    let label = (stage.stage_kind == RuntimeOperatorStageKind::MlNative)
                        .then(|| stage.strategy_label.clone())
                        .flatten();
                    Some((stage.stage_kind, label, stage.chunk_index))
                },
                _ => None,
            })
            .collect()
    }

    fn host_output(result: Result<QueryOutput, QueryError>) -> RecordBatch {
        let QueryOutput::HostArrow(output) = result.expect("forest prediction") else { panic!("host output") };
        output
    }

    fn float_column(output: &RecordBatch, name: &str) -> Vec<f32> {
        output.column_by_name(name).unwrap().as_any().downcast_ref::<Float32Array>().unwrap().values().to_vec()
    }

    fn assert_close(actual: &[f32], expected: impl IntoIterator<Item = f64>, context: &str) {
        let expected = expected.into_iter().collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len(), "{context}");
        for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((f64::from(*actual) - expected).abs() <= 1e-5, "{context} row {row}: {actual} vs {expected}");
        }
    }

    #[test]
    fn forest_predict_scores_fixture_models_after_one_import_stage() {
        let snapshot =
            forest_snapshot("xgb_binary_logistic.json", MlForestModelFormat::XgboostJson, ForestOutput::Scores);
        let digest = snapshot.digest().short_hex();
        let rows = forest_csv::<f32>("xgb_binary_logistic.input.csv");
        let (result, observation) = execute_forest(snapshot, &rows);
        let output = host_output(result);
        assert_eq!(
            output.schema().as_ref(),
            ml_forest_output_schema(DataType::Int64, MlForestOutputContract::Score).as_ref()
        );
        let ids = output.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ids.values().as_ref(), &(100..116).collect::<Vec<_>>());
        let expected = forest_csv::<f64>("xgb_binary_logistic.gtil.csv");
        assert_close(&float_column(&output, "prediction"), expected.iter().map(|row| row[0]), "binary logistic");
        assert_eq!(
            forest_stages(&observation),
            [
                (RuntimeOperatorStageKind::MlNative, Some(format!("forest_import:{digest}:single")), None),
                (RuntimeOperatorStageKind::MlPack, None, Some(0)),
                (RuntimeOperatorStageKind::MlNative, Some(format!("forest_predict:{digest}")), Some(0)),
                (RuntimeOperatorStageKind::MlAssemble, None, None),
            ]
        );
        let version = forest_inference_version().unwrap();
        assert!(observation.events().iter().all(|event| match &event.payload {
            RuntimeObservationPayload::OperatorStage(stage)
                if stage.stage_kind == RuntimeOperatorStageKind::MlNative =>
            {
                stage.ml_algorithm.as_deref() == Some("forest_predict")
                    && (stage.native_version_major, stage.native_version_minor, stage.native_version_patch)
                        == (Some(version.0), Some(version.1), Some(version.2))
            },
            _ => true,
        }));

        let rows = forest_csv::<f32>("xgb_multiclass_softprob.input.csv");
        let expected = forest_csv::<f64>("xgb_multiclass_softprob.gtil.csv");
        for (file, format) in [
            ("xgb_multiclass_softprob.json", MlForestModelFormat::XgboostJson),
            ("xgb_multiclass_softprob_ubj.ubj", MlForestModelFormat::XgboostUbjson),
        ] {
            let (result, _) = execute_forest(forest_snapshot(file, format, ForestOutput::Scores), &rows);
            let output = host_output(result);
            assert_eq!(
                output.schema().as_ref(),
                ml_forest_output_schema(DataType::Int64, MlForestOutputContract::Vector(3)).as_ref()
            );
            for class in 0..3 {
                assert_close(
                    &float_column(&output, &format!("output_{class}")),
                    expected.iter().map(|row| row[class]),
                    &format!("{file} output_{class}"),
                );
            }
        }

        for name in ["xgb_multiclass_softmax", "xgb_multiclass_tie"] {
            let snapshot =
                forest_snapshot(&format!("{name}.json"), MlForestModelFormat::XgboostJson, ForestOutput::ClassIndex);
            let (result, _) = execute_forest(snapshot, &forest_csv::<f32>(&format!("{name}.input.csv")));
            let output = host_output(result);
            let predictions =
                output.column_by_name("prediction").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
            let expected = forest_csv::<f64>(&format!("{name}.expected.csv"));
            assert_eq!(
                predictions.values().as_ref(),
                expected.iter().map(|row| row[0] as i32).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    #[test]
    fn double_forest_widens_features_and_matches_the_reference() {
        let snapshot =
            forest_snapshot("lgb_regression_double.txt", MlForestModelFormat::LightGbm, ForestOutput::Scores);
        assert_eq!(snapshot.precision(), MlForestPrecision::Double);
        let digest = snapshot.digest().short_hex();
        let (result, observation) = execute_forest(snapshot, &forest_csv::<f32>("lgb_regression_double.input.csv"));
        let expected = forest_csv::<f64>("lgb_regression_double.gtil.csv");
        assert_close(
            &float_column(&host_output(result), "prediction"),
            expected.iter().map(|row| row[0]),
            "double regression",
        );
        assert_eq!(forest_stages(&observation)[0].1, Some(format!("forest_import:{digest}:double")));

        // A Float32 copy of this model's threshold equals 1.0 and flips the branch for input 1.0.
        let snapshot =
            forest_snapshot("lgb_threshold_boundary.txt", MlForestModelFormat::LightGbm, ForestOutput::Scores);
        let (result, _) = execute_forest(snapshot, &forest_csv::<f32>("lgb_threshold_boundary.input.csv"));
        let expected = forest_csv::<f64>("lgb_threshold_boundary.expected.csv");
        assert_close(
            &float_column(&host_output(result), "prediction"),
            expected.iter().map(|row| row[0]),
            "threshold boundary",
        );
    }

    #[test]
    fn forest_batches_share_one_import_and_reproduce_unbatched_output() {
        for (file, limit, columns, batches) in [
            // 4 features and stride 1: 20 elements hold 5 rows, so 16 rows take 4 batches.
            ("xgb_binary_logistic", 20, vec!["prediction".to_owned()], 4),
            // 4 features and stride 3: 12 elements hold 3 rows, so 16 rows take 6 batches.
            ("xgb_multiclass_softprob", 12, (0..3).map(|class| format!("output_{class}")).collect(), 6),
        ] {
            let snapshot =
                forest_snapshot(&format!("{file}.json"), MlForestModelFormat::XgboostJson, ForestOutput::Scores);
            let rows = forest_csv::<f32>(&format!("{file}.input.csv"));
            let (whole, _) = execute_forest(snapshot.clone(), &rows);
            set_forest_batch_element_limit_for_tests(Some(limit));
            let (batched, observation) = execute_forest(snapshot, &rows);
            set_forest_batch_element_limit_for_tests(None);
            let (whole, batched) = (host_output(whole), host_output(batched));
            assert_eq!(batched.schema(), whole.schema());
            for column in &columns {
                assert_eq!(float_column(&batched, column), float_column(&whole, column), "{file} {column}");
            }
            let stages = forest_stages(&observation);
            assert_eq!(stages.len(), 2 + 2 * batches, "{file}: {stages:?}");
            assert!(stages[0].1.as_deref().is_some_and(|label| label.starts_with("forest_import:")));
            for batch in 0..batches {
                assert_eq!(stages[1 + 2 * batch].0, RuntimeOperatorStageKind::MlPack);
                assert_eq!(stages[1 + 2 * batch].2, Some(batch));
                assert_eq!(stages[2 + 2 * batch].0, RuntimeOperatorStageKind::MlNative);
                assert_eq!(stages[2 + 2 * batch].2, Some(batch));
            }
            assert_eq!(stages.last().unwrap().0, RuntimeOperatorStageKind::MlAssemble);
        }
    }

    fn cancel_execution(context: &ExecutionContext) {
        context.cancellation().cancel();
    }

    #[test]
    fn cancellation_after_forest_import_skips_prediction_and_releases_the_attempt() {
        let snapshot =
            forest_snapshot("xgb_binary_logistic.json", MlForestModelFormat::XgboostJson, ForestOutput::Scores);
        set_forest_after_import_hook_for_tests(Some(cancel_execution));
        let (result, observation) = execute_forest(snapshot, &forest_csv::<f32>("xgb_binary_logistic.input.csv"));
        set_forest_after_import_hook_for_tests(None);
        assert_eq!(result.err().expect("cancelled forest").code(), QueryErrorCode::Cancelled);
        let stages = forest_stages(&observation);
        assert_eq!(stages.len(), 1, "{stages:?}");
        assert!(stages[0].1.as_deref().is_some_and(|label| label.starts_with("forest_import:")));
        assert!(!observation.events().iter().any(|event| matches!(&event.payload,
            RuntimeObservationPayload::OperatorStage(stage) if stage.stage_kind == RuntimeOperatorStageKind::MlPack)));
    }

    #[test]
    fn imported_forest_facts_that_differ_from_the_snapshot_are_an_invariant_violation() {
        let planned =
            forest_snapshot("xgb_binary_logistic.json", MlForestModelFormat::XgboostJson, ForestOutput::Scores);
        let tampered = MlForestModelSnapshot::new(
            &forest_fixture("xgb_binary_logistic.json"),
            planned.checkpoint().to_vec(),
            planned.format(),
            MlForestModelFacts { num_trees: planned.facts().num_trees + 1, ..planned.facts() },
            planned.output(),
        )
        .unwrap();
        let (result, observation) = execute_forest(tampered, &forest_csv::<f32>("xgb_binary_logistic.input.csv"));
        let error = result.err().expect("snapshot mismatch");
        assert_eq!(error.code(), QueryErrorCode::InternalInvariantViolation);
        assert_eq!(error.fact_value("reason"), Some("forest_snapshot_mismatch"));
        assert!(forest_stages(&observation).is_empty());
    }
}

#[cfg(not(feature = "cuml"))]
#[test]
fn cuml_disabled_rejects_before_local_source_staging() {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use query_engine::{
        error::QueryErrorCode,
        exec::QueryExecutionRequest,
        plan::{
            MlAlgorithm, MlAlgorithmParams, MlAlgorithmSpec, MlRelationBinding, MlTsvdParams, QueryPlanBuilder,
            VectorRowIdBinding, vector_pca_output_schema,
        },
        source::{LocalParquetSource, SourceNode},
    };

    use super::{backend_context, execute_query_plan_direct};

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/algeon-cuml-disabled-source-must-not-open.parquet",
        Arc::clone(&schema),
        vec!["id".into(), "x".into(), "y".into()],
    )));
    let root = builder.add_ml_algorithm(MlAlgorithmSpec::new(
        MlAlgorithm::Tsvd,
        MlAlgorithmParams::Tsvd(MlTsvdParams::new(
            MlRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x", "y"]),
            1,
        )),
        vector_pca_output_schema(DataType::Int64, 1),
    ));
    let plan = builder.finish_with_host_sink(root);
    let context = backend_context();
    let admitted = context.admit_plan(&plan).expect("admission without cuML");
    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let (result, metrics, _) =
        execute_query_plan_direct(admitted.context.context(), QueryExecutionRequest::admitted(&admitted_plan))
            .into_parts();
    let error = result.err().expect("disabled cuML must reject before source staging");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("reason"), Some("required_feature_disabled"));
    assert_eq!(error.fact_value("required_feature"), Some("cuml"));
    assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
    assert_eq!(metrics.source_read_time, std::time::Duration::ZERO);
    admitted.finish();
}
