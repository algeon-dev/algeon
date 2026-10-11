mod gpu_exclusive {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use algeon_datafusion::planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig};
    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{ParquetReadOptions, SessionContext},
    };
    use query_runtime::runtime::GpuDeviceMemoryLimit;
    use workloads::tpcds::write_tiny_tpcds_parquet_dataset;

    use crate::gpu_backend::{build_report_gpu_backend, close_report_gpu_backend};

    #[tokio::test]
    async fn report_backend_preserves_native_required_rejection_of_partial_plans() {
        let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("algeon-report-required-{id}"));
        write_tiny_tpcds_parquet_dataset(&root).unwrap();
        let config = AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired);
        let backend =
            build_report_gpu_backend(config, GpuDeviceMemoryLimit::Explicit { limit_bytes: 512 * 1024 * 1024 })
                .unwrap();
        let result = {
            let state = backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build();
            let ctx = SessionContext::new_with_state(state);
            ctx.register_parquet(
                "customer",
                root.join("customer.parquet").to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await
            .unwrap();
            ctx.sql("SELECT c_customer_sk, repeat(c_first_name, 2) FROM customer")
                .await
                .unwrap()
                .create_physical_plan()
                .await
        };
        close_report_gpu_backend(backend).unwrap();
        fs::remove_dir_all(root).unwrap();
        assert!(result.is_err(), "a report requiring native execution must reject a CPU projection over a GPU scan");
    }
}
