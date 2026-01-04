use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BenchmarkSourceLayout {
    SingleFileMultiRowGroup,
    MultiFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BenchmarkScalePoint {
    ThreeGiB,
    SixGiB,
    TenGiB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhaseCMatrixCase {
    pub(crate) scale: BenchmarkScalePoint,
    pub(crate) compression: BenchmarkCompression,
    pub(crate) source_layout: BenchmarkSourceLayout,
    pub(crate) native_plan_shape: BenchmarkPlanShape,
    pub(crate) expected_outcome: BenchmarkExpectedOutcome,
}

impl BenchmarkSourceLayout {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::SingleFileMultiRowGroup => "single_file_multi_row_group",
            Self::MultiFile => "multi_file",
        }
    }

    pub(crate) fn file_count(self) -> usize {
        match self {
            Self::SingleFileMultiRowGroup => 1,
            Self::MultiFile => 4,
        }
    }
}

impl BenchmarkScalePoint {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::ThreeGiB => "3gib",
            Self::SixGiB => "6gib",
            Self::TenGiB => "10gib",
        }
    }

    pub(crate) fn input_rows(self, compression: BenchmarkCompression) -> usize {
        match (self, compression) {
            (Self::ThreeGiB, BenchmarkCompression::Uncompressed) => 12_250_000,
            (Self::ThreeGiB, BenchmarkCompression::Snappy) => 16_000_000,
            (Self::SixGiB, BenchmarkCompression::Uncompressed) => 24_500_000,
            (Self::SixGiB, BenchmarkCompression::Snappy) => 32_000_000,
            (Self::TenGiB, BenchmarkCompression::Uncompressed) => 40_900_000,
            (Self::TenGiB, BenchmarkCompression::Snappy) => 52_000_000,
        }
    }

    pub(crate) fn minimum_file_bytes(self) -> u64 {
        match self {
            Self::ThreeGiB => THREE_GIB,
            Self::SixGiB => SIX_GIB,
            Self::TenGiB => TEN_GIB,
        }
    }
}

impl PhaseCMatrixCase {
    pub(crate) fn name(self) -> String {
        format!(
            "{}-{}-{}-{}-{}",
            self.scale.name(),
            self.compression.name(),
            self.source_layout.name(),
            self.native_plan_shape.name(),
            self.expected_outcome.name()
        )
    }

    #[cfg(test)]
    pub(crate) fn env_overrides(self) -> Vec<(&'static str, String)> {
        let mut overrides = vec![
            ("CUDF_PARQUET_BENCH_ROWS", self.scale.input_rows(self.compression).to_string()),
            ("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", self.scale.minimum_file_bytes().to_string()),
            ("CUDF_PARQUET_BENCH_COMPRESSION", self.compression.name().to_string()),
            ("CUDF_PARQUET_BENCH_FILES", self.source_layout.file_count().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE", self.native_plan_shape.name().to_string()),
            ("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME", self.expected_outcome.name().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY", "bounded".to_string()),
        ];
        if self.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure {
            overrides.extend([
                ("CUDF_PARQUET_BENCH_NATIVE_DEVICE_BUDGET_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_RETRY_ATTEMPTS", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS", "1".to_string()),
            ]);
        }
        overrides
    }

    pub(crate) fn configured_case(
        self,
        base: &LocalParquetBenchmarkConfig,
        output_dir: PathBuf,
    ) -> LocalParquetBenchmarkConfig {
        let mut config = base.clone();
        config.input_rows = self.scale.input_rows(self.compression);
        config.file_count = self.source_layout.file_count();
        config.minimum_file_bytes = self.scale.minimum_file_bytes();
        config.compression = self.compression;
        config.native_plan_shape = self.native_plan_shape;
        config.expected_outcome = self.expected_outcome;
        config.benchmark_paths = BenchmarkPathSet::NativeOnly;
        config.output_dir = Some(output_dir);
        if self.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure {
            config.native_memory_control_policy = forced_resource_failure_policy();
        }
        config
    }
}

pub(crate) fn phase_c_decision_matrix() -> Vec<PhaseCMatrixCase> {
    let mut cases = Vec::new();
    let compressions = [BenchmarkCompression::Uncompressed, BenchmarkCompression::Snappy];
    let layouts = [BenchmarkSourceLayout::SingleFileMultiRowGroup, BenchmarkSourceLayout::MultiFile];
    let plan_shapes = [BenchmarkPlanShape::DefaultSql, BenchmarkPlanShape::DirectFriendly];

    for scale in [BenchmarkScalePoint::ThreeGiB, BenchmarkScalePoint::SixGiB] {
        for compression in compressions {
            for source_layout in layouts {
                for native_plan_shape in plan_shapes {
                    cases.push(PhaseCMatrixCase {
                        scale,
                        compression,
                        source_layout,
                        native_plan_shape,
                        expected_outcome: BenchmarkExpectedOutcome::Success,
                    });
                }
            }
        }
    }

    for compression in compressions {
        for source_layout in layouts {
            for native_plan_shape in plan_shapes {
                cases.push(PhaseCMatrixCase {
                    scale: BenchmarkScalePoint::TenGiB,
                    compression,
                    source_layout,
                    native_plan_shape,
                    expected_outcome: BenchmarkExpectedOutcome::ResourceFailure,
                });
            }
        }
    }

    cases
}
