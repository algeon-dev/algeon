// Generated from main-1a7c85a7f/tpch/*/q*.tsv, layout.json and page-cache.json.
// Do not hand-edit benchmark values; regenerate from the run directory.

export type TpchBenchmarkRow = {
  queryId: string;
  cpuMs: number;
  gpuMs: number;
  speedup: number;
  cpuPlanningMs: number;
  gpuPlanningMs: number;
  cpuExecutionMs: number;
  gpuExecutionMs: number;
  planningOutcome: string;
  runtimeOutcome: string;
  parityStatus: string;
  rowGroupsSelected: number;
  parquetFiles: number;
  peakMemoryOperatorClass: string;
  peakMemoryOperatorClassBytes: number;
  sourcePredicatePushdownStatus: string;
  partitionedJoinCount: number;
  joinPartitioningStatus: string;
  aggregateMergeStatus: string;
  parquetReadTimeMs: number | null;
  sourceReadTimeMs: number | null;
  operatorTimeMs: number | null;
  materializationTimeMs: number | null;
};

export const tpchBenchmark = {
  "environment": {
    "instanceType": "g7e.8xlarge",
    "cpu": "Intel Xeon Platinum 8559C",
    "cpuCores": 16,
    "cpuThreads": 32,
    "memoryGiB": 256,
    "memoryVisibleGiB": 249.7,
    "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "gpuMemoryGB": 96,
    "gpuDriver": "595.99.02",
    "cudaToolkit": "13.2",
    "storage": "EC2 instance-store NVMe",
    "parquetIo": "KvikIO compatibility mode with Algeon's default 64 I/O threads and 4 MiB tasks; GPUDirect Storage not used",
    "dataFusionVersion": "55.1.0",
    "cpuTargetPartitions": 32,
    "cpuBatchSize": 32768,
    "nativeProfile": "sf100_gpu_throughput",
    "executionMode": "NativeRequired",
    "cpuParquetPushdownFilters": false,
    "cpuParquetReorderFilters": false,
    "buildProfile": "release",
    "cpuTarget": "x86-64-v3",
    "cpuSettings": {
      "datafusion.execution.target_partitions": 32,
      "datafusion.execution.batch_size": 32768,
      "datafusion.optimizer.repartition_file_scans": true,
      "datafusion.optimizer.repartition_aggregations": true,
      "datafusion.optimizer.repartition_joins": true,
      "datafusion.optimizer.repartition_sorts": true,
      "datafusion.optimizer.enable_dynamic_filter_pushdown": true,
      "datafusion.optimizer.enable_join_dynamic_filter_pushdown": true,
      "datafusion.optimizer.enable_topk_dynamic_filter_pushdown": true,
      "datafusion.optimizer.enable_aggregate_dynamic_filter_pushdown": true,
      "datafusion.execution.parquet.pushdown_filters": false,
      "datafusion.execution.parquet.reorder_filters": false,
      "datafusion.execution.collect_statistics": true,
      "datafusion.execution.parquet.pruning": true,
      "datafusion.execution.parquet.enable_page_index": true
    },
    "sourceRevision": "1a7c85a7f2d77ad7eb63e035ca7e621b853b38c2",
    "cudfRevision": "576ceb28a90b7c0238108f6780bf20ee6de83f8d",
    "treeHash": "48e6cfdc5e51d1b95df21051bbb7a26759846286",
    "measuredAtUtcDate": "2026-11-13",
    "worktreeChanges": false,
    "binarySha256": {
      "native_tpch_benchmark": "fd80dab7f86719d9e2771aef0406b70c18194923e3d7e6a76e72dc42b3132228",
      "tpcds_gpu_cpu_bench": "5acccc9ea919375d09217507490b85846ad32ea00d345f4c0331a7f7f021594b",
      "native_tpcds_parity_report": "6af73d3c6c6c46e257b21536fefed39f2d8ff6c29ca49fbffb330fbd9f8c3330",
      "algeon_server": "0e2883e4d2cad1998eae88c031be4f6d1d2d60386efd993939f2e45719fcd896",
      "algeon_tools": "04508104954ae5c89f52e5d7358571962fd83c819f201c87f27b2bdb1a059c04"
    },
    "nativeLibSha256": {
      "libcudf_rust.so": "be5cdc36ff61b07eac1ddb47ec2b93cd71e42f03caf12acf9580589bb38fbb99",
      "libcudf.so": "ef778779de7ca6738b2e988fa6de78a3c61cc25e9ec924a31fe2a0361a6d7451"
    }
  },
  "config": {
    "queryIds": [
      "q01",
      "q02",
      "q03",
      "q04",
      "q05",
      "q06",
      "q07",
      "q08",
      "q09",
      "q10",
      "q11",
      "q12",
      "q13",
      "q14",
      "q15",
      "q16",
      "q17",
      "q18",
      "q19",
      "q20",
      "q21",
      "q22"
    ],
    "warmups": 1,
    "iterations": 5,
    "wallTimeFields": "per_measured_iteration_median",
    "wallTimeBoundary": "sql_planning_and_collect",
    "parquetPushdownFilters": "session_default",
    "relationalExecution": "native_no_datafusion_cpu",
    "nativeMemoryPolicy": "bounded",
    "nativeProfileName": "sf100_gpu_throughput",
    "nativeProfileSource": "env_profile",
    "nativeDeviceBudgetPolicy": "automatic",
    "nativeDeviceBudgetBytes": 101390548992,
    "nativeMaxSourceChunkBytes": 34359738368,
    "nativeMaxRowGroupsPerChunk": 1024,
    "nativeMaxRetryAttempts": 4,
    "nativeMinRetryChunkRowGroups": 1,
    "nativeSourceChunkPlanningMode": "parquet_throughput",
    "nativeMaxSourceReadsInFlight": 2,
    "nativeRuntimePolicy": "shared_per_benchmark_run",
    "parity": {
      "floatToleranceAbsolute": 1e-08,
      "floatToleranceRelative": 1e-08
    }
  },
  "datasets": [
    {
      "id": "sf100_sorted",
      "label": "SF100 sorted",
      "scaleFactor": "sf100",
      "generationCommand": "python3 fixture/tpc-h/generate.py --scale 100 --out /mnt/algeon-nvme/algeon/fixture/tpch/sf100-spark/parquet",
      "layout": {
        "generation": {
          "layout": "single_file_per_table_sorted_row_groups",
          "scale_factor": 100,
          "writer": "spark",
          "spark_version": "4.0.1",
          "parquet_java_version": "1.15.2",
          "parquet_block_size": 134217728,
          "parquet_page_size": 1048576,
          "parquet_page_row_count_limit": 20000,
          "compression": "SNAPPY",
          "parquet_sorting_columns_metadata": false,
          "tables": [
            {
              "name": "customer",
              "sort_keys": [
                "c_nationkey",
                "c_custkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "lineitem",
              "sort_keys": [
                "l_shipdate",
                "l_receiptdate",
                "l_commitdate",
                "l_orderkey",
                "l_linenumber"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "nation",
              "sort_keys": [
                "n_regionkey",
                "n_nationkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "orders",
              "sort_keys": [
                "o_orderdate",
                "o_orderkey",
                "o_custkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "part",
              "sort_keys": [
                "p_brand",
                "p_type",
                "p_size",
                "p_partkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "partsupp",
              "sort_keys": [
                "ps_partkey",
                "ps_suppkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "region",
              "sort_keys": [
                "r_regionkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "supplier",
              "sort_keys": [
                "s_nationkey",
                "s_suppkey"
              ],
              "field_renames": [],
              "field_casts": []
            }
          ]
        },
        "partitioning": {
          "kind": "single_file_per_table",
          "fileCount": 8,
          "hiveKeys": []
        },
        "tables": {
          "customer": {
            "file": "customer.parquet",
            "bytes": 1224902134,
            "rows": 15000000,
            "columns": 8,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 10,
            "rowGroupRows": {
              "min": 287225,
              "median": 1635125,
              "max": 1635125
            },
            "rowGroupUncompressedBytes": {
              "min": 46769380,
              "median": 266334081,
              "max": 266351855
            },
            "dataPages": 6453,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 20,
            "columnChunks": 80,
            "nonOverlappingColumns": [
              "c_nationkey"
            ]
          },
          "lineitem": {
            "file": "lineitem.parquet",
            "bytes": 22405075300,
            "rows": 600037902,
            "columns": 16,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 168,
            "rowGroupRows": {
              "min": 2151202,
              "median": 3570100,
              "max": 3590100
            },
            "rowGroupUncompressedBytes": {
              "min": 140761378,
              "median": 233977800,
              "max": 234811315
            },
            "dataPages": 481360,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 1848,
            "columnChunks": 2688,
            "nonOverlappingColumns": [
              "l_linestatus",
              "l_shipdate"
            ]
          },
          "nation": {
            "file": "nation.parquet",
            "bytes": 3141,
            "rows": 25,
            "columns": 4,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 25,
              "median": 25,
              "max": 25
            },
            "rowGroupUncompressedBytes": {
              "min": 2629,
              "median": 2629,
              "max": 2629
            },
            "dataPages": 4,
            "rowsPerPage": {
              "median": 25,
              "max": 25
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 1,
            "columnChunks": 4,
            "nonOverlappingColumns": []
          },
          "orders": {
            "file": "orders.parquet",
            "bytes": 6406156485,
            "rows": 150000000,
            "columns": 9,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 48,
            "rowGroupRows": {
              "min": 2925300,
              "median": 3130100,
              "max": 3130100
            },
            "rowGroupUncompressedBytes": {
              "min": 280535678,
              "median": 300157483,
              "max": 300312115
            },
            "dataPages": 68291,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 192,
            "columnChunks": 432,
            "nonOverlappingColumns": [
              "o_orderdate"
            ]
          },
          "part": {
            "file": "part.parquet",
            "bytes": 653513899,
            "rows": 20000000,
            "columns": 9,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 5,
            "rowGroupRows": {
              "min": 3639600,
              "median": 4090100,
              "max": 4090100
            },
            "rowGroupUncompressedBytes": {
              "min": 258601682,
              "median": 290580029,
              "max": 290612152
            },
            "dataPages": 9018,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 25,
            "columnChunks": 45,
            "nonOverlappingColumns": [
              "p_mfgr",
              "p_brand"
            ]
          },
          "partsupp": {
            "file": "partsupp.parquet",
            "bytes": 4386215383,
            "rows": 80000000,
            "columns": 5,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 33,
            "rowGroupRows": {
              "min": 1316800,
              "median": 2460100,
              "max": 2460100
            },
            "rowGroupUncompressedBytes": {
              "min": 199760670,
              "median": 374993744,
              "max": 375119370
            },
            "dataPages": 26528,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 66,
            "columnChunks": 165,
            "nonOverlappingColumns": [
              "ps_partkey"
            ]
          },
          "region": {
            "file": "region.parquet",
            "bytes": 1609,
            "rows": 5,
            "columns": 3,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 5,
              "median": 5,
              "max": 5
            },
            "rowGroupUncompressedBytes": {
              "min": 533,
              "median": 533,
              "max": 533
            },
            "dataPages": 3,
            "rowsPerPage": {
              "median": 5,
              "max": 5
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 0,
            "columnChunks": 3,
            "nonOverlappingColumns": []
          },
          "supplier": {
            "file": "supplier.parquet",
            "bytes": 78247418,
            "rows": 1000000,
            "columns": 7,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 1000000,
              "median": 1000000,
              "max": 1000000
            },
            "rowGroupUncompressedBytes": {
              "min": 152506871,
              "median": 152506871,
              "max": 152506871
            },
            "dataPages": 369,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 1,
            "columnChunks": 7,
            "nonOverlappingColumns": []
          }
        }
      },
      "pageCache": {
        "totalBytes": 35154115369,
        "residentFractionBefore": 0.6446,
        "residentFractionAfter": 0.6446
      },
      "summary": {
        "queryCount": 22,
        "cpuTotalMs": 50256.189,
        "gpuTotalMs": 9304.964,
        "speedup": 5.401,
        "cpuPlanningTotalMs": 97.458,
        "gpuPlanningTotalMs": 137.52,
        "cpuPlanningShare": 0.00194,
        "gpuPlanningShare": 0.01478,
        "nativeQueryCount": 22,
        "parityMatchedCount": 22,
        "allPlanningAccepted": true,
        "allRuntimeSucceeded": true,
        "allParityMatched": true,
        "maxPeakMemoryOperatorClassBytes": 45172982356,
        "parquetFilesSummedAcrossQueries": 395,
        "rowGroupsSelectedSummedAcrossQueries": 15085,
        "fastestSpeedupQuery": "Q16",
        "slowestSpeedupQuery": "Q19"
      },
      "rows": [
        {
          "queryId": "Q01",
          "cpuMs": 2113.72,
          "gpuMs": 669.811,
          "speedup": 3.156,
          "cpuPlanningMs": 3.307,
          "gpuPlanningMs": 4.818,
          "cpuExecutionMs": 2110.311,
          "gpuExecutionMs": 665.213,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 830,
          "parquetFiles": 5,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 10166718356,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "no_joins",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 357.357,
          "sourceReadTimeMs": 369.585,
          "operatorTimeMs": 270.125,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q02",
          "cpuMs": 956.971,
          "gpuMs": 123.945,
          "speedup": 7.721,
          "cpuPlanningMs": 7.145,
          "gpuPlanningMs": 9.02,
          "cpuExecutionMs": 950.044,
          "gpuExecutionMs": 114.925,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 215,
          "parquetFiles": 35,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 4928611233,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 82.714,
          "sourceReadTimeMs": 82.832,
          "operatorTimeMs": 81.226,
          "materializationTimeMs": 0.198
        },
        {
          "queryId": "Q03",
          "cpuMs": 1313.658,
          "gpuMs": 400.762,
          "speedup": 3.278,
          "cpuPlanningMs": 4.076,
          "gpuPlanningMs": 5.851,
          "cpuExecutionMs": 1309.582,
          "gpuExecutionMs": 394.97,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 625,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 30723032211,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 239.71,
          "sourceReadTimeMs": 239.865,
          "operatorTimeMs": 78.009,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q04",
          "cpuMs": 738.395,
          "gpuMs": 269.708,
          "speedup": 2.738,
          "cpuPlanningMs": 2.911,
          "gpuPlanningMs": 4.247,
          "cpuExecutionMs": 735.441,
          "gpuExecutionMs": 265.547,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 855,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 19642131225,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 203.982,
          "sourceReadTimeMs": 203.99,
          "operatorTimeMs": 62.537,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q05",
          "cpuMs": 2435.445,
          "gpuMs": 731.931,
          "speedup": 3.327,
          "cpuPlanningMs": 5.519,
          "gpuPlanningMs": 7.843,
          "cpuExecutionMs": 2429.926,
          "gpuExecutionMs": 724.088,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 945,
          "parquetFiles": 30,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 45172982356,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 444.632,
          "sourceReadTimeMs": 447.78,
          "operatorTimeMs": 147.835,
          "materializationTimeMs": 0.045
        },
        {
          "queryId": "Q06",
          "cpuMs": 300.801,
          "gpuMs": 70.899,
          "speedup": 4.243,
          "cpuPlanningMs": 1.704,
          "gpuPlanningMs": 3.095,
          "cpuExecutionMs": 298.945,
          "gpuExecutionMs": 67.866,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 135,
          "parquetFiles": 5,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 6142202297,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "no_joins",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 72.823,
          "sourceReadTimeMs": 72.96,
          "operatorTimeMs": 10.775,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q07",
          "cpuMs": 2353.274,
          "gpuMs": 340.004,
          "speedup": 6.921,
          "cpuPlanningMs": 5.931,
          "gpuPlanningMs": 8.349,
          "cpuExecutionMs": 2347.308,
          "gpuExecutionMs": 331.828,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 560,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 22790351297,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 219.826,
          "sourceReadTimeMs": 219.964,
          "operatorTimeMs": 66.332,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q08",
          "cpuMs": 2397.904,
          "gpuMs": 662.234,
          "speedup": 3.621,
          "cpuPlanningMs": 7.315,
          "gpuPlanningMs": 10.62,
          "cpuExecutionMs": 2390.538,
          "gpuExecutionMs": 651.615,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1015,
          "parquetFiles": 40,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 35982925932,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 556.239,
          "sourceReadTimeMs": 556.57,
          "operatorTimeMs": 92.395,
          "materializationTimeMs": 0.045
        },
        {
          "queryId": "Q09",
          "cpuMs": 4728.565,
          "gpuMs": 1147.622,
          "speedup": 4.12,
          "cpuPlanningMs": 5.692,
          "gpuPlanningMs": 7.643,
          "cpuExecutionMs": 4723.118,
          "gpuExecutionMs": 1140.356,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1280,
          "parquetFiles": 30,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 41266454000,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 644.463,
          "sourceReadTimeMs": 645.183,
          "operatorTimeMs": 389.703,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q10",
          "cpuMs": 1897.742,
          "gpuMs": 461.904,
          "speedup": 4.109,
          "cpuPlanningMs": 5.516,
          "gpuPlanningMs": 7.842,
          "cpuExecutionMs": 1892.223,
          "gpuExecutionMs": 453.88,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 490,
          "parquetFiles": 20,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 26435308048,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 246.536,
          "sourceReadTimeMs": 246.747,
          "operatorTimeMs": 127.31,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q11",
          "cpuMs": 486.385,
          "gpuMs": 135.414,
          "speedup": 3.592,
          "cpuPlanningMs": 4.667,
          "gpuPlanningMs": 6.511,
          "cpuExecutionMs": 481.842,
          "gpuExecutionMs": 128.903,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 175,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 6770312902,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 55.325,
          "sourceReadTimeMs": 55.396,
          "operatorTimeMs": 7.34,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q12",
          "cpuMs": 797.78,
          "gpuMs": 147.39,
          "speedup": 5.413,
          "cpuPlanningMs": 3.349,
          "gpuPlanningMs": 4.745,
          "cpuExecutionMs": 794.227,
          "gpuExecutionMs": 142.666,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 385,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 5613036133,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 111.612,
          "sourceReadTimeMs": 111.621,
          "operatorTimeMs": 25.329,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q13",
          "cpuMs": 1721.55,
          "gpuMs": 324.139,
          "speedup": 5.311,
          "cpuPlanningMs": 2.518,
          "gpuPlanningMs": 3.667,
          "cpuExecutionMs": 1718.724,
          "gpuExecutionMs": 320.349,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 290,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 17572292214,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 270.292,
          "sourceReadTimeMs": 270.3,
          "operatorTimeMs": 168.601,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q14",
          "cpuMs": 354.976,
          "gpuMs": 63.041,
          "speedup": 5.631,
          "cpuPlanningMs": 3.241,
          "gpuPlanningMs": 4.322,
          "cpuExecutionMs": 351.543,
          "gpuExecutionMs": 58.783,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 40,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 2285770355,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 43.939,
          "sourceReadTimeMs": 44.024,
          "operatorTimeMs": 10.607,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q15",
          "cpuMs": 1211.652,
          "gpuMs": 77.122,
          "speedup": 15.711,
          "cpuPlanningMs": 4.634,
          "gpuPlanningMs": 6.319,
          "cpuExecutionMs": 1206.898,
          "gpuExecutionMs": 70.803,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 45,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 2431229190,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 49.424,
          "sourceReadTimeMs": 49.799,
          "operatorTimeMs": 15.827,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q16",
          "cpuMs": 2800.884,
          "gpuMs": 114.995,
          "speedup": 24.357,
          "cpuPlanningMs": 3.649,
          "gpuPlanningMs": 4.433,
          "cpuExecutionMs": 2797.235,
          "gpuExecutionMs": 110.754,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 195,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 3365458003,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 57.935,
          "sourceReadTimeMs": 57.945,
          "operatorTimeMs": 32.31,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q17",
          "cpuMs": 3980.662,
          "gpuMs": 964.64,
          "speedup": 4.127,
          "cpuPlanningMs": 3.67,
          "gpuPlanningMs": 5.474,
          "cpuExecutionMs": 3977.1,
          "gpuExecutionMs": 959.166,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1685,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 31750258429,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 550.413,
          "sourceReadTimeMs": 568.416,
          "operatorTimeMs": 418.216,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q18",
          "cpuMs": 10636.376,
          "gpuMs": 906.741,
          "speedup": 11.73,
          "cpuPlanningMs": 3.956,
          "gpuPlanningMs": 5.839,
          "cpuExecutionMs": 10632.42,
          "gpuExecutionMs": 900.847,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1970,
          "parquetFiles": 20,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 31380820809,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 532.579,
          "sourceReadTimeMs": 539.921,
          "operatorTimeMs": 319.817,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q19",
          "cpuMs": 1318.107,
          "gpuMs": 543.65,
          "speedup": 2.425,
          "cpuPlanningMs": 3.884,
          "gpuPlanningMs": 5.79,
          "cpuExecutionMs": 1312.666,
          "gpuExecutionMs": 537.914,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 855,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 19500285808,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 416.146,
          "sourceReadTimeMs": 418.133,
          "operatorTimeMs": 110.25,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q20",
          "cpuMs": 1407.328,
          "gpuMs": 257.25,
          "speedup": 5.471,
          "cpuPlanningMs": 4.904,
          "gpuPlanningMs": 6.695,
          "cpuExecutionMs": 1402.278,
          "gpuExecutionMs": 250.678,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 335,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 7322916270,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 149.375,
          "sourceReadTimeMs": 150.04,
          "operatorTimeMs": 68.473,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q21",
          "cpuMs": 5855.144,
          "gpuMs": 758.255,
          "speedup": 7.722,
          "cpuPlanningMs": 5.91,
          "gpuPlanningMs": 9.293,
          "cpuExecutionMs": 5849.235,
          "gpuExecutionMs": 749.088,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1820,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 29101877916,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 596.248,
          "sourceReadTimeMs": 596.267,
          "operatorTimeMs": 201.714,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q22",
          "cpuMs": 448.87,
          "gpuMs": 133.507,
          "speedup": 3.362,
          "cpuPlanningMs": 3.96,
          "gpuPlanningMs": 5.104,
          "cpuExecutionMs": 445.106,
          "gpuExecutionMs": 128.222,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 340,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 3626746074,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 74.158,
          "sourceReadTimeMs": 74.288,
          "operatorTimeMs": 50.39,
          "materializationTimeMs": 0.0
        }
      ]
    },
    {
      "id": "sf100_dbgen_order",
      "label": "SF100 dbgen order",
      "scaleFactor": "sf100",
      "generationCommand": "python3 fixture/tpc-h/generate.py --scale 100 --unsorted --out /mnt/algeon-nvme/algeon/fixture/tpch/sf100-unsorted/parquet",
      "layout": {
        "generation": {
          "layout": "single_file_per_table_unsorted",
          "scale_factor": 100,
          "writer": "spark",
          "spark_version": "4.0.1",
          "parquet_java_version": "1.15.2",
          "parquet_block_size": 134217728,
          "parquet_page_size": 1048576,
          "parquet_page_row_count_limit": 20000,
          "compression": "SNAPPY",
          "parquet_sorting_columns_metadata": false,
          "tables": [
            {
              "name": "customer",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "lineitem",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "nation",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "orders",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "part",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "partsupp",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "region",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "supplier",
              "sort_keys": [],
              "field_renames": [],
              "field_casts": []
            }
          ]
        },
        "partitioning": {
          "kind": "single_file_per_table",
          "fileCount": 8,
          "hiveKeys": []
        },
        "tables": {
          "customer": {
            "file": "customer.parquet",
            "bytes": 1238811107,
            "rows": 15000000,
            "columns": 8,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 10,
            "rowGroupRows": {
              "min": 369000,
              "median": 1628450,
              "max": 1628450
            },
            "rowGroupUncompressedBytes": {
              "min": 60334271,
              "median": 266253628,
              "max": 266298040
            },
            "dataPages": 6477,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 20,
            "columnChunks": 80,
            "nonOverlappingColumns": [
              "c_custkey",
              "c_name"
            ]
          },
          "lineitem": {
            "file": "lineitem.parquet",
            "bytes": 22664120577,
            "rows": 600037902,
            "columns": 16,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 170,
            "rowGroupRows": {
              "min": 71002,
              "median": 3550100,
              "max": 3550100
            },
            "rowGroupUncompressedBytes": {
              "min": 4714497,
              "median": 247168708,
              "max": 247215370
            },
            "dataPages": 481376,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 2040,
            "columnChunks": 2720,
            "nonOverlappingColumns": [
              "l_orderkey"
            ]
          },
          "nation": {
            "file": "nation.parquet",
            "bytes": 3126,
            "rows": 25,
            "columns": 4,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 25,
              "median": 25,
              "max": 25
            },
            "rowGroupUncompressedBytes": {
              "min": 2628,
              "median": 2628,
              "max": 2628
            },
            "dataPages": 4,
            "rowsPerPage": {
              "median": 25,
              "max": 25
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 1,
            "columnChunks": 4,
            "nonOverlappingColumns": []
          },
          "orders": {
            "file": "orders.parquet",
            "bytes": 6497082273,
            "rows": 150000000,
            "columns": 9,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 49,
            "rowGroupRows": {
              "min": 1675200,
              "median": 3090100,
              "max": 3090100
            },
            "rowGroupUncompressedBytes": {
              "min": 163582627,
              "median": 301754263,
              "max": 301821428
            },
            "dataPages": 68299,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 196,
            "columnChunks": 441,
            "nonOverlappingColumns": [
              "o_orderkey"
            ]
          },
          "part": {
            "file": "part.parquet",
            "bytes": 620190414,
            "rows": 20000000,
            "columns": 9,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 5,
            "rowGroupRows": {
              "min": 2749848,
              "median": 4310100,
              "max": 4319852
            },
            "rowGroupUncompressedBytes": {
              "min": 186347159,
              "median": 292170486,
              "max": 292852027
            },
            "dataPages": 9018,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 30,
            "columnChunks": 45,
            "nonOverlappingColumns": [
              "p_partkey"
            ]
          },
          "partsupp": {
            "file": "partsupp.parquet",
            "bytes": 4386194160,
            "rows": 80000000,
            "columns": 5,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 33,
            "rowGroupRows": {
              "min": 1326800,
              "median": 2460100,
              "max": 2460100
            },
            "rowGroupUncompressedBytes": {
              "min": 201290365,
              "median": 374981687,
              "max": 375119384
            },
            "dataPages": 26528,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 66,
            "columnChunks": 165,
            "nonOverlappingColumns": [
              "ps_partkey"
            ]
          },
          "region": {
            "file": "region.parquet",
            "bytes": 1609,
            "rows": 5,
            "columns": 3,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 5,
              "median": 5,
              "max": 5
            },
            "rowGroupUncompressedBytes": {
              "min": 533,
              "median": 533,
              "max": 533
            },
            "dataPages": 3,
            "rowsPerPage": {
              "median": 5,
              "max": 5
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 0,
            "columnChunks": 3,
            "nonOverlappingColumns": []
          },
          "supplier": {
            "file": "supplier.parquet",
            "bytes": 79161212,
            "rows": 1000000,
            "columns": 7,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 1000000,
              "median": 1000000,
              "max": 1000000
            },
            "rowGroupUncompressedBytes": {
              "min": 153133797,
              "median": 153133797,
              "max": 153133797
            },
            "dataPages": 369,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 1,
            "columnChunks": 7,
            "nonOverlappingColumns": []
          }
        }
      },
      "pageCache": {
        "totalBytes": 35485564478,
        "residentFractionBefore": 0.0,
        "residentFractionAfter": 0.6518
      },
      "summary": {
        "queryCount": 22,
        "cpuTotalMs": 51694.723,
        "gpuTotalMs": 12349.092,
        "speedup": 4.186,
        "cpuPlanningTotalMs": 101.984,
        "gpuPlanningTotalMs": 138.195,
        "cpuPlanningShare": 0.00197,
        "gpuPlanningShare": 0.01119,
        "nativeQueryCount": 22,
        "parityMatchedCount": 22,
        "allPlanningAccepted": true,
        "allRuntimeSucceeded": true,
        "allParityMatched": true,
        "maxPeakMemoryOperatorClassBytes": 45234555380,
        "parquetFilesSummedAcrossQueries": 395,
        "rowGroupsSelectedSummedAcrossQueries": 21540,
        "fastestSpeedupQuery": "Q16",
        "slowestSpeedupQuery": "Q10"
      },
      "rows": [
        {
          "queryId": "Q01",
          "cpuMs": 2205.055,
          "gpuMs": 895.512,
          "speedup": 2.462,
          "cpuPlanningMs": 3.358,
          "gpuPlanningMs": 4.716,
          "cpuExecutionMs": 2201.59,
          "gpuExecutionMs": 890.796,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 850,
          "parquetFiles": 5,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 14740751943,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "no_joins",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 403.702,
          "sourceReadTimeMs": 411.301,
          "operatorTimeMs": 485.691,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q02",
          "cpuMs": 944.245,
          "gpuMs": 128.249,
          "speedup": 7.363,
          "cpuPlanningMs": 7.041,
          "gpuPlanningMs": 9.086,
          "cpuExecutionMs": 937.219,
          "gpuExecutionMs": 119.163,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 215,
          "parquetFiles": 35,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 4928611649,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 84.71,
          "sourceReadTimeMs": 84.833,
          "operatorTimeMs": 83.416,
          "materializationTimeMs": 0.223
        },
        {
          "queryId": "Q03",
          "cpuMs": 1442.211,
          "gpuMs": 534.542,
          "speedup": 2.698,
          "cpuPlanningMs": 4.219,
          "gpuPlanningMs": 5.745,
          "cpuExecutionMs": 1437.914,
          "gpuExecutionMs": 528.72,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1145,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 27888562905,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 372.587,
          "sourceReadTimeMs": 375.116,
          "operatorTimeMs": 93.73,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q04",
          "cpuMs": 730.129,
          "gpuMs": 311.063,
          "speedup": 2.347,
          "cpuPlanningMs": 3.06,
          "gpuPlanningMs": 4.193,
          "cpuExecutionMs": 727.069,
          "gpuExecutionMs": 306.87,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1095,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 19642131717,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 241.535,
          "sourceReadTimeMs": 241.544,
          "operatorTimeMs": 66.675,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q05",
          "cpuMs": 2228.146,
          "gpuMs": 710.433,
          "speedup": 3.136,
          "cpuPlanningMs": 5.419,
          "gpuPlanningMs": 7.742,
          "cpuExecutionMs": 2222.59,
          "gpuExecutionMs": 702.492,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1160,
          "parquetFiles": 30,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 45234555380,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 447.302,
          "sourceReadTimeMs": 447.953,
          "operatorTimeMs": 122.989,
          "materializationTimeMs": 0.043
        },
        {
          "queryId": "Q06",
          "cpuMs": 743.943,
          "gpuMs": 242.501,
          "speedup": 3.068,
          "cpuPlanningMs": 2.072,
          "gpuPlanningMs": 3.08,
          "cpuExecutionMs": 741.906,
          "gpuExecutionMs": 239.421,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 850,
          "parquetFiles": 5,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 24535628640,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "no_joins",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 313.938,
          "sourceReadTimeMs": 324.725,
          "operatorTimeMs": 62.843,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q07",
          "cpuMs": 4016.276,
          "gpuMs": 610.874,
          "speedup": 6.575,
          "cpuPlanningMs": 6.153,
          "gpuPlanningMs": 8.35,
          "cpuExecutionMs": 4010.123,
          "gpuExecutionMs": 602.524,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1155,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 28816123963,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 465.481,
          "sourceReadTimeMs": 465.778,
          "operatorTimeMs": 91.609,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q08",
          "cpuMs": 2356.215,
          "gpuMs": 650.921,
          "speedup": 3.62,
          "cpuPlanningMs": 7.433,
          "gpuPlanningMs": 10.459,
          "cpuExecutionMs": 2348.782,
          "gpuExecutionMs": 640.374,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1190,
          "parquetFiles": 40,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 34031526822,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 544.607,
          "sourceReadTimeMs": 544.933,
          "operatorTimeMs": 91.753,
          "materializationTimeMs": 0.042
        },
        {
          "queryId": "Q09",
          "cpuMs": 4667.774,
          "gpuMs": 1122.099,
          "speedup": 4.16,
          "cpuPlanningMs": 5.441,
          "gpuPlanningMs": 7.542,
          "cpuExecutionMs": 4662.549,
          "gpuExecutionMs": 1114.557,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1295,
          "parquetFiles": 30,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 39990650613,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 592.008,
          "sourceReadTimeMs": 592.901,
          "operatorTimeMs": 421.194,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q10",
          "cpuMs": 1982.968,
          "gpuMs": 1574.131,
          "speedup": 1.26,
          "cpuPlanningMs": 5.341,
          "gpuPlanningMs": 8.0,
          "cpuExecutionMs": 1977.704,
          "gpuExecutionMs": 1566.131,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1150,
          "parquetFiles": 20,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 24084647427,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 406.046,
          "sourceReadTimeMs": 406.384,
          "operatorTimeMs": 580.525,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q11",
          "cpuMs": 599.367,
          "gpuMs": 135.36,
          "speedup": 4.428,
          "cpuPlanningMs": 4.953,
          "gpuPlanningMs": 6.508,
          "cpuExecutionMs": 594.375,
          "gpuExecutionMs": 128.852,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 175,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 6770312902,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 55.759,
          "sourceReadTimeMs": 55.835,
          "operatorTimeMs": 7.297,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q12",
          "cpuMs": 1214.312,
          "gpuMs": 485.503,
          "speedup": 2.501,
          "cpuPlanningMs": 3.423,
          "gpuPlanningMs": 4.708,
          "cpuExecutionMs": 1210.871,
          "gpuExecutionMs": 480.837,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1095,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 34363230487,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 380.631,
          "sourceReadTimeMs": 380.649,
          "operatorTimeMs": 51.669,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q13",
          "cpuMs": 1687.102,
          "gpuMs": 314.696,
          "speedup": 5.361,
          "cpuPlanningMs": 2.571,
          "gpuPlanningMs": 3.82,
          "cpuExecutionMs": 1684.384,
          "gpuExecutionMs": 311.13,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 295,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 17850078957,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 254.762,
          "sourceReadTimeMs": 254.771,
          "operatorTimeMs": 168.936,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q14",
          "cpuMs": 894.831,
          "gpuMs": 424.378,
          "speedup": 2.109,
          "cpuPlanningMs": 3.458,
          "gpuPlanningMs": 4.745,
          "cpuExecutionMs": 891.373,
          "gpuExecutionMs": 419.523,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 875,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 24452385427,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 348.157,
          "sourceReadTimeMs": 348.395,
          "operatorTimeMs": 47.922,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q15",
          "cpuMs": 1575.826,
          "gpuMs": 427.497,
          "speedup": 3.686,
          "cpuPlanningMs": 4.561,
          "gpuPlanningMs": 6.757,
          "cpuExecutionMs": 1571.694,
          "gpuExecutionMs": 420.739,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 855,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 10166955263,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 413.668,
          "sourceReadTimeMs": 420.398,
          "operatorTimeMs": 85.31,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q16",
          "cpuMs": 2829.461,
          "gpuMs": 115.449,
          "speedup": 24.508,
          "cpuPlanningMs": 3.587,
          "gpuPlanningMs": 4.518,
          "cpuExecutionMs": 2825.801,
          "gpuExecutionMs": 110.917,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 195,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 3365458003,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 57.553,
          "sourceReadTimeMs": 57.562,
          "operatorTimeMs": 31.551,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q17",
          "cpuMs": 3977.778,
          "gpuMs": 967.529,
          "speedup": 4.111,
          "cpuPlanningMs": 3.817,
          "gpuPlanningMs": 5.438,
          "cpuExecutionMs": 3973.944,
          "gpuExecutionMs": 962.172,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1725,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 31712464429,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 555.649,
          "sourceReadTimeMs": 575.572,
          "operatorTimeMs": 417.97,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q18",
          "cpuMs": 7504.478,
          "gpuMs": 682.728,
          "speedup": 10.992,
          "cpuPlanningMs": 3.862,
          "gpuPlanningMs": 5.853,
          "cpuExecutionMs": 7500.615,
          "gpuExecutionMs": 676.919,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1995,
          "parquetFiles": 20,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 31312356665,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 432.098,
          "sourceReadTimeMs": 433.417,
          "operatorTimeMs": 185.22,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q19",
          "cpuMs": 1380.783,
          "gpuMs": 544.324,
          "speedup": 2.537,
          "cpuPlanningMs": 7.735,
          "gpuPlanningMs": 5.757,
          "cpuExecutionMs": 1376.841,
          "gpuExecutionMs": 538.462,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 875,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 19572364926,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 418.609,
          "sourceReadTimeMs": 419.367,
          "operatorTimeMs": 112.774,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q20",
          "cpuMs": 1698.08,
          "gpuMs": 608.27,
          "speedup": 2.792,
          "cpuPlanningMs": 4.873,
          "gpuPlanningMs": 6.852,
          "cpuExecutionMs": 1693.299,
          "gpuExecutionMs": 601.418,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1050,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 13742645499,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 419.57,
          "sourceReadTimeMs": 419.876,
          "operatorTimeMs": 118.602,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q21",
          "cpuMs": 6570.258,
          "gpuMs": 727.035,
          "speedup": 9.037,
          "cpuPlanningMs": 5.717,
          "gpuPlanningMs": 9.197,
          "cpuExecutionMs": 6564.522,
          "gpuExecutionMs": 718.042,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1955,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 29101878552,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 551.199,
          "sourceReadTimeMs": 551.218,
          "operatorTimeMs": 215.674,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q22",
          "cpuMs": 445.485,
          "gpuMs": 135.998,
          "speedup": 3.276,
          "cpuPlanningMs": 3.89,
          "gpuPlanningMs": 5.129,
          "cpuExecutionMs": 441.679,
          "gpuExecutionMs": 130.869,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 345,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 3626630130,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 77.78,
          "sourceReadTimeMs": 77.887,
          "operatorTimeMs": 50.716,
          "materializationTimeMs": 0.0
        }
      ]
    },
    {
      "id": "sf500_sorted",
      "label": "SF500 sorted",
      "scaleFactor": "sf500",
      "generationCommand": "python3 fixture/tpc-h/generate.py --scale 500 --out /mnt/algeon-nvme/algeon/fixture/tpch/sf500-spark/parquet",
      "layout": {
        "generation": {
          "layout": "single_file_per_table_sorted_row_groups",
          "scale_factor": 500,
          "writer": "spark",
          "spark_version": "4.0.1",
          "parquet_java_version": "1.15.2",
          "parquet_block_size": 134217728,
          "parquet_page_size": 1048576,
          "parquet_page_row_count_limit": 20000,
          "compression": "SNAPPY",
          "parquet_sorting_columns_metadata": false,
          "tables": [
            {
              "name": "customer",
              "sort_keys": [
                "c_nationkey",
                "c_custkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "lineitem",
              "sort_keys": [
                "l_shipdate",
                "l_receiptdate",
                "l_commitdate",
                "l_orderkey",
                "l_linenumber"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "nation",
              "sort_keys": [
                "n_regionkey",
                "n_nationkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "orders",
              "sort_keys": [
                "o_orderdate",
                "o_orderkey",
                "o_custkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "part",
              "sort_keys": [
                "p_brand",
                "p_type",
                "p_size",
                "p_partkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "partsupp",
              "sort_keys": [
                "ps_partkey",
                "ps_suppkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "region",
              "sort_keys": [
                "r_regionkey"
              ],
              "field_renames": [],
              "field_casts": []
            },
            {
              "name": "supplier",
              "sort_keys": [
                "s_nationkey",
                "s_suppkey"
              ],
              "field_renames": [],
              "field_casts": []
            }
          ]
        },
        "partitioning": {
          "kind": "single_file_per_table",
          "fileCount": 8,
          "hiveKeys": []
        },
        "tables": {
          "customer": {
            "file": "customer.parquet",
            "bytes": 6124467492,
            "rows": 75000000,
            "columns": 8,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 46,
            "rowGroupRows": {
              "min": 1438650,
              "median": 1635125,
              "max": 1640100
            },
            "rowGroupUncompressedBytes": {
              "min": 234316513,
              "median": 266340524,
              "max": 267174065
            },
            "dataPages": 32246,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 92,
            "columnChunks": 368,
            "nonOverlappingColumns": [
              "c_nationkey"
            ]
          },
          "lineitem": {
            "file": "lineitem.parquet",
            "bytes": 114607955977,
            "rows": 3000028242,
            "columns": 16,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 858,
            "rowGroupRows": {
              "min": 332542,
              "median": 3510100,
              "max": 3510100
            },
            "rowGroupUncompressedBytes": {
              "min": 21751144,
              "median": 229178540,
              "max": 229788904
            },
            "dataPages": 2406816,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 9438,
            "columnChunks": 13728,
            "nonOverlappingColumns": [
              "l_linestatus",
              "l_shipdate"
            ]
          },
          "nation": {
            "file": "nation.parquet",
            "bytes": 3141,
            "rows": 25,
            "columns": 4,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 25,
              "median": 25,
              "max": 25
            },
            "rowGroupUncompressedBytes": {
              "min": 2629,
              "median": 2629,
              "max": 2629
            },
            "dataPages": 4,
            "rowsPerPage": {
              "median": 25,
              "max": 25
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 1,
            "columnChunks": 4,
            "nonOverlappingColumns": []
          },
          "orders": {
            "file": "orders.parquet",
            "bytes": 32842540438,
            "rows": 750000000,
            "columns": 9,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 246,
            "rowGroupRows": {
              "min": 468839,
              "median": 3060100,
              "max": 3060100
            },
            "rowGroupUncompressedBytes": {
              "min": 44959646,
              "median": 293443464,
              "max": 293557416
            },
            "dataPages": 342367,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 984,
            "columnChunks": 2214,
            "nonOverlappingColumns": [
              "o_orderdate"
            ]
          },
          "part": {
            "file": "part.parquet",
            "bytes": 3306746158,
            "rows": 100000000,
            "columns": 9,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 25,
            "rowGroupRows": {
              "min": 3277600,
              "median": 4030100,
              "max": 4030100
            },
            "rowGroupUncompressedBytes": {
              "min": 232788572,
              "median": 286241353,
              "max": 286258335
            },
            "dataPages": 45108,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 125,
            "columnChunks": 225,
            "nonOverlappingColumns": [
              "p_mfgr",
              "p_brand"
            ]
          },
          "partsupp": {
            "file": "partsupp.parquet",
            "bytes": 21929961206,
            "rows": 400000000,
            "columns": 5,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 163,
            "rowGroupRows": {
              "min": 1633800,
              "median": 2460100,
              "max": 2460100
            },
            "rowGroupUncompressedBytes": {
              "min": 248340413,
              "median": 374989206,
              "max": 375151700
            },
            "dataPages": 132648,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 326,
            "columnChunks": 815,
            "nonOverlappingColumns": [
              "ps_partkey"
            ]
          },
          "region": {
            "file": "region.parquet",
            "bytes": 1609,
            "rows": 5,
            "columns": 3,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 1,
            "rowGroupRows": {
              "min": 5,
              "median": 5,
              "max": 5
            },
            "rowGroupUncompressedBytes": {
              "min": 533,
              "median": 533,
              "max": 533
            },
            "dataPages": 3,
            "rowsPerPage": {
              "median": 5,
              "max": 5
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 0,
            "columnChunks": 3,
            "nonOverlappingColumns": []
          },
          "supplier": {
            "file": "supplier.parquet",
            "bytes": 391215855,
            "rows": 5000000,
            "columns": 7,
            "createdBy": "parquet-mr version 1.15.2 (build 859eac165b08f927fa14590c33bc5f476405fb68)",
            "rowGroups": 3,
            "rowGroupRows": {
              "min": 1579800,
              "median": 1710100,
              "max": 1710100
            },
            "rowGroupUncompressedBytes": {
              "min": 240955770,
              "median": 260774110,
              "max": 260814197
            },
            "dataPages": 1848,
            "rowsPerPage": {
              "median": 20000,
              "max": 20000
            },
            "pageSource": "offset_index",
            "codecs": [
              "SNAPPY"
            ],
            "dictionaryChunks": 3,
            "columnChunks": 21,
            "nonOverlappingColumns": [
              "s_nationkey"
            ]
          }
        }
      },
      "pageCache": {
        "totalBytes": 179202891876,
        "residentFractionBefore": 0.0,
        "residentFractionAfter": 0.52
      },
      "summary": {
        "queryCount": 22,
        "cpuTotalMs": 262235.631,
        "gpuTotalMs": 49282.937,
        "speedup": 5.321,
        "cpuPlanningTotalMs": 113.009,
        "gpuPlanningTotalMs": 184.013,
        "cpuPlanningShare": 0.00043,
        "gpuPlanningShare": 0.00373,
        "nativeQueryCount": 22,
        "parityMatchedCount": 22,
        "allPlanningAccepted": true,
        "allRuntimeSucceeded": true,
        "allParityMatched": true,
        "maxPeakMemoryOperatorClassBytes": 144022499424,
        "parquetFilesSummedAcrossQueries": 400,
        "rowGroupsSelectedSummedAcrossQueries": 80190,
        "fastestSpeedupQuery": "Q16",
        "slowestSpeedupQuery": "Q19"
      },
      "rows": [
        {
          "queryId": "Q01",
          "cpuMs": 10026.895,
          "gpuMs": 3546.977,
          "speedup": 2.827,
          "cpuPlanningMs": 3.997,
          "gpuPlanningMs": 7.246,
          "cpuExecutionMs": 10022.873,
          "gpuExecutionMs": 3539.554,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 4230,
          "parquetFiles": 5,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 11173924207,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "no_joins",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 1678.216,
          "sourceReadTimeMs": 1719.667,
          "operatorTimeMs": 1295.706,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q02",
          "cpuMs": 3594.396,
          "gpuMs": 347.083,
          "speedup": 10.356,
          "cpuPlanningMs": 7.298,
          "gpuPlanningMs": 9.658,
          "cpuExecutionMs": 3587.051,
          "gpuExecutionMs": 337.539,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 985,
          "parquetFiles": 35,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 19542821765,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 262.745,
          "sourceReadTimeMs": 262.861,
          "operatorTimeMs": 278.304,
          "materializationTimeMs": 0.397
        },
        {
          "queryId": "Q03",
          "cpuMs": 5739.345,
          "gpuMs": 1849.986,
          "speedup": 3.102,
          "cpuPlanningMs": 4.988,
          "gpuPlanningMs": 8.336,
          "cpuExecutionMs": 5734.545,
          "gpuExecutionMs": 1841.637,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 3140,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "partition",
          "peakMemoryOperatorClassBytes": 65094786488,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 1157.681,
          "sourceReadTimeMs": 1158.788,
          "operatorTimeMs": 393.537,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q04",
          "cpuMs": 4168.063,
          "gpuMs": 1568.188,
          "speedup": 2.658,
          "cpuPlanningMs": 3.867,
          "gpuPlanningMs": 6.196,
          "cpuExecutionMs": 4164.182,
          "gpuExecutionMs": 1561.992,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 4345,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 34415007043,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 1162.334,
          "sourceReadTimeMs": 1162.353,
          "operatorTimeMs": 411.76,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q05",
          "cpuMs": 13308.804,
          "gpuMs": 3578.61,
          "speedup": 3.719,
          "cpuPlanningMs": 6.341,
          "gpuPlanningMs": 9.946,
          "cpuExecutionMs": 13302.464,
          "gpuExecutionMs": 3568.664,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 4735,
          "parquetFiles": 30,
          "peakMemoryOperatorClass": "partition",
          "peakMemoryOperatorClassBytes": 144022499424,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 2158.781,
          "sourceReadTimeMs": 2162.258,
          "operatorTimeMs": 639.767,
          "materializationTimeMs": 0.049
        },
        {
          "queryId": "Q06",
          "cpuMs": 1364.634,
          "gpuMs": 308.318,
          "speedup": 4.426,
          "cpuPlanningMs": 2.699,
          "gpuPlanningMs": 4.976,
          "cpuExecutionMs": 1361.936,
          "gpuExecutionMs": 303.392,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 655,
          "parquetFiles": 5,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 11848990833,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "no_joins",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 423.538,
          "sourceReadTimeMs": 428.648,
          "operatorTimeMs": 75.582,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q07",
          "cpuMs": 10642.681,
          "gpuMs": 1585.169,
          "speedup": 6.714,
          "cpuPlanningMs": 6.967,
          "gpuPlanningMs": 10.666,
          "cpuExecutionMs": 10635.714,
          "gpuExecutionMs": 1574.503,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 2785,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 61002018167,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 1016.184,
          "sourceReadTimeMs": 1016.648,
          "operatorTimeMs": 299.803,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q08",
          "cpuMs": 11340.716,
          "gpuMs": 3252.26,
          "speedup": 3.487,
          "cpuPlanningMs": 8.578,
          "gpuPlanningMs": 12.945,
          "cpuExecutionMs": 11332.123,
          "gpuExecutionMs": 3239.384,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 5050,
          "parquetFiles": 40,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 37378335165,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 15,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 2727.665,
          "sourceReadTimeMs": 2729.056,
          "operatorTimeMs": 446.5,
          "materializationTimeMs": 0.047
        },
        {
          "queryId": "Q09",
          "cpuMs": 23025.227,
          "gpuMs": 5633.72,
          "speedup": 4.087,
          "cpuPlanningMs": 6.542,
          "gpuPlanningMs": 9.995,
          "cpuExecutionMs": 23018.754,
          "gpuExecutionMs": 5623.814,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 6480,
          "parquetFiles": 30,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 80406043916,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 15,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 3146.524,
          "sourceReadTimeMs": 3148.706,
          "operatorTimeMs": 1936.64,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q10",
          "cpuMs": 8329.236,
          "gpuMs": 2353.399,
          "speedup": 3.539,
          "cpuPlanningMs": 6.159,
          "gpuPlanningMs": 10.704,
          "cpuExecutionMs": 8322.903,
          "gpuExecutionMs": 2342.872,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 2435,
          "parquetFiles": 20,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 38909811740,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 10,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 1157.654,
          "sourceReadTimeMs": 1158.546,
          "operatorTimeMs": 667.097,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q11",
          "cpuMs": 2043.672,
          "gpuMs": 628.047,
          "speedup": 3.254,
          "cpuPlanningMs": 5.173,
          "gpuPlanningMs": 6.943,
          "cpuExecutionMs": 2038.527,
          "gpuExecutionMs": 620.93,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 835,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 33602344162,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 215.201,
          "sourceReadTimeMs": 215.254,
          "operatorTimeMs": 21.827,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q12",
          "cpuMs": 3347.275,
          "gpuMs": 832.017,
          "speedup": 4.023,
          "cpuPlanningMs": 4.214,
          "gpuPlanningMs": 7.775,
          "cpuExecutionMs": 3343.014,
          "gpuExecutionMs": 824.204,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1940,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 58753759069,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 575.733,
          "sourceReadTimeMs": 575.744,
          "operatorTimeMs": 115.356,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q13",
          "cpuMs": 9788.322,
          "gpuMs": 1719.506,
          "speedup": 5.693,
          "cpuPlanningMs": 2.915,
          "gpuPlanningMs": 4.272,
          "cpuExecutionMs": 9785.504,
          "gpuExecutionMs": 1715.244,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1460,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 20174973509,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 836.879,
          "sourceReadTimeMs": 836.904,
          "operatorTimeMs": 611.385,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q14",
          "cpuMs": 2085.566,
          "gpuMs": 267.927,
          "speedup": 7.784,
          "cpuPlanningMs": 4.036,
          "gpuPlanningMs": 6.521,
          "cpuExecutionMs": 2081.673,
          "gpuExecutionMs": 261.296,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 185,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 11654535167,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 166.626,
          "sourceReadTimeMs": 167.205,
          "operatorTimeMs": 52.005,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q15",
          "cpuMs": 6790.116,
          "gpuMs": 277.033,
          "speedup": 24.51,
          "cpuPlanningMs": 5.391,
          "gpuPlanningMs": 9.452,
          "cpuExecutionMs": 6784.687,
          "gpuExecutionMs": 267.582,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 185,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 10956460522,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 175.135,
          "sourceReadTimeMs": 178.372,
          "operatorTimeMs": 74.858,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q16",
          "cpuMs": 14823.902,
          "gpuMs": 398.027,
          "speedup": 37.243,
          "cpuPlanningMs": 3.699,
          "gpuPlanningMs": 4.997,
          "cpuExecutionMs": 14819.978,
          "gpuExecutionMs": 393.105,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 955,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 16726690345,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 181.345,
          "sourceReadTimeMs": 181.356,
          "operatorTimeMs": 133.276,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q17",
          "cpuMs": 22654.347,
          "gpuMs": 5239.372,
          "speedup": 4.324,
          "cpuPlanningMs": 4.419,
          "gpuPlanningMs": 7.78,
          "cpuExecutionMs": 22650.119,
          "gpuExecutionMs": 5231.55,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 8590,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 35946737269,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 3764.932,
          "sourceReadTimeMs": 3765.983,
          "operatorTimeMs": 2132.91,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q18",
          "cpuMs": 63232.827,
          "gpuMs": 5364.284,
          "speedup": 11.788,
          "cpuPlanningMs": 4.967,
          "gpuPlanningMs": 8.381,
          "cpuExecutionMs": 63228.132,
          "gpuExecutionMs": 5355.827,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 10040,
          "parquetFiles": 20,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 38268751464,
          "sourcePredicatePushdownStatus": "absent,pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 3822.357,
          "sourceReadTimeMs": 3832.919,
          "operatorTimeMs": 1848.371,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q19",
          "cpuMs": 5800.767,
          "gpuMs": 2653.501,
          "speedup": 2.186,
          "cpuPlanningMs": 4.641,
          "gpuPlanningMs": 8.674,
          "cpuExecutionMs": 5791.625,
          "gpuExecutionMs": 2644.907,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 4320,
          "parquetFiles": 10,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 22542699322,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "partitioned",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 2054.967,
          "sourceReadTimeMs": 2131.578,
          "operatorTimeMs": 469.409,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q20",
          "cpuMs": 6330.947,
          "gpuMs": 1390.491,
          "speedup": 4.553,
          "cpuPlanningMs": 5.533,
          "gpuPlanningMs": 9.038,
          "cpuExecutionMs": 6325.441,
          "gpuExecutionMs": 1381.459,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1615,
          "parquetFiles": 25,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 20006196783,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "thresholded",
          "parquetReadTimeMs": 626.301,
          "sourceReadTimeMs": 627.21,
          "operatorTimeMs": 504.495,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q21",
          "cpuMs": 32116.534,
          "gpuMs": 5892.616,
          "speedup": 5.45,
          "cpuPlanningMs": 6.472,
          "gpuPlanningMs": 13.652,
          "cpuExecutionMs": 32110.127,
          "gpuExecutionMs": 5878.964,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 13535,
          "parquetFiles": 30,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 46757880258,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 5,
          "joinPartitioningStatus": "mixed",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 4793.331,
          "sourceReadTimeMs": 4793.413,
          "operatorTimeMs": 939.966,
          "materializationTimeMs": 0.0
        },
        {
          "queryId": "Q22",
          "cpuMs": 1681.359,
          "gpuMs": 596.406,
          "speedup": 2.819,
          "cpuPlanningMs": 4.113,
          "gpuPlanningMs": 5.86,
          "cpuExecutionMs": 1677.132,
          "gpuExecutionMs": 590.685,
          "planningOutcome": "selected",
          "runtimeOutcome": "succeeded",
          "parityStatus": "matched",
          "rowGroupsSelected": 1690,
          "parquetFiles": 15,
          "peakMemoryOperatorClass": "aggregate",
          "peakMemoryOperatorClassBytes": 13361997185,
          "sourcePredicatePushdownStatus": "pruning_only",
          "partitionedJoinCount": 0,
          "joinPartitioningStatus": "not_partitioned",
          "aggregateMergeStatus": "not_applicable",
          "parquetReadTimeMs": 297.312,
          "sourceReadTimeMs": 297.46,
          "operatorTimeMs": 288.534,
          "materializationTimeMs": 0.0
        }
      ]
    }
  ],
  "dbgenOrderComparison": {
    "identicalOutputQueryCount": 22
  }
} as const satisfies { datasets: readonly { rows: readonly TpchBenchmarkRow[]; [key: string]: unknown }[]; [key: string]: unknown };
