// GPU runs: target/merrec-reuse-review/gather-fix-full/gpu-summary.json, one server binary.
// CPU pipeline totals: docs/research/barbie_cpu_comparison/results (2026-10-12, same host).
// FAISS runs and kNN validation: target/merrec-reuse-review.

export type GpuRun = {
  repeat: number;
  elapsedSeconds: number;
  rows: number;
  runtimeComplete: boolean;
  hostStagesSeconds: Record<string, number>;
};

export type GpuCapture = {
  case: string;
  sourceArtifact: string;
  warmups: number;
  runs: GpuRun[];
};

export type CpuKnnRun = {
  sourceArtifact: string;
  sourceRepeat: number;
  includeInMedian: boolean;
  excludedReason: string | null;
  elapsedSeconds: number;
  searchSeconds: number;
  loadSortPackSeconds: number;
  indexCopySeconds: number;
  assembleSeconds: number;
};

export type CpuKnnCapture = {
  threads: number;
  blasThreshold: number;
  faissVersion: string;
  numpyVersion: string;
  pyarrowVersion: string;
  threadEnvironment: Record<string, string>;
  runs: CpuKnnRun[];
  reportedDistanceMatches: boolean;
  maxReportedDistanceError: number;
  referenceOutputSha256: string;
};

export type GpuQualityRun = {
  repeat: number;
  candidateDistanceMatches: boolean;
  reportedDistanceMatches: boolean;
  maxReportedDistanceError: number;
  maxCandidateDistanceError: number;
  idRecallAtK: number;
  withinReferenceKthFraction: number;
  validOutput: boolean;
};

export type CpuPipeline = {
  pipeline: string;
  measuredDate: string;
  totalSeconds: number[];
};

export type MerrecQuery = {
  label: string;
  guideAnchor: string;
  algorithms: string;
  gpu: GpuCapture;
  cpu: CpuPipeline;
  resultCheck: string;
};

export type ExactKnnCase = {
  queryRows: number;
  gpu: GpuCapture;
  cpu: CpuKnnCapture;
  quality: GpuQualityRun[];
};

export function medianAfterWarmups(values: readonly number[], warmups: number): number {
  const measured = values.slice(warmups).sort((left, right) => left - right);
  const middle = Math.floor(measured.length / 2);
  return measured.length % 2 === 0
    ? (measured[middle - 1] + measured[middle]) / 2
    : measured[middle];
}

export function gpuMedianSeconds(capture: GpuCapture): number {
  return medianAfterWarmups(capture.runs.map((run) => run.elapsedSeconds), capture.warmups);
}

export function cpuPipelineMedianSeconds(cpu: CpuPipeline): number {
  return medianAfterWarmups(cpu.totalSeconds, 0);
}

export function cpuMedianSeconds(capture: CpuKnnCapture, field: 'elapsedSeconds' | 'searchSeconds'): number {
  return medianAfterWarmups(capture.runs.filter((run) => run.includeInMedian).map((run) => run[field]), 0);
}

export const merrecAlgorithmBenchmark = {
  "environment": {
    "sourceRevision": "512503695da15e5111c4fbe11fbd97b2b19ab37e",
    "worktreeChanges": true,
    "serverBinarySha256": "7b1fbd69f03975073dbf50b8e5d73eeaf68165fa2a90457aa8d18b26a458f8d2",
    "nativeCuvsRevision": "ce00140ce1e4b715fd6433145a9a22804a4de592",
    "rustBinaryRebuiltForNativeFix": true,
    "nativeBuildPolicy": "Canonical cuVS install rebuilt from corrected fork; cuDF, cuGraph and cuML native installs retained",
    "installedLibrarySha256": [
      "42d7677d07c5df966de801a1772bb0fc71f10097cf374d4481a7f89b14dc6f33  target/native/cuvs-install/lib/libcuvs.so",
      "00f2fb8806e40c168fa9a3f2279e142b62764ed6e6fa38b3089bfecb76bd542f  target/native/cuvs-install/lib/libcuvs_c.so"
    ],
    "cpu": "Intel Xeon Platinum 8559C",
    "cpuCores": 16,
    "cpuThreads": 32,
    "memoryGiB": 256,
    "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "gpuMemoryGB": 96,
    "gpuDriver": "595.91.07",
    "cudaToolkit": "13.2",
    "buildProfile": "release",
    "cpuTarget": "x86-64-v3",
    "gpuMeasuredDates": [
      "2026-10-15"
    ]
  },
  "dataset": {
    "tasteSource": "fixture/demo/merrec/scale-results/full-demos/14_taste64.parquet",
    "tasteSha256": "8cc5f237629d78f202697b5bda4966d75d15db953aedf6cd1ab523ca127c7b6c",
    "tasteRows": 2080743,
    "dimensions": 64,
    "k": 10,
    "metric": "squared_l2",
    "selfMatches": true,
    "querySelection": "First IDs in increasing order; full dataset at every query cohort"
  },
  "queries": [
    {
      "label": "Barbie tie-ins",
      "guideAnchor": "barbie",
      "algorithms": "cuGraph personalized PageRank ×2, PageRank ×2",
      "gpu": {
        "case": "10_barbie_tie_ins",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 3.058884863043204,
            "rows": 25,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.311483169
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 3.048391876043752,
            "rows": 25,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.30742847100000004
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 3.0169681920669973,
            "rows": 25,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.30730571400000006
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 3.011409148108214,
            "rows": 25,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.30724487300000003
            }
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB 1.5.6 + igraph 1.0.0",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          23.75,
          23.266,
          23.076
        ]
      },
      "resultCheck": "Top 25 identical"
    },
    {
      "label": "Session audit",
      "guideAnchor": "sessions",
      "algorithms": "cuGraph weakly connected components",
      "gpu": {
        "case": "12_session_audit",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 1.8998997029848397,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.414288349
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 1.8821459219325334,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.420825721
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 1.8802138809114695,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.411553237
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 1.8692932380363345,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.409525572
            }
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB + scipy 1.18.1",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          13.119,
          12.228,
          12.349
        ]
      },
      "resultCheck": "Identical components"
    },
    {
      "label": "Audience funnel",
      "guideAnchor": "audience-funnel",
      "algorithms": "cuGraph personalized PageRank, cuVS brute-force kNN, cuML forest inference",
      "gpu": {
        "case": "30_audience_funnel",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 8.259018758079037,
            "rows": 8,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.120975917
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 8.257321076001972,
            "rows": 8,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.12207626
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 8.274395982036367,
            "rows": 8,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.121982462
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 8.265892094001174,
            "rows": 8,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 0.122211358
            }
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB + igraph + FAISS + XGBoost",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          44.404,
          40.34,
          39.738
        ]
      },
      "resultCheck": "Identical buyer counts"
    },
    {
      "label": "Propensity model",
      "guideAnchor": "model",
      "algorithms": "cuML forest inference",
      "gpu": {
        "case": "27_propensity_xgboost",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 6.627020346000791,
            "rows": 1,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          },
          {
            "repeat": 2,
            "elapsedSeconds": 6.586721200030297,
            "rows": 1,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          },
          {
            "repeat": 3,
            "elapsedSeconds": 6.591813851846382,
            "rows": 1,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          },
          {
            "repeat": 4,
            "elapsedSeconds": 6.574452277971432,
            "rows": 1,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB + XGBoost 3.4",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          16.784,
          16.119,
          15.767
        ]
      },
      "resultCheck": "Identical apart from 3 tied scores"
    },
    {
      "label": "Brand twins",
      "guideAnchor": "brand-twins",
      "algorithms": "cuGraph Jaccard",
      "gpu": {
        "case": "08_brand_twins",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 4.023640512023121,
            "rows": 30,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 1.725797133
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 3.3362505370751023,
            "rows": 30,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 1.614404623
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 3.324651040136814,
            "rows": 30,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 1.616815269
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 3.3255083081312478,
            "rows": 30,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 1.613305124
            }
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB SQL Jaccard",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          6.584,
          6.57,
          6.437
        ]
      },
      "resultCheck": "30 pairs identical"
    },
    {
      "label": "Pokemon lookalikes",
      "guideAnchor": "pokemon",
      "algorithms": "cuVS brute-force kNN",
      "gpu": {
        "case": "13_pokemon_lookalikes",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 13.961489233886823,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          },
          {
            "repeat": 2,
            "elapsedSeconds": 13.850861615967005,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          },
          {
            "repeat": 3,
            "elapsedSeconds": 13.807080084923655,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          },
          {
            "repeat": 4,
            "elapsedSeconds": 13.796136000193655,
            "rows": 3,
            "runtimeComplete": true,
            "hostStagesSeconds": {}
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB + FAISS 1.15",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          25.956,
          25.633,
          25.999
        ]
      },
      "resultCheck": "Identical buyer counts"
    },
    {
      "label": "Shopping worlds",
      "guideAnchor": "shopping-worlds",
      "algorithms": "cuGraph Jaccard, Leiden",
      "gpu": {
        "case": "09_shopping_worlds",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 5.889316993067041,
            "rows": 190,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.150201066999999
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 5.829822584055364,
            "rows": 190,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.095586759
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 5.798623761162162,
            "rows": 190,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.074248894
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 5.827277451055124,
            "rows": 190,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.091003007
            }
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB + igraph 1.0.0",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          9.701,
          9.64,
          9.296
        ]
      },
      "resultCheck": "Leiden partitions differ"
    },
    {
      "label": "World bridges",
      "guideAnchor": "bridges",
      "algorithms": "cuGraph Jaccard, Leiden, betweenness",
      "gpu": {
        "case": "31_world_bridges",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 8.166950152954087,
            "rows": 15,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.916052946000001
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 8.138080921024084,
            "rows": 15,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.894494776
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 8.140956039074808,
            "rows": 15,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.892715351
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 8.155634578084573,
            "rows": 15,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "graph_task": 4.899837034000001
            }
          }
        ]
      },
      "cpu": {
        "pipeline": "DuckDB + networkit 11.2.2",
        "measuredDate": "2026-10-12",
        "totalSeconds": [
          11.776,
          11.504,
          12.096
        ]
      },
      "resultCheck": "Top 15 identical"
    }
  ],
  "exactKnn": [
    {
      "queryRows": 100,
      "gpu": {
        "case": "knn_exact_100",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 0.10731129301711917,
            "rows": 1000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.3732e-05,
              "vector_native:search": 0.003533929
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 0.10458244103938341,
            "rows": 1000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.286e-05,
              "vector_native:search": 0.003428311
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 0.10514762601815164,
            "rows": 1000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.0739e-05,
              "vector_native:search": 0.003451005
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 0.10438823909498751,
            "rows": 1000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.2869e-05,
              "vector_native:search": 0.003459343
            }
          }
        ]
      },
      "cpu": {
        "threads": 8,
        "blasThreshold": 0,
        "faissVersion": "1.15.1",
        "numpyVersion": "2.5.3",
        "pyarrowVersion": "25.0.1",
        "threadEnvironment": {
          "OPENBLAS_CORETYPE": "SkylakeX",
          "OPENBLAS_NUM_THREADS": "8",
          "OMP_NUM_THREADS": "8"
        },
        "runs": [
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100.json",
            "sourceRepeat": 1,
            "includeInMedian": false,
            "excludedReason": "warmup",
            "elapsedSeconds": 3.2230813710484654,
            "searchSeconds": 0.20445423806086183,
            "loadSortPackSeconds": 2.7630505189299583,
            "indexCopySeconds": 0.25542314001359046,
            "assembleSeconds": 0.00015347404405474663
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100.json",
            "sourceRepeat": 2,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 3.1591920810751617,
            "searchSeconds": 0.2150724409148097,
            "loadSortPackSeconds": 2.665575424209237,
            "indexCopySeconds": 0.2783876589965075,
            "assembleSeconds": 0.00015655695460736752
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100.json",
            "sourceRepeat": 3,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 3.104213239857927,
            "searchSeconds": 0.21999625395983458,
            "loadSortPackSeconds": 2.6061530360020697,
            "indexCopySeconds": 0.2779278210364282,
            "assembleSeconds": 0.0001361288595944643
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100.json",
            "sourceRepeat": 4,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 3.1003935080952942,
            "searchSeconds": 0.21612779097631574,
            "loadSortPackSeconds": 2.605135028017685,
            "indexCopySeconds": 0.2789733191020787,
            "assembleSeconds": 0.00015736999921500683
          }
        ],
        "reportedDistanceMatches": true,
        "maxReportedDistanceError": 7.032270316204015e-07,
        "referenceOutputSha256": "bb8ef16a5e227293e2563a3e50f8c4b9dc2c6a1ac3e760d62252dd120f908f93"
      },
      "quality": [
        {
          "repeat": 1,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": true,
          "maxReportedDistanceError": 7.032270316204015e-07,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 0.996,
          "withinReferenceKthFraction": 1.0,
          "validOutput": true
        },
        {
          "repeat": 2,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": true,
          "maxReportedDistanceError": 7.032270316204015e-07,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 0.996,
          "withinReferenceKthFraction": 1.0,
          "validOutput": true
        },
        {
          "repeat": 3,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": true,
          "maxReportedDistanceError": 7.032270316204015e-07,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 0.996,
          "withinReferenceKthFraction": 1.0,
          "validOutput": true
        },
        {
          "repeat": 4,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": true,
          "maxReportedDistanceError": 7.032270316204015e-07,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 0.996,
          "withinReferenceKthFraction": 1.0,
          "validOutput": true
        }
      ]
    },
    {
      "queryRows": 1000,
      "gpu": {
        "case": "knn_exact_1000",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 0.1312618050724268,
            "rows": 10000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.1384e-05,
              "vector_native:search": 0.027155312
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 0.12970678810961545,
            "rows": 10000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.1451e-05,
              "vector_native:search": 0.027129198
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 0.12955461419187486,
            "rows": 10000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 1.9576e-05,
              "vector_native:search": 0.027124497
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 0.12883287598378956,
            "rows": 10000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.0785e-05,
              "vector_native:search": 0.026924484
            }
          }
        ]
      },
      "cpu": {
        "threads": 8,
        "blasThreshold": 0,
        "faissVersion": "1.15.1",
        "numpyVersion": "2.5.3",
        "pyarrowVersion": "25.0.1",
        "threadEnvironment": {
          "OPENBLAS_CORETYPE": "SkylakeX",
          "OPENBLAS_NUM_THREADS": "8",
          "OMP_NUM_THREADS": "8"
        },
        "runs": [
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_1000.json",
            "sourceRepeat": 1,
            "includeInMedian": false,
            "excludedReason": "warmup",
            "elapsedSeconds": 4.262130697956309,
            "searchSeconds": 1.2873136010020971,
            "loadSortPackSeconds": 2.720165621023625,
            "indexCopySeconds": 0.25439495383761823,
            "assembleSeconds": 0.0002565220929682255
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_1000.json",
            "sourceRepeat": 2,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 4.1405739698093385,
            "searchSeconds": 1.24931551492773,
            "loadSortPackSeconds": 2.6165948209818453,
            "indexCopySeconds": 0.2744002139661461,
            "assembleSeconds": 0.000263419933617115
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_1000.json",
            "sourceRepeat": 3,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 4.124838707968593,
            "searchSeconds": 1.1944592830259353,
            "loadSortPackSeconds": 2.652398006990552,
            "indexCopySeconds": 0.2777257990092039,
            "assembleSeconds": 0.0002556189429014921
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_1000.json",
            "sourceRepeat": 4,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 4.151156581938267,
            "searchSeconds": 1.2705972320400178,
            "loadSortPackSeconds": 2.603224040940404,
            "indexCopySeconds": 0.27709487406536937,
            "assembleSeconds": 0.000240434892475605
          }
        ],
        "reportedDistanceMatches": false,
        "maxReportedDistanceError": 1.0728836059570312e-06,
        "referenceOutputSha256": "055178a37693cb562c9102fb4c88105eafc712e20b2d20598249ee93f345a335"
      },
      "quality": [
        {
          "repeat": 1,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.0728836059570312e-06,
          "maxCandidateDistanceError": 2.9182312566433666e-08,
          "idRecallAtK": 0.9958,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 2,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.0728836059570312e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 0.9958,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 3,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.0728836059570312e-06,
          "maxCandidateDistanceError": 2.9182312566433666e-08,
          "idRecallAtK": 0.9958,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 4,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.0728836059570312e-06,
          "maxCandidateDistanceError": 2.9182312566433666e-08,
          "idRecallAtK": 0.9957,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        }
      ]
    },
    {
      "queryRows": 10000,
      "gpu": {
        "case": "knn_exact_10000",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 0.32062005484476686,
            "rows": 100000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.1581e-05,
              "vector_native:search": 0.211513217
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 0.320347593864426,
            "rows": 100000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 1.9527e-05,
              "vector_native:search": 0.211442315
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 0.3212265761103481,
            "rows": 100000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.0662e-05,
              "vector_native:search": 0.211328381
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 0.3249092639889568,
            "rows": 100000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 1.9949e-05,
              "vector_native:search": 0.21632985
            }
          }
        ]
      },
      "cpu": {
        "threads": 8,
        "blasThreshold": 128000,
        "faissVersion": "1.15.1",
        "numpyVersion": "2.5.3",
        "pyarrowVersion": "25.0.1",
        "threadEnvironment": {
          "OPENBLAS_CORETYPE": "SkylakeX",
          "OPENBLAS_NUM_THREADS": "8",
          "OMP_NUM_THREADS": "8"
        },
        "runs": [
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_10000.json",
            "sourceRepeat": 1,
            "includeInMedian": false,
            "excludedReason": "warmup",
            "elapsedSeconds": 13.178156951908022,
            "searchSeconds": 10.178091906011105,
            "loadSortPackSeconds": 2.7432704500388354,
            "indexCopySeconds": 0.255759019870311,
            "assembleSeconds": 0.0010355759877711535
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_10000.json",
            "sourceRepeat": 2,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 12.554792687064037,
            "searchSeconds": 9.58138315891847,
            "loadSortPackSeconds": 2.6949601219967008,
            "indexCopySeconds": 0.2773418561555445,
            "assembleSeconds": 0.0011075499933212996
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_10000.json",
            "sourceRepeat": 3,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 12.848905079998076,
            "searchSeconds": 9.89731801301241,
            "loadSortPackSeconds": 2.6730368838179857,
            "indexCopySeconds": 0.27738428907468915,
            "assembleSeconds": 0.0011658940929919481
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_10000.json",
            "sourceRepeat": 4,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 12.991496911039576,
            "searchSeconds": 10.078531198902056,
            "loadSortPackSeconds": 2.6340917870402336,
            "indexCopySeconds": 0.27777347806841135,
            "assembleSeconds": 0.001100447028875351
          }
        ],
        "reportedDistanceMatches": false,
        "maxReportedDistanceError": 1.1920928955078125e-06,
        "referenceOutputSha256": "05e455b2d8a2f078160c0c800a643d3f2e861af1c5e314031c44d0be15ffba6c"
      },
      "quality": [
        {
          "repeat": 1,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.3113021850585938e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 2,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.3113021850585938e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 3,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.3113021850585938e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 4,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.3113021850585938e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        }
      ]
    },
    {
      "queryRows": 100000,
      "gpu": {
        "case": "knn_exact_100000",
        "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
        "warmups": 1,
        "runs": [
          {
            "repeat": 1,
            "elapsedSeconds": 1.9733310658484697,
            "rows": 1000000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 3.2777e-05,
              "vector_native:search": 1.810684586
            }
          },
          {
            "repeat": 2,
            "elapsedSeconds": 1.9720034359488636,
            "rows": 1000000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.1133e-05,
              "vector_native:search": 1.810828763
            }
          },
          {
            "repeat": 3,
            "elapsedSeconds": 1.9743309491313994,
            "rows": 1000000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.6461e-05,
              "vector_native:search": 1.810288272
            }
          },
          {
            "repeat": 4,
            "elapsedSeconds": 1.9721313810441643,
            "rows": 1000000,
            "runtimeComplete": true,
            "hostStagesSeconds": {
              "vector_native:build_index": 2.297e-05,
              "vector_native:search": 1.81027653
            }
          }
        ]
      },
      "cpu": {
        "threads": 8,
        "blasThreshold": 128000,
        "faissVersion": "1.15.1",
        "numpyVersion": "2.5.3",
        "pyarrowVersion": "25.0.1",
        "threadEnvironment": {
          "OPENBLAS_CORETYPE": "SkylakeX",
          "OPENBLAS_NUM_THREADS": "8",
          "OMP_NUM_THREADS": "8"
        },
        "runs": [
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100000.json",
            "sourceRepeat": 1,
            "includeInMedian": false,
            "excludedReason": "warmup",
            "elapsedSeconds": 104.5719949840568,
            "searchSeconds": 101.61728177801706,
            "loadSortPackSeconds": 2.6959730989765376,
            "indexCopySeconds": 0.251410260098055,
            "assembleSeconds": 0.007329846965149045
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100000.json",
            "sourceRepeat": 2,
            "includeInMedian": false,
            "excludedReason": "concurrent_compilation",
            "elapsedSeconds": 103.89682252099738,
            "searchSeconds": 100.81574612786062,
            "loadSortPackSeconds": 2.7809662129729986,
            "indexCopySeconds": 0.2924224380403757,
            "assembleSeconds": 0.007687742123380303
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100000.json",
            "sourceRepeat": 3,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 97.78611902985722,
            "searchSeconds": 94.83319136989303,
            "loadSortPackSeconds": 2.663796448847279,
            "indexCopySeconds": 0.2814977371599525,
            "assembleSeconds": 0.0076334739569574594
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu/t8/knn_faiss_100000.json",
            "sourceRepeat": 4,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 101.40113845118321,
            "searchSeconds": 98.45298141683452,
            "loadSortPackSeconds": 2.6594407851807773,
            "indexCopySeconds": 0.2810794049873948,
            "assembleSeconds": 0.007636844180524349
          },
          {
            "sourceArtifact": "target/merrec-reuse-review/cpu-replacement/t8/knn_faiss_100000.json",
            "sourceRepeat": 1,
            "includeInMedian": true,
            "excludedReason": null,
            "elapsedSeconds": 102.36240083514713,
            "searchSeconds": 99.3685096888803,
            "loadSortPackSeconds": 2.734169509029016,
            "indexCopySeconds": 0.2526650810614228,
            "assembleSeconds": 0.007056556176394224
          }
        ],
        "reportedDistanceMatches": false,
        "maxReportedDistanceError": 1.9073486328125e-06,
        "referenceOutputSha256": "44b07fb1e36756dab8015c456a66dcd7071530cd3ed6604eccdef7f668d5c555"
      },
      "quality": [
        {
          "repeat": 1,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.9073486328125e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 2,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.9073486328125e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 3,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.9073486328125e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        },
        {
          "repeat": 4,
          "candidateDistanceMatches": true,
          "reportedDistanceMatches": false,
          "maxReportedDistanceError": 1.9073486328125e-06,
          "maxCandidateDistanceError": 0.0,
          "idRecallAtK": 1.0,
          "withinReferenceKthFraction": 1.0,
          "validOutput": false
        }
      ]
    }
  ],
  "itemPageRank": {
    "gpu": {
      "case": "04_item_pagerank",
      "sourceArtifact": "target/merrec-reuse-review/gather-fix-full/gpu-summary.json",
      "warmups": 1,
      "runs": [
        {
          "repeat": 1,
          "elapsedSeconds": 20.989827630110085,
          "rows": 20,
          "runtimeComplete": true,
          "hostStagesSeconds": {
            "graph_task": 7.921028386
          }
        },
        {
          "repeat": 2,
          "elapsedSeconds": 20.913689757231623,
          "rows": 20,
          "runtimeComplete": true,
          "hostStagesSeconds": {
            "graph_task": 7.920131322
          }
        },
        {
          "repeat": 3,
          "elapsedSeconds": 20.873537689913064,
          "rows": 20,
          "runtimeComplete": true,
          "hostStagesSeconds": {
            "graph_task": 7.92118748
          }
        },
        {
          "repeat": 4,
          "elapsedSeconds": 20.137228369014338,
          "rows": 20,
          "runtimeComplete": true,
          "hostStagesSeconds": {
            "graph_task": 7.921960751
          }
        }
      ]
    },
    "normalizedDirectedEdges": 1682503472,
    "vertices": 87055073,
    "iterations": 75,
    "alpha": 0.85,
    "epsilon": 1e-05,
    "weighted": true,
    "directed": false
  }
} satisfies {
  queries: readonly MerrecQuery[];
  exactKnn: readonly ExactKnnCase[];
  [key: string]: unknown;
};
