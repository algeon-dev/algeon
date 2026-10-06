# EMBER2024 scale evaluation

The EMBER2024 dataset contains extracted features, metadata, and labels for files uploaded
to VirusTotal between September 2023 and December 2024. Following deduplication of duplicate
lines in raw upstream archives, the full dataset comprises 3,238,244 distinct files:
- **Training set (Weeks 1 to 52)**: 2,626,000 unique files (~1.56M Win32 executables).
- **Test set (Weeks 53 to 64)**: 605,929 unique files (359,994 Win32 executables).
- **Challenge set**: 6,315 malicious files (3,225 Win32 executables) evading antivirus engines.

Every record preserves its exact SHA-256 digest, split isolation, and disjoint ID allocation
(train starting at 0, test at 10,000,000, challenge at 20,000,000).

## Scale evaluation suite results

All queries were executed against `algeon_server` in `native_required` execution mode on an
NVIDIA RTX PRO 6000 Blackwell GPU (96 GB device budget, 90 GiB query grant, 1 GiB backend reserve,
max 1 active attempt).

| Query | Analytical Operation / Algorithm | Input Scale | Output Rows | Execution Path | Host Boundaries | 1st Run (s) | Median (r2..r5) (s) | Peak GPU Alloc (GiB) |
|---|---|---|---:|---|---:|---:|---:|---:|
| `01_historical_exact_search.sql` | Exact squared-L2 kNN (`cuvs_brute_force_knn`, k=10) | ~1.56M corpus × 10,000 test queries (512-dim) | 100,000 | native | 0 | 281.53 | 281.53* | 17.60 |
| `02_historical_cagra_search.sql` | CAGRA ANN graph search (`cuvs_cagra`, k=10, deg 32/64) | ~1.56M corpus × 10,000 test queries (512-dim) | 0 | native (failed) | 0 | Failed (RAFT) | Failed (RAFT) | 17.60 |
| `03_historical_ivf_flat_search.sql` | IVF-Flat ANN search (`cuvs_ivf_flat`, 1024 lists, 32 probes) | ~1.56M corpus × 10,000 test queries (512-dim) | 100,000 | native | 0 | 302.90 | 302.90* | 17.60 |
| `04_temporal_logistic_regression.sql` | Supervised logistic regression (`cuml_logistic_regression`) | ~1.56M train, 359,994 test (512-dim) | 1 | native | 0 | 3.23 | 3.13 | 16.79 |
| `05_temporal_random_forest.sql` | Supervised random forest (`cuml_random_forest_classifier`, 100 trees, depth 12) | ~1.56M train, 359,994 test (512-dim) | 1 | native | 0 | 62.92 | 62.02 | 16.79 |
| `06_temporal_model_comparison.sql` | Unified query comparing Logistic Regression and Random Forest | ~1.56M train, 359,994 test each | 2 | native | 0 | 65.80 | 65.80* | 16.80 |
| `07_challenge_import_review.sql` | Personalized PageRank + Jaccard similarity on challenge import graph | 926 files, 4,988 symbols, 80,673 edges | 15 | native | 0 | 0.098 | 0.098 | 0.01 |
| `08_challenge_forest_review.sql` | LightGBM forest inference (`cuml_forest_predict`) + cuVS Top-3 attribution | 6,315 challenge files (2,568-dim) | 30 | native | 0 | 0.217 | 0.199 | 0.24 |

\*Single run captured for full-corpus vector search and combined model comparison; multi-run repeats captured for supervised classifiers and challenge review queries.

## Technical findings and domain analysis

### 1. Removal of artificial downsampling

Earlier preparation scripts applied `int(digest_hex[:8], 16) % 100 == 0`, producing an arbitrary
1% sample of train/test vectors (26,277 train and 6,102 test rows; 15,588 and 3,618 Win32 rows).
In this scale suite, this artificial constraint is completely eliminated:
- `ember_research_vectors_train`: all 2,626,000 unique files.
- `ember_research_vectors_test`: all 605,929 unique files.
- `ember_metadata_train` and `ember_metadata_test`: all 2,626,000 and 605,929 files.
- Zero label disagreements across raw JSON lines, feature memmaps, and Parquet projections.

### 2. Search corpus scale vs query batching

The historical reference search constructs its searchable corpus over the entire deduplicated
Win32 training dataset (~1.56M files) across weeks 0 to 51. The query batch selects 10,000
held-out Win32 test queries from weeks 52 to 63, returning Top-10 reference neighbors
(100,000 result rows). Distinguishing the multi-million indexed corpus from the query batch
reflects real-world malware triage workflows where analysts search against historical reference
repositories.

### 3. cuVS vector search scalability and CAGRA duplicate vector degeneration

- **Exact Search (`cuvs_brute_force_knn`)**:
  Computes exact squared-L2 distances between 10,000 queries and the ~1.56M corpus (15.6 billion
  pairwise distances, 512 dimensions) and executes Top-10 reduction entirely in GPU memory in
  281.53s, joining query and neighbor metadata post-search with zero host boundaries.

- **IVF-Flat Search (`cuvs_ivf_flat`)**:
  Trains 1024 inverted lists via GPU k-means clustering across the 1.56M vectors and searches
  32 probes for 10,000 queries in 302.90s with zero host boundaries.

- **CAGRA Index Build Degeneration (`cuvs_cagra`)**:
  CAGRA failed during query-local index build with a structured cuVS RAFT dependency error:
  `Could not generate an intermediate CAGRA graph because the initial kNN graph contains too many invalid or duplicated neighbor nodes.`
  In PE malware analysis, byte histograms (normalized byte frequencies and byte entropies)
  frequently produce identical feature representations across different binaries (e.g. shared
  compiler stubs, standard PE section padding, uniform resource blocks). While brute force and
  IVF-Flat handle identical vectors seamlessly, CAGRA's intermediate IVF-PQ graph construction
  (intermediate_graph_degree=64) encounters zero-distance duplicate nodes that break reverse
  edge generation. Documenting this structural characteristic is critical for vector index selection
  on malware feature representations.

### 4. Supervised temporal generalization on full cohorts

Both logistic regression and random forest train on the complete ~1.56M Win32 training cohort
and evaluate every single held-out Win32 test file (359,994 files) without downsampling or truncation.

Confusion metrics evaluated over 359,994 test files:
- **Logistic Regression (`cuml_logistic_regression`)**:
  - Training time: ~3.13s median.
  - True Positives: 149,414 | False Positives: 24,679 | False Negatives: 30,586 | True Negatives: 155,315.
  - Accuracy: 84.65%.
- **Random Forest (`cuml_random_forest_classifier`, 100 trees, max depth 12)**:
  - Training time: ~62.02s median (100% GPU utilization).
  - True Positives: 156,532 | False Positives: 9,239 | False Negatives: 23,468 | True Negatives: 170,755.
  - Accuracy: 90.91%.

Random forest achieves a 6.26% accuracy improvement over logistic regression, reducing false
positives from 24,679 down to 9,239 on the held-out quarter.

### 5. Challenge graph and forest attribution analysis

- **Import Graph Review (`cugraph_personalized_pagerank` + `cugraph_jaccard`)**:
  The challenge split contains 6,315 evasive malicious files (3,225 Win32). The bipartite import
  graph links 926 of these files to 4,988 symbols with 80,673 edges (the remaining 2,299 Win32 files
  have no retained imports within the frequency bounds of 3 to 100 occurrences).
  Personalized PageRank seeded from unlabeled sample `20005740` followed by Jaccard similarity
  identifies the top 15 candidate matches in 0.098s median.

- **Forest Inference & Attribution Ranking (`cuml_forest_predict` + `cuvs_top_k`)**:
  Scores 6,315 challenge files with `EMBER2024_all.model` (2,568 dimensions) and computes
  top-3 feature group attributions via cuVS Top-K in 0.199s median.

## Server configuration

The evaluation executed on an NVIDIA RTX PRO 6000 Blackwell (96 GB device budget, 90 GiB query grant,
1 GiB backend reserve, 1 active attempt):

```toml
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 100931731456
backend_reserve_bytes = 1073741824
cache_cap_bytes = 0
default_query_limit_bytes = 96636764160
max_active_attempts = 1
source_work_per_attempt = 2
stream_pool_size = 8
worker_count = 4
lane_count = 4

[native]
execution_mode = "native_required"

[models]
root = "fixture/demo/ember2024/models"
```

## Reproducing the evaluation

1. Prepare the full deduplicated Parquet relations:
   ```bash
   python3 fixture/demo/ember2024/scripts/research_prepare.py
   python3 fixture/demo/ember2024/scripts/research_graph_prepare.py
   python3 fixture/demo/ember2024/scripts/research_attributions.py
   ```

2. Start `algeon_server` with the GPU wrapper lock:
   ```bash
   ALGEON_SERVER_CONFIG_FILE="$PWD/fixture/demo/ember2024/scripts/scale/server.toml" \
   ALGEON_SERVER_EXECUTION_MODE=native_required \
   ALGEON_SERVER_BIND=127.0.0.1:50062 \
   flock -x /tmp/cudf-gpu.lock target/release/algeon_server
   ```

3. Execute the scale queries:
   ```bash
   python3 scripts/dev/ember_run.py --uri grpc://127.0.0.1:50062 --register --repeats 5 \
     --report-dir target/ember2024-scale/report --results-dir target/ember2024-scale/results \
     fixture/demo/ember2024/scripts/scale/04_temporal_logistic_regression.sql \
     fixture/demo/ember2024/scripts/scale/05_temporal_random_forest.sql \
     fixture/demo/ember2024/scripts/scale/07_challenge_import_review.sql \
     fixture/demo/ember2024/scripts/scale/08_challenge_forest_review.sql
   ```
