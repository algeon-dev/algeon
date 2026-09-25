# EMBER2024 Fixture

EMBER2024 is a benchmark dataset for malware detection and classification, published in ACM SIGKDD 2025:
- **Paper**: *EMBER2024 - A Benchmark Dataset for Holistic Evaluation of Malware Classifiers* (Robert J. Joyce et al., arXiv: [2506.05074](https://arxiv.org/abs/2506.05074)).
- **Repository**: [`joyce8/EMBER2024`](https://huggingface.co/datasets/joyce8/EMBER2024)
- **Upstream Code**: [`FutureComputing4AI/EMBER2024`](https://github.com/FutureComputing4AI/EMBER2024)

## Dataset Specifications

EMBER2024 contains extracted raw features and labels for 3,232,000 files uploaded to VirusTotal between September 24, 2023, and December 14, 2024 (50,500 files per week):
- **Training set (Weeks 1 to 52)**: 2,626,000 files
- **Test set (Weeks 53 to 64)**: 606,000 files (for temporal generalization evaluation)
- **Challenge set**: 6,315 malicious files that originally evaded approximately 70 antivirus vendors on VirusTotal

### File Manifest and Disk Footprint

All 13 archives (totaling ~24.3 GB compressed, ~48 GB uncompressed) are located in this directory:

| Subset | Archive Name | Compressed Size | Uncompressed Size | Weekly Volume | Train Total | Test Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Challenge Set** | `challenge.zip` | 32.1 MB | ~126 MB | N/A | N/A | 6,315 |
| **Linux ELF** | `ELF_train.zip`<br>`ELF_test.zip` | 69.2 MB<br>16.8 MB | ~100 MB<br>~24 MB | 500 | 26,000 | 6,000 |
| **PDF Documents** | `PDF_train.zip`<br>`PDF_test.zip` | 122.6 MB<br>28.8 MB | ~197 MB<br>~46 MB | 1,000 | 52,000 | 12,000 |
| **Android APK** | `APK_train.zip`<br>`APK_test.zip` | 783.6 MB<br>177.5 MB | ~1.0 GB<br>~234 MB | 4,000 | 208,000 | 48,000 |
| **.NET Executables** | `Dot_Net_train.zip`<br>`Dot_Net_test.zip` | 937.1 MB<br>220.5 MB | ~1.8 GB<br>~425 MB | 5,000 | 260,000 | 60,000 |
| **Windows 64-bit PE** | `Win64_train.zip`<br>`Win64_test.zip` | 5.85 GB<br>1.18 GB | ~12.9 GB<br>~2.5 GB | 10,000 | 520,000 | 120,000 |
| **Windows 32-bit PE** | `Win32_train.zip`<br>`Win32_test.zip` | 12.32 GB<br>2.59 GB | ~23.7 GB<br>~4.9 GB | 30,000 | 1,560,000 | 360,000 |
| **Total** | 13 archives | **~24.3 GB** | **~48 GB** | 50,500 / wk | 2,626,000 | 606,000 |

## Python Environment

All commands below use a dedicated virtualenv, not `uvx`, so the toolchain is
pinned and reproducible without a network fetch on every invocation:

```bash
python3 -m venv ~/.venvs/forest
~/.venvs/forest/bin/pip install lightgbm xgboost pyarrow huggingface_hub numpy \
    "thrember @ git+https://github.com/FutureComputing4AI/EMBER2024.git"
```

`~/.venvs/forest/bin/hf` (the CLI entry point `huggingface_hub` installs) and
`~/.venvs/forest/bin/python` are used directly in every example instead of
`uvx --from huggingface_hub hf ...` or a bare `pip`/`python3`.

## Download Instructions

### Method 1: Hugging Face CLI (`hf download`)

```bash
# Full dataset download (24.3 GB)
~/.venvs/forest/bin/hf download joyce8/EMBER2024 \
  --repo-type dataset \
  --local-dir ./fixture/ember2024

# Single file download (e.g. challenge set only)
~/.venvs/forest/bin/hf download joyce8/EMBER2024 challenge.zip \
  --repo-type dataset \
  --local-dir ./fixture/ember2024
```

### Method 2: Upstream `thrember` Python Library

```python
import thrember

# Download dataset slices
thrember.download_dataset("./fixture/ember2024", file_type="PE", split="test")

# Download 14 pre-trained benchmark LightGBM models
thrember.download_models("./fixture/ember2024/models")
```

## Feature Vectorization and Usage

EMBER2024 features use format version 3 (`thrember`), using `pefile` instead of `LIEF 0.9.0`:

```python
import thrember

# Vectorize raw features to binary .dat files
thrember.create_vectorized_features("./fixture/ember2024")

# Vectorize for specific label tasks (family, behavior, file_property, packer, exploit, group)
thrember.create_vectorized_features("./fixture/ember2024", label_type="family", class_min=10)

# Read vectorized features into numpy arrays
X_train, y_train = thrember.read_vectorized_features("./fixture/ember2024", split="train")
X_test, y_test = thrember.read_vectorized_features("./fixture/ember2024", split="test")
```

`create_vectorized_features` always (re)writes all three splits (train, test,
challenge) in one call; it has no per-split mode.

## `prepare.py`: Parquet Export

`prepare.py` wraps the three steps above (models, raw data, vectorization,
each skipped if its output already exists) and adds a fourth: exporting a
requested split as Parquet with columns `id` (Int64, row ordinal within the
split), `f0..f{d-1}` (Float32, `d` read from the vectorized files rather than
hardcoded), and `label` (Int32, EMBER's raw label). All fields are
non-nullable. Parquet is written in row-group chunks straight from the
`X_*.dat` / `y_*.dat` memmaps to bound RAM (`--row-group-rows`, default
100,000); `train` alone is 51 GB vectorized, so never load a full split into
memory at once.

```bash
# From the venv directly:
~/.venvs/forest/bin/python fixture/ember2024/prepare.py --split test

# Or via the fixture entrypoint (resolves ~/.venvs/forest automatically,
# override with ALGEON_EMBER_PYTHON):
fixture/fixture.sh ember2024 prepare --split test --split challenge

# Cap rows per split, e.g. for a bench subset; requires --force to regenerate
# an existing output at a different row count:
fixture/fixture.sh ember2024 prepare --split train --rows 1000000 --force
```

Output lands at `fixture/ember2024/parquet/<split>.parquet`. Every step is
skipped if its output already exists; pass `--force` to redo a step (a
`--force` re-run of the Parquet step also redownloads/revectorizes upstream
steps that are missing, but not ones already present).

`prepare.py` fails loudly (and deletes the partial output) if any feature
value in the split is non-finite (NaN or Inf): the `cuml_forest_predict` SQL
function requires finite input by contract (see
`docs/plan/2026-09-24-nvforest-forest-inference-evaluation-and-plan.md`,
decision D3). As of 2026-09-25 all three splits are fully finite.

### Row count anomaly

The official EMBER2024 paper reports 606,000 test files and 2,626,000 train
files. The vectorized `X_test.dat` / `X_train.dat` this fixture reads from
actually contain **2x** those counts (1,212,000 and 5,252,000 rows): the
upstream `*_train.zip` / `*_test.zip` archives on
`joyce8/EMBER2024` embed each weekly JSONL file with its lines duplicated
back-to-back (verified byte-identical, e.g.
`2024-09-22_2024-09-28_ELF_test.jsonl` in `ELF_test.zip` has 1,000 lines for
a week the paper's own per-week volume table says should hold 500). The
`challenge` split is not affected (6,315 rows, matching the paper). `id` is
the row ordinal in this (duplicated) split as vectorized, not a dedup key;
`prepare.py` does not deduplicate. Use `--rows 606000` / `--rows 2626000` to
reproduce the paper's row counts for a `test` / `train` Parquet export.

Labels observed in this fixture (the default `label_type="label"` binary
task; `thrember` only writes -1 for classes below `class_min` under a
non-default multiclass `label_type`, which this fixture does not use):
`test` and `train` are exactly balanced 50/50 between label 0 (benign) and
label 1 (malicious); `challenge` is 100% label 1.

## Version Control Policy

Everything under this directory (`*.zip`, `*.jsonl`, `.cache/`, `*.dat`,
`models/`, `parquet/`) is gitignored except `prepare.py`, `prepare.sh`, this
`README.md`, and `.gitkeep`; see the `fixture/ember2024/` block in the
repository root [`.gitignore`](file:///home/ubuntu/algeon/.gitignore).
