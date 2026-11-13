#!/usr/bin/env python3
"""Relational benchmark driver behind the website's TPC-H, TPC-DS and server-load numbers.

One run directory holds one build and every measurement the benchmarks page shows:
TPC-H at SF100 on the sorted layout, SF100 in dbgen order (tpchgen's generation order,
unsorted), SF500 on the sorted layout, TPC-DS SF100, and the server load matrix. Each
stage writes DONE on success or STOP (with reasons) and the driver then exits non-zero.
Re-running a stage skips a stage directory that holds DONE and refuses one that exists
without it, except `server-load`, which resumes through server_load_driver.py.

This driver does not take /tmp/cudf-gpu.lock. bench_one.sh, native_tpcds_parity_report
(through flock) and server_load_driver.py take it for each GPU process, so an outer lock
would deadlock them.

Every subprocess runs without LD_LIBRARY_PATH, so each binary loads the libraries its
RUNPATH names and the recorded library hashes are the ones the measurements loaded.

Prerequisites
  * A clean worktree: the build stage and every measurement stage stop on `git status
    --porcelain` output, on a HEAD change and on a binary or native library sha256 change.
  * The four Parquet directories written by fixture/tpc-h/generate.py and
    fixture/tpc-ds/generate.py, each with its _generation.json, one table per file:
    --tpch-sf100-sorted, --tpch-sf100-dbgen-order (generated with --unsorted),
    --tpch-sf500-sorted and --tpcds-sf100.
  * --host-facts: a JSON file with the hardware description and the TPC-DS `cpuSettings`
    (instanceType, cpu, cpuThreads, memoryGiB, gpu, gpuMemoryGB, gpuDriver, cudaToolkit,
    dataFusionVersion, cpuSettings, and any other fixed fact for the page). It must not
    contain the keys this driver owns: sourceRevision, cudfRevision, treeHash,
    measuredAtUtcDate, worktreeChanges, binarySha256, nativeLibSha256, or the retired keys
    cpuTuning, warmups, iterations and sourceCachePolicy.
  * Python 3 with pyarrow (describe_layout reads the Parquet footers) and the native
    cuDF install under target/native.

Commands (global options first, then the command)
  python3 scripts/dev/relational_benchmarks_driver.py --out-dir target/relational-benchmarks/RUN \\
      --tpch-sf100-sorted DIR --tpch-sf100-dbgen-order DIR --tpch-sf500-sorted DIR \\
      --tpcds-sf100 DIR --host-facts FILE run-all

  build           clean-worktree check, release build of the three bench binaries,
                  algeon_server and algeon_tools; records HEAD, tree hash, cuDF revision,
                  binary sha256, the sha256 of libcudf.so and libcudf_rust.so as the bench
                  binary's RUNPATH resolves them, and a hardware probe
  tpch ID         ID is sf100_sorted, sf100_dbgen_order or sf500_sorted; evicts the other
                  datasets from the page cache, runs the 22 queries one process each with
                  1 warm-up and 5 measured iterations, checks every row as it lands, then
                  writes layout.json (describe_layout) and page-cache.json
  tpcds           parity report, then the CPU and GPU runs of all 99 queries (1 warm-up,
                  5 measured iterations), layout.json and page-cache.json
  server-load     server_load_driver.py stage and run-all on the sf100_sorted directory,
                  then server_load_analyze.py; stops when any run directory holds STOP
  check           the release gate below, printed as a list of reasons; exit 1 when any
  website         check, environment.json, then the three website generators
  run-all         build, tpch sf100_sorted, tpch sf100_dbgen_order, tpcds, tpch
                  sf500_sorted, check, server-load, website (about 6.5 hours of GPU time)

Release gate (a violated condition writes STOP and publishes no data file)
  * The worktree is not clean at the start or end of a stage, or HEAD or a binary sha256
    changed during the run.
  * A TPC-H row has planning_outcome other than selected, runtime_outcome other than
    succeeded, parity_status other than matched, or a sample count other than 5.
  * An output_fingerprint or output row count differs between sf100_sorted and
    sf100_dbgen_order.
  * A TPC-DS parity row is not matched, or a measured timing row did not succeed.
  * describe_layout finds a manifest and footer mismatch, or the manifest is not the
    expected layout and scale factor.
  * A server-load run directory holds STOP.
  * `git diff X HEAD` is not empty over crates, components, fixture, scripts, Cargo.toml
    and Cargo.lock, where X is the measured revision. Any code change means a full re-run.

Output layout under --out-dir
  build/                       build.json, build.log
  tpch/ID/                     qNN.tsv, artifacts/ (sidecars), logs/, layout.json, page-cache.json
  tpcds/                       parity.tsv, cpu.tsv, gpu.tsv, logs/, layout.json, page-cache.json
  server-load/                 server_load_driver.py output directory
  environment.json             written by `website`: host facts plus the keys this driver owns
  server-load-environment.json written by `website` for the server-load generator

Every option except the command also reads an environment variable (RELATIONAL_BENCH_OUT,
RELATIONAL_BENCH_TPCH_SF100_SORTED, RELATIONAL_BENCH_TPCH_SF100_DBGEN_ORDER,
RELATIONAL_BENCH_TPCH_SF500_SORTED, RELATIONAL_BENCH_TPCDS_SF100, RELATIONAL_BENCH_HOST_FACTS).
"""
import argparse, datetime, hashlib, json, os, signal, subprocess, sys
from pathlib import Path

import page_cache

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'target/relational-benchmarks/run'
# cargo writes release binaries under CARGO_TARGET_DIR when it is set.
RELEASE = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target')) / 'release'
BENCH_BINARIES = ('native_tpch_benchmark', 'tpcds_gpu_cpu_bench', 'native_tpcds_parity_report')
SERVER_BINARIES = ('algeon_server', 'algeon_tools')
NATIVE_LIBS = ('libcudf.so', 'libcudf_rust.so')
BUILD_COMMANDS = (
    'cargo build --release -p bench --all-features --bins',
    'cargo build --release -p algeon-server --bin algeon_server',
    'cargo build --release -p tools --features flight-sql-stress-check',
)
CODE_PATHS = ('crates', 'components', 'fixture', 'scripts', 'Cargo.toml', 'Cargo.lock')

WARMUPS = 1
ITERATIONS = 5
TPCH_QUERIES = [f'q{i:02d}' for i in range(1, 23)]
# id: (scale factor, expected manifest layout, extra generate.py flags)
TPCH_DATASETS = {
    'sf100_sorted': (100, 'single_file_per_table_sorted_row_groups', ''),
    'sf100_dbgen_order': (100, 'single_file_per_table_unsorted', ' --unsorted'),
    'sf500_sorted': (500, 'single_file_per_table_sorted_row_groups', ''),
}
TPCDS_LAYOUT = 'single_file_per_table_sorted_row_groups'
TPCDS_QUERY_COUNT = 99
QUERY_TIMEOUT_SECONDS = 3 * 3600
TPCDS_CPU_ENV = {'ALGEON_TPCDS_CPU_TARGET_PARTITIONS': '32', 'ALGEON_TPCDS_CPU_BATCH_SIZE': '32768'}
TPCDS_NATIVE_ENV = {
    'ALGEON_TPCDS_NATIVE_PROFILE': 'sf100_gpu_throughput',
    'ALGEON_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE': 'gpu_native_tolerant',
}
PARITY_TOLERANCE_ENV = {
    'ALGEON_TPCDS_PARITY_FLOAT_ABS_TOLERANCE': '0.00000001',
    'ALGEON_TPCDS_PARITY_FLOAT_REL_TOLERANCE': '0.00000001',
}
TPCH_ENV = {'ALGEON_TPCH_CPU_TARGET_PARTITIONS': '32', 'ALGEON_TPCH_BENCH_NATIVE_PROFILE': 'sf100_gpu_throughput'}
DRIVER_ENVIRONMENT_KEYS = ('sourceRevision', 'cudfRevision', 'treeHash', 'measuredAtUtcDate', 'worktreeChanges',
                           'binarySha256', 'nativeLibSha256')
RETIRED_ENVIRONMENT_KEYS = ('cpuTuning', 'warmups', 'iterations', 'sourceCachePolicy')
REQUIRED_HOST_FACTS = ('instanceType', 'cpu', 'cpuThreads', 'gpu', 'gpuDriver', 'dataFusionVersion', 'cpuSettings')
SERVER_LOAD_CPU_PATH_SETTINGS = {'datafusion.execution.target_partitions': 32, 'datafusion.execution.batch_size': 32768}

OUT = None
DATA = {}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log(msg):
    print(now(), msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


# ---------------------------------------------------------------- stage markers

class Stop(Exception):
    pass


def stage_dir(*parts):
    return OUT.joinpath(*parts)


def begin(directory, resumable=False):
    """Create the stage directory; False when it already holds DONE."""
    if (directory / 'DONE').exists():
        log(f'{directory.relative_to(OUT)}: DONE, skipping')
        return False
    if (directory / 'STOP').exists():
        raise SystemExit(f'{directory} holds STOP:\n{(directory / "STOP").read_text()}Remove the directory to run it again.')
    if directory.exists() and not resumable:
        raise SystemExit(f'{directory} exists without DONE; remove it to run the stage again')
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / 'STARTED').exists():
        (directory / 'STARTED').write_text(now() + '\n')
    return True


def finish(directory, reasons):
    if reasons:
        (directory / 'STOP').write_text(''.join(f'{r}\n' for r in reasons))
        log(f'{directory.relative_to(OUT)}: STOP')
        for reason in reasons:
            log(f'  {reason}')
        raise SystemExit(1)
    (directory / 'DONE').write_text(now() + '\n')
    log(f'{directory.relative_to(OUT)}: DONE')


# ---------------------------------------------------------------- build state

def worktree_state():
    return {
        'head': git('rev-parse', 'HEAD'),
        'treeHash': git('rev-parse', 'HEAD^{tree}'),
        'cudfRevision': git('-C', 'components/cudf/cudf', 'rev-parse', 'HEAD'),
        'dirty': git('status', '--porcelain').splitlines(),
    }


def runpath_env():
    """os.environ without LD_LIBRARY_PATH, so every binary loads the libraries its RUNPATH names.

    target/native/cudf-install/lib can hold an older libcudf_rust.so than the shim the build
    writes under the build output; an LD_LIBRARY_PATH that names that directory would put the
    older shim ahead of the RUNPATH.
    """
    return {key: value for key, value in os.environ.items() if key != 'LD_LIBRARY_PATH'}


def resolved_native_libs():
    """Paths of NATIVE_LIBS as the TPC-H bench binary resolves them through its RUNPATH."""
    text = subprocess.run(['ldd', str(RELEASE / 'native_tpch_benchmark')], env=runpath_env(), capture_output=True,
                          text=True, check=True).stdout
    paths = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in NATIVE_LIBS and parts[1] == '=>':
            paths[parts[0]] = Path(parts[2])
    missing = [name for name in NATIVE_LIBS if name not in paths]
    if missing:
        raise SystemExit(f'ldd of native_tpch_benchmark does not resolve {missing}')
    return paths


def artifact_hashes():
    binaries = {name: sha256(RELEASE / name) for name in (*BENCH_BINARIES, *SERVER_BINARIES)}
    libs = {name: sha256(path) for name, path in resolved_native_libs().items()}
    return binaries, libs


def hardware_probe():
    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    gpu = sh('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits')
    name, driver, memory = ([part.strip() for part in gpu.splitlines()[0].split(',')] + [None] * 3)[:3] if gpu else (None,) * 3
    return {
        'cpuModel': sh("lscpu | sed -n 's/^Model name:[[:space:]]*//p'"),
        'cpuThreads': os.cpu_count(),
        'memTotalKiB': int(page_cache.meminfo()['MemTotal'].split()[0]),
        'gpu': name, 'gpuDriver': driver, 'gpuMemoryMiB': memory,
        'nvcc': sh('/usr/local/cuda/bin/nvcc --version | tail -1'),
        'kernel': sh('uname -r'),
    }


def load_build():
    path = stage_dir('build', 'build.json')
    if not (stage_dir('build', 'DONE')).exists():
        raise SystemExit('the build stage has not finished; run `build` first')
    return json.loads(path.read_text())


def build_drift(build, strict):
    """Reasons the current worktree and artifacts differ from the recorded build.

    `strict` also requires a clean worktree and an unchanged HEAD, as every measurement
    stage does; the release gate instead accepts new commits that leave CODE_PATHS alone.
    """
    reasons = []
    state = worktree_state()
    if strict and state['dirty']:
        reasons.append(f'the worktree is not clean: {state["dirty"][:5]}')
    if strict and state['head'] != build['head']:
        reasons.append(f'HEAD changed from {build["head"]} to {state["head"]}')
    if not strict:
        if state['head'] != build['head'] and subprocess.run(
                ['git', '-C', str(ROOT), 'diff', '--quiet', build['head'], state['head'], '--', *CODE_PATHS]).returncode:
            reasons.append(f'code changed between the measured revision {build["head"]} and {state["head"]}; re-measure everything')
        dirty_code = git('status', '--porcelain', '--', *CODE_PATHS)
        if dirty_code:
            reasons.append(f'uncommitted changes under the code paths: {dirty_code.splitlines()[:5]}')
    binaries, libs = artifact_hashes()
    for kind, now_hashes, then in (('binary', binaries, build['binarySha256']), ('native library', libs, build['nativeLibSha256'])):
        for name, digest in now_hashes.items():
            if then.get(name) != digest:
                reasons.append(f'{kind} {name} sha256 changed since the build stage')
    return reasons


def guard(directory):
    """Stop the stage when the worktree or the build artifacts differ from the build stage."""
    reasons = build_drift(load_build(), strict=True)
    if reasons:
        finish(directory, reasons)


# ---------------------------------------------------------------- build stage

def run_logged(cmd, log_path, env=None, timeout=None, shell=False, stdout_path=None):
    """Run `cmd` in its own process group; stdout and stderr go to `log_path`, or stdout to `stdout_path`.

    Returns the exit code, or 'timeout' after killing the group.
    """
    with open(log_path, 'w') as log_file, open(stdout_path or log_path, 'a' if stdout_path is None else 'w') as out:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=log_file if stdout_path else subprocess.STDOUT,
                                start_new_session=True, shell=shell, executable='/bin/bash' if shell else None)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 'timeout'


def cmd_build():
    directory = stage_dir('build')
    if not begin(directory):
        return
    state = worktree_state()
    if state['dirty']:
        finish(directory, [f'the worktree is not clean: {state["dirty"][:5]}'])
    script = 'set -euo pipefail; [ -f env.sh ] && source env.sh; ' + ' && '.join(BUILD_COMMANDS)
    rc = run_logged(script, directory / 'build.log', env=runpath_env(), shell=True)
    if rc != 0:
        finish(directory, [f'cargo build failed with {rc}; see {directory / "build.log"}'])
    after = worktree_state()
    if after['dirty'] or after['head'] != state['head']:
        finish(directory, [f'the build changed the worktree: {after["dirty"][:5]}'])
    binaries, libs = artifact_hashes()
    (directory / 'build.json').write_text(json.dumps({
        'head': state['head'], 'treeHash': state['treeHash'], 'cudfRevision': state['cudfRevision'],
        'builtAtUtc': now(), 'commands': list(BUILD_COMMANDS),
        'binarySha256': binaries, 'nativeLibSha256': libs, 'hardware': hardware_probe(),
    }, indent=1))
    finish(directory, [])


# ---------------------------------------------------------------- TSV helpers

def read_tsv(path):
    """`# key=value` comments and data rows as dicts."""
    meta, header, rows = {}, None, []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        if line.startswith('#'):
            key, sep, value = line[1:].strip().partition('=')
            if sep:
                meta[key.strip()] = value.strip()
            continue
        cells = line.split('\t')
        if header is None:
            header = cells
        else:
            rows.append(dict(zip(header, cells)))
    return meta, rows


def sample_count(text):
    return len([s for s in (text or '').split(',') if s.strip()])


def check_tpch_query(path, query):
    """Reasons a TPC-H qNN.tsv cannot be published; the row is returned for later checks."""
    if not Path(path).exists():
        return [f'{path} is missing'], None
    meta, rows = read_tsv(path)
    if len(rows) != 1 or rows[0].get('query_id') != query:
        return [f'{path} holds {len(rows)} rows, expected one row for {query}'], None
    row = rows[0]
    reasons = []
    for key, wanted in (('tpch_benchmark_wall_time_boundary', 'sql_planning_and_collect'),
                        ('tpch_benchmark_wall_time_fields', 'per_measured_iteration_median')):
        if meta.get(key) != wanted:
            reasons.append(f'{query}: {key}={meta.get(key)}, expected {wanted}; the bench binary predates the planning boundary')
    if meta.get('tpch_benchmark_warmups') != str(WARMUPS) or meta.get('tpch_benchmark_iterations') != str(ITERATIONS):
        reasons.append(f'{query}: warmups={meta.get("tpch_benchmark_warmups")} iterations={meta.get("tpch_benchmark_iterations")}, '
                       f'expected {WARMUPS} and {ITERATIONS}')
    for column, wanted in (('planning_outcome', 'selected'), ('runtime_outcome', 'succeeded'), ('parity_status', 'matched')):
        if row.get(column) != wanted:
            reasons.append(f'{query}: {column}={row.get(column)}, expected {wanted}')
    for column in ('cpu_wall_ms_samples', 'native_wall_ms_samples'):
        if sample_count(row.get(column)) != ITERATIONS:
            reasons.append(f'{query}: {column} holds {sample_count(row.get(column))} samples, expected {ITERATIONS}')
    return reasons, row


def check_tpch_dataset(dataset_id):
    directory = stage_dir('tpch', dataset_id)
    reasons, rows = [], {}
    for query in TPCH_QUERIES:
        query_reasons, row = check_tpch_query(directory / f'{query}.tsv', query)
        reasons += query_reasons
        if row is not None:
            rows[query] = row
    scale, layout_name, _ = TPCH_DATASETS[dataset_id]
    reasons += check_layout_file(directory / 'layout.json', layout_name, scale)
    return reasons, rows


def check_layout_file(path, layout_name, scale):
    if not Path(path).exists():
        return [f'{path} is missing']
    generation = json.loads(Path(path).read_text())['generation']
    reasons = []
    if generation.get('layout') != layout_name:
        reasons.append(f'{path}: manifest layout {generation.get("layout")}, expected {layout_name}')
    if scale is not None and generation.get('scale_factor') != scale:
        reasons.append(f'{path}: manifest scale_factor {generation.get("scale_factor")}, expected {scale}')
    return reasons


def check_dbgen_order_matches_sorted(sorted_rows, dbgen_rows):
    reasons = []
    for query in TPCH_QUERIES:
        a, b = sorted_rows.get(query), dbgen_rows.get(query)
        if a is None or b is None:
            continue
        for column in ('output_fingerprint', 'cpu_output_rows', 'native_output_rows'):
            if a.get(column) != b.get(column):
                reasons.append(f'{query}: {column} differs between sf100_sorted ({a.get(column)}) and sf100_dbgen_order ({b.get(column)})')
    return reasons


def check_tpcds(directory):
    reasons = []
    for name in ('parity.tsv', 'cpu.tsv', 'gpu.tsv', 'layout.json'):
        if not (directory / name).exists():
            reasons.append(f'{directory / name} is missing')
    if reasons:
        return reasons
    _, parity = read_tsv(directory / 'parity.tsv')
    if len(parity) != TPCDS_QUERY_COUNT:
        reasons.append(f'parity.tsv holds {len(parity)} queries, expected {TPCDS_QUERY_COUNT}')
    for row in parity:
        if row.get('parity_status') != 'matched':
            reasons.append(f'{row.get("query_id")}: parity_status={row.get("parity_status")}')
    for name in ('cpu.tsv', 'gpu.tsv'):
        meta, rows = read_tsv(directory / name)
        if meta.get('tpcds_benchmark_wall_time_boundary') != 'sql_planning_and_collect':
            reasons.append(f'{name}: tpcds_benchmark_wall_time_boundary={meta.get("tpcds_benchmark_wall_time_boundary")}')
        measured = {}
        for row in rows:
            if row.get('warmup') != 'true':
                measured.setdefault(row['query_id'], []).append(row)
        if len(measured) != TPCDS_QUERY_COUNT:
            reasons.append(f'{name} holds {len(measured)} queries, expected {TPCDS_QUERY_COUNT}')
        for query, query_rows in measured.items():
            if len(query_rows) != ITERATIONS:
                reasons.append(f'{name} {query}: {len(query_rows)} measured rows, expected {ITERATIONS}')
            for row in query_rows:
                if row.get('outcome') != 'succeeded':
                    reasons.append(f'{name} {query} iteration {row.get("iteration")}: outcome={row.get("outcome")}')
    return reasons + check_layout_file(directory / 'layout.json', TPCDS_LAYOUT, None)


def check_server_load(directory):
    stops = sorted(directory.rglob('STOP'))
    return [f'{stop.parent} holds STOP' for stop in stops]


def check_all():
    """Every release-gate reason over the run directory; empty when the data may be published."""
    reasons = []
    if not stage_dir('build', 'DONE').exists():
        return ['the build stage has not finished']
    reasons += build_drift(load_build(), strict=False)
    rows = {}
    for dataset_id in TPCH_DATASETS:
        if not stage_dir('tpch', dataset_id, 'DONE').exists():
            reasons.append(f'tpch/{dataset_id} has not finished')
            continue
        dataset_reasons, rows[dataset_id] = check_tpch_dataset(dataset_id)
        reasons += dataset_reasons
    if 'sf100_sorted' in rows and 'sf100_dbgen_order' in rows:
        reasons += check_dbgen_order_matches_sorted(rows['sf100_sorted'], rows['sf100_dbgen_order'])
    if stage_dir('tpcds', 'DONE').exists():
        reasons += check_tpcds(stage_dir('tpcds'))
    else:
        reasons.append('tpcds has not finished')
    if stage_dir('server-load', 'DONE').exists():
        reasons += check_server_load(stage_dir('server-load'))
    else:
        reasons.append('server-load has not finished')
    return reasons


def cmd_check():
    reasons = check_all()
    for reason in reasons:
        print(reason)
    if reasons:
        raise SystemExit(1)
    log('check passed')


# ---------------------------------------------------------------- page cache

def residency_summary(data_dir):
    r = page_cache.residency(data_dir)
    return {'total_bytes': r['total_bytes'], 'resident_fraction_by_bytes': r['resident_fraction_by_bytes'],
            'min_file_fraction': min(v['resident_fraction'] or 0 for v in r['files'].values())}


def evict_others(current):
    others = [str(path) for path in DATA.values() if path is not None and Path(path) != Path(current)]
    for path in others:
        page_cache.evict(path)
    return others


def measure_page_cache(directory, data_dir, body):
    evicted = evict_others(data_dir)
    record = {'evicted': evicted, 'meminfoBefore': page_cache.meminfo(), 'before': residency_summary(data_dir)}
    body()
    record['after'] = residency_summary(data_dir)
    record['meminfoAfter'] = page_cache.meminfo()
    (directory / 'page-cache.json').write_text(json.dumps(record, indent=1))


def write_layout(directory, data_dir):
    cmd = [sys.executable, str(ROOT / 'fixture/tpc/describe_layout.py'), str(data_dir), '--out', str(directory / 'layout.json')]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return [f'describe_layout {data_dir} failed: {r.stderr.strip().splitlines()[-1:]}']
    return []


# ---------------------------------------------------------------- measurement stages

def bench_env(extra):
    env = runpath_env()
    env.update(extra)
    return env


def cmd_tpch(dataset_id):
    scale, _, flags = TPCH_DATASETS[dataset_id]
    data = DATA[dataset_id]
    if data is None:
        raise SystemExit(f'--tpch-{dataset_id.replace("_", "-")} is required for tpch {dataset_id}')
    directory = stage_dir('tpch', dataset_id)
    if not begin(directory):
        return
    guard(directory)
    (directory / 'logs').mkdir()
    env = bench_env({
        **TPCH_ENV,
        'ALGEON_TPCH_BENCH_ARTIFACT_DIR': str(directory / 'artifacts'),
        'ALGEON_TPCH_SCALE_FACTOR': f'sf{scale}',
        'ALGEON_TPCH_GENERATION_COMMAND': f'python3 fixture/tpc-h/generate.py --scale {scale}{flags} --out {data}',
    })
    (directory / 'artifacts').mkdir()

    def body():
        for query in TPCH_QUERIES:
            tsv = directory / f'{query}.tsv'
            cmd = ['bash', 'scripts/dev/bench_one.sh', 'tpch', query, '--scale', str(scale), '--warmups', str(WARMUPS),
                   '--iterations', str(ITERATIONS), '--parquet-dir', str(data), '--out', str(tsv), '--no-review']
            log(f'tpch {dataset_id} {query}')
            rc = run_logged(cmd, directory / 'logs' / f'{query}.log', env=env, timeout=QUERY_TIMEOUT_SECONDS)
            if rc != 0:
                finish(directory, [f'{query}: bench_one.sh returned {rc}; see {directory / "logs" / f"{query}.log"}'])
            reasons, _ = check_tpch_query(tsv, query)
            if reasons:
                finish(directory, reasons)

    measure_page_cache(directory, data, body)
    reasons = write_layout(directory, data)
    if not reasons:
        reasons = check_layout_file(directory / 'layout.json', TPCH_DATASETS[dataset_id][1], scale)
    finish(directory, reasons + build_drift(load_build(), strict=True))


def cmd_tpcds():
    data = DATA['tpcds_sf100']
    if data is None:
        raise SystemExit('--tpcds-sf100 is required for tpcds')
    directory = stage_dir('tpcds')
    if not begin(directory):
        return
    guard(directory)
    (directory / 'logs').mkdir()
    env = bench_env({
        **TPCDS_CPU_ENV, **TPCDS_NATIVE_ENV, **PARITY_TOLERANCE_ENV,
        'ALGEON_TPCDS_SCALE_FACTOR': 'sf100',
        'ALGEON_TPCDS_GENERATION_COMMAND': f'python3 fixture/tpc-ds/generate.py --scale 100 --out {data}',
        'ALGEON_PARQUET_DIR': str(data),
    })

    def body():
        log('tpcds parity report')
        parity = ['flock', '/tmp/cudf-gpu.lock', 'cargo', 'run', '--release', '-p', 'bench', '--all-features', '--bin',
                  'native_tpcds_parity_report']
        rc = run_logged(parity, directory / 'logs/parity.log', env=env, timeout=QUERY_TIMEOUT_SECONDS,
                        stdout_path=directory / 'parity.tsv')
        if rc != 0:
            finish(directory, [f'native_tpcds_parity_report returned {rc}'])
        for engine, name in (('cpu_datafusion', 'cpu'), ('native_gpu', 'gpu')):
            log(f'tpcds {engine}')
            cmd = ['bash', 'scripts/dev/bench_one.sh', 'tpcds', 'all', '--engine', engine, '--scale', '100',
                   '--warmups', str(WARMUPS), '--iterations', str(ITERATIONS), '--parquet-dir', str(data),
                   '--out', str(directory / f'{name}.tsv'), '--no-review']
            rc = run_logged(cmd, directory / 'logs' / f'{name}.log', env=env, timeout=QUERY_TIMEOUT_SECONDS)
            if rc != 0:
                finish(directory, [f'tpcds {engine} returned {rc}; see {directory / "logs" / f"{name}.log"}'])

    measure_page_cache(directory, data, body)
    reasons = write_layout(directory, data)
    if not reasons:
        reasons = check_tpcds(directory)
    finish(directory, reasons + build_drift(load_build(), strict=True))


def cmd_server_load():
    data = DATA['sf100_sorted']
    if data is None:
        raise SystemExit('--tpch-sf100-sorted is required for server-load')
    directory = stage_dir('server-load')
    if not begin(directory, resumable=True):
        return
    guard(directory)
    evict_others(data)
    driver = [sys.executable, str(ROOT / 'scripts/dev/server_load_driver.py'), '--out-dir', str(directory), '--data', str(data),
              '--release-dir', str(RELEASE)]
    for step in (['stage'], ['run-all']):
        rc = subprocess.run([*driver, *step], cwd=ROOT, env=runpath_env()).returncode
        if rc != 0:
            finish(directory, [f'server_load_driver.py {step[0]} returned {rc}'] + check_server_load(directory))
    rc = subprocess.run([sys.executable, str(ROOT / 'scripts/dev/server_load_analyze.py'), '--out-dir', str(directory)], cwd=ROOT).returncode
    reasons = ([f'server_load_analyze.py returned {rc}'] if rc else []) + check_server_load(directory) + build_drift(load_build(), strict=True)
    finish(directory, reasons)


# ---------------------------------------------------------------- website stage

def load_host_facts(path):
    if path is None:
        raise SystemExit('--host-facts is required')
    facts = json.loads(Path(path).read_text())
    owned = [key for key in (*DRIVER_ENVIRONMENT_KEYS, *RETIRED_ENVIRONMENT_KEYS) if key in facts]
    if owned:
        raise SystemExit(f'{path} contains {owned}; the driver writes the first group and the page no longer carries the retired keys')
    missing = [key for key in REQUIRED_HOST_FACTS if key not in facts]
    if missing:
        raise SystemExit(f'{path} is missing {missing}')
    return facts


def hardware_mismatches(facts, probe):
    reasons = []
    for fact, measured in (('cpuThreads', probe['cpuThreads']), ('gpu', probe['gpu']), ('gpuDriver', probe['gpuDriver'])):
        if facts[fact] != measured:
            reasons.append(f'host facts {fact}={facts[fact]!r} but this host reports {measured!r}')
    return reasons


def measured_date():
    stages = [stage_dir('tpch', dataset_id) for dataset_id in TPCH_DATASETS] + [stage_dir('tpcds')]
    return min((stage / 'STARTED').read_text().strip() for stage in stages)[:10]


def server_load_environment(facts, environment):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import server_load_driver
    return {
        **{key: facts[key] for key in ('instanceType', 'cpu', 'cpuThreads', 'memoryGiB', 'gpu', 'gpuMemoryGB', 'gpuDriver',
                                       'cudaToolkit', 'dataFusionVersion') if key in facts},
        'measuredDate': environment['measuredAtUtcDate'],
        'sourceRevision': environment['sourceRevision'],
        'serverBinarySha256': environment['binarySha256']['algeon_server'],
        'serverBuild': 'cargo build --release -p algeon-server --bin algeon_server',
        'serverCpuThreads': len(server_load_driver.cpu_set(server_load_driver.DEFAULT_SERVER_CPUS)),
        'clientCpuThreads': len(server_load_driver.cpu_set(server_load_driver.DEFAULT_CLIENT_CPUS)),
        'cpuPathSettings': SERVER_LOAD_CPU_PATH_SETTINGS,
    }


def generator_commands(environment_path, server_environment_path, data_dir):
    scripts = ROOT / 'website/scripts'
    sl = stage_dir('server-load')
    profiles = [arg for name in ('gpu-a1', 'gpu-a2', 'gpu-a3', 'cpu') for arg in ('--profile', str(sl / 'configs' / f'{name}.toml'))]
    py = sys.executable
    return [
        [py, str(scripts / 'generate_tpch_benchmark_ts.py'), '--run-dir', str(OUT), '--out', str(data_dir / 'tpchBenchmark.ts')],
        [py, str(scripts / 'generate_tpcds_sf100_benchmark_ts.py'),
         '--cpu-tsv', str(stage_dir('tpcds', 'cpu.tsv')), '--gpu-tsv', str(stage_dir('tpcds', 'gpu.tsv')),
         '--parity-tsv', str(stage_dir('tpcds', 'parity.tsv')), '--environment', str(environment_path),
         '--layout', str(stage_dir('tpcds', 'layout.json')), '--page-cache', str(stage_dir('tpcds', 'page-cache.json')),
         '--out', str(data_dir / 'tpcdsSf100Benchmark.ts')],
        [py, str(scripts / 'generate_server_load_benchmark_ts.py'), *profiles,
         '--streams-root', str(sl / 'streams'), '--overload-root', str(sl / 'overload'), '--reference', str(sl / 'reference/run'),
         '--environment', str(server_environment_path), '--out', str(data_dir / 'serverLoadBenchmark.ts')],
    ]


def cmd_website(host_facts_path):
    facts = load_host_facts(host_facts_path)
    directory = stage_dir('website')
    if not begin(directory):
        return
    reasons = check_all()
    if reasons:
        finish(directory, reasons)
    build = load_build()
    reasons = build_drift(build, strict=True) + hardware_mismatches(facts, build['hardware'])
    if reasons:
        finish(directory, reasons)
    environment = {**facts, 'sourceRevision': build['head'], 'cudfRevision': build['cudfRevision'], 'treeHash': build['treeHash'],
                   'measuredAtUtcDate': measured_date(), 'worktreeChanges': False,
                   'binarySha256': build['binarySha256'], 'nativeLibSha256': build['nativeLibSha256']}
    environment_path = OUT / 'environment.json'
    environment_path.write_text(json.dumps(environment, indent=1))
    server_environment_path = OUT / 'server-load-environment.json'
    server_environment_path.write_text(json.dumps(server_load_environment(facts, environment), indent=1))
    for cmd in generator_commands(environment_path, server_environment_path, ROOT / 'website/src/data'):
        log(' '.join(cmd))
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            finish(directory, [f'{Path(cmd[1]).name} returned {rc}'])
    finish(directory, [])


# ---------------------------------------------------------------- command line

def env_default(name, default=None):
    return os.environ.get(name, default if default is None else str(default))


def configure(args):
    global OUT, DATA
    OUT = Path(args.out_dir).resolve()
    DATA = {'sf100_sorted': args.tpch_sf100_sorted, 'sf100_dbgen_order': args.tpch_sf100_dbgen_order,
            'sf500_sorted': args.tpch_sf500_sorted, 'tpcds_sf100': args.tpcds_sf100}
    DATA = {key: (Path(value).absolute() if value else None) for key, value in DATA.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out-dir', default=env_default('RELATIONAL_BENCH_OUT', DEFAULT_OUT), help='run directory (default: %(default)s)')
    ap.add_argument('--tpch-sf100-sorted', default=env_default('RELATIONAL_BENCH_TPCH_SF100_SORTED'), help='TPC-H SF100 sorted Parquet directory')
    ap.add_argument('--tpch-sf100-dbgen-order', default=env_default('RELATIONAL_BENCH_TPCH_SF100_DBGEN_ORDER'),
                    help='TPC-H SF100 Parquet directory in dbgen order (generated with --unsorted)')
    ap.add_argument('--tpch-sf500-sorted', default=env_default('RELATIONAL_BENCH_TPCH_SF500_SORTED'), help='TPC-H SF500 sorted Parquet directory')
    ap.add_argument('--tpcds-sf100', default=env_default('RELATIONAL_BENCH_TPCDS_SF100'), help='TPC-DS SF100 Parquet directory')
    ap.add_argument('--host-facts', default=env_default('RELATIONAL_BENCH_HOST_FACTS'), help='JSON file with the hardware description and cpuSettings')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, text in (('build', 'clean-worktree check and release build, with hashes and a hardware probe'),
                       ('tpcds', 'parity report and the CPU and GPU runs of all 99 queries'),
                       ('server-load', 'server_load_driver.py stage and run-all, then server_load_analyze.py'),
                       ('check', 'print every release-gate reason; exit 1 when any'),
                       ('website', 'check, write environment.json and run the three website generators'),
                       ('run-all', 'build, the four dataset stages, check, server-load and website')):
        sub.add_parser(name, help=text)
    t = sub.add_parser('tpch', help='TPC-H, one query per process')
    t.add_argument('dataset', choices=sorted(TPCH_DATASETS))
    a = ap.parse_args()
    configure(a)
    if a.cmd == 'build':
        cmd_build()
    elif a.cmd == 'tpch':
        cmd_tpch(a.dataset)
    elif a.cmd == 'tpcds':
        cmd_tpcds()
    elif a.cmd == 'server-load':
        cmd_server_load()
    elif a.cmd == 'check':
        cmd_check()
    elif a.cmd == 'website':
        cmd_website(a.host_facts)
    elif a.cmd == 'run-all':
        missing = [name for name, path in (*DATA.items(), ('host_facts', a.host_facts)) if not path]
        if missing:
            raise SystemExit(f'run-all needs all dataset directories and --host-facts; missing {missing}')
        load_host_facts(a.host_facts)
        cmd_build()
        for dataset_id in ('sf100_sorted', 'sf100_dbgen_order'):
            cmd_tpch(dataset_id)
        cmd_tpcds()
        cmd_tpch('sf500_sorted')
        cmd_server_load()
        cmd_website(a.host_facts)
        log('ALL_DONE')


if __name__ == '__main__':
    main()
