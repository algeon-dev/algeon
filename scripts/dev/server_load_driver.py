#!/usr/bin/env python3
"""Server load benchmark driver behind the website's server-load numbers.

Runs `algeon_server` under a fresh process per measurement and drives it with
`algeon_tools flight-sql-stress` over loopback: a CPU reference pass, a
closed-loop streams matrix (GPU A=1, A=2, A=3 and the CPU path at 1, 2, 4, 8
and 16 streams, three repeats each) and open-loop overload runs at 0.5, 1.0,
1.5 and 2.0 times the A=2 capacity. Every run directory gets DONE on success
or STOP (with reasons) on a stop condition, and the driver then exits
non-zero. The run needs the GPU, so it takes /tmp/cudf-gpu.lock for each
server lifetime.

Prerequisites
  * An SF100 TPC-H Parquet directory (--data) with the eight table files.
  * Release binaries from the revision under test, with a tools build that
    reads `tpch_query` and `tpch_scale_factor` in run profiles:
      cargo build --release -p algeon-server --bin algeon_server
      cargo build --release -p tools --features flight-sql-stress-check
  * Python 3 with pyarrow, and the native cuDF install under target/native.
  * Enough free page cache for the Parquet files; the driver warms it and
    re-warms before every run that finds a file below 99.9 percent resident.
  * A host where --server-cpus and --client-cpus name disjoint logical CPUs.
    The defaults are the 32-vCPU g7e.8xlarge layout of the published runs
    (hyperthread siblings i and i+16; the client gets physical cores 14-15).

Full matrix (about 4.5 hours of GPU time: 60 stream runs plus 12 overload runs)
  python3 scripts/dev/server_load_driver.py --out-dir target/server-load/RUN stage
  python3 scripts/dev/server_load_driver.py --out-dir target/server-load/RUN run-all
  python3 scripts/dev/server_load_analyze.py --out-dir target/server-load/RUN

Re-running a command skips run directories that already hold DONE and refuses
one that exists without it. `run-all` is `configs`, `env`, `warm`, `reference`,
`matrix` and `overload-all` in order.

Website data
  scripts/dev/relational_benchmarks_driver.py runs this driver in its `server-load`
  stage and builds website-environment.json and the generator call in its `website`
  stage, so the server-load numbers share one build and one environment file with the
  TPC-H and TPC-DS numbers. Run this driver by itself only for experiments.

Output layout under --out-dir
  bin/                  staged algeon_server, algeon_tools and bin/lib shim libraries
  configs/              per-profile server TOML, effective TOML and the init SQL
  reference/run         CPU reference run; reference/arrow holds the recorded results
  streams/PROFILE/cN/rR closed-loop runs
  overload/rR/LABEL     open-loop runs; overload/capacity.json names C
  environment.json, page-cache-warm.json

Every option except the subcommand also reads an environment variable
(SERVER_LOAD_OUT, SERVER_LOAD_DATA, SERVER_LOAD_RELEASE_DIR,
SERVER_LOAD_SERVER_CPUS, SERVER_LOAD_CLIENT_CPUS).
"""
import argparse, datetime, fcntl, hashlib, json, os, re, shutil, signal, subprocess, time
from pathlib import Path

import page_cache

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'crates/tools/fixtures/stress'
DEFAULT_OUT = ROOT / 'target/server-load/run'
DEFAULT_DATA = ROOT / 'fixture/tpc-h/data/sf100/parquet'
DEFAULT_RELEASE_DIR = ROOT / 'target/release'
DEFAULT_SERVER_CPUS = '0-13,16-29'
DEFAULT_CLIENT_CPUS = '14,15,30,31'
SHIM_LIBS = ('libcudf_rust.so', 'libcudf_rust_rmm_interop.so')
GPU_LOCK = '/tmp/cudf-gpu.lock'
URL = 'http://127.0.0.1:50051'
GRPC = 'grpc://127.0.0.1:50051'
SEED = 20261013
CPU_ENV = {'DATAFUSION_EXECUTION_TARGET_PARTITIONS': '32', 'DATAFUSION_EXECUTION_BATCH_SIZE': '32768'}
TOL = {'float_abs_tolerance': 1e-8, 'float_rel_tolerance': 1e-8}
PROFILES = {'gpu-a1': 'gpu', 'gpu-a2': 'gpu', 'gpu-a3': 'gpu', 'cpu': 'cpu'}
EXPECTED_A = {'gpu-a1': 1, 'gpu-a2': 2, 'gpu-a3': 3}
DF_SETTINGS_SQL = ("SELECT name, value FROM information_schema.df_settings WHERE name IN "
                   "('datafusion.execution.target_partitions','datafusion.execution.batch_size') ORDER BY name")
SUMMARY_FIELDS = ['query_id', 'outcome', 'sealed', 'elapsed_ms', 'attempt_disposition', 'attempt_outcome',
                  'attempt_cleanup_disposition', 'attempt_cleanup_failure_count',
                  'attempt_cleanup_failure_count_complete', 'attempt_capacity_disposition',
                  'attempt_health_transition', 'attempt_retry_recovery', 'incomplete', 'unavailable_reason']

# Set by configure() from the command line before any subcommand runs.
OUT = BIN_SERVER = BIN_TOOLS = DATA = INIT_SQL = REFERENCE = LIB_PATH = None
SERVER_CPUS, CLIENT_CPUS = DEFAULT_SERVER_CPUS, DEFAULT_CLIENT_CPUS


def configure(out, data, server_cpus, client_cpus):
    global OUT, BIN_SERVER, BIN_TOOLS, DATA, INIT_SQL, REFERENCE, LIB_PATH, SERVER_CPUS, CLIENT_CPUS
    OUT, DATA = Path(out).resolve(), Path(data).absolute()
    BIN_SERVER, BIN_TOOLS = OUT / 'bin/algeon_server', OUT / 'bin/algeon_tools'
    INIT_SQL = OUT / 'configs/tpch-sf100-spark-init.sql'
    REFERENCE = OUT / 'reference/arrow'
    # The binaries' RUNPATH names target/release/build/.../out/lib, which a later build can
    # replace, and an environment that puts target/native/cudf-install/lib first can supply an
    # older libcudf_rust.so. The run uses the shim libraries staged next to the binaries.
    LIB_PATH = f"{OUT / 'bin/lib'}:{ROOT / 'target/native/cudf-install/lib'}"
    os.environ['LD_LIBRARY_PATH'] = LIB_PATH
    SERVER_CPUS, CLIENT_CPUS = server_cpus, client_cpus
    os.sched_setaffinity(0, cpu_set(client_cpus))


def cpu_set(spec):
    cpus = set()
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


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


def base_env(kind):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ALGEON_SERVER_', 'DATAFUSION_'))}
    if kind == 'cpu':
        env.update(CPU_ENV)
    return env


# ---------------------------------------------------------------- configs

def write_configs():
    d = OUT / 'configs'; d.mkdir(exist_ok=True)
    init = (FIXTURES / 'tpch-sf100-init.sql').read_text()
    spark_init = init.replace("'fixture/tpc-h/data/sf100/parquet/", f"'{DATA}/")
    assert spark_init.count(str(DATA)) == 8
    INIT_SQL.write_text(spark_init)
    shipped_text = (FIXTURES / 'tpch-sf100-server.toml').read_text()
    shipped = shipped_text.replace('init_sql = "crates/tools/fixtures/stress/tpch-sf100-init.sql"', f'init_sql = "{INIT_SQL}"')
    assert shipped != shipped_text
    # A=2 is the shipped file with only init_sql replaced (derived admission defaults).
    (d / 'gpu-a2.toml').write_text(shipped)
    for a in (1, 3):
        text = shipped.replace('\n[native]\n',
                               f'\n[[admission.device_profiles]]\ndevice_ordinal = 0\nmax_active_attempts = {a}\n\n[native]\n')
        assert text != shipped
        (d / f'gpu-a{a}.toml').write_text(text)
    cpu = shipped.replace('"native_required"', '"functions_only"')
    assert cpu != shipped
    (d / 'cpu.toml').write_text(cpu)
    for p in PROFILES:
        cfg = d / f'{p}.toml'
        r = subprocess.run([str(BIN_SERVER), '--check-config', '--config', str(cfg)], cwd=ROOT,
                           env=base_env(PROFILES[p]), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f'--check-config failed for {p}: {r.stderr}')
        (d / f'{p}.effective.toml').write_text(r.stdout)


def statements():
    return [{'name': f'q{i:02d}', 'tpch_query': f'q{i:02d}'} for i in range(1, 23)]


def stream_profile(name, cfg, clients, passes, warmup_passes, results):
    return {
        'schema': 'algeon_datafusion.flight_sql_stress.run_profile', 'name': name,
        'server': {'url': URL, 'config_file': str(cfg)},
        'tpch_scale_factor': 'sf100',
        'statements': statements(),
        'load': {'arrival_mode': 'streams', 'clients': clients, 'seed': SEED, 'passes': passes,
                 'warmup_passes': warmup_passes, 'connection_mode': 'pooled'},
        'admission_sampling': {'interval_ms': 1000},
        'results': results,
    }


def rate_profile(name, cfg, rate):
    return {
        'schema': 'algeon_datafusion.flight_sql_stress.run_profile', 'name': name,
        'server': {'url': URL, 'config_file': str(cfg)},
        'tpch_scale_factor': 'sf100',
        'statements': statements(),
        'load': {'arrival_mode': 'rate', 'clients': 256, 'seed': SEED, 'warmup_seconds': 30,
                 'measurement_seconds': 300, 'rate_per_second': rate, 'connection_mode': 'pooled'},
        'admission_sampling': {'interval_ms': 1000},
        'results': {'check_reference': str(REFERENCE), **TOL},
    }


# ---------------------------------------------------------------- page cache

def warm(dest):
    before = {'meminfo': page_cache.meminfo(), 'residency': page_cache.residency(DATA)}
    t = time.perf_counter()
    for f in sorted(DATA.glob('*.parquet')):
        with open(f, 'rb') as fh:
            while fh.read(16 << 20):
                pass
    elapsed = time.perf_counter() - t
    after = {'meminfo': page_cache.meminfo(), 'residency': page_cache.residency(DATA)}
    rec = {'utc': now(), 'warm_seconds': elapsed, 'before': before, 'after': after}
    Path(dest).write_text(json.dumps(rec, indent=1))
    log(f'warmed page cache in {elapsed:.1f}s: {before["residency"]["resident_fraction_by_bytes"]:.4f} -> '
        f'{after["residency"]["resident_fraction_by_bytes"]:.4f}')
    return rec


def ensure_warm(run_dir):
    r = page_cache.residency(DATA)
    rec = {'utc': now(), 'meminfo': page_cache.meminfo(), 'resident_fraction_by_bytes': r['resident_fraction_by_bytes'],
           'min_file_fraction': min(v['resident_fraction'] for v in r['files'].values())}
    if rec['min_file_fraction'] < 0.999:
        rec['rewarm'] = warm(run_dir / 'page-cache-rewarm.json')['after']['residency']['resident_fraction_by_bytes']
    (run_dir / 'page-cache-before.json').write_text(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- server

def flight_action(name, body=b'', timeout=30):
    import pyarrow.flight as flight
    c = flight.FlightClient(GRPC)
    try:
        return b''.join(bytes(x.body) for x in c.do_action(flight.Action(name, body), flight.FlightCallOptions(timeout=timeout)))
    finally:
        c.close()


def headers(text):
    return dict(line[2:].split('=', 1) for line in text.splitlines() if line.startswith('# ') and '=' in line)


class Server:
    def __init__(self, profile, run_dir):
        self.profile, self.run_dir, self.kind = profile, run_dir, PROFILES[profile]
        self.cfg = OUT / 'configs' / f'{profile}.toml'

    def __enter__(self):
        self.lock = open(GPU_LOCK, 'a')
        log(f'{self.run_dir.relative_to(OUT)}: waiting for GPU lock')
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        env = base_env(self.kind)
        (self.run_dir / 'server-env.json').write_text(json.dumps(
            {k: v for k, v in env.items() if k.startswith(('DATAFUSION_', 'ALGEON_', 'LD_LIBRARY', 'KVIKIO', 'MALLOC', 'CUDA', 'RUST'))}, indent=1))
        cmd = ['taskset', '-c', SERVER_CPUS, str(BIN_SERVER), '--config', str(self.cfg)]
        self.cmd = cmd
        self.slog = open(self.run_dir / 'server.log', 'w')
        self.started = time.monotonic()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=self.slog, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 300
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f'server exited {self.proc.returncode} during startup')
            if 'starting Algeon Flight SQL server' in (self.run_dir / 'server.log').read_text(errors='replace'):
                try:
                    flight_action('GetAdmissionStats', timeout=2); break
                except Exception:
                    pass
            if time.monotonic() > deadline:
                raise RuntimeError('server not ready after 300 s')
            time.sleep(0.2)
        self.ready_seconds = time.monotonic() - self.started
        return self

    def proc_status(self):
        try:
            # taskset execs the server, so the child pid is the server pid.
            st = Path(f'/proc/{self.proc.pid}/status').read_text()
            return {k: v.strip() for k, v in (l.split(':', 1) for l in st.splitlines()) if k in ('VmHWM', 'VmRSS', 'RssAnon', 'Threads', 'Cpus_allowed_list')}
        except Exception as e:
            return {'error': str(e)}

    def __exit__(self, *exc):
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait()
            self.rc = self.proc.returncode
            log(f'{self.run_dir.relative_to(OUT)}: server stopped rc={self.rc}')
        finally:
            self.slog.close()
            fcntl.flock(self.lock, fcntl.LOCK_UN); self.lock.close()


def client_cmd(*args):
    return ['taskset', '-c', CLIENT_CPUS, str(BIN_TOOLS), *args]


def collect_query_summaries(run_dir):
    text = (run_dir / 'server.log').read_text(errors='replace')
    ids = list(dict.fromkeys(re.findall(r'finished Flight SQL statement query_id="([0-9a-f-]+)"', text)))
    rows = []
    for qid in ids:
        try:
            h = {}
            for _ in range(50):
                h = headers(flight_action('GetQueryStatsSummary', qid.encode(), timeout=10).decode(errors='replace'))
                if h.get('sealed') == 'true' or h.get('unavailable_reason'):
                    break
                time.sleep(0.1)
        except Exception as e:
            h = {'query_id': qid, 'error': f'{type(e).__name__}: {str(e)[:200]}'}
        h.setdefault('query_id', qid)
        rows.append(h)
    with open(run_dir / 'query-summaries.tsv', 'w') as f:
        f.write('\t'.join(SUMMARY_FIELDS + ['error']) + '\n')
        for h in rows:
            f.write('\t'.join(str(h.get(k, '')).replace('\t', ' ').replace('\n', ' ') for k in SUMMARY_FIELDS + ['error']) + '\n')
    return rows


# ---------------------------------------------------------------- checks

def check_run(run_dir, kind, mode, profile):
    reasons, notes = [], []
    s = json.loads((run_dir / 'client/summary.json').read_text())
    ov = s['overall']
    if ov.get('cancelled', 0):
        reasons.append(f'cancelled={ov["cancelled"]}')
    if mode == 'streams' and ov.get('failed', 0):
        reasons.append(f'failed={ov["failed"]} errors={ov.get("errors")}')
    # Every failed attempt (both phases) must be an admission rejection in rate mode, none in streams mode.
    lines = (run_dir / 'client/attempts.tsv').read_text().splitlines()
    hdr = lines[0].split('\t'); rows = [dict(zip(hdr, l.split('\t'))) for l in lines[1:]]
    bad = [r for r in rows if r['outcome'] != 'succeeded' and not (mode == 'rate' and r['outcome'] == 'failed' and r['error_code'] == 'service_overloaded')]
    if bad:
        reasons.append(f'{len(bad)} non-succeeded attempts that are not service_overloaded: '
                       + '; '.join(f"{r['statement']}/{r['phase']}/{r['outcome']}/{r['error_class']}/{r['error_code']}/{r['error_reason']}" for r in bad[:10]))
    res = s.get('results') or {}
    ch = res.get('checks_all_phases') or {}
    if ch.get('mismatch', 0) or ch.get('error', 0):
        mm = [f"{r['statement']}/{r['phase']}/{r['check']}" for r in rows if r['check'] in ('mismatch', 'error')]
        reasons.append(f'checks mismatch={ch.get("mismatch")} error={ch.get("error")}: {mm[:20]}')
    if res.get('unchecked_successes', 0):
        reasons.append(f'unchecked_successes={res["unchecked_successes"]}')
    adm = s.get('admission') or {}
    if adm.get('after_run_idle') is not True:
        reasons.append(f'admission after_run_idle={adm.get("after_run_idle")} after_run={adm.get("after_run")}')
    alines = (run_dir / 'client/admission.tsv').read_text().splitlines()
    ah = alines[0].split('\t')
    health = {dict(zip(ah, l.split('\t')))['health'] for l in alines[1:]}
    if health - {'Open'}:
        reasons.append(f'admission health values {sorted(health)}')
    end = (run_dir / 'admission-stats-end.txt').read_text()
    m = re.search(r'\[backend_device_admission\]\n(.*)\n(.*)', end)
    if m:
        d = dict(zip(m.group(1).split('\t'), m.group(2).split('\t')))
        if d.get('live_grant_bytes') != '0' or d.get('active_attempts') != '0' or d.get('health') != 'Open':
            reasons.append(f'end admission stats {d}')
    if kind == 'gpu':
        expected = EXPECTED_A[profile]
        hdrs = headers(end)
        if hdrs.get('device_0_max_active_attempts') != str(expected):
            reasons.append(f'max_active_attempts {hdrs.get("device_0_max_active_attempts")} != {expected}')
    ql = (run_dir / 'query-summaries.tsv').read_text().splitlines()
    qh = ql[0].split('\t'); qrows = [dict(zip(qh, l.split('\t'))) for l in ql[1:]]
    cleanup = [q for q in qrows if q.get('attempt_cleanup_failure_count') not in ('', '0')]
    if cleanup:
        reasons.append(f'{len(cleanup)} queries with cleanup failures: {[q["query_id"] for q in cleanup[:5]]}')
    unhealthy = [q for q in qrows if q.get('attempt_health_transition') not in ('', 'healthy_reuse', 'none')]
    if unhealthy:
        reasons.append(f'{len(unhealthy)} queries with health transition {sorted({q["attempt_health_transition"] for q in unhealthy})}')
    unreleased = [q for q in qrows if q.get('attempt_capacity_disposition') not in ('', 'released', 'not_admitted')]
    if unreleased:
        reasons.append(f'{len(unreleased)} queries with capacity disposition {sorted({q["attempt_capacity_disposition"] for q in unreleased})}')
    if kind == 'gpu':
        nonnative = [q for q in qrows if q.get('outcome') == 'succeeded' and q.get('attempt_disposition') and q.get('attempt_disposition') != 'native']
        if nonnative:
            notes.append(f'{len(nonnative)} succeeded summaries without native disposition: {sorted({q.get("attempt_disposition") for q in nonnative})}')
    unavailable = [q for q in qrows if q.get('unavailable_reason') or q.get('error')]
    if unavailable:
        notes.append(f'{len(unavailable)} query summaries unavailable: {sorted({q.get("unavailable_reason") or q.get("error")[:60] for q in unavailable})}')
    slog = (run_dir / 'server.log').read_text(errors='replace')
    closed = [l[:300] for l in slog.splitlines() if re.search(r'admission (is )?closed|health=Closed|device_admission_closed|quarantin', l, re.I)]
    if closed:
        reasons.append(f'server log admission closed lines: {closed[:3]}')
    warns = [l[:300] for l in slog.splitlines() if re.search(r'\s(WARN|ERROR)\s', l) and 'auth DISABLED' not in l]
    if warns:
        notes.append(f'{len(warns)} WARN/ERROR server log lines (first: {warns[0]})')
    return reasons, notes, s


def finish(run_dir, record, reasons, notes):
    record.update(stop_reasons=reasons, notes=notes, finished_utc=now())
    (run_dir / 'run.json').write_text(json.dumps(record, indent=1))
    if reasons:
        (run_dir / 'STOP').write_text('\n'.join(reasons) + '\n')
        log(f'STOP {run_dir.relative_to(OUT)}: {reasons}')
        raise SystemExit(3)
    (run_dir / 'DONE').write_text(now() + '\n')


def run_unit(run_dir, profile, profile_json, mode, tool_timeout):
    if (run_dir / 'DONE').exists():
        log(f'skip {run_dir.relative_to(OUT)} (DONE)'); return
    if run_dir.exists():
        raise SystemExit(f'{run_dir} exists without DONE; inspect before rerunning')
    run_dir.mkdir(parents=True)
    kind = PROFILES[profile]
    pfile = run_dir / 'profile.flight-profile'
    pfile.write_text(json.dumps(profile_json, indent=1))
    ensure_warm(run_dir)
    record = {'profile': profile, 'mode': mode, 'started_utc': now(), 'server_config': str(OUT / 'configs' / f'{profile}.toml'),
              'server_config_sha256': sha256(OUT / 'configs' / f'{profile}.toml'), 'profile_sha256': sha256(pfile)}
    with Server(profile, run_dir) as srv:
        record['server_cmd'] = srv.cmd; record['server_ready_seconds'] = srv.ready_seconds
        r = subprocess.run(client_cmd('flight-sql-query', URL, DF_SETTINGS_SQL), cwd=ROOT, capture_output=True, text=True, timeout=60)
        (run_dir / 'df-settings.txt').write_text(r.stdout + r.stderr)
        (run_dir / 'admission-stats-start.txt').write_bytes(flight_action('GetAdmissionStats'))
        cmd = client_cmd('flight-sql-stress', '--config', str(pfile), '--out-dir', str(run_dir / 'client'))
        record['client_cmd'] = cmd
        log(f'{run_dir.relative_to(OUT)}: client start')
        t = time.monotonic()
        with open(run_dir / 'client.log', 'w') as clog:
            try:
                cr = subprocess.run(cmd, cwd=ROOT, stdout=clog, stderr=subprocess.STDOUT, timeout=tool_timeout)
                record['client_rc'] = cr.returncode
            except subprocess.TimeoutExpired:
                record['client_rc'] = 'timeout'
        record['client_seconds'] = time.monotonic() - t
        log(f'{run_dir.relative_to(OUT)}: client rc={record["client_rc"]} in {record["client_seconds"]:.1f}s')
        record['server_proc_status_end'] = srv.proc_status()
        try:
            (run_dir / 'admission-stats-end.txt').write_bytes(flight_action('GetAdmissionStats'))
        except Exception as e:
            (run_dir / 'admission-stats-end.txt').write_text(f'error: {e}\n')
        record['query_summaries'] = len(collect_query_summaries(run_dir))
    record['server_rc'] = srv.rc
    if record['client_rc'] != 0 or not (run_dir / 'client/summary.json').exists():
        finish(run_dir, record, [f'client rc={record["client_rc"]}; see client.log'], []); return
    reasons, notes, s = check_run(run_dir, kind, mode, profile)
    if srv.rc not in (0, -2, 130):
        notes.append(f'server rc={srv.rc}')
    ov = s['overall']
    record['headline'] = {
        'completed_qps': ov['throughput']['completed_qps'], 'offered_qps': ov['throughput']['offered_qps'],
        'p50_ms': (ov['latency_ns'].get('p50') or 0) / 1e6, 'p95_ms': (ov['latency_ns'].get('p95') or 0) / 1e6,
        'attempts': ov['attempts'], 'succeeded': ov['succeeded'], 'failed': ov['failed'],
        'window_s': s['window'].get('measured_window_seconds')}
    log(f'{run_dir.relative_to(OUT)}: {record["headline"]}')
    finish(run_dir, record, reasons, notes)


# ---------------------------------------------------------------- commands

def cmd_stage(release_dir):
    release_dir = Path(release_dir).resolve()
    (OUT / 'bin/lib').mkdir(parents=True, exist_ok=True)
    for name in ('algeon_server', 'algeon_tools'):
        shutil.copy2(release_dir / name, OUT / 'bin' / name)
    # Resolve the shim libraries the way the server's RUNPATH does, without this run's override.
    env = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH'}
    resolved = subprocess.run(['ldd', str(BIN_SERVER)], env=env, capture_output=True, text=True, check=True).stdout
    for lib in SHIM_LIBS:
        m = re.search(rf'^\s*{re.escape(lib)} => (\S+)', resolved, re.M)
        if not m:
            raise SystemExit(f'ldd of {BIN_SERVER} does not resolve {lib}')
        shutil.copy2(m.group(1), OUT / 'bin/lib' / lib)
    log(f'staged binaries and shim libraries from {release_dir} into {OUT / "bin"}')


def cmd_env():
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    env = {
        'measured_utc': now(),
        'source_revision': sh(f'git -C {ROOT} rev-parse HEAD'),
        'git_status_short': sh(f'git -C {ROOT} status --short').splitlines(),
        'binaries': {
            'algeon_server': {'path': str(BIN_SERVER), 'sha256': sha256(BIN_SERVER), 'version': sh(f'{BIN_SERVER} --version'),
                              'build': 'cargo build --release -p algeon-server --bin algeon_server (no features)'},
            'algeon_tools': {'path': str(BIN_TOOLS), 'sha256': sha256(BIN_TOOLS),
                             'build': 'cargo build --release -p tools --features flight-sql-stress-check'},
        },
        'ld_library_path': LIB_PATH,
        'native_libraries': {str(p): sha256(p) for p in [ROOT / 'target/native/cudf-install/lib/libcudf.so',
                                                         ROOT / 'target/native/cudf-install/lib/librmm.so',
                                                         *(OUT / 'bin/lib' / lib for lib in SHIM_LIBS)]},
        'gpu': sh('nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id,power.limit,clocks.max.sm,persistence_mode --format=csv'),
        'nvidia_smi_header': sh('nvidia-smi | head -4'),
        'cuda_runtime': sh('/usr/local/cuda/bin/nvcc --version | tail -2'),
        'cpu': sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket|NUMA node|MHz'"),
        'cpu_governor': sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unavailable'),
        'memory': sh('free -b'),
        'os': sh('cat /etc/os-release | grep PRETTY_NAME'),
        'kernel': sh('uname -a'),
        'instance': sh('cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null'),
        'dataset': {'path': str(DATA), 'registered_by': str(INIT_SQL), 'init_sql_text': INIT_SQL.read_text(),
                    'generation': json.loads((DATA / '_generation.json').read_text()) if (DATA / '_generation.json').exists() else None},
        'pinning': {'server_cpus': SERVER_CPUS, 'client_cpus': CLIENT_CPUS,
                    'note': 'logical CPUs i and i+16 are hyperthread siblings; the client gets physical cores 14-15 (4 logical CPUs), the server the other 14 physical cores (28 logical CPUs)'},
        'cpu_path_settings': CPU_ENV,
        'gpu_path_settings': 'DATAFUSION_* unset; native_required forces target_partitions=1',
        'server_env_cleared_prefixes': ['ALGEON_SERVER_', 'DATAFUSION_'],
        'configs': {p: {'path': str(OUT / 'configs' / f'{p}.toml'), 'sha256': sha256(OUT / 'configs' / f'{p}.toml'),
                        'text': (OUT / 'configs' / f'{p}.toml').read_text()} for p in PROFILES},
        'shipped_config_sha256': sha256(FIXTURES / 'tpch-sf100-server.toml'),
        'shipped_profile_sha256': sha256(FIXTURES / 'tpch-sf100-streams.flight-profile'),
        'init_sql_sha256': sha256(FIXTURES / 'tpch-sf100-init.sql'),
        'seed': SEED, 'float_tolerance': TOL,
        'commands': {
            'server': f'cd {ROOT} && env -u ALGEON_SERVER_* -u DATAFUSION_* [CPU: {CPU_ENV}] taskset -c {SERVER_CPUS} {BIN_SERVER} --config <configs/PROFILE.toml> (under flock {GPU_LOCK})',
            'client': f'taskset -c {CLIENT_CPUS} {BIN_TOOLS} flight-sql-stress --config <run>/profile.flight-profile --out-dir <run>/client',
            'driver': f'python3 scripts/dev/server_load_driver.py --out-dir {OUT} --data {DATA} --server-cpus {SERVER_CPUS} --client-cpus {CLIENT_CPUS} ...',
        },
    }
    p = OUT / 'environment.json'
    if p.exists():
        old = json.loads(p.read_text()); old.update(env); env = old
    p.write_text(json.dumps(env, indent=1))


def cmd_reference():
    run_dir = OUT / 'reference/run'
    prof = stream_profile('tpch-sf100-reference', OUT / 'configs/cpu.toml', 1, 1, 0, {'record_reference': str(REFERENCE)})
    if REFERENCE.exists() and not (run_dir / 'DONE').exists():
        raise SystemExit('reference dir exists without DONE')
    # The reference run checks nothing; skip the check-only conditions.
    if (run_dir / 'DONE').exists():
        return
    run_dir.mkdir(parents=True)
    pfile = run_dir / 'profile.flight-profile'; pfile.write_text(json.dumps(prof, indent=1))
    ensure_warm(run_dir)
    record = {'profile': 'cpu', 'mode': 'reference', 'started_utc': now()}
    with Server('cpu', run_dir) as srv:
        r = subprocess.run(client_cmd('flight-sql-query', URL, DF_SETTINGS_SQL), cwd=ROOT, capture_output=True, text=True, timeout=60)
        (run_dir / 'df-settings.txt').write_text(r.stdout + r.stderr)
        (run_dir / 'admission-stats-start.txt').write_bytes(flight_action('GetAdmissionStats'))
        cmd = client_cmd('flight-sql-stress', '--config', str(pfile), '--out-dir', str(run_dir / 'client'))
        record['client_cmd'] = cmd
        with open(run_dir / 'client.log', 'w') as clog:
            record['client_rc'] = subprocess.run(cmd, cwd=ROOT, stdout=clog, stderr=subprocess.STDOUT, timeout=3600).returncode
        (run_dir / 'admission-stats-end.txt').write_bytes(flight_action('GetAdmissionStats'))
        collect_query_summaries(run_dir)
    s = json.loads((run_dir / 'client/summary.json').read_text())
    reasons = []
    if record['client_rc'] != 0 or s['overall']['failed'] or s['overall']['succeeded'] != 22:
        reasons.append(f'reference run rc={record["client_rc"]} overall={ {k: s["overall"][k] for k in ("attempts","succeeded","failed")} }')
    finish(run_dir, record, reasons, [])


def cmd_stream(profile, clients, rep):
    run_dir = OUT / 'streams' / profile / f'c{clients}' / f'r{rep}'
    prof = stream_profile(f'tpch-sf100-{profile}-c{clients}-r{rep}', OUT / 'configs' / f'{profile}.toml', clients, 2, 1,
                          {'check_reference': str(REFERENCE), **TOL})
    run_unit(run_dir, profile, prof, 'streams', 4 * 3600)


def cmd_overload(label, rate, rep):
    run_dir = OUT / 'overload' / f'r{rep}' / label
    prof = rate_profile(f'tpch-sf100-rate-{label}-r{rep}', OUT / 'configs/gpu-a2.toml', rate)
    run_unit(run_dir, 'gpu-a2', prof, 'rate', 3600)


def env_default(name, default):
    return os.environ.get(name, str(default))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out-dir', default=env_default('SERVER_LOAD_OUT', DEFAULT_OUT), help='run directory (default: %(default)s)')
    ap.add_argument('--data', default=env_default('SERVER_LOAD_DATA', DEFAULT_DATA), help='SF100 TPC-H Parquet directory (default: %(default)s)')
    ap.add_argument('--release-dir', default=env_default('SERVER_LOAD_RELEASE_DIR', DEFAULT_RELEASE_DIR),
                    help='Cargo release directory that `stage` copies the binaries from (default: %(default)s)')
    ap.add_argument('--server-cpus', default=env_default('SERVER_LOAD_SERVER_CPUS', DEFAULT_SERVER_CPUS), help='taskset CPU list of the server (default: %(default)s)')
    ap.add_argument('--client-cpus', default=env_default('SERVER_LOAD_CLIENT_CPUS', DEFAULT_CLIENT_CPUS),
                    help='taskset CPU list of the load client and of this driver (default: %(default)s)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, text in (('stage', 'copy the release binaries and their shim libraries into OUT/bin'),
                       ('configs', 'write and check the per-profile server configs'),
                       ('env', 'write environment.json'),
                       ('warm', 'read every Parquet file once, record residency'),
                       ('reference', 'CPU functions_only single-stream pass recording references'),
                       ('overload-all', 'derive capacity C from the A=2 streams, then run the open-loop rates'),
                       ('run-all', 'configs, env, warm, reference, matrix and overload-all')):
        sub.add_parser(name, help=text)
    s = sub.add_parser('stream', help='one streams run (fresh server)')
    s.add_argument('profile', choices=PROFILES); s.add_argument('clients', type=int); s.add_argument('rep', type=int)
    o = sub.add_parser('overload', help='one open-loop rate run on GPU A=2 (fresh server)')
    o.add_argument('label'); o.add_argument('rate', type=float, help='queries per second'); o.add_argument('rep', type=int)
    m = sub.add_parser('matrix', help='full streams matrix, three repeats at 1, 2, 4, 8 and 16 streams')
    m.add_argument('profiles', nargs='?', default='gpu-a2,gpu-a1,gpu-a3,cpu', help='comma-separated profiles, in run order (default: %(default)s)')
    a = ap.parse_args()
    configure(a.out_dir, a.data, a.server_cpus, a.client_cpus)
    if a.cmd == 'stage':
        cmd_stage(a.release_dir)
    elif a.cmd == 'configs':
        write_configs()
    elif a.cmd == 'env':
        cmd_env()
    elif a.cmd == 'warm':
        warm(OUT / 'page-cache-warm.json')
    elif a.cmd == 'reference':
        cmd_reference()
    elif a.cmd == 'stream':
        cmd_stream(a.profile, a.clients, a.rep)
    elif a.cmd == 'overload':
        cmd_overload(a.label, a.rate, a.rep)
    elif a.cmd == 'matrix':
        cmd_matrix(a.profiles.split(','))
    elif a.cmd == 'overload-all':
        cmd_overload_all()
    elif a.cmd == 'run-all':
        write_configs()
        cmd_env()
        warm(OUT / 'page-cache-warm.json')
        cmd_reference()
        cmd_matrix(['gpu-a2', 'gpu-a1', 'gpu-a3', 'cpu'])
        cmd_overload_all()
        log('ALL_DONE')


def cmd_matrix(profiles):
    for p in profiles:
        if p not in PROFILES:
            raise SystemExit(f'unknown profile {p!r}; choose from {sorted(PROFILES)}')
    for p in profiles:
        for rep in (1, 2, 3):
            for c in (1, 2, 4, 8, 16):
                cmd_stream(p, c, rep)


def cmd_overload_all():
    import statistics
    best = {}
    for c in (1, 2, 4, 8, 16):
        vals = [json.loads((OUT / f'streams/gpu-a2/c{c}/r{r}/run.json').read_text())['headline']['completed_qps'] for r in (1, 2, 3)]
        best[c] = statistics.median(vals)
    cap = max(best.values())
    (OUT / 'overload').mkdir(exist_ok=True)
    (OUT / 'overload/capacity.json').write_text(json.dumps({'median_completed_qps_by_clients': best, 'C': cap,
        'rates': {k: f * cap for k, f in (('0.5C', .5), ('1.0C', 1.0), ('1.5C', 1.5), ('2.0C', 2.0))}}, indent=1))
    log(f'C = {cap} QPS from {best}')
    for rep in (1, 2, 3):
        for label, f in (('0.5C', .5), ('1.0C', 1.0), ('1.5C', 1.5), ('2.0C', 2.0)):
            cmd_overload(label, f * cap, rep)


if __name__ == '__main__':
    main()
