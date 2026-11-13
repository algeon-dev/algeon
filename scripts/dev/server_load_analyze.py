#!/usr/bin/env python3
"""Reduce the run directories of server_load_driver.py into tables.md and results.json.

Reads RUN/streams and RUN/overload and writes both files into RUN. The website
generator does not read them; they summarize a run for review.
"""
import argparse, json, statistics
from pathlib import Path

from server_load_driver import DEFAULT_OUT, env_default

OUT = None
PROFILES = [('gpu-a1', 'GPU A=1'), ('gpu-a2', 'GPU A=2 (default)'), ('gpu-a3', 'GPU A=3'), ('cpu', 'CPU functions_only')]
CLIENTS = (1, 2, 4, 8, 16)
RATES = ('0.5C', '1.0C', '1.5C', '2.0C')


def med_range(vals, fmt):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 'n/a'
    if len(vals) == 1:
        return fmt(vals[0]) + ' (1 run)'
    return f'{fmt(statistics.median(vals))} ({fmt(min(vals))}-{fmt(max(vals))})'


def load(run):
    s = json.loads((run / 'client/summary.json').read_text())
    r = json.loads((run / 'run.json').read_text())
    return s, r


def stream_rows():
    out, rows = {}, []
    for p, label in PROFILES:
        for c in CLIENTS:
            runs = sorted((OUT / 'streams' / p / f'c{c}').glob('r*/DONE'))
            if not runs:
                continue
            data = [load(d.parent) for d in runs]
            qps = [s['overall']['throughput']['completed_qps'] for s, _ in data]
            p50 = [s['overall']['latency_ns']['p50'] / 1e9 for s, _ in data]
            p95 = [s['overall']['latency_ns']['p95'] / 1e9 for s, _ in data]
            failed = sum(s['overall']['failed'] + s['overall']['cancelled'] for s, _ in data)
            ch = {k: sum(s['results']['checks_all_phases'].get(k, 0) for s, _ in data) for k in ('match', 'mismatch', 'error', 'not_checked')}
            unchecked = sum(s['results'].get('unchecked_successes', 0) for s, _ in data)
            idle = all(s['admission']['after_run_idle'] for s, _ in data)
            hwm = [int(r['server_proc_status_end'].get('VmHWM', '0 kB').split()[0]) / 2**20 for _, r in data]
            out[f'{p}/c{c}'] = {'runs': len(data), 'completed_qps': qps, 'p50_s': p50, 'p95_s': p95, 'failed_or_cancelled': failed,
                                'checks_all_phases': ch, 'unchecked_successes': unchecked, 'after_run_idle_all': idle,
                                'measured_attempts': [s['overall']['attempts'] for s, _ in data],
                                'window_s': [s['window']['measured_window_seconds'] for s, _ in data],
                                'server_vmhwm_gib': hwm, 'notes': [r['notes'] for _, r in data]}
            rows.append(f'| {label} | {c} | {med_range([q * 3600 for q in qps], lambda v: f"{v:,.0f}")} | '
                        f'{med_range(qps, lambda v: f"{v:.3f}")} | {med_range(p50, lambda v: f"{v:.2f}")} | '
                        f'{med_range(p95, lambda v: f"{v:.2f}")} | {failed} | {ch["match"]}/{ch["match"] + ch["mismatch"] + ch["error"]}'
                        f'{"" if unchecked == 0 else f" ({unchecked} unchecked)"} | {"yes" if idle else "NO"} | {len(data)} |')
    head = ('| Profile | Streams | Completed queries/hour | Completed QPS | p50 latency (s) | p95 latency (s) | Errors | '
            'Checks matched (both phases) | Grants idle after run | Runs |\n|---|---|---|---|---|---|---|---|---|---|')
    return out, head + '\n' + '\n'.join(rows)


def overload_rows():
    out, rows = {}, []
    cap_path = OUT / 'overload/capacity.json'
    cap = json.loads(cap_path.read_text()) if cap_path.exists() else None
    for label in RATES:
        runs = sorted(OUT.glob(f'overload/r*/{label}/DONE'))
        if not runs:
            continue
        data = [load(d.parent) for d in runs]
        rec = {'runs': len(data)}
        comp = [s['overall']['throughput']['completed_qps'] for s, _ in data]
        offered = [s['overall']['throughput']['offered_qps'] for s, _ in data]
        frac, by_reason, rej_lat, p95, lag, bound, idle, tti, sched, sub = [], {}, {}, [], [], [], [], [], [], []
        for s, _ in data:
            ov = s['overall']
            so = (ov.get('errors') or {}).get('service_overloaded') or {}
            n = sum(so.values()) if isinstance(so, dict) else so
            frac.append(n / ov['attempts'] if ov['attempts'] else 0)
            for reason, cnt in (so.items() if isinstance(so, dict) else []):
                by_reason.setdefault(reason, []).append(cnt)
            for reason, ns in ((ov.get('rejected_latency_p50_ns') or {}).items()):
                rej_lat.setdefault(reason, []).append(ns / 1e9 if ns is not None else None)
            p95.append(ov['latency_ns']['p95'] / 1e9 if ov['latency_ns'].get('p95') else None)
            ol = s.get('open_loop') or ov.get('open_loop') or {}
            lag.append((ol.get('max_schedule_lag_ns') or 0) / 1e6)
            bound.append(ol.get('client_bound_reached'))
            sched.append(ol.get('scheduled')); sub.append(ol.get('submitted'))
            idle.append(s['admission']['after_run_idle']); tti.append((s['admission'].get('time_to_idle_ns') or 0) / 1e9)
        ch = {k: sum(s['results']['checks_all_phases'].get(k, 0) for s, _ in data) for k in ('match', 'mismatch', 'error')}
        rec.update(configured_rate=[(s.get('open_loop') or s['overall'].get('open_loop') or {}).get('configured_rate_per_second') for s, _ in data],
                   completed_qps=comp, offered_qps=offered, overloaded_fraction=frac, overloaded_by_reason=by_reason,
                   rejected_latency_p50_s=rej_lat, completed_p95_s=p95, max_schedule_lag_ms=lag, client_bound_reached=bound,
                   scheduled=sched, submitted=sub, after_run_idle=idle, time_to_idle_s=tti, checks_all_phases=ch,
                   attempts=[s['overall']['attempts'] for s, _ in data], succeeded=[s['overall']['succeeded'] for s, _ in data])
        out[label] = rec
        rl = lambda reason: med_range(rej_lat.get(reason, []), lambda v: f'{v:.3f}')
        reasons = ', '.join(f'{k} {sum(v)}' for k, v in sorted(by_reason.items())) or 'none'
        rows.append(f'| {label} | {med_range(rec["configured_rate"], lambda v: f"{v:.3f}")} | {med_range(offered, lambda v: f"{v:.3f}")} | '
                    f'{med_range(comp, lambda v: f"{v:.3f}")} | {med_range([f * 100 for f in frac], lambda v: f"{v:.1f}%")} | {reasons} | '
                    f'{rl("admission_queue_full")} | {rl("queue_wait_elapsed")} | {med_range(p95, lambda v: f"{v:.2f}")} | '
                    f'{ch["match"]}/{sum(ch.values())} | {med_range(lag, lambda v: f"{v:.1f}")} | {"/".join(str(b).lower() for b in bound)} | '
                    f'{"/".join("yes" if i else "NO" for i in idle)} | {len(data)} |')
    head = ('| Level | Configured rate (QPS) | Offered QPS | Completed QPS | service_overloaded | Rejections by reason (sum of runs) | '
            'Rejection p50 queue full (s) | Rejection p50 queue wait (s) | Completed p95 (s) | Checks matched | Max schedule lag (ms) | '
            'Client bound reached | live_grant_bytes back to 0 | Runs |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    return out, cap, head + '\n' + '\n'.join(rows)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out-dir', default=env_default('SERVER_LOAD_OUT', DEFAULT_OUT), help='run directory (default: %(default)s)')
    OUT = Path(ap.parse_args().out_dir).resolve()
    s_out, s_tab = stream_rows()
    o_out, cap, o_tab = overload_rows()
    (OUT / 'results.json').write_text(json.dumps({'streams': s_out, 'overload': o_out, 'capacity': cap}, indent=1))
    (OUT / 'tables.md').write_text('## Streams\n\n' + s_tab + '\n\n## Overload\n\n' + (f'C = {cap["C"]:.4f} QPS\n\n' if cap else '') + o_tab + '\n')
    print((OUT / 'tables.md').read_text())


if __name__ == '__main__':
    main()
