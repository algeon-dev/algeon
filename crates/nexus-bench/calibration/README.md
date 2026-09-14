# Goodput calibration store

`goodput_slo.tsv` and `goodput_slo_graph.tsv` freeze, for one workload item
each, the single-flight median and 95th-percentile end-to-end latency at the
configured request cap, the attempt's aligned allocator peak, and the output
fingerprint. With `NEXUS_GOODPUT_REQUEST_CAP_MANIFEST`, the calibration header
records the resolved cap for every item in `calibration_caps`.

`query_service_goodput` uses them for two things a comparison run must not
derive from itself:

- the SLO deadline, as a fixed multiple of the frozen median, and
- the correctness check, by comparing each completed query's output
  fingerprint against the frozen one.

The header lines record the GPU the calibration ran on and the dataset
fingerprints it used. A measurement run refuses to start when the live device
or dataset differs, because a deadline calibrated elsewhere says nothing about
this machine. Regenerate with:

```bash
crates/nexus-bench/scripts/run_goodput_matrix.sh calibrate
```

These files are therefore host- and dataset-specific, and are not committed
(`.gitignore` excludes `*.tsv` here). They are regenerated per host rather than
shared, because a deadline calibrated on other hardware, or against a different
dataset build, is not a fact about this machine.

A published comparison must therefore quote the calibration values it was judged
against inline in its evidence, rather than pointing at this directory.

The optional request-cap manifest is an independent TSV input with schema
`nexus-goodput-cap-assignment-v1` and columns `kind`, `key`, `cap_bytes`. Rows
may assign `default`, `family`, `item`, or `sequence` caps. Sequence, item,
family, and default rules are applied in that order for arrivals; calibration
requires an item, family, or default rule so its cap is unambiguous.
