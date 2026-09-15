# Repository configuration

This is the repository's only configuration directory.

- `nextest.toml` stays at this standard nextest discovery path. It is the
  only test-lane source: CPU tests are opt-in by a `_cpu` target suffix or a
  listed binary, `_gpu_exclusive` targets run alone and last, and every other
  test runs in the `gpu` group. Its `[scripts.wrapper.gpu-lock]` entry runs
  every test through `scripts/ci/nextest_gpu_lock.sh`, which takes
  `/tmp/cudf-gpu.lock` shared for `gpu` and exclusive for `gpu-exclusive`.
- `policies/` contains repository-owned validation and CPU packaging policy
  data.
- Component crates consume this root policy; no component-local workspace,
  lockfile, lane catalog, or configuration directory is authoritative.

Executable implementations belong under `scripts/`, not in this directory.
