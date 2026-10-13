# Continuous integration

GitHub Actions runs only checks that need no GPU. GPU validation runs on a
local host through the gate described in
[Tests](../../website/docs/build/tests.md), and each pull request carries that
evidence as described in
[CONTRIBUTING.md](../../CONTRIBUTING.md#pull-requests).

## Hosted checks

[`ci.yml`](ci.yml) runs on every pull request and every push to `main`:

- `bash scripts/check_all.sh --native-free`, which covers rustfmt, the
  native-source and policy checks, `DOCS_RS=1` clippy and rustdoc, MSRV,
  packaged docs.rs builds, the API consumer probe and the tests of packages
  that link no native library. The
  [native-free checks](../../website/docs/build/tests.md#native-free-checks)
  section lists its phases. Failed runs upload the phase logs.
- The website's `npm ci`, `npm run typecheck` and `npm run build`.

[`dependabot.yml`](../dependabot.yml) proposes monthly GitHub Actions and
website npm updates. It leaves Cargo out, since no hosted run can execute the
GPU tests a dependency bump needs.

## GPU test requirements

Most of the suite executes GPU kernels. `.config/nextest.toml` places every
test outside an explicit CPU list in the `gpu` or `gpu-exclusive` group.
Apart from the few packages `scripts/ci/native_free_packages.py` selects,
the workspace crates depend on the `-sys` crates that link libcudf,
libcugraph, libcuvs and libcuml. Their test binaries, including the ones
whose tests stay on the CPU, fail to link without those libraries.

A runner for the full gate needs:

- An NVIDIA GPU with a driver that supports CUDA 13 (R580 or newer). The gate
  serializes GPU tests through `/tmp/cudf-gpu.lock` and assumes no other
  workload shares the device.
- An x86-64-v3 host CPU (AVX2), the target `.cargo/config.toml` sets.
- The four native libraries built from the pinned submodule sources for the
  runner's CUDA architecture. Their build trees take about 5 GB and must
  persist between runs, since a cold native build dominates any job time.
- The host tools listed in
  [Prerequisites](../../website/docs/build/prerequisites.md).

GitHub-hosted GPU runners are paid larger runners, billed per minute on
public repositories too, and each run would have to restore the native build
trees.

## Sponsoring a runner

A machine that meets the requirements above, registered as a self-hosted
runner, would let the full gate run on every push to `main`. Cloud GPU credits
for such a machine work as well. Open a
[GitHub issue](https://github.com/algeon-dev/algeon/issues) to discuss
hardware or credits.

A public repository's self-hosted runner executes whatever code a workflow
checks out. Workflows on a sponsored runner would run only for pushes to
`main` and for pull requests a maintainer has approved, never directly for
pull requests from forks.
