# Build and CI maintenance

Follow the root [AGENTS.md](../AGENTS.md) for validation selection and cleanup
restrictions. This file also applies when changing build, CI, or packaging
configuration outside `scripts/`.

- Read [Harness internals](../website/docs/build/tests.md#harness-internals)
  for script ownership and test scheduling, and
  [Feature coverage](../website/docs/build/tests.md#feature-coverage) before
  changing Cargo validation or packaging features.
- Read [Workspace and publication contracts](../website/docs/build/overview.md#workspace-and-publication-contracts)
  before changing manifests, source registration or shared build support.
- Use [README.md](README.md) for the directory map; update it and all callers
  when moving or adding scripts.
- Read the affected component's `AGENTS.md` before changing its native build,
  discovery, or binding scripts.
