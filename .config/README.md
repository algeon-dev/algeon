# Repository configuration

This is the repository's only configuration directory.

- `nextest.toml` stays at this standard nextest discovery path.
- `policies/` contains repository-owned validation, scheduling, and CPU
  packaging policy data.
- Component crates consume this root policy; no component-local workspace,
  lockfile, lane catalog, or configuration directory is authoritative.

Executable implementations belong under `scripts/`, not in this directory.
