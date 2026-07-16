# Repository configuration

This is the repository's only configuration directory.

- `nextest.toml` stays at this standard nextest discovery path.
- `policies/` contains repository-owned validation, scheduling, and CPU
  packaging policy data.

Executable implementations belong under `scripts/`, not in this directory.
