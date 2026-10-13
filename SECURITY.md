# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/algeon-dev/algeon/security/advisories/new).
If that form is unavailable, email <morristai01@gmail.com>. Do not open a
public issue or pull request for a security problem until a fix is released.

A useful report names the affected crate or binary, the commit or release,
the sanitized server configuration, and the steps that reproduce the
problem. A proof of concept helps but is not required.

Algeon is maintained by one person. Expect an acknowledgement within seven
days. Fixes land on `main`, and the advisory is published once a release
that contains the fix is available. There is no bug bounty.

## Supported versions

Algeon is pre-release. Security fixes go to `main` and to the latest release
of the published workspace crates under `crates/` and `components/`. Older
releases are not patched.

## Scope

In scope:

- The Rust crates under `crates/` and `components/`, including the
  `algeon_server` Flight SQL binary and its authentication, token and TLS
  handling.
- The Docker image and the build scripts under `scripts/`.
- The patches under `scripts/build/patches`, and the way Algeon calls cuDF,
  cuGraph, cuVS and cuML.

Out of scope:

- Vulnerabilities in cuDF, cuGraph, cuVS, cuML, DataFusion or Arrow
  themselves. Report those to the upstream project. If an Algeon patch or
  call introduces the problem, report it here.
- The server running with authentication disabled or without TLS. Both are
  the documented defaults for a trusted local network; see
  [Configuration](https://algeon.dev/integration/configuration#authentication).
- Any authenticated user reading or replacing any registered table. The DDL
  namespace is server-wide by design; there is no per-user authorization yet.
