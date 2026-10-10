# Docker packaging

This directory packages `algeon_server` and the locally built RAPIDS libraries
it links into a CUDA runtime image. Building, running and troubleshooting the
image are documented in
[Building with Docker](../website/docs/build/docker.md).

| File | Purpose |
| --- | --- |
| `stage-docker-payload.sh` | Runs the packaging build of `algeon_server`, stages the binary and its native libraries into `dist/`, validates the loader closure, and writes `dist/payload-manifest.json`. |
| `build-image.sh` | Verifies the staged manifest and hashes, then runs `docker build` with the build arguments the Dockerfile requires. |
| `Dockerfile` | Two stages: harvest the CUDA toolkit libraries the payload links from an `nvidia/cuda` runtime image, then assemble the runtime image on the `nvidia/cuda` base image. |
| `dist/` | Generated staging output; not checked in. |

The native libraries must already be built under `target/native` (or the root
named by `ALGEON_NATIVE_DIR`); see
[Building from Source](../website/docs/build/from-source.md). Then run:

```bash
bash docker/stage-docker-payload.sh
bash docker/build-image.sh
```

`scripts/ci/check_docker_scripts.sh` checks the syntax of these scripts and of
every Dockerfile `RUN` body in the gate's policy phase.
