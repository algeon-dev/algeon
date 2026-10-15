# Docker packaging

`build-image.sh` builds `algeon_server`, stages it with the native libraries
it links into `dist/`, and builds the image with `dist/` as the context.
`validate-loader-closure.sh` checks the staged loader closure on the host and
again in both `Dockerfile` stages. Usage, tags and troubleshooting are in
[Building with Docker](../website/docs/build/docker.md).
