# Docker packaging

`build-image.sh` builds `algeon_server`, stages it with the native libraries
it links into `dist/`, and builds the image with `dist/` as the context.
`validate-loader-closure.sh` checks the staged loader closure on the host and
again in both `Dockerfile` stages. `stage_licenses.py` writes `dist/licenses/`
from the native build trees and the build's `cargo metadata` capture;
`license-texts/` holds the standard texts for crates that ship no license
file. `publish-image.sh` checks a built image's labels against the release
tag and pushes it. Usage, tags and troubleshooting are in
[Building with Docker](../website/docs/build/docker.md).
