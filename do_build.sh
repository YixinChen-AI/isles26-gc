#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image_tag="${ISLES26_IMAGE_TAG:-isles26-dataset503:preliminary}"
base_image="${ISLES26_BASE_IMAGE:-pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime}"
docker build \
  --platform=linux/amd64 \
  --build-arg BASE_IMAGE="$base_image" \
  --tag "$image_tag" \
  "$script_dir"
