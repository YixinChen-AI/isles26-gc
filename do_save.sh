#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image_tag="${ISLES26_IMAGE_TAG:-isles26-dataset503:preliminary}"
artifact_dir="${ISLES26_ARTIFACT_DIR:-$script_dir/artifacts}"
model_dir="${ISLES26_MODEL_STAGE:?set ISLES26_MODEL_STAGE}"
stage="${artifact_dir}.stage-$$"

test -f "$model_dir/isles26_model_manifest.json"
test ! -e "$artifact_dir"
test ! -e "$stage"
mkdir -p "$(dirname -- "$artifact_dir")"
mkdir "$stage"
cleanup() {
  rm -f \
    "$stage/container.tar.gz" \
    "$stage/model.tar.gz" \
    "$stage/model_archive_members.txt" \
    "$stage/SHA256SUMS"
  rmdir "$stage" 2>/dev/null || true
}
trap cleanup EXIT

"$script_dir/do_build.sh"
docker save "$image_tag" | gzip -1 > "$stage/container.tar.gz"
tar -czf "$stage/model.tar.gz" -C "$model_dir" .
gzip -t "$stage/container.tar.gz"
tar -tzf "$stage/model.tar.gz" \
  > "$stage/model_archive_members.txt"
grep -qx './isles26_model_manifest.json' \
  "$stage/model_archive_members.txt"
(
  cd "$stage"
  sha256sum container.tar.gz model.tar.gz > SHA256SUMS
)
test ! -e "$artifact_dir"
mv "$stage" "$artifact_dir"
trap - EXIT
echo "ISLES26_SAVED_ARTIFACTS=$artifact_dir"
