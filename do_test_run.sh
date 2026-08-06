#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image_tag="${ISLES26_IMAGE_TAG:-isles26-dataset503:preliminary}"
model_dir="${ISLES26_MODEL_STAGE:?set ISLES26_MODEL_STAGE}"
input_dir="${ISLES26_TEST_INPUT:?set ISLES26_TEST_INPUT}"
artifact_dir="${ISLES26_TEST_ARTIFACT_DIR:-$script_dir/test-artifacts}"
run_root=$(mktemp -d /tmp/isles26-container-test.XXXXXX)
staged_input="$run_root/input"
staged_output="$run_root/output"
scratch="$run_root/tmp"
network="isles26-test-${SLURM_JOB_ID:-$$}"
container="isles26-test-${SLURM_JOB_ID:-$$}"

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker network rm "$network" >/dev/null 2>&1 || true
  rm -rf "$run_root"
}
trap cleanup EXIT

test -f "$model_dir/isles26_model_manifest.json"
test -f "$input_dir/inputs.json"
mkdir -p \
  "$staged_input" \
  "$staged_output" \
  "$scratch" \
  "$artifact_dir/output"
chmod 0777 "$staged_output" "$scratch"
cp -aL "$input_dir/." "$staged_input/"
chmod -R a+rX "$staged_input"
test -z "$(find "$staged_input" -type l -print -quit)"

"$script_dir/do_build.sh"
test "$(docker inspect --format='{{index .Config.Labels "org.grand-challenge.api-method"}}' "$image_tag")" = invoke
docker network create --internal "$network" >/dev/null

gpu_args=()
if test -n "${CUDA_VISIBLE_DEVICES:-}"; then
  gpu_args=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
fi
docker run --detach \
  --name "$container" \
  --platform=linux/amd64 \
  --network "$network" \
  --shm-size 8g \
  --memory 32g \
  --memory-swap 32g \
  "${gpu_args[@]}" \
  --volume "$model_dir":/opt/ml/model:ro \
  --volume "$staged_input":/input:ro \
  --volume "$staged_output":/output \
  --volume "$scratch":/tmp \
  "$image_tag" >/dev/null

container_http_status() {
  local method="$1"
  local url="$2"
  local request_timeout="$3"
  docker exec --interactive "$container" python - \
    "$method" "$url" "$request_timeout" <<'PY'
import sys
import urllib.error
import urllib.request

method, url, request_timeout = sys.argv[1:]
request = urllib.request.Request(url, method=method)
try:
    with urllib.request.urlopen(
        request, timeout=float(request_timeout)
    ) as response:
        print(response.status)
except urllib.error.HTTPError as error:
    print(error.code)
PY
}

healthy=0
for attempt in $(seq 1 60); do
  status=$(container_http_status \
    GET http://127.0.0.1:4743/health 10 2>/dev/null || true)
  if test "$status" = 200; then
    healthy=1
    break
  fi
  sleep 5
done
test "$healthy" -eq 1

status=$(container_http_status \
  POST http://127.0.0.1:4743/invoke 420)
test "$status" = 201
docker logs "$container" > "$artifact_dir/container.log" 2>&1
cp -a "$staged_output/." "$artifact_dir/output/"
python "$script_dir/validate_outputs.py" \
  --input-dir "$staged_input" \
  --output-dir "$staged_output" \
  --manifest "$model_dir/isles26_model_manifest.json" \
  | tee "$artifact_dir/output_validation.txt"
docker inspect "$container" > "$artifact_dir/container_inspect.json"
echo "CONTAINER_OFFLINE_INVOKE_E2E_PASS=$artifact_dir"
