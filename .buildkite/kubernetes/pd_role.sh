#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Drives the Kimi-K3 AgentX 1P1D lane: a prefill/decode JobSet pod that sets PD_ROLE, PD_CONFIG,
# PD_HOSTS, PD_CHIPS and JS_NAME (cdk recipe vllm-torchtpu-pd-bench) gets kimi-k3-agentx-1p1d.yaml's
# container env for one config. PD_ROLE=check only validates the run factors in PD_RUN_ENV_KEYS.
set -uo pipefail
: "${PD_ROLE:?}" "${PD_CONFIG:?}" "${PD_HOSTS:?}" "${PD_CHIPS:?}" "${JS_NAME:?}"
case "$PD_CONFIG" in *,*) echo "pd_role.sh: one config per job, not '$PD_CONFIG'" >&2; exit 2 ;; esac
# Run factors only: a different server is a different config file.
keys="${PD_RUN_ENV_KEYS:-}"
for key in ${keys//,/ }; do
  case "$key" in
    AGENTX_DURATION | AGENTX_FAST | AGENTX_SKIP_POST_EVAL | AGENTX_READY_WAIT_MIN) ;;
    *) echo "pd_role.sh: $key is not a run factor; change the server in a config file" >&2; exit 2 ;;
  esac
done
[ "$PD_ROLE" != check ] || exit 0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_DIR K3_PD_CONFIG="$PD_CONFIG" K3_PD_EXTERNAL_CLIENT=1 PYTHONUNBUFFERED=1
out="${PD_OUT:-${CDK_OUTPUT_DIR:-$REPO_DIR}}"
cd "$REPO_DIR" || exit 2
case "$PD_ROLE" in
  prefill | decode)
    # TPU_ACCELERATOR_TYPE as the lane sets it over GKE's tpu7x-32; K3_ENGINE_IMAGE keys the
    # compilation caches (recipe-base.sh).
    export ROLE_HOST_PREFIX="$JS_NAME-$PD_ROLE-0-" ROLE_HOST_DOMAIN="$JS_NAME" NUM_HOSTS="$PD_HOSTS" \
      PREFILL_HOST="$JS_NAME-prefill-0-0.$JS_NAME" DECODE_HOST="$JS_NAME-decode-0-0.$JS_NAME" \
      TMPDIR=/dev/shm VLLM_XLA_CACHE_PATH=/cache/jax/xla VLLM_CACHE_ROOT=/cache/jax/vllm \
      TPU_ACCELERATOR_TYPE=tpu7x K3_ENGINE_IMAGE="${PD_IMAGE:-unknown}"
    mkdir -p /cache/jax/xla /cache/jax/vllm
    exec bash .buildkite/kubernetes/k3_1p1d.sh "$PD_ROLE"
    ;;
  client)
    export WORKLOAD_NAME="$JS_NAME" AGENTX_CHIPS="$PD_CHIPS" ARTIFACTS_DIR="$out/perf_eval_results" \
      VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 TMPDIR=/tmp
    bash .buildkite/kubernetes/k3_agentx_client_pod.sh
    rc=$?
    if ! python3 scripts/vllm/benchmark/inferencex/agentx_point_metrics.py --config "$PD_CONFIG" \
        --results "$ARTIFACTS_DIR/k3-agentx-1p1d" --chips "$PD_CHIPS" --out "$out/result.json"; then
      [ "$rc" -ne 0 ] || rc=1
    fi
    exit "$rc"
    ;;
  *) echo "pd_role.sh: PD_ROLE=$PD_ROLE; expected prefill, decode, client or check" >&2; exit 2 ;;
esac
