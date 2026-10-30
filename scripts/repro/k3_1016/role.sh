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

# One pod of a Kimi-K3 Oct 16 recipe point on a dev_nexus `nexus-pd` JobSet:
# the role's half of main's AgentX 1P1D lane (kimi-k3-agentx-1p1d.yaml), with
# the environment that manifest gives its containers set here instead.
#
#   role.sh <point> [<agentx duration s>]     e.g. role.sh p3-c8
#
# NEXUS_ROLE (prefill | decode | client), JS_NAME, POD_IP and RESULT_DIR come
# from the pod. Engines run k3_1p1d.sh and, when the client asks, copy their
# profile out (profile.sh). The client runs k3_agentx_client_pod.sh, then
# captures a profile under load, then writes the point's numbers (extract.py).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../../.." && pwd)"
export REPO_DIR
POINT="${1:?usage: role.sh <point, e.g. p3-c8> [duration]}"
DURATION="${2:-}"
ROLE="${NEXUS_ROLE:?the nexus-pd recipe sets NEXUS_ROLE}"
: "${JS_NAME:?}" "${RESULT_DIR:?}"
export K3_PD_CONFIG="k3-repro-1016/${POINT}"
[ -f "$REPO_DIR/scripts/vllm/benchmarking/configs/${K3_PD_CONFIG}.sh" ] \
  || { echo "role.sh: no such point: $POINT" >&2; exit 2; }
export K3_PD_EXTERNAL_CLIENT=1
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/repro/k3_1016/profile.sh
source "$HERE/profile.sh"

case "$ROLE" in
  prefill|decode)
    : "${POD_IP:?}"
    # kimi-k3-agentx-1p1d.yaml's container env, per role.
    export ROLE_HOST_PREFIX="${JS_NAME}-${ROLE}-0-"
    export ROLE_HOST_DOMAIN="${JS_NAME}"
    export PREFILL_HOST="${JS_NAME}-prefill-0-0.${JS_NAME}"
    export DECODE_HOST="${JS_NAME}-decode-0-0.${JS_NAME}"
    export NUM_HOSTS=4
    export TPU_ACCELERATOR_TYPE=tpu7x
    export TMPDIR=/dev/shm
    export VLLM_XLA_CACHE_PATH=/cache/jax/xla VLLM_CACHE_ROOT=/cache/jax/vllm
    export K3_ENGINE_IMAGE="${NEXUS_IMAGE:-unknown}"
    # Traces small enough to bring back: device and host activity, no Python stacks.
    export PYTHON_TRACER_LEVEL=0 HOST_TRACER_LEVEL=1
    export K3R_PROFILE_DIR=/dev/shm/k3mk/profile
    mkdir -p /cache/jax/xla /cache/jax/vllm "$K3R_PROFILE_DIR"
    cd "$REPO_DIR" || exit 2
    k3r_engine_profile_watch &
    exec bash .buildkite/kubernetes/k3_1p1d.sh "$ROLE"
    ;;
  client)
    export WORKLOAD_NAME="$JS_NAME"
    # The two slices' physical chips: what per-chip throughput divides by.
    export AGENTX_CHIPS=32
    export ARTIFACTS_DIR="$RESULT_DIR/perf_eval_results"
    # A chipless pod: vLLM's CLI must not load the TPU plugin (the manifest's).
    export VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 TMPDIR=/tmp
    [ -z "$DURATION" ] || export AGENTX_DURATION="$DURATION"
    cd "$REPO_DIR" || exit 2
    bash .buildkite/kubernetes/k3_agentx_client_pod.sh
    rc=$?
    echo "[role.sh] the AgentX client finished with rc=$rc"
    if [ "${K3R_PROFILE:-1}" = 1 ]; then
      k3r_client_profile "$POINT" || echo "[role.sh] WARNING: no profile for this point (see above)"
    fi
    python3 "$HERE/extract.py" --point "$POINT" --results "$ARTIFACTS_DIR/k3-agentx-1p1d" \
      --out "$RESULT_DIR" || { echo "[role.sh] extract.py failed"; [ "$rc" -ne 0 ] || rc=1; }
    exit "$rc"
    ;;
  *)
    echo "role.sh: NEXUS_ROLE='$ROLE'; expected prefill, decode or client" >&2
    exit 2
    ;;
esac
