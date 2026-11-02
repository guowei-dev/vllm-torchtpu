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

# The ENTRY of a Kimi-K3 roofline capture slot (cdk recipe vllm-torchtpu-pd-bench; see README.md here).
# Engines are the lane adapter's, unchanged (pd_role.sh), plus a watcher that copies each window's traces out;
# the client drives the point's own AgentX load and captures inside its profiling phase (capture.sh), scoring
# nothing. PD_ROLE=check validates the run factors (the adapter's four) and exits.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../../.." && pwd)"
ADAPTER="$REPO_DIR/.buildkite/kubernetes/pd_role.sh"
role="${PD_ROLE:?}"
# The lane adapter's run factors, the same four (the check runs this file alone, without the repo).
keys="${PD_RUN_ENV_KEYS:-}"
for key in ${keys//,/ }; do
  case "$key" in
    AGENTX_DURATION | AGENTX_FAST | AGENTX_SKIP_POST_EVAL | AGENTX_READY_WAIT_MIN) ;;
    *) echo "entry.sh: $key is not a run factor; change the server in a config file" >&2; exit 2 ;;
  esac
done
case "$role" in
  check)
    exit 0
    ;;
  prefill | decode)
    # Only in a wrapped pod (the wrapper names the slot's resolved dir and the outputs bucket is mounted); detached,
    # so an offline render of this entry never waits on it. Nothing is exported: the engine env stays the adapter's.
    root="${CDK_OUTPUT_DIR:-}"
    root="${root%/*/*/*}"
    if [ -n "${PD_RESOLVED_OUT:-}" ] && [ -n "$root" ] && [ -d "$root" ]; then
      case "${PD_CONFIG:-}" in
        *-scopes)
          # A different program: nothing an earlier slot of this pod compiled may be read (all compile caches).
          echo "entry.sh: $PD_CONFIG: clearing the pod's compile caches"
          find /cache/jax -mindepth 1 -maxdepth 1 ! -name vllm -exec rm -rf {} + 2>/dev/null
          find /cache/jax/vllm -mindepth 1 -maxdepth 1 ! -name profile_cache -exec rm -rf {} + 2>/dev/null
          rm -rf /dev/shm/torch_tpu_cache
          ;;
      esac
      (
        # shellcheck source=scripts/roofline/k3/capture.sh
        source "$HERE/capture.sh"
        k3cap_engine_watch
      ) < /dev/null >> "${CDK_OUTPUT_DIR}/k3cap_engine.log" 2>&1 &
    fi
    exec bash "$ADAPTER"
    ;;
  client)
    : "${PD_OUT:?the wrapper sets PD_OUT per slot}"
    # shellcheck source=scripts/roofline/k3/capture.sh
    source "$HERE/capture.sh"
    k3cap_client "$ADAPTER"
    exit $?
    ;;
  *)
    echo "entry.sh: PD_ROLE=$role; expected prefill, decode, client or check" >&2
    exit 2
    ;;
esac
