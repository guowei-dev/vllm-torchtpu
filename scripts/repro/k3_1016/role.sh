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

# One pod of Kimi-K3 Oct 16 recipe points on a dev_nexus `serve-pd-bench` JobSet (`--pd`):
# the role's half of main's AgentX 1P1D lane (kimi-k3-agentx-1p1d.yaml), with
# the environment that manifest gives its containers set here instead.
#
#   role.sh <point>[,<point>…] [<agentx duration s>]     e.g. role.sh p3-c8  ·  role.sh p2-c4,p3-c8
#
# NEXUS_ROLE (prefill | decode | client), JS_NAME, POD_IP and RESULT_DIR come
# from the pod. Engines run k3_1p1d.sh and, when the client asks, copy their
# profile out (profile.sh). The client runs k3_agentx_client_pod.sh, then
# captures a profile under load, then writes the point's numbers (extract.py).
#
# Several points run one after another in the same job, each on FRESH servers
# (the doc ran every point as its own run, and the decode-host store servers
# never evict): the client writes result/points/<point>/POINT_DONE when a point
# is finished; each engine then stops its whole k3_1p1d.sh process group
# (engine, registry, store server) and starts the next point's; the client
# waits for both heads to go down before it waits for them to come back up.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../../.." && pwd)"
export REPO_DIR
IFS=',' read -r -a POINTS <<< "${1:?usage: role.sh <point>[,<point>…] [duration]}"
DURATION="${2:-}"
ROLE="${NEXUS_ROLE:?the serve-pd-bench recipe sets NEXUS_ROLE}"
: "${JS_NAME:?}" "${RESULT_DIR:?}"
# Each role is TP32 on one 2x2x4 slice: four hosts, or the engines wait for hosts that never come.
[ "${NEXUS_HOSTS:-4}" = 4 ] || { echo "role.sh: NEXUS_HOSTS=${NEXUS_HOSTS}; the K3 points need --hosts 4" >&2; exit 2; }
for p in "${POINTS[@]}"; do
  [ -f "$REPO_DIR/scripts/vllm/benchmarking/configs/k3-repro-1016/${p}.sh" ] \
    || { echo "role.sh: no such point: $p" >&2; exit 2; }
done
MULTI=0
[ "${#POINTS[@]}" -eq 1 ] || MULTI=1
export K3_PD_EXTERNAL_CLIENT=1
export PYTHONUNBUFFERED=1
# The engines' and the client's command (and the heads' health URLs below),
# overridable only to rehearse the point loop off-TPU.
ENGINE_CMD="${K3R_ENGINE_CMD:-bash .buildkite/kubernetes/k3_1p1d.sh}"
# How often a role relaunches a point's engines that died before the client finished it.
TRIES="${K3R_ENGINE_TRIES:-3}"
CLIENT_CMD="${K3R_CLIENT_CMD:-bash .buildkite/kubernetes/k3_agentx_client_pod.sh}"
# shellcheck source=scripts/repro/k3_1016/profile.sh
source "$HERE/profile.sh"

log() { echo "[role.sh] $(date -u +%T) $*"; }

# point_done_marker <point>: the client's POINT_DONE for <point> written after this
# pod started (an earlier attempt's, under the same job id, does not count).
point_done_marker() {
  find "$(k3r_outputs_root)" -path "*-client-0/*/result/points/$1/POINT_DONE" -newer "$STAMP" 2>/dev/null | head -n 1
}

# earlier_ok <point>: an EARLIER attempt's client dir (a preemption retry keeps the job id, so
# its dirs sit beside ours) that finished <point> green (POINT_OK); both roles skip such a point.
# The client tells earlier attempts by path, an engine by the marker predating its own start.
earlier_ok() {
  local f
  for f in "$(k3r_outputs_root)"/*-client-0/*/*/result/points/"$1"/POINT_OK; do
    [ -f "$f" ] || continue
    if [ "$ROLE" = client ]; then
      case "$f" in "$RESULT_DIR"/*) continue ;; esac
    elif [ "$f" -nt "$STAMP" ]; then
      continue
    fi
    dirname "$f"
    return 0
  done
  return 1
}

# role_retry_marker <point> <try>: any host of THIS role asked the role to relaunch <point>'s engines
# after try <try> failed (written after this pod started).
role_retry_marker() {
  find "$(k3r_outputs_root)" -path "*-${ROLE}-0/*/result/ENGINE_RETRY_$1_$2" -newer "$STAMP" 2>/dev/null | head -n 1
}

# heads_state: "up" when both heads answer /health, "down" when neither does.
heads_state() {
  local p d
  p=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${K3R_PREFILL_HEALTH:-http://${JS_NAME}-prefill-0-0.${JS_NAME}:8000/health}" || true)
  d=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${K3R_DECODE_HEALTH:-http://${JS_NAME}-decode-0-0.${JS_NAME}:8000/health}" || true)
  if [ "$p" = 200 ] && [ "$d" = 200 ]; then echo up
  elif [ "$p" != 200 ] && [ "$d" != 200 ]; then echo down
  else echo mixed; fi
}

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
    if [ "$MULTI" = 0 ]; then
      export K3_PD_CONFIG="k3-repro-1016/${POINTS[0]}"
      exec $ENGINE_CMD "$ROLE"
    fi
    STAMP=$(mktemp)
    for p in "${POINTS[@]}"; do
      if earlier_ok "$p" >/dev/null; then log "an earlier attempt finished $p; skipping it"; continue; fi
      export K3_PD_CONFIG="k3-repro-1016/${p}"
      try=1
      while :; do
        log "starting $ROLE for $p (try $try of $TRIES)"
        setsid $ENGINE_CMD "$ROLE" &
        eng=$!
        while kill -0 "$eng" 2>/dev/null && [ -z "$(point_done_marker "$p")" ] \
            && [ -z "$(role_retry_marker "$p" "$try")" ]; do sleep 20; done
        outcome=done
        if [ -n "$(role_retry_marker "$p" "$try")" ]; then
          outcome=retry
        elif ! kill -0 "$eng" 2>/dev/null; then
          wait "$eng"
          rc=$?
          # A follower's engine exits when its head was stopped first (each host polls on its own
          # cycle): if the client's POINT_DONE or the role's retry turns up (gcsfuse lags), that was it.
          for _ in 1 2 3 4; do
            [ -n "$(point_done_marker "$p")" ] && break
            [ -n "$(role_retry_marker "$p" "$try")" ] && { outcome=retry; break; }
            sleep 20
          done
          if [ "$outcome" = done ] && [ -z "$(point_done_marker "$p")" ]; then
            log "the $ROLE engine for $p exited (rc=$rc) before the client finished it (try $try of $TRIES)"
            [ "$try" -lt "$TRIES" ] || exit $(( rc == 0 ? 1 : rc ))
            # The whole role relaunches: TP32 spans its four hosts, so one host cannot restart alone.
            # (A TPU "START_SESSION failed" at init struck the same two decode nodes twice.)
            mkdir -p "$RESULT_DIR" && date -u +%FT%TZ > "$RESULT_DIR/ENGINE_RETRY_${p}_${try}"
            outcome=retry
          fi
        fi
        [ "$outcome" = retry ] && log "relaunching the $ROLE servers for $p" \
          || log "the client finished $p; stopping the $ROLE servers (process group $eng)"
        kill -TERM -- "-$eng" 2>/dev/null
        for _ in $(seq 60); do kill -0 -- "-$eng" 2>/dev/null || break; sleep 2; done
        kill -KILL -- "-$eng" 2>/dev/null
        wait "$eng" 2>/dev/null
        rm -rf "${K3R_PROFILE_DIR:?}"/* 2>/dev/null
        if [ "$outcome" = retry ]; then
          sleep "${K3R_RETRY_PAUSE:-90}"  # every host of the role is down, and the TPU runtime has settled
          try=$((try + 1))
          continue
        fi
        sleep 20  # the chips and the ports are free once every process of the group is gone
        break
      done
    done
    log "all points served; waiting for the client to finish"
    while sleep 3600; do :; done
    ;;
  client)
    export WORKLOAD_NAME="$JS_NAME"
    # The two slices' physical chips: what per-chip throughput divides by.
    export AGENTX_CHIPS=32
    # A chipless pod: vLLM's CLI must not load the TPU plugin (the manifest's).
    export VLLM_PLUGINS="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 TMPDIR=/tmp
    [ -z "$DURATION" ] || export AGENTX_DURATION="$DURATION"
    cd "$REPO_DIR" || exit 2
    rc_all=0
    first=1
    for p in "${POINTS[@]}"; do
      export K3_PD_CONFIG="k3-repro-1016/${p}"
      out="$RESULT_DIR"
      [ "$MULTI" = 0 ] || out="$RESULT_DIR/points/$p"
      mkdir -p "$out"
      export ARTIFACTS_DIR="$out/perf_eval_results"
      if [ "$MULTI" = 1 ] && prev=$(earlier_ok "$p"); then
        log "an earlier attempt finished $p green ($prev); keeping its results"
        cp -r "$prev/." "$out/" && echo "$prev" > "$out/RESUMED_FROM"
        [ ! -f "$out/cdk_points.jsonl" ] || cat "$out/cdk_points.jsonl" >> "$RESULT_DIR/cdk_points.jsonl"
        continue
      fi
      if [ "$first" = 0 ]; then
        # The previous point's servers must be gone first, or the client would measure them.
        log "waiting for the previous point's servers to stop"
        for _ in $(seq 180); do [ "$(heads_state)" = down ] && break; sleep 10; done
        [ "$(heads_state)" = down ] || { log "FATAL: the previous servers never went down"; rc_all=1; break; }
      fi
      first=0
      log "===== point $p ====="
      $CLIENT_CMD
      rc=$?
      log "the AgentX client finished $p with rc=$rc"
      if [ "${K3R_PROFILE:-1}" = 1 ] && [ "$rc" = 0 ]; then
        K3R_PROFILE_TAG="$p" k3r_client_profile "$p" "$out" \
          || log "WARNING: no profile for $p (see above)"
      fi
      python3 "$HERE/extract.py" --point "$p" --results "$ARTIFACTS_DIR/k3-agentx-1p1d" --out "$out" \
        || { log "extract.py failed for $p"; [ "$rc" -ne 0 ] || rc=1; }
      [ "$rc" = 0 ] || rc_all=$rc
      # each point's dash row as soon as it exists: the live dash reads $RESULT_DIR/cdk_points.jsonl while we run
      if [ "$MULTI" = 1 ] && [ -f "$out/cdk_points.jsonl" ]; then cat "$out/cdk_points.jsonl" >> "$RESULT_DIR/cdk_points.jsonl"; fi
      [ "$rc" != 0 ] || date -u +%FT%TZ > "$out/POINT_OK"
      date -u +%FT%TZ > "$out/POINT_DONE"
      # A point whose servers failed leaves nothing to measure the next point on.
      [ "$rc" = 0 ] || [ "$MULTI" = 0 ] || { log "stopping the batch after $p"; break; }
    done
    exit "$rc_all"
    ;;
  *)
    echo "role.sh: NEXUS_ROLE='$ROLE'; expected prefill, decode or client" >&2
    exit 2
    ;;
esac
