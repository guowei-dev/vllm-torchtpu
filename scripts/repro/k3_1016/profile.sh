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

# A profile of both engines under the point's own load, captured AFTER the
# measured window so it cannot touch the numbers. Sourced by role.sh.
#
# Pods meet through the job's outputs bucket, which every pod of a cdk job
# mounts (/cdk-outputs/outputs/<rjob>/<pod>/<container>/):
#   client   replays the point's AgentX trace for K3R_LOAD_SECONDS, POSTs
#            /start_profile to both heads after K3R_PROFILE_AFTER s, /stop_profile
#            K3R_PROFILE_SECONDS later (it returns once every rank has written),
#            then writes result/PROFILE_STOPPED holding a token
#   engine   on that token copies its ranks' traces to result/profile/ and writes
#            result/PROFILE_UPLOADED holding the same token
#   client   waits for all eight engines' PROFILE_UPLOADED (K3R_UPLOAD_WAIT s)

: "${K3R_LOAD_SECONDS:=300}" "${K3R_PROFILE_AFTER:=150}" "${K3R_PROFILE_SECONDS:=5}"
: "${K3R_UPLOAD_WAIT:=1500}"

k3r_outputs_root() {  # /cdk-outputs/outputs: three levels above this pod's CDK_OUTPUT_DIR
  local d="${CDK_OUTPUT_DIR:?}"
  echo "${d%/*/*/*}"
}

k3r_log() { echo "[k3r-profile] $(date -u +%T) $*"; }

# k3r_engine_profile_watch: on each new client token, copy this host's traces out.
k3r_engine_profile_watch() {
  local root marker token done_tok=""
  root="$(k3r_outputs_root)"
  while sleep 20; do
    marker=$(ls -t "$root"/*-client-0/*/*/result/PROFILE_STOPPED 2>/dev/null | head -n 1)
    [ -n "$marker" ] || continue
    token=$(cat "$marker" 2>/dev/null) || continue
    [ -n "$token" ] && [ "$token" != "$done_tok" ] || continue
    k3r_log "profile stopped (token $token); copying $K3R_PROFILE_DIR"
    mkdir -p "$RESULT_DIR/profile"
    if cp -r "$K3R_PROFILE_DIR/." "$RESULT_DIR/profile/"; then
      find "$RESULT_DIR/profile" -type f -printf '%s %P\n' > "$RESULT_DIR/profile_files.txt"
      k3r_log "copied $(wc -l < "$RESULT_DIR/profile_files.txt") files"
    else
      k3r_log "ERROR: copying the traces failed"
    fi
    echo "$token" > "$RESULT_DIR/PROFILE_UPLOADED"
    done_tok="$token"
  done
}

# k3r_post <url>: POST, printing the HTTP code; 20 min covers writing 32 ranks' traces.
k3r_post() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 1200 -X POST "$1" || echo "curl-failed"
}

# k3r_client_profile <point>: the load, the capture, the wait for the uploads.
k3r_client_profile() {
  local point="$1" res="$ARTIFACTS_DIR/k3-agentx-1p1d" cmdf prefill decode token root
  local prefill_url decode_url router_pid="" load_pid="" n up rc=0
  prefill="${JS_NAME}-prefill-0-0.${JS_NAME}"
  decode="${JS_NAME}-decode-0-0.${JS_NAME}"
  prefill_url="http://${prefill}:8000"
  decode_url="http://${decode}:8000"
  mkdir -p "$RESULT_DIR/profile_load"
  cmdf=$(ls "$res"/conc_*/benchmark_command.txt 2>/dev/null | head -n 1)
  [ -n "$cmdf" ] || { k3r_log "no benchmark_command.txt under $res; no profile"; return 1; }
  for u in "$prefill_url" "$decode_url"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$u/health")" = 200 ] \
      || { k3r_log "$u is not healthy; no profile"; return 1; }
  done
  # The client pod's own router died with k3_agentx_client_pod.sh; the same one again.
  vllm-router --host 0.0.0.0 --port 8100 --vllm-pd-disaggregation \
    --prefill "$prefill_url" --decode "$decode_url" --policy consistent_hash \
    --request-timeout-secs 5000 --worker-startup-timeout-secs 300 \
    --health-check-interval-secs 30 --disable-retries \
    > "$RESULT_DIR/profile_load/router.log" 2>&1 &
  router_pid=$!
  timeout 120 bash -c 'until curl -sf -o /dev/null http://127.0.0.1:8100/health; do sleep 2; done' \
    || { k3r_log "the router did not come up"; kill "$router_pid" 2>/dev/null; return 1; }
  # The measured run's own aiperf command, shortened, into its own directory. Under the
  # scenario's 900 s minimum, which AIPERF_UNSAFE_OVERRIDE allows.
  local load
  load=$(sed -E -e 's/--benchmark-duration [0-9]+/--benchmark-duration '"$K3R_LOAD_SECONDS"'/' \
      -e 's/--warmup-requests-per-lane [0-9]+/--warmup-requests-per-lane 1/' \
      -e 's/--warmup-grace-period [0-9]+/--warmup-grace-period 300/' \
      -e 's#--output-artifact-dir [^ ]+#--output-artifact-dir '"$RESULT_DIR"'/profile_load/aiperf#' \
      -e 's#--server-metrics( [^ -][^ ]*)+##' "$cmdf")
  echo "$load" > "$RESULT_DIR/profile_load/command.txt"
  (
    ix=/tmp/.agentx_scratch/InferenceX/inferencex-e2e
    # shellcheck disable=SC1091
    [ -f "$ix/benchmarks/runtime_settings.sh" ] && source "$ix/benchmarks/runtime_settings.sh" >/dev/null 2>&1
    if [ -w /root/.cache/huggingface ]; then export HF_HOME=/root/.cache/huggingface
    else export HF_HOME=/dev/shm/.agentx_scratch/hf_home; fi
    export AIPERF_RUNTIME_DIR=/tmp/.agentx_scratch/inferencex-agentic AIPERF_UNSAFE_OVERRIDE=true
    export TMPDIR=/tmp/k3r-load
    mkdir -p "$TMPDIR"
    exec bash -c "$load"
  ) > "$RESULT_DIR/profile_load/aiperf.log" 2>&1 &
  load_pid=$!
  k3r_log "load started (pid $load_pid): $load"
  sleep "$K3R_PROFILE_AFTER"
  if ! kill -0 "$load_pid" 2>/dev/null; then
    k3r_log "the load exited early; see profile_load/aiperf.log"
    tail -20 "$RESULT_DIR/profile_load/aiperf.log"
    rc=1
  else
    # Both roles at once: a request in flight crosses both.
    k3r_log "start_profile: prefill $(k3r_post "$prefill_url/start_profile") decode $(k3r_post "$decode_url/start_profile")"
    sleep "$K3R_PROFILE_SECONDS"
    local sp sd
    sp=$(mktemp) sd=$(mktemp)
    local wp wd
    k3r_post "$prefill_url/stop_profile" > "$sp" &
    wp=$!
    k3r_post "$decode_url/stop_profile" > "$sd" &
    wd=$!
    wait "$wp" "$wd"
    k3r_log "stop_profile: prefill $(cat "$sp") decode $(cat "$sd")"
    grep -qx 200 "$sp" && grep -qx 200 "$sd" || rc=1
  fi
  pkill -TERM -f "aiperf" 2>/dev/null; sleep 15; pkill -KILL -f "aiperf" 2>/dev/null
  kill "$load_pid" "$router_pid" 2>/dev/null
  [ "$rc" = 0 ] || return "$rc"
  token="${HOSTNAME:-client}-$(date +%s)"
  echo "$token" > "$RESULT_DIR/PROFILE_STOPPED"
  root="$(k3r_outputs_root)"
  local deadline=$((SECONDS + K3R_UPLOAD_WAIT))
  while :; do
    n=0
    for up in "$root"/*-prefill-0/*/*/result/PROFILE_UPLOADED "$root"/*-decode-0/*/*/result/PROFILE_UPLOADED; do
      [ -f "$up" ] && [ "$(cat "$up" 2>/dev/null)" = "$token" ] && n=$((n + 1))
    done
    [ "$n" -ge 8 ] && break
    [ "$SECONDS" -lt "$deadline" ] || { k3r_log "only $n of 8 engines uploaded their traces"; return 1; }
    sleep 20
  done
  k3r_log "all 8 engines uploaded their traces"
}
