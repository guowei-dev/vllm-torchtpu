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

# Profile windows of a Kimi-K3 1P1D pair INSIDE the point's own AgentX profiling phase (sourced by entry.sh).
#
# The client runs the lane's AgentX client unchanged (pd_role.sh client), waits for aiperf's "Phase profiling
# (profiling) started", and opens a window only while the decode head's vllm:num_requests_running sits inside the
# golden run's p10-p90 for the point (steady_refs.json). Each window: /start_profile on both heads at once, sleep,
# /stop_profile on both (it returns once every rank has written), a token file; every engine host copies its new
# traces out and answers with the same token; check.py reads both heads' rank-0 xplanes in this pod and accepts or
# rejects the window. After K3CAP_WANT accepted windows (or K3CAP_ATTEMPTS) the AgentX client is stopped: nothing is
# scored. Pods meet through the job's outputs bucket (/cdk-outputs/outputs/<rjob>/<pod>/<container>/).
#
#   client   $PD_OUT/capture/: w<n>.json (gate + timing + check), STOPPED_w<n>, summary.json; $PD_OUT/result.json
#   engine   $CDK_OUTPUT_DIR/result/profile/<slot>/w<n>/ (this host's ranks), UPLOADED_<slot>_w<n>

K3CAP_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/roofline/k3/plan.env
source "$K3CAP_HERE/plan.env"

k3cap_log() { echo "[k3cap] $(date -u +%T) $*"; }
k3cap_root() { local d="${CDK_OUTPUT_DIR:?}"; echo "${d%/*/*/*}"; }

# k3cap_slot: the wrapper's slot key <i>-<name> (client: PD_OUT's basename; engine: PD_RESOLVED_OUT's)
k3cap_slot() {
  local p="${PD_OUT:-${PD_RESOLVED_OUT:-}}"
  if [ -n "$p" ]; then basename "$p"; else echo "${PD_CONFIG##*/}"; fi
}

# ----------------------------------------------------------------------------------------------- engine side
# k3cap_engine_watch: on each new STOPPED_w<n> token of this slot, copy this host's traces to
# result/profile/<slot>/w<n>/, empty the profile dir, answer UPLOADED_<slot>_w<n> with the token.
k3cap_engine_watch() {
  local root slot pdir out marker token n done_list=" "
  root="$(k3cap_root)"
  slot="$(k3cap_slot)"
  pdir="${K3R_PROFILE_DIR:-/dev/shm/k3mk/profile}"
  out="$CDK_OUTPUT_DIR/result/profile/$slot"
  k3cap_log "engine watch: slot $slot, traces from $pdir"
  while sleep "$K3CAP_POLL_S"; do
    for marker in "$root"/*-client-0/*/*/result/points/"$slot"/capture/STOPPED_w*; do
      [ -f "$marker" ] || continue
      token=$(cat "$marker" 2>/dev/null) || continue
      [ -n "$token" ] || continue
      case "$done_list" in *" $token "*) continue ;; esac
      n="${marker##*/STOPPED_}"
      mkdir -p "$out/$n"
      if cp -r "$pdir/." "$out/$n/"; then
        k3cap_log "window $n: copied $(find "$out/$n" -name '*.xplane.pb' | wc -l) xplanes"
      else
        k3cap_log "ERROR: window $n: copying the traces failed"
      fi
      rm -rf "${pdir:?}"/* 2>/dev/null
      echo "$token" > "$CDK_OUTPUT_DIR/result/UPLOADED_${slot}_$n"
      done_list="$done_list$token "
    done
  done
}

# ----------------------------------------------------------------------------------------------- client side
k3cap_post() {  # k3cap_post <url>: the HTTP code (20 min covers writing 32 ranks' traces)
  curl -s -o /dev/null -w '%{http_code}' --max-time 1200 -X POST "$1" || echo "curl-failed"
}

# k3cap_gauge <head url> <metric>: the head's gauge value (sum over label sets), or "nan"
k3cap_gauge() {
  curl -s --max-time 5 "$1/metrics" 2>/dev/null | awk -v m="$2" '
    $1 ~ "^"m"([{]|$)" { s += $NF; f = 1 } END { if (f) printf "%g\n", s; else print "nan" }'
}

# k3cap_band <point>: "<lo> <hi>" of the golden decode running gauge (lo >= 1: an idle decode is never steady state)
k3cap_band() {
  python3 - "$K3CAP_HERE/steady_refs.json" "$1" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["points"][sys.argv[2]]["decode"]["num_requests_running"]
lo = max(1.0, r["p10"])
print(f"{lo:g} {max(lo, r['p90']):g}")
EOF
}

in_band() { awk -v v="$1" -v lo="$2" -v hi="$3" 'BEGIN { exit !(v == v + 0 && v >= lo && v <= hi) }'; }

# k3cap_phase_started <dir>: aiperf's profiling phase has started (its log line, anywhere under <dir>)
k3cap_phase_started() {
  grep -rqs --include='*.log' "Phase profiling (profiling) started" "$1"
}

# k3cap_uploads <slot> <n> <token>: how many engine hosts answered window <n> with <token>
k3cap_uploads() {
  local f n=0
  for f in "$(k3cap_root)"/*-prefill-0/*/*/result/UPLOADED_"$1_$2" "$(k3cap_root)"/*-decode-0/*/*/result/UPLOADED_"$1_$2"; do
    [ -f "$f" ] && [ "$(cat "$f" 2>/dev/null)" = "$3" ] && n=$((n + 1))
  done
  echo "$n"
}

# k3cap_rank0 <role> <slot> <n>: host 0's rank-0 xplane of window <n>
k3cap_rank0() {
  find "$(k3cap_root)"/*-"$1"-0/*-"$1"-0-0-*/*/result/profile/"$2/$3" -name "rank0_*.xplane.pb" 2>/dev/null | head -n 1
}

# k3cap_window <n> <kind>: one window (kind: decode = gated on the decode gauge, prefill = on prefill work);
# writes capture/w<n>.json; returns 0 when accepted for its kind
k3cap_window() {
  local n="$1" kind="$2" t0 g_dec g_pre waited=0 gated=1 sp sd ps pd tok deadline up rc
  local dir="$PD_OUT/capture" wdir="w$n"
  while :; do
    g_dec=$(k3cap_gauge "$K3CAP_DECODE" vllm:num_requests_running)
    g_pre=$(k3cap_gauge "$K3CAP_PREFILL" vllm:num_requests_running)
    if [ "$kind" = decode ]; then in_band "$g_dec" "$K3CAP_LO" "$K3CAP_HI" && break
    else
      local w; w=$(k3cap_gauge "$K3CAP_PREFILL" vllm:num_requests_waiting)
      in_band "$g_pre" 1 1000000 || in_band "$w" 1 1000000 && break
    fi
    [ "$waited" -lt "$K3CAP_GATE_WAIT_S" ] || { gated=0; break; }
    sleep "$K3CAP_GATE_POLL_S"
    waited=$((waited + K3CAP_GATE_POLL_S))
  done
  k3cap_log "window $n ($kind): decode running $g_dec (band $K3CAP_LO-$K3CAP_HI), prefill running $g_pre," \
    "waited ${waited}s, in band: $gated"
  t0=$(date +%s.%N)
  sp=$(mktemp) sd=$(mktemp)
  k3cap_post "$K3CAP_PREFILL/start_profile" > "$sp" &
  ps=$!
  k3cap_post "$K3CAP_DECODE/start_profile" > "$sd" &
  pd=$!
  wait "$ps" "$pd"
  local started; started=$(date +%s.%N)
  local start_codes
  start_codes="$(cat "$sp")/$(cat "$sd")"
  sleep "$K3CAP_WIN"
  local g_dec_mid; g_dec_mid=$(k3cap_gauge "$K3CAP_DECODE" vllm:num_requests_running)
  local stopping; stopping=$(date +%s.%N)
  k3cap_post "$K3CAP_PREFILL/stop_profile" > "$sp" &
  ps=$!
  k3cap_post "$K3CAP_DECODE/stop_profile" > "$sd" &
  pd=$!
  wait "$ps" "$pd"
  local stopped; stopped=$(date +%s.%N)
  local stop_codes
  stop_codes="$(cat "$sp")/$(cat "$sd")"
  rm -f "$sp" "$sd"
  local g_dec_after; g_dec_after=$(k3cap_gauge "$K3CAP_DECODE" vllm:num_requests_running)
  k3cap_log "window $n: start $start_codes stop $stop_codes; decode running mid $g_dec_mid after $g_dec_after"
  K3CAP_LAST_STOP=$(date +%s)
  tok="$K3CAP_SLOT/$wdir@${HOSTNAME:-client}-$(date +%s)"
  echo "$tok" > "$dir/STOPPED_$wdir"
  deadline=$((SECONDS + K3CAP_UPLOAD_WAIT_S))
  while up=$(k3cap_uploads "$K3CAP_SLOT" "$wdir" "$tok"); [ "$up" -lt "$K3CAP_HOSTS2" ]; do
    [ "$SECONDS" -lt "$deadline" ] || break
    sleep "$K3CAP_POLL_S"
  done
  k3cap_log "window $n: $up of $K3CAP_HOSTS2 engine hosts uploaded"
  local dx px
  dx=$(k3cap_rank0 decode "$K3CAP_SLOT" "$wdir")
  px=$(k3cap_rank0 prefill "$K3CAP_SLOT" "$wdir")
  JAX_PLATFORMS=cpu python3 -I "$K3CAP_HERE/check.py" ${dx:+--decode "$dx"} ${px:+--prefill "$px"} \
    --min-steps "$K3CAP_MIN_DECODE_STEPS" --min-forwards "$K3CAP_MIN_PREFILL_FORWARDS" "${K3CAP_CHECK_ARGS[@]}" \
    --out "$dir/$wdir.check.json" \
    2>&1 | grep -v -e '^WARNING: All log' -e '^I0000' -e 'DeprecationWarning'
  python3 - "$dir/$wdir.json" "$dir/$wdir.check.json" <<EOF
import json, sys
try:
    check = json.load(open(sys.argv[2]))
except (OSError, ValueError):
    check = {"decode_ok": False, "prefill_ok": False, "why": ["no check output"]}
f = lambda v: None if v in ("nan", "") else float(v)
doc = {"window": "$wdir", "kind": "$kind", "slot": "$K3CAP_SLOT", "point": "$K3CAP_POINT",
       "gate": {"band": [$K3CAP_LO, $K3CAP_HI], "decode_running_before": f("$g_dec"), "prefill_running_before": f("$g_pre"),
                "decode_running_mid": f("$g_dec_mid"), "decode_running_after": f("$g_dec_after"),
                "waited_s": $waited, "in_band": bool($gated)},
       "time": {"gate_s": $t0, "started_s": $started, "stopping_s": $stopping, "stopped_s": $stopped,
                "window_s": $K3CAP_WIN, "since_phase_s": $t0 - $K3CAP_PHASE_T},
       "http": {"start": "$start_codes", "stop": "$stop_codes"},
       "uploads": {"hosts": $up, "want": $K3CAP_HOSTS2}, "rank0": {"decode": "$dx", "prefill": "$px"},
       "check": check}
gate_ok = doc["gate"]["in_band"] and (doc["gate"]["decode_running_after"] or 0) >= $K3CAP_LO
ok = doc["http"]["start"] == "200/200" and doc["http"]["stop"] == "200/200" and $up >= $K3CAP_HOSTS2
doc["accepted_decode"] = bool(ok and gate_ok and check.get("decode_ok")) if "$kind" == "decode" else False
doc["accepted_prefill"] = bool(ok and check.get("prefill_ok"))
json.dump(doc, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print("window $wdir:", "decode ACCEPTED" if doc["accepted_decode"] else "decode no",
      "/ prefill ACCEPTED" if doc["accepted_prefill"] else "/ prefill no", "|", "; ".join(check.get("why") or []))
sys.exit(0 if (doc["accepted_decode"] if "$kind" == "decode" else doc["accepted_prefill"]) else 1)
EOF
  rc=$?
  return "$rc"
}

# k3cap_stop_load: the AgentX client and its aiperf services, whole process group, then stragglers
k3cap_stop_load() {
  [ -n "${K3CAP_LOAD_PID:-}" ] || return 0
  kill -TERM -- "-$K3CAP_LOAD_PID" 2>/dev/null
  for _ in $(seq 30); do kill -0 -- "-$K3CAP_LOAD_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$K3CAP_LOAD_PID" 2>/dev/null
  pkill -TERM -f aiperf 2>/dev/null; sleep 5; pkill -KILL -f aiperf 2>/dev/null
  wait "$K3CAP_LOAD_PID" 2>/dev/null
  K3CAP_LOAD_PID=""
}

# k3cap_client <adapter>: the whole client slot; 0 when the windows ran (accepted or not), else 1
k3cap_client() {
  local adapter="$1" dir n acc_d=0 acc_p=0 tries=0 ptries=0 rc=0 phase_deadline
  K3CAP_SLOT="$(k3cap_slot)"
  K3CAP_POINT="${PD_CONFIG##*/}"
  local ref="${K3CAP_POINT%-scopes}" conc
  conc="${ref##*-c}"
  dir="$PD_OUT/capture"
  mkdir -p "$dir"
  read -r K3CAP_LO K3CAP_HI < <(k3cap_band "$ref") || { k3cap_log "no steady reference for $ref"; return 1; }
  K3CAP_WIN=$K3CAP_WINDOW_S
  [ "$conc" -gt "$K3CAP_LOWCONC_MAX" ] || K3CAP_WIN=$K3CAP_WINDOW_S_LOWCONC
  local scopes=()
  case "$K3CAP_POINT" in *-scopes) K3CAP_WIN=$K3CAP_WINDOW_S_SCOPES; scopes=(--want-scopes) ;; esac
  K3CAP_CHECK_ARGS=("${scopes[@]}")
  : "${K3CAP_PREFILL:=http://${JS_NAME}-prefill-0-0.${JS_NAME}:8000}" "${K3CAP_DECODE:=http://${JS_NAME}-decode-0-0.${JS_NAME}:8000}"
  K3CAP_HOSTS2=$((2 * ${PD_HOSTS:-4}))
  k3cap_log "slot $K3CAP_SLOT point $K3CAP_POINT: decode band $K3CAP_LO-$K3CAP_HI, window ${K3CAP_WIN}s," \
    "want $K3CAP_WANT of $K3CAP_ATTEMPTS"
  trap 'k3cap_stop_load; exit 143' TERM
  setsid bash "$adapter" > "$dir/agentx_client.log" 2>&1 &
  K3CAP_LOAD_PID=$!
  phase_deadline=$((SECONDS + K3CAP_PHASE_WAIT_S))
  until k3cap_phase_started "$PD_OUT/perf_eval_results"; do
    if ! kill -0 "$K3CAP_LOAD_PID" 2>/dev/null; then
      k3cap_log "FATAL: the AgentX client exited before its profiling phase; tail of its log:"
      tail -30 "$dir/agentx_client.log"
      rc=1; break
    fi
    [ "$SECONDS" -lt "$phase_deadline" ] || { k3cap_log "FATAL: no profiling phase in ${K3CAP_PHASE_WAIT_S}s"; rc=1; break; }
    sleep "$K3CAP_POLL_S"
  done
  if [ "$rc" = 0 ]; then
    K3CAP_PHASE_T=$(date +%s.%N)
    k3cap_log "aiperf profiling phase started; first window in ${K3CAP_MIN_DELAY_S}s"
    sleep "$K3CAP_MIN_DELAY_S"
    K3CAP_LAST_STOP=0
    n=0
    while [ "$acc_d" -lt "$K3CAP_WANT" ] && [ "$tries" -lt "$K3CAP_ATTEMPTS" ]; do
      local rest=$((K3CAP_LAST_STOP + K3CAP_SPACING_S - $(date +%s)))
      [ "$n" = 0 ] || [ "$rest" -le 0 ] || sleep "$rest"
      kill -0 "$K3CAP_LOAD_PID" 2>/dev/null || { k3cap_log "the AgentX client ended; no more windows"; break; }
      n=$((n + 1)); tries=$((tries + 1))
      if k3cap_window "$n" decode; then acc_d=$((acc_d + 1)); fi
      python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("accepted_prefill") else 1)' \
        "$dir/w$n.json" 2>/dev/null && acc_p=$((acc_p + 1))
    done
    while [ "$acc_p" -lt 1 ] && [ "$ptries" -lt "$K3CAP_PREFILL_ATTEMPTS" ] && kill -0 "$K3CAP_LOAD_PID" 2>/dev/null; do
      local rest=$((K3CAP_LAST_STOP + K3CAP_SPACING_S - $(date +%s)))
      [ "$rest" -le 0 ] || sleep "$rest"
      n=$((n + 1)); ptries=$((ptries + 1))
      if k3cap_window "$n" prefill; then acc_p=$((acc_p + 1)); fi
    done
  fi
  k3cap_stop_load
  python3 - "$dir" "$PD_OUT/result.json" "$K3CAP_POINT" "$conc" "$acc_d" "$acc_p" "$n" "$rc" <<'EOF'
import glob, json, os, statistics, sys
d, out, point, conc, acc_d, acc_p, n, rc = sys.argv[1:]
wins = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "w*.json"))) if ".check." not in f]
acc = [w for w in wins if w.get("accepted_decode")]
def med(key, sub):
    v = [((w["check"].get(sub) or {}).get(key) or {}).get("p50") for w in acc]
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None
metrics = {"concurrency": int(conc), "windows": len(wins), "accepted_decode": int(acc_d), "accepted_prefill": int(acc_p)}
for k, v in {"decode_running_at_capture": statistics.median([w["gate"]["decode_running_before"] for w in acc]) if acc else None,
             "decode_slots_p50": med("slots", "decode"), "decode_rows_p50": med("rows", "decode"),
             "decode_period_us_p50": med("period_us", "decode"), "decode_stack_us_p50": med("stack_us", "decode"),
             "decode_idle_us_p50": med("idle_us", "decode"),
             "decode_busy_frac": statistics.median([w["check"]["decode"]["busy_frac"] for w in acc
                                                    if w["check"]["decode"].get("busy_frac") is not None] or [float("nan")]),
             }.items():
    if v is not None and v == v:
        metrics[k] = v
pf = [w["check"]["prefill"] for w in wins if w.get("accepted_prefill")]
if pf:
    metrics["prefill_forwards"] = sum(p["forwards"] for p in pf)
    metrics["prefill_busy_frac"] = statistics.median([p["busy_frac"] for p in pf if p.get("busy_frac") is not None])
summary = {"point": point, "rc": int(rc), "metrics": metrics, "windows": wins}
json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1, sort_keys=True)
json.dump({"bench": point, "metrics": metrics}, open(out, "w"), indent=1, sort_keys=True)
print("summary:", json.dumps(metrics, sort_keys=True))
EOF
  return "$rc"
}
