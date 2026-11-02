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

# A CPU rehearsal of capture.sh: two fake heads (HTTP /metrics, /start_profile, /stop_profile, the stop writing a
# given xplane as the head's rank 0), a fake AgentX client (its aiperf log announces the profiling phase), the real
# engine watchers, the real check.py, in a fake outputs bucket. Needs a python with jaxlib.
#
#   rehearse.sh <decode rank-0 xplane> <prefill rank-0 xplane> [<decode running gauge>]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DX="${1:?decode xplane}" PX="${2:?prefill xplane}" GAUGE="${3:-30}"
T=$(mktemp -d -t k3cap-rehearse.XXXXXX)
J=jtest
ROOT="$T/cdk-outputs/outputs"
mkdir -p "$ROOT/$J-client-0/$J-client-0-0-aaaaa/client" "$ROOT/$J-decode-0/$J-decode-0-0-bbbbb/decode" \
  "$ROOT/$J-prefill-0/$J-prefill-0-0-ccccc/prefill" "$T/shm/decode" "$T/shm/prefill"

fake_head() {  # fake_head <port> <role> <xplane> <profile dir>
  python3 - "$@" <<'EOF' &
import http.server, os, shutil, sys, time
port, role, xplane, pdir = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
gauge = os.environ.get("GAUGE", "30")
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _ok(self, body=b""):
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/metrics":
            v = gauge if role == "decode" else "1"
            self._ok((f'vllm:num_requests_running{{engine="0",model_name="m"}} {v}\n'
                      f'vllm:num_requests_waiting{{engine="0",model_name="m"}} 0\n').encode())
        else:
            self._ok()
    def do_POST(self):
        if self.path == "/stop_profile":
            d = os.path.join(pdir, "plugins", "profile", time.strftime("%Y_%m_%d_%H_%M_%S"))
            os.makedirs(d, exist_ok=True)
            shutil.copy(xplane, os.path.join(d, f"rank0_jtest-{role}-0-0.xplane.pb"))
        self._ok()
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
EOF
}
export GAUGE
fake_head 18501 decode "$DX" "$T/shm/decode"; H1=$!
fake_head 18502 prefill "$PX" "$T/shm/prefill"; H2=$!

engine() {  # engine <role>: the real watcher, as entry.sh starts it
  ( export CDK_OUTPUT_DIR="$ROOT/$J-$1-0/$J-$1-0-0-$2/$1" PD_RESOLVED_OUT="x/resolved/0-p8-c72" \
      K3R_PROFILE_DIR="$T/shm/$1"
    # shellcheck source=scripts/roofline/k3/capture.sh
    source "$HERE/capture.sh"; K3CAP_POLL_S=1; k3cap_engine_watch ) > "$T/$1.log" 2>&1 &
}
engine decode bbbbb; E1=$!
engine prefill ccccc; E2=$!

cat > "$T/fake_agentx.sh" <<'EOF'
mkdir -p "$PD_OUT/perf_eval_results/k3-agentx-1p1d/conc_72/aiperf_artifacts/logs"
sleep 3
echo "2026-11-02 00:00:00.000 - PhaseRunner - NOTICE - Phase profiling (profiling) started | target: 1800.0s" \
  >> "$PD_OUT/perf_eval_results/k3-agentx-1p1d/conc_72/aiperf_artifacts/logs/aiperf.log"
exec sleep 3600
EOF
(
  export CDK_OUTPUT_DIR="$ROOT/$J-client-0/$J-client-0-0-aaaaa/client" JS_NAME=$J PD_HOSTS=1 PD_CONFIG=k3-roofline/p8-c72
  export PD_OUT="$CDK_OUTPUT_DIR/result/points/0-p8-c72" K3CAP_PREFILL=http://127.0.0.1:18502 K3CAP_DECODE=http://127.0.0.1:18501
  # shellcheck source=scripts/roofline/k3/capture.sh
  source "$HERE/capture.sh"
  K3CAP_MIN_DELAY_S=1 K3CAP_SPACING_S=2 K3CAP_WINDOW_S=1 K3CAP_POLL_S=1 K3CAP_GATE_WAIT_S=4 K3CAP_UPLOAD_WAIT_S=60
  k3cap_client "$T/fake_agentx.sh"
  echo "client rc=$?"
) 2>&1 | tee "$T/client.log"
kill "$H1" "$H2" "$E1" "$E2" 2>/dev/null
echo "== outputs under $T"
find "$ROOT" -type f | sed "s#$ROOT/##" | sort
echo "== result.json"; cat "$ROOT/$J-client-0/$J-client-0-0-aaaaa/client/result/points/0-p8-c72/result.json"
