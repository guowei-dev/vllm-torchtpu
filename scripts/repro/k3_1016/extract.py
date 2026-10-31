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
"""One recipe point's numbers, as inferact-k3-recipes-1016.md defines them.

    python3 extract.py --point p3-c8 --results <ARTIFACTS_DIR>/k3-agentx-1p1d --out <dir>

Reads aiperf's profile_export_aiperf.json and server_metrics_export.csv for the
point's one concurrency and writes <out>/k3r_metrics.json and
<out>/cdk_result.json (the job's published metrics) and <out>/cdk_points.jsonl
(its point for `nexus cdk dash`). The doc's definitions:
P90 interactivity is 1000 / P90 inter_token_latency (ms); total tok/s is
total_token_throughput; per chip divides by the 32 chips; the chip and CPU
cache hit rates are the PREFILL's prefix_cache_hits and
external_prefix_cache_hits over its prefix_cache_queries, all over the
profiling phase; the theoretical rate is theoretical_prefix_cache_hit.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

CHIPS = 32


def metric(d: dict, name: str, stat: str = "avg"):
    v = d.get(name)
    return v.get(stat) if isinstance(v, dict) else None


def counters(path: Path) -> dict:
    """{(endpoint role, metric): total} from the counter section of aiperf's server metrics CSV."""
    out, header = {}, None
    for row in csv.reader(line for line in path.read_text().splitlines() if not line.startswith("#")):
        if not row:
            header = None
            continue
        if row[0] == "Endpoint":
            header = row
            continue
        if header is None or len(row) < 5 or row[1] != "counter" or "total" not in header:
            continue
        rec = dict(zip(header, row))
        role = "prefill" if "-prefill-" in rec["Endpoint"] else "decode" if "-decode-" in rec["Endpoint"] else "?"
        key = (role, rec["Metric"])
        out[key] = out.get(key, 0.0) + float(rec["total"] or 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    concs = sorted(a.results.glob("conc_*/aiperf_artifacts/profile_export_aiperf.json"))
    if len(concs) != 1:
        print(f"extract: expected one concurrency under {a.results}, found {len(concs)}", file=sys.stderr)
        return 1
    art = concs[0].parent
    d = json.loads(concs[0].read_text())
    itl_p90 = metric(d, "inter_token_latency", "p90")
    total = metric(d, "total_token_throughput")
    m = {
        "point": a.point,
        "conc": int(art.parent.name.removeprefix("conc_")),
        "ttft_p90_s": (metric(d, "time_to_first_token", "p90") or 0) / 1000,
        "ttft_p50_s": (metric(d, "time_to_first_token", "p50") or 0) / 1000,
        "itl_p90_ms": itl_p90,
        "interactivity_p90": 1000 / itl_p90 if itl_p90 else None,
        "total_tok_s": total,
        "tok_s_per_chip": total / CHIPS if total else None,
        "output_tok_s": metric(d, "output_token_throughput"),
        "input_tok_s": metric(d, "input_token_throughput"),
        "requests": metric(d, "request_count"),
        "benchmark_duration_s": metric(d, "benchmark_duration"),
        "theoretical_hit_pct": metric(d, "theoretical_prefix_cache_hit"),
        "was_cancelled": d.get("was_cancelled"),
        "errors": sum(int(e.get("count", 0)) for e in d.get("error_summary") or [] if isinstance(e, dict)),
        "aiperf_version": d.get("aiperf_version"),
    }
    smf = art / "server_metrics_export.csv"
    if smf.exists():
        c = counters(smf)
        q = c.get(("prefill", "vllm:prefix_cache_queries"))
        if q:
            m["chip_hit_pct"] = 100 * c.get(("prefill", "vllm:prefix_cache_hits"), 0) / q
            m["cpu_hit_pct"] = 100 * c.get(("prefill", "vllm:external_prefix_cache_hits"), 0) / q
        drafts = c.get(("decode", "vllm:spec_decode_num_drafts"))
        if drafts:
            m["decode_acceptance_length"] = 1 + c.get(("decode", "vllm:spec_decode_num_accepted_tokens"), 0) / drafts
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "k3r_metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    numeric = {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    (a.out / "cdk_result.json").write_text(json.dumps(numeric, indent=2) + "\n")
    # The run's point for `nexus cdk dash` (one sheet for the whole study). cdk's result-updater files a bench
    # ending in -c<N> and a `conc` key under c<N>.metrics, so the dash gets `p3` and `concurrency`.
    dash = {("concurrency" if k == "conc" else k): v for k, v in numeric.items()}
    (a.out / "cdk_points.jsonl").write_text(json.dumps(
        {"sheet": "k3-1016-repro", "bench": a.point.split("-")[0], "metrics": dash}) + "\n")
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
