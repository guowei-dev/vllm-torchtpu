#!/usr/bin/env python3
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
"""One AgentX point's numbers, as the Kimi-K3 1P1D recipe doc defines them.

    agentx_point_metrics.py --config k3-repro-1016/p5-c32 \\
        --results <ARTIFACTS_DIR>/k3-agentx-1p1d --chips 32 --out result.json \\
        [--expect expected.json --window 3600]

Reads aiperf's profile_export_aiperf.json and server_metrics_export.csv for the
point's one concurrency and writes {bench, metrics, info}, bench being the config
with '/' as '.' and no trailing -c<N>. P90 interactivity is 1000 / P90
inter-token latency (ms); per chip divides total tok/s by --chips, the physical
chips of both roles; the chip and CPU hit rates are the PREFILL's
prefix_cache_hits and external_prefix_cache_hits over its prefix_cache_queries;
the acceptance length is the decode's 1 + accepted tokens / drafts. A metric the
exports lack is left out of metrics.

--expect scores the point against points[bench][window] of a reference file,
each gate applying where that window's reference has the metric:

    {"gates": {"total_tok_s": {"rel": 0.06}, "chip_hit_pct": {"abs": 3.0}, ...},
     "points": {"k3-repro-1016.p5": {"3600": {"total_tok_s": <ref>, ...}}}}

A gated metric the run lacks fails; a rel gate on a zero reference passes only 0.

Exit 1 when the run has not exactly one concurrency, errors, was cancelled or
completed no request, or a gate fails; 3 when no reference covers the window.
"""

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

# Where agentx_client.sh's HF_HOME puts the unpinned tokenizer and trace corpus.
HF_HUBS = (
    "/root/.cache/huggingface/hub",
    "/tmp/.agentx_scratch/hf_home/hub",
    "/dev/shm/.agentx_scratch/hf_home/hub",
)


def stat(d: dict, name: str, key: str = "avg"):
    v = d.get(name)
    return v.get(key) if isinstance(v, dict) else None


def counters(path: Path) -> dict:
    """{(endpoint role, metric): total} from the counter section of the CSV."""
    out, header = {}, None
    lines = (line for line in path.read_text().splitlines() if not line.startswith("#"))
    for row in csv.reader(lines):
        if not row:
            header = None
            continue
        if row[0] == "Endpoint":
            header = row if "total" in row else None
            continue
        if header is None or len(row) < 5 or row[1] != "counter":
            continue
        rec = dict(zip(header, row))
        roles = [r for r in ("prefill", "decode") if f"-{r}-" in rec["Endpoint"]]
        key = (roles[0] if roles else "?", rec["Metric"])
        out[key] = out.get(key, 0.0) + float(rec["total"] or 0)
    return out


def hf_revisions() -> dict:
    """{repo: commit} of every Hub repo the client fetched at main."""
    revs = {}
    refs = sorted(f for hub in HF_HUBS for f in glob.glob(f"{hub}/*--*/refs/main"))
    for ref in map(Path, refs):
        kind, repo = ref.parent.parent.name.split("--", 1)
        repo = repo.replace("--", "/")
        revs[repo if kind == "models" else f"{kind}/{repo}"] = ref.read_text().strip()
    return revs


def score(metrics: dict, expected: dict, bench: str, window: int) -> dict:
    ref = expected.get("points", {}).get(bench, {}).get(str(window))
    if ref is None:
        print(f"no reference for {bench} at {window} s", file=sys.stderr)
        return {"window": window, "result": "NO_REFERENCE"}
    gates = {}
    for name, gate in expected["gates"].items():
        if name not in ref:
            continue
        [(kind, band)] = gate.items()
        value, want = metrics.get(name), ref[name]
        if value is None:
            print(f"FAIL {name}: absent, reference {want}")
            gates[name] = {"value": None, "ref": want, "ok": False}
            continue
        miss = value - want
        if kind == "rel" and want:
            miss /= want
        elif kind == "rel" and miss:
            miss = float("inf")
        ok = abs(miss) <= band
        shown = f"{miss:+.2%}" if kind == "rel" else f"{miss:+.2f}"
        verdict = "PASS" if ok else "FAIL"
        print(f"{verdict} {name}: {value:.6g} vs {want:.6g} ({shown}, {kind} {band})")
        gates[name] = {"value": value, "ref": want, "ok": ok}
    result = "PASS" if all(g["ok"] for g in gates.values()) else "FAIL"
    print(f"{bench} at {window} s: {result}")
    return {"window": window, "result": result, "gates": gates}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="<dir>/<point> under configs/")
    ap.add_argument("--results", required=True, type=Path, help="<ARTIFACTS_DIR>/...")
    ap.add_argument("--chips", required=True, type=int, help="both roles' chips")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect", type=Path, help="reference numbers and gates")
    ap.add_argument("--window", type=int, help="the run's window (s) for --expect")
    a = ap.parse_args()
    if a.expect and not a.window:
        ap.error("--expect needs --window")
    concs = sorted(a.results.glob("conc_*/aiperf_artifacts/profile_export_aiperf.json"))
    if len(concs) != 1:
        print(f"{len(concs)} concurrencies under {a.results}, not 1", file=sys.stderr)
        return 1
    art = concs[0].parent
    d = json.loads(concs[0].read_text())
    itl_p90 = stat(d, "inter_token_latency", "p90")
    ttft_p90 = stat(d, "time_to_first_token", "p90")
    ttft_p50 = stat(d, "time_to_first_token", "p50")
    total = stat(d, "total_token_throughput")
    errors = d.get("error_summary") or []
    m = {
        "concurrency": int(art.parent.name.removeprefix("conc_")),
        "total_tok_s": total,
        "tok_s_per_chip": total / a.chips if total else None,
        "interactivity_p90": 1000 / itl_p90 if itl_p90 else None,
        "itl_p90_ms": itl_p90,
        "ttft_p90_s": ttft_p90 / 1000 if ttft_p90 is not None else None,
        "ttft_p50_s": ttft_p50 / 1000 if ttft_p50 is not None else None,
        "output_tok_s": stat(d, "output_token_throughput"),
        "input_tok_s": stat(d, "input_token_throughput"),
        "requests": stat(d, "request_count"),
        "benchmark_duration_s": stat(d, "benchmark_duration"),
        "theoretical_hit_pct": stat(d, "theoretical_prefix_cache_hit"),
        "errors": sum(int(e.get("count", 0)) for e in errors if isinstance(e, dict)),
    }
    smf = art / "server_metrics_export.csv"
    c = counters(smf) if smf.exists() else {}
    queries = c.get(("prefill", "vllm:prefix_cache_queries"))
    if queries:
        hits = c.get(("prefill", "vllm:prefix_cache_hits"), 0)
        m["chip_hit_pct"] = 100 * hits / queries
        hits = c.get(("prefill", "vllm:external_prefix_cache_hits"), 0)
        m["cpu_hit_pct"] = 100 * hits / queries
    drafts = c.get(("decode", "vllm:spec_decode_num_drafts"))
    if drafts:
        accepted = c.get(("decode", "vllm:spec_decode_num_accepted_tokens"), 0)
        m["decode_acceptance_length"] = 1 + accepted / drafts
    result = {
        "bench": re.sub(r"-c\d+$", "", a.config.replace("/", ".")),
        "metrics": {k: v for k, v in m.items() if v is not None},
        "info": {
            "config": a.config,
            "aiperf_version": d.get("aiperf_version"),
            "was_cancelled": d.get("was_cancelled"),
            "hf_revisions": hf_revisions(),
        },
    }
    print(json.dumps(result, indent=2))
    problems = []
    if m["errors"]:
        problems.append(f"{m['errors']} requests failed")
    if d.get("was_cancelled"):
        problems.append("the run was cancelled")
    if not m["requests"]:
        problems.append("no request completed")
    for p in problems:
        print(f"agentx_point_metrics: {p}", file=sys.stderr)
    rc = 1 if problems else 0
    if a.expect:
        expected = json.loads(a.expect.read_text())
        result["verdict"] = score(m, expected, result["bench"], a.window)
        if result["verdict"]["result"] == "FAIL":
            rc = 1
        elif result["verdict"]["result"] == "NO_REFERENCE" and rc == 0:
            rc = 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
