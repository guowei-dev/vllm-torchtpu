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
"""What one profile window of a Kimi-K3 1P1D pair holds, read from the xplanes in the client pod.

Decode (the decode head's rank 0, TPU:0): every ``kimi_decoder_stack`` launch is a step. Its rows are output 0's
first dimension; with DSpark the draft's ``s32[slots,128]`` output gives the request slots (the step's padded batch),
without it a slot is a row. A step's period runs to the next launch; its busy time is the union of the XLA modules
inside the period, the rest is the device idling for the host. Prefill (the prefill head's rank 0): forwards counted
from the MLA prefill attention calls (24 MLA layers per forward). The programs are SPMD, so rank 0's shapes are
every rank's.

    python3 check.py --decode <xplane> --prefill <xplane> --min-steps 20 --min-forwards 1 --out window.json

Exit 0 when the window is accepted, 1 when it is not, 2 when a file cannot be read.
"""
import argparse
import json
import os
import re
import statistics
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

MLA_LAYERS = 24
KDA_LAYERS = 69
_STACK = re.compile(r"^%kimi_decoder_stack(?:\.\d+)? = \(bf16\[(\d+),7168\]")
_DRAFT = re.compile(r"^%dspark_mla_fused_draft(?:\.\d+)? = \(s32\[(\d+),128\]")
_COMMIT = re.compile(r"^%kimi_commit_unified_pools(?:\.\d+)? = .*?custom-call\(.*?f8e4m3fn\[(\d+),24,640\]")
_OPNAME = re.compile(r"^%([A-Za-z_][A-Za-z0-9_\-]*?)(?:\.\d+)? = ")


def _plane(path):
    from jax.profiler import ProfileData  # noqa: PLC0415  (jaxlib; CPU only)

    return ProfileData.from_file(path).find_plane_with_name("/device:TPU:0")


def _union(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def _q(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def _lines(plane):
    out = {}
    for line in plane.lines:
        out.setdefault(line.name, []).extend(
            (e.name, e.start_ns, e.start_ns + e.duration_ns) for e in line.events)
    return out


def _span(lines):
    starts = [s for evs in lines.values() for _, s, _ in evs]
    ends = [e for evs in lines.values() for _, _, e in evs]
    return (min(starts), max(ends)) if starts else (0, 0)


def decode(path):
    plane = _plane(path)
    if plane is None:
        return {"file": path, "device_plane": False, "steps": 0}
    lines = _lines(plane)
    ops = lines.get("XLA Ops", [])
    modules = [(s, e) for _, s, e in lines.get("XLA Modules", [])]
    stacks, drafts, commits = [], [], []
    for name, s, e in ops:
        if m := _STACK.match(name):
            stacks.append((s, e, int(m.group(1))))
        elif m := _DRAFT.match(name):
            drafts.append((s, int(m.group(1))))
        elif m := _COMMIT.match(name):
            commits.append((s, int(m.group(1))))
    stacks.sort()
    drafts.sort()
    steps = []
    for i, (s, e, rows) in enumerate(stacks):
        nxt = stacks[i + 1][0] if i + 1 < len(stacks) else None
        slot = next((n for t, n in drafts if s <= t and (nxt is None or t < nxt)), None)
        step = {"start_ns": s, "stack_us": (e - s) / 1e3, "rows": rows, "slots": slot or rows}
        if nxt is not None:
            busy = _union([(max(a, s), min(b, nxt)) for a, b in modules if b > s and a < nxt])
            step.update(period_us=(nxt - s) / 1e3, busy_us=busy / 1e3, idle_us=(nxt - s - busy) / 1e3)
        steps.append(step)
    full = [x for x in steps if "period_us" in x]
    lo, hi = _span(lines)
    scopes = sorted({n for n, _, _ in lines.get("XLA TraceMe", []) if n != "barrier-cores"})

    def dist(key, src):
        v = [x[key] for x in src]
        return {"n": len(v), "min": min(v), "p10": _q(v, .1), "p50": _q(v, .5), "p90": _q(v, .9), "max": max(v),
                "mean": statistics.fmean(v)} if v else None

    hist = {}
    for x in steps:
        hist[f"{x['slots']}x{x['rows'] // max(1, x['slots'])}"] = hist.get(f"{x['slots']}x{x['rows'] // max(1, x['slots'])}", 0) + 1
    return {"file": path, "device_plane": True, "steps": len(steps), "span_s": (hi - lo) / 1e9,
            "slots_x_rows_per_slot": hist, "rows": dist("rows", steps), "slots": dist("slots", steps),
            "stack_us": dist("stack_us", steps), "period_us": dist("period_us", full),
            "busy_us": dist("busy_us", full), "idle_us": dist("idle_us", full),
            "busy_frac": (sum(x["busy_us"] for x in full) / sum(x["period_us"] for x in full)) if full else None,
            "rows_committed_by_rank0": sorted({k for _, k in commits}), "draft": bool(drafts),
            "scope_names": scopes[:64], "scope_count": len(scopes)}


def prefill(path):
    plane = _plane(path)
    if plane is None:
        return {"file": path, "device_plane": False, "forwards": 0}
    lines = _lines(plane)
    calls = {}
    for name, _, _ in lines.get("XLA Ops", []):
        if m := _OPNAME.match(name):
            calls[m.group(1)] = calls.get(m.group(1), 0) + 1
    mla = sum(n for k, n in calls.items() if k.startswith("kimi_mla_flash_prefill"))
    kda = calls.get("kimi_prefill_norm_gather_kda_sp", 0)
    lo, hi = _span(lines)
    busy = _union([(s, e) for _, s, e in lines.get("XLA Modules", [])])
    return {"file": path, "device_plane": True, "span_s": (hi - lo) / 1e9, "mla_prefill_calls": mla,
            "kda_prefill_calls": kda, "forwards": round(mla / MLA_LAYERS), "forwards_by_kda": round(kda / KDA_LAYERS),
            "busy_frac": busy / (hi - lo) if hi > lo else None,
            "top_ops": dict(sorted(calls.items(), key=lambda kv: -kv[1])[:12])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode")
    ap.add_argument("--prefill")
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--min-forwards", type=int, default=1)
    ap.add_argument("--want-scopes", action="store_true", help="decode needs named phase scopes (a -scopes slot)")
    ap.add_argument("--out")
    a = ap.parse_args()
    doc, why = {}, []
    try:
        if a.decode:
            doc["decode"] = decode(a.decode)
            if doc["decode"]["steps"] < a.min_steps:
                why.append(f"decode: {doc['decode']['steps']} steps < {a.min_steps}")
            if a.want_scopes and not doc["decode"].get("scope_count"):
                why.append("decode: no named phase scopes on the XLA TraceMe line")
        if a.prefill:
            doc["prefill"] = prefill(a.prefill)
            if doc["prefill"]["forwards"] < a.min_forwards:
                why.append(f"prefill: {doc['prefill']['forwards']} forwards < {a.min_forwards}")
    except (OSError, RuntimeError, ValueError) as e:
        print(f"check.py: cannot read a trace: {e}", file=sys.stderr)
        return 2
    doc.update(decode_ok=not any(w.startswith("decode") for w in why),
               prefill_ok=not any(w.startswith("prefill") for w in why), why=why)
    text = json.dumps(doc, indent=1, sort_keys=True)
    if a.out:
        with open(a.out + ".part", "w") as f:
            f.write(text + "\n")
        os.replace(a.out + ".part", a.out)
    d, p = doc.get("decode") or {}, doc.get("prefill") or {}
    print(f"check: decode steps={d.get('steps')} slots={((d.get('slots') or {}).get('p50'))} "
          f"rows={((d.get('rows') or {}).get('p50'))} busy={d.get('busy_frac')} | prefill forwards={p.get('forwards')} "
          f"busy={p.get('busy_frac')} | {'; '.join(why) or 'accepted'}")
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main())
