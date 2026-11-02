# Kimi-K3 roofline captures: profile windows inside the serving steady state

A cdk entry for recipe `vllm-torchtpu-pd-bench` (the Kimi-K3 1P1D lane) that captures profile windows of both
engines **inside the point's own AgentX profiling phase**, judges each window in the client pod, and scores nothing.
It feeds the roofline / headroom study in dev_nexus `project/k3-roofline/artifact/roofline/`.

    cdk job create vllm-torchtpu-pd-bench … ENTRY=scripts/roofline/k3/entry.sh CONFIG=k3-roofline/p8-c72 …
    # nexus: pd.toml entry = scripts/roofline/k3/entry.sh, profile = false

| Role | What runs |
|---|---|
| `prefill`, `decode` | `.buildkite/kubernetes/pd_role.sh`, unchanged, plus (in a wrapped pod) a detached watcher that copies each window's traces to `result/profile/<slot>/w<n>/` |
| `client` | `pd_role.sh client` (the lane's AgentX client) in the background; `capture.sh` waits for aiperf's `Phase profiling (profiling) started`, then opens windows; the client is stopped once enough windows are accepted |
| `check` | the run-factor check, with `pd_role.sh`'s four factors |

## A window

1. **Gate**: the decode head's `vllm:num_requests_running` inside the golden run's p10–p90 for the point
   (`steady_refs.json`, from the golden's `server_metrics_export.csv`; the floor is 1: an idle decode is never steady
   state). Polled every 2 s for up to 15 min, else the window is captured and recorded out of band.
2. `/start_profile` on both heads at once, 6 s (15 s at concurrency ≤ 16, where the engines idle most of the time),
   `/stop_profile` on both (it returns once every rank has written), a token file `capture/STOPPED_w<n>`.
3. Every engine host copies its traces out and answers `result/UPLOADED_<slot>_w<n>` with the token.
4. `check.py` reads both heads' rank-0 xplanes here (jaxlib `ProfileData`, CPU): decode steps (each
   `kimi_decoder_stack` launch), rows and request slots per step from the op shapes, period / device busy / device
   idle per step; prefill forwards (MLA prefill calls / 24) and busy fraction. A decode window is accepted when the gate
   held before it, the decode still ran ≥ the band's floor after it, both heads answered 200, every host uploaded and
   the trace holds ≥ 20 steps. Two accepted decode windows (≤ 5 attempts), ≥ 300 s apart; then up to three
   prefill-gated windows if none held a prefill forward.

`capture/w<n>.json` keeps the gate, the timing, the HTTP codes, the uploads and the check; `capture/summary.json` and
`result.json` (the dash row: capture statistics, not benchmark numbers) close the slot. The plan's constants are
`plan.env`: a different plan is a different commit.

## What differs from the golden

The configs under `configs/k3-roofline/` are `k3-repro-1016/<point>` plus `HOST_TRACER_LEVEL=2` on the engines (read
only at `/start_profile`), and the entry is this one, so a `--golden` run labels both departures (`--variant`).

`rehearse.sh <decode xplane> <prefill xplane> [<gauge>]` runs all of it on a CPU with fake heads.
