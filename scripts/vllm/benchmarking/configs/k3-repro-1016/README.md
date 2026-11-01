# k3-repro-1016: the Kimi-K3 1P1D AgentX points of the Oct 16 recipe set

Each `pN-cM.sh` here is main's `kimi-k3-megakernel-1016/pN-cM.sh`, unchanged, plus
`profile.sh`: every engine also registers `/start_profile` and `/stop_profile`, and
traces at small tracer levels once asked. Nothing traces until `/start_profile`, so
the measured window serves the same program as main's point; a launcher may POST
both heads after the window to capture one.

| Point | Decode | Point | Decode |
|---|---|---|---|
| `p1-c1` … `p5-c32` | DSpark, DCP32 | `p7-c48`, `p8-c72` | no speculation, attention DP2 x DCP16 |
| `p6-c48` | DSpark, attention DP2 x DCP16 | `p9-c48` | no speculation, DCP32 |

The point files say the rest (concurrency, max sequences, drafts, store servers);
`../kimi-k3-megakernel/recipe-base.sh` turns them into the serve flags and
environment of both roles. Each role is TP32 on its own 2x2x4 tpu7x slice: four
hosts per role, 32 chips in all.

## Running a point: `.buildkite/kubernetes/pd_role.sh`

Every pod of the job runs `bash .buildkite/kubernetes/pd_role.sh` from the repo root,
one point per job. It maps a neutral environment onto main's lane
(`kimi-k3-agentx-1p1d.yaml`): the engines run `k3_1p1d.sh`, the client runs
`k3_agentx_client_pod.sh` and then `agentx_point_metrics.py`.

| Variable | Meaning |
|---|---|
| `PD_ROLE` | `prefill`, `decode`, `client`, or `check` (validate the run factors and exit) |
| `PD_CONFIG` | the point, `k3-repro-1016/p5-c32`; one per job |
| `PD_HOSTS` | hosts per role (`4` for these points) |
| `PD_CHIPS` | physical chips of both roles, what per-chip throughput divides by |
| `JS_NAME` | the JobSet; host `i` of role `r` is `<JS_NAME>-<r>-0-<i>.<JS_NAME>` |
| `POD_IP`, `JOB_COMPLETION_INDEX` | the engine's address and host index |
| `PD_IMAGE` | the engine image reference; keys the compilation caches |
| `PD_RUN_ENV_KEYS` | comma list of the run factors the job sets (below) |
| `PD_OUT` | where the client writes; default `$CDK_OUTPUT_DIR` |

Engines serve until the JobSet ends and never exit on success. The client writes
`$PD_OUT/perf_eval_results/k3-agentx-1p1d/` (aiperf's artifacts) and
`$PD_OUT/result.json`; its exit code is the run's verdict. The adapter needs no
cdk: any JobSet with this environment, the mounts below, `/dev/shm` and `/cache/jax`
runs it. On cdk, recipe `vllm-torchtpu-pd-bench` provides all of it:

    cdk job create vllm-torchtpu-pd-bench IMAGE=<repo>@sha256:<digest> \
      CODE_REPO=<this repo> CODE_REF=<40-hex commit> ENTRY=.buildkite/kubernetes/pd_role.sh \
      CONFIG=k3-repro-1016/p5-c32 HOSTS=4 MEMORY=880Gi SHM=256Gi \
      'MOUNTS=<bucket>:<dir>:/ckpt/k3;<bucket>:<dir>:/cache/hf/hub/presharded/kimi-k3-tp32;<bucket>:<dir>:/cache/hf/hub/models--Inferact--Kimi-K3-DSpark'

The points were validated on `docker.io/vllm/vllm-torchtpu@sha256:c86637bddf84176f91154df5109f6f36d9bdfb59155ee7796cf9686eb2795945`.

## Run factors and server changes

`RUN_ENV` may set only `AGENTX_DURATION` (the window, seconds), `AGENTX_FAST`,
`AGENTX_SKIP_POST_EVAL=1` (replay only) and `AGENTX_READY_WAIT_MIN`; any other key
fails every pod at start (exit 2). `recipe-base.sh` reads server knobs from the
environment (`RECIPE_*`, `AGENTX_EVAL_JOB`), so a different server is a different
config file: copy the point and change one `RECIPE_*` before it sources
`recipe-base.sh`, or source the point and append to `SERVE_ARGS` under
`[ "${K3_1P1D_ENGINE:-0}" = 1 ]`. Then commit, and run that commit.

## Inputs

| Path | Content |
|---|---|
| `/ckpt/k3` | `moonshotai/Kimi-K3`, revision `9f62e4e9` |
| `/cache/hf/hub/presharded/kimi-k3-tp32` | the TP32 pre-shard (1.55 TiB): `python -m tools.megakernel preshard k3`, `prefill` and `dspark` at vllm-torchtpu `3af67bd47`, from the checkpoint and the draft |
| `/cache/hf/hub/models--Inferact--Kimi-K3-DSpark` | `Inferact/Kimi-K3-DSpark`, snapshot `cf6b8244620e7ea4b0651d214f28e89eac75bed6` |

`k3_1p1d.sh` refuses to boot without `config.json`, the pre-shard's `layout.json`,
`prefill-sp-cache/layout.json` and, on a DSpark point, the draft's `config.json` and
the pre-shard's `dspark-cf6b8244`. The client
fetches its tokenizer (`moonshotai/Kimi-K3`) and the trace corpus from the Hub at
`main`, unpinned; `result.json` records what it got under `info.hf_revisions`.

## Metrics: `scripts/vllm/benchmark/inferencex/agentx_point_metrics.py`

`result.json` is `{bench, metrics, info}`: `bench` is the config with `/` as `.`
and no `-c<N>` (`k3-repro-1016.p5`); `metrics` holds `concurrency`, `total_tok_s`,
`tok_s_per_chip` (total over `PD_CHIPS`), `interactivity_p90` (1000 / P90
inter-token latency in ms), `itl_p90_ms`, `ttft_p90_s`, `ttft_p50_s`,
`output_tok_s`, `input_tok_s`, `requests`, `benchmark_duration_s`,
`theoretical_hit_pct`, `chip_hit_pct` and `cpu_hit_pct` (the prefill's
`prefix_cache_hits` and `external_prefix_cache_hits` over its
`prefix_cache_queries`), `decode_acceptance_length` (1 + accepted / drafts) and
`errors`; a metric the exports lack is left out. The client fails on errors, a
cancelled run, no request, or other than one concurrency. `--expect <file> --window <s>` scores a run against reference
numbers and gates (format in the script's docstring): exit 1 on a miss, 3 when the
file has no reference for that window.

## Off-TPU render

What a point serves, per role, without a TPU: `k3_1p1d.sh` prints the `vllm serve`
command and its environment when `K3_1P1D_DRY_RUN=1` (the role's hosts must resolve).

    docker run --rm --entrypoint bash -w /workspace/vllm-torchtpu \
      -v "$PWD:/workspace/vllm-torchtpu:ro" -e K3_1P1D_DRY_RUN=1 \
      -e PD_ROLE=decode -e PD_CONFIG=k3-repro-1016/p5-c32 -e PD_HOSTS=4 -e PD_CHIPS=32 \
      -e JS_NAME=js -e POD_IP=10.0.0.10 -e TPU_TOPOLOGY=2x2x4 -e TPU_ACCELERATOR_TYPE=tpu7x \
      --add-host js-prefill-0-0.js:10.0.0.20 --add-host js-decode-0-1.js:10.0.0.11 \
      --add-host js-decode-0-2.js:10.0.0.12 --add-host js-decode-0-3.js:10.0.0.13 \
      <image> .buildkite/kubernetes/pd_role.sh
