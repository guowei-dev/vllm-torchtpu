#!/bin/bash
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

# shellcheck disable=SC2034  # SERVE_ARGS is read by k3_1p1d.sh
# The one departure of the k3-repro-1016 points from main's: each engine also
# registers /start_profile and /stop_profile (vllm-rs registers them when
# --profiler-config names a profiler). Nothing traces until /start_profile, and
# the TPU worker reads the config only then (worker/tpu_worker.py profile()),
# so the measured window serves main's program; the launcher captures after
# the window. Engines only: the client sources this for the cases.
[ "${K3_1P1D_ENGINE:-0}" = 1 ] || return 0
# Traces small enough to bring back: device and host activity, no Python stacks.
export PYTHON_TRACER_LEVEL=0 HOST_TRACER_LEVEL=1
: "${K3R_PROFILE_DIR:=${TMPDIR:-/dev/shm}/k3mk/profile}"
mkdir -p "$K3R_PROFILE_DIR"
SERVE_ARGS+=(--profiler-config "$(printf '{"profiler":"torch","torch_profiler_dir":"%s","ignore_frontend":true,"torch_profiler_with_stack":false}' "$K3R_PROFILE_DIR")")
