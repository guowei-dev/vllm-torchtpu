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

# The one departure of the k3-roofline points from k3-repro-1016's: the engines trace the host at level 2 (the
# plugin's default; k3-repro-1016/profile.sh lowered it to 1), so the runtime's launches, transfers and syncs
# between decode steps are in the trace. The TPU worker reads the level only at /start_profile
# (tracing/options.py), so the served program is k3-repro-1016's. Python stays untraced (its tracer would
# inflate the host gaps under study). Engines only.
[ "${K3_1P1D_ENGINE:-0}" = 1 ] || return 0
export HOST_TRACER_LEVEL=2
