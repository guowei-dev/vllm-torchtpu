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

# k3-repro-1016/p4-c16, unchanged, with the engines' host tracer at the plugin default (see host_tracer.sh here).
# shellcheck source=scripts/vllm/benchmarking/configs/k3-repro-1016/p4-c16.sh
source "$(dirname "${BASH_SOURCE[0]}")/../k3-repro-1016/p4-c16.sh" || return $?
# shellcheck source=scripts/vllm/benchmarking/configs/k3-roofline/host_tracer.sh
source "$(dirname "${BASH_SOURCE[0]}")/host_tracer.sh"
