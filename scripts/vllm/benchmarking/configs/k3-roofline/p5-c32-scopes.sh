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

# k3-roofline/p5-c32 with named phase scopes in the decode megakernel: the decode engine installs
# KernelConfig(phase_scopes=True) (TPU_K3_PHASE_SCOPES) and libtpu traces custom-call regions, so each phase
# of the decoder step is a scope on the XLA TraceMe line. This compiles a different decode program (the
# bench measured ~+1.9 %): run it beside k3-roofline/p5-c32 in one job, LAST (its entry clears the pod's
# compile caches first). Prefill is k3-roofline/p5-c32's.
# shellcheck source=scripts/vllm/benchmarking/configs/k3-roofline/p5-c32.sh
source "$(dirname "${BASH_SOURCE[0]}")/p5-c32.sh" || return $?
[ "${K3_1P1D_ENGINE:-0}" = 1 ] && [ "${K3_1P1D_ROLE:-}" = decode ] || return 0
export TPU_K3_PHASE_SCOPES=1
export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
