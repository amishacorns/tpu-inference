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
"""Ask the CPU backend for eight devices so this directory's mesh suites
have a mesh to build. The flag must be set before jax initializes a
backend, which is why it sits at import rather than in a fixture.
"""

import os

_HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"
_xla_flags = os.environ.get("XLA_FLAGS", "")
if not any(f.split("=")[0] == _HOST_DEVICE_FLAG for f in _xla_flags.split()):
    os.environ["XLA_FLAGS"] = (f"{_xla_flags} {_HOST_DEVICE_FLAG}=8").strip()
