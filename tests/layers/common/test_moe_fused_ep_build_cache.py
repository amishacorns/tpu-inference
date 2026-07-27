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
"""The fused expert-parallel kernel's build cache: one program per config."""

import inspect
import threading

from absl.testing import absltest
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_moe.v2.kernel import (
    _BUILD_CACHE, build_fused_ep_moe_kernel)

# The build needs a VMEM bound, which a CPU cannot answer.
SERVED_CHIP = pltpu.ChipVersion.TPU_7X


def _restore_cache(saved):
    _BUILD_CACHE.clear()
    _BUILD_CACHE.update(saved)


class KernelBuildCacheTest(jtu.JaxTestCase):
    """Pallas keys its tracing cache on the kernel function's identity, so a
    second build of one configuration retraces every layer using it."""

    BASE_KWARGS = dict(g_local=1,
                       capacity=256,
                       hidden=512,
                       inter=512,
                       ep=8,
                       ragged_rows_alloc=512,
                       rhs_fp4=False,
                       rhs_qb=512)

    def setUp(self):
        super().setUp()
        info = pltpu.get_tpu_info_for_chip(SERVED_CHIP, 1)
        original = pltpu.get_tpu_info
        pltpu.get_tpu_info = lambda: info
        self.addCleanup(setattr, pltpu, "get_tpu_info", original)
        saved = dict(_BUILD_CACHE)
        self.addCleanup(_restore_cache, saved)

    def build(self, **overrides):
        kwargs = dict(self.BASE_KWARGS, **overrides)
        return build_fused_ep_moe_kernel(**kwargs)

    def test_one_key_returns_one_program(self):
        first = self.build(refill_priority=1)
        second = self.build(refill_priority=1)
        self.assertIs(first, second)

    def test_a_changed_schedule_knob_returns_a_different_program(self):
        """Both refill_priority values build, so it has to be in the key."""
        self.assertIsNot(self.build(refill_priority=0),
                         self.build(refill_priority=1))

    def test_a_changed_shape_returns_a_different_program(self):
        self.assertIsNot(self.build(refill_priority=1),
                         self.build(refill_priority=1, capacity=512))

    def test_no_environment_variable_reaches_the_build(self):
        """An environment read is a program switch outside the key's reach."""
        source = inspect.getsource(
            inspect.getmodule(build_fused_ep_moe_kernel))
        self.assertNotIn("os.environ", source)
        self.assertNotIn("getenv", source)

    def test_racing_builds_settle_on_one_program(self):
        _BUILD_CACHE.clear()
        start = threading.Barrier(8)
        built = []
        lock = threading.Lock()

        def race():
            start.wait()
            program = self.build(refill_priority=1)
            with lock:
                built.append(program)

        threads = [threading.Thread(target=race) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertLen(built, 8)
        self.assertLen({id(program) for program in built}, 1)
        self.assertLen(_BUILD_CACHE, 1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
