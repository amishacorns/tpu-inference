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

"""The fused grouped-matmul path, on this branch's own tree.

The same layer entry the production pair arm takes, on the tree this kit
ships in. There the fused pair is unconditional: the knob that used to
choose between fused and unfused is gone, because the layer builds the fused
pair in and picks it wherever the fused kernel supports the shape, so the
plain call IS the fused expert compute, on a tree whose layer signature
carries no use_fused_gmm parameter at all.

Everything else is the pair arm's builder, so the two cannot drift apart in
the frame, the operands, the topology or the exit: only the tree and which
expert compute that tree's plain call is differ. The builder refuses if the
bound tree's plain call is not the fused program, so this arm can never
publish the unfused pair under the fused name.

THE TREE IS THIS CHECKOUT WHERE THIS CHECKOUT CARRIES THE FUSED PAIR, and
this arm's own pin where it does not. The fused grouped matmul is not on
upstream main, so on a clone of upstream the builder's own refusal would be
the only outcome; the arm falls back to its pins.json entry instead,
fetched and hash verified like an external, and binds that checkout. A kit
sitting on the serving tree still measures that tree.

THE TOPOLOGY REVIEW APPLIES HERE TOO, and it lands through the same builder:
this arm's attention data-parallelism follows its scatter flag, the routing
all-gather is gated on the same axis extent on this tree (fused_moe_gmm.py
:701, is_attn_dp at :710, the two combines at :383 and :391), and the two
forms are the same two the production arm has. A cell of this arm at the
upstream-default profile therefore measures the replicated-entry, psum-combine
form rather than a mixture of the two.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _sources  # noqa: E402
import production_pair  # noqa: E402

REQUIRES_EXTERNAL = False

_PIN = "fused_gmm"
# What a tree has to carry to be this arm's tree: the layer entry, the fused
# grouped matmul its plain call reaches, and the two modules that pair reads.
# These are the paths the pin names, so a tree that satisfies them is a tree
# the pin could have been taken from.
_NEEDS = ("tpu_inference/layers/common/fused_moe_gmm.py",
          "tpu_inference/kernels/megablox/gmm_fused.py",
          "tpu_inference/kernels/megablox/gmm_v2.py",
          "tpu_inference/kernels/megablox/gmm_v2_fused_support.py")


def bind(tree_path, shape, profile_flags=None):
    tree, fetched = _sources.tree_for_arm(_PIN, _NEEDS, tree_path)
    bound = production_pair.bind_family(
        tree, shape, profile_flags, fused=True,
        expert_compute="one fused grouped matmul pair (gmm_fused)")
    if fetched:
        # The pin was taken, so the cell names it and the source gate checks
        # the files this arm executed against the pinned hashes.
        bound["pin"] = _PIN
        bound["tree_sha"] = fetched["commit"]
    return bound
