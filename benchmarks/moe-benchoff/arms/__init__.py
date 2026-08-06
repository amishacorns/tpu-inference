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

"""The arms. The registry is this directory: an arm exists when its file does.

Each arms/<name>.py exposes bind(tree_path, shape, profile_flags) and returns
the dict run.py's docstring specifies. Nothing is imported here, so importing
one arm never drags another arm's tree or external source into the process:
two arms that bind different trees have to run in different processes, and
_sources.import_from_tree refuses rather than silently binding the wrong one.
"""
