# Copyright 2023 The GPJax Contributors. All Rights Reserved.
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
# ==============================================================================

from jax import config
import pytest

from gpjax.decision_making.utility_functions.base import AbstractUtilityFunctionBuilder

config.update("jax_enable_x64", True)


def test_abstract_utility_function_builder():
    with pytest.raises(TypeError):
        AbstractUtilityFunctionBuilder()
