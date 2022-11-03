# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

from ml_collections import ConfigDict
from gpjax.config import add_parameter, get_defaults, Identity
import distrax as dx


def test_add_parameter():
    add_parameter("test_parameter", Identity)
    config = get_defaults()
    assert "test_parameter" in config.transformations
    assert "test_parameter_transform" in config.transformations
    assert config.transformations["test_parameter"] == "test_parameter_transform"
    assert isinstance(config.transformations["test_parameter_transform"], dx.Bijector)


def test_add_parameter():
    config = get_defaults()
    add_parameter("test_parameter", Identity)
    config = get_defaults()
    assert "test_parameter" in config.transformations
    assert "test_parameter_transform" in config.transformations
    assert config.transformations["test_parameter"] == "test_parameter_transform"
    assert isinstance(config.transformations["test_parameter_transform"], dx.Bijector)


def test_get_defaults():
    config = get_defaults()
    assert isinstance(config, ConfigDict)
    assert isinstance(config.transformations, ConfigDict)
