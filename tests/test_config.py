# # Copyright 2022 The GPJax Contributors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

# import jax
# import distrax as dx
# from jax.config import config
# from ml_collections import ConfigDict

# from gpjax.config import (
#     Identity,
#     get_global_config,
# )

# # Enable Float64 for more stable matrix inversions.
# config.update("jax_enable_x64", True)

# # TODO: Fix this test.
# # This test needs to be run first to ensure that the global config is not set on library import.
# # def test_config_on_library_import():
# #     assert config is None


# def test_get_global_config():
#     config = get_global_config()
#     assert isinstance(config, ConfigDict)
#     assert isinstance(config.transformations, ConfigDict)


# def test_x64_based_config_update():
#     cached_jax_precision = jax.config.x64_enabled

#     jax.config.update("jax_enable_x64", True)
#     config = get_global_config()
#     assert config.x64_state is True

#     jax.config.update("jax_enable_x64", False)
#     config = get_global_config()
#     assert config.x64_state is False

#     # Reset the JAX precision to the original value.
#     jax.config.update("jax_enable_x64", cached_jax_precision)
#     get_global_config()
