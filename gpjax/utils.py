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

from beartype.typing import Union
from jaxtyping import Bool, UInt32, Int, Float, Array
from jax.random import KeyArray as JAXKeyArray

OldKeyArray = UInt32[Array, "2"]
KeyArray = Union[OldKeyArray, JAXKeyArray]  # for compatibility regardless of enable_custom_prng setting

ScalarBool = Union[bool, Bool[Array, ""]]
ScalarInt = Union[int, Int[Array, ""]]
ScalarFloat = Union[float, Float[Array, ""]]

VecNOrMatNM = Union[Float[Array, "N"], Float[Array, "N M"]]

__all__ = [
    "KeyArray",
    "ScalarBool",
    "ScalarInt",
    "ScalarFloat",
]
