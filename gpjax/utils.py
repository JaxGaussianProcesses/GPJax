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

import jaxutils
import deprecation

depreciate = deprecation.deprecated(
    deprecated_in="0.5.6",
    removed_in="0.6.0",
    details="Use method from jaxutils.config instead.",
)


concat_dictionaries = depreciate(jaxutils.dict.concat_dictionaries)
merge_dictionaries = depreciate(jaxutils.dict.merge_dictionaries)
sort_dictionary = depreciate(jaxutils.dict.sort_dictionary)
dict_array_coercion = depreciate(jaxutils.dict.dict_array_coercion)


__all__ = [
    "concat_dictionaries",
    "merge_dictionaries",
    "sort_dictionary",
    "dict_array_coercion",
]
