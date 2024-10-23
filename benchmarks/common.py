# Copyright 2024 Intel Corporation
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

from onedal import _is_dpc_backend

if _is_dpc_backend:
    from onedal._device_offload import support_input_format
    from onedal.datatypes._data_conversion import from_table, to_table
    from onedal.utils._array_api import _get_sycl_namespace

    @support_input_format(freefunc=True, queue_param=True)
    def dummy_fit_with_functional_support(X, y=None, queue=None):
        X_table = to_table(X)
        result = from_table(X_table)
        return result

    def dummy_fit_with_sua_support(X, y=None, queue=None):
        sua_iface, xp, _ = _get_sycl_namespace(X)
        X_table = to_table(X, sua_iface=sua_iface)
        result = from_table(
            X_table,
            sua_iface=sua_iface,
            sycl_queue=X.sycl_queue,
            xp=xp,
        )
        return result
