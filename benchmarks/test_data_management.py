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

import pytest
from onedal import _is_dpc_backend

DATA_SHAPE = [(100, 100), (1000, 100), (10000, 100), (100000, 100), (1000000, 100) ]

ROUNDS = 10
ITERATIONS = 1


if _is_dpc_backend:
    import numpy as np

    from .common import (dummy_fit_with_functional_support,
                         dummy_fit_with_sua_support)

    # TODO:
    # use in test cases.
    ORDER_DICT = {"F": np.asfortranarray, "C": np.ascontiguousarray}
    from onedal.tests.utils._dataframes_support import (
        _convert_to_dataframe, get_dataframes_and_queues)
    from sklearn.datasets import make_blobs


# TODO:
# add dtypes, order.
# TODO:
# different shapes to show constant number for the zero-copy.
# TODO:
# 1D and 2D arrays as well.
@pytest.mark.skipif(
    not _is_dpc_backend,
    reason="__sycl_usm_array_interface__ support requires DPC backend.",
)
@pytest.mark.parametrize(
    # "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu, gpu")
    "dataframe,queue", get_dataframes_and_queues("dpnp", "gpu")
)
@pytest.mark.parametrize(
    "datashape", DATA_SHAPE
)
@pytest.mark.parametrize(
    "backend_impl",
    [
        pytest.param(dummy_fit_with_functional_support, id="functional_support"),
        pytest.param(dummy_fit_with_sua_support, id="sua_support"),
    ],
)
def test_data_management_flows(benchmark, backend_impl, datashape, dataframe, queue):
    n_samples, n_features = datashape
    X, _ = make_blobs(
        n_samples=n_samples, centers=3, n_features=n_features, random_state=0
    )
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    result = benchmark.pedantic(
        target=backend_impl,
        args=(X, None),
        kwargs={
            "queue": queue,
        },
        rounds=ROUNDS,
        iterations=ITERATIONS,
    )
