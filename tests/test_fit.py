# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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

from jaxutils.dataset import Dataset
from jaxutils.fit import fit
from jaxutils.bijectors import Identity
from jaxutils.module import param, Module
from jaxutils.objective import Objective

import jax.numpy as jnp
import jax.random as jr
import optax as ox


def test_simple_linear_model():
    # (1) Create a dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    # (2) Define your model:
    class LinearModel(Module):
        weight: float = param(Identity)
        bias: float = param(Identity)

        def __call__(self, x):
            return self.weight * x + self.bias

    model = LinearModel(weight=1.0, bias=1.0)

    # (3) Define your loss function:
    class MeanSqaureError(Objective):
        def evaluate(self, model: LinearModel, train_data: Dataset) -> float:
            return jnp.mean((train_data.y - model(train_data.X)) ** 2)

    loss = MeanSqaureError()

    # (4) Train!
    trained_model, hist = fit(
        model=model, objective=loss, train_data=D, optim=ox.sgd(0.001), num_iters=100
    )

    assert len(hist) == 100
    assert isinstance(trained_model, LinearModel)
    assert loss(trained_model, D) < loss(model, D)
