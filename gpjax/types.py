import typing as tp

import jax.numpy as jnp
import numpy as np
from chex import dataclass

NoneType = type(None)
Array = tp.Union[np.ndarray, jnp.ndarray]

import tensorflow.data as tfd


@dataclass
class Dataset:
    """GPJax Dataset."""

    X: Array
    y: Array = None
    _tf_dataset: tp.Optional[tfd.Dataset] = None
            

    def __repr__(self) -> str:
        return f"- Number of datapoints: {self.X.shape[0]}\n- Dimension:" f" {self.X.shape[1]}"

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        return self.y.shape[1]

    def _make_tfd_if_none(self) -> None:
        if self._tf_dataset is None:
            self._tf_dataset = tfd.Dataset.from_tensor_slices((self.X, self.y))
    
    def cache(self, *args, **kwargs) -> "Dataset":
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.cache(*args, **kwargs)
        return self
    
    def repeat(self, *args, **kwargs) -> "Dataset":
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.repeat(*args, **kwargs)
        return self
    
    def shuffle(self, *args, **kwargs) -> "Dataset":
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.shuffle(buffer_size=self.n, *args, **kwargs)
        return self
    
    def batch(self, *args, **kwargs) -> "Dataset":
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.batch(*args, **kwargs)
        return self
    
    def prefetch(self, *args, **kwargs) -> "Dataset":
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.prefetch(*args, **kwargs)
        return self
    
    def get_batches(self) -> tp.Callable[[None], "Dataset"]:
        self._make_tfd_if_none()
        tfd_iter = iter(self._tf_dataset)

        def next_batch() -> "Dataset":
            X_batch, y_batch = next(tfd_iter)
            
            return Dataset(X=X_batch.numpy(), y=y_batch.numpy())
       
        return next_batch


def verify_dataset(ds: Dataset) -> None:
    assert ds.X.ndim == 2, (
        "2-dimensional training inputs are required. Current dimension:"
        f" {ds.X.ndim}."
    )
    if ds.y is not None:
        assert ds.y.ndim == 2, (
            "2-dimensional training outputs are required. Current dimension:"
            f" {ds.y.ndim}."
        )
        assert ds.X.shape[0] == ds.y.shape[0], (
            "Number of inputs must equal the number of outputs. \nCurrent"
            f" counts:\n- X: {ds.X.shape[0]}\n- y: {ds.y.shape[0]}"
        )
