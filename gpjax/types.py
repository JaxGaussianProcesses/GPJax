import typing as tp

import jax.numpy as jnp
import numpy as np
from chex import dataclass

NoneType = type(None)
Array = tp.Union[np.ndarray, jnp.ndarray]

import tensorflow.data as tfd


@dataclass
class Dataset:
    """GPJax Dataset class. This supports batching through tensorflow.data with the '_tf_dataset' variable."""

    X: Array
    y: Array = None
    _tf_dataset: tp.Optional[tfd.Dataset] = None

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension: {self.X.shape[1]}"
        )

    @property
    def n(self) -> int:
        """The number of observations in the dataset."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        """The dimension of the input data."""
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        """The dimension of the output data."""
        return self.y.shape[1]

    def _make_tfd_if_none(self) -> None:
        """Creates TensorFlow Dataset to facilitate data batching operations."""
        if self._tf_dataset is None:
            self._tf_dataset = tfd.Dataset.from_tensor_slices((self.X, self.y))

    def cache(self, *args, **kwargs) -> "Dataset":
        """Caches dataset elements (same syntax as tensorflow.data.Dataset).
        Args:
            filename: (Optional[str]) The name of a directory on the filesystem to use for caching elements in this Dataset.
            If a filename is not provided, the dataset will be cached in memory.
        Returns:
            Dataset: A GPJax Dataset.
        """
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.cache(*args, **kwargs)
        return self

    def repeat(self, *args, **kwargs) -> "Dataset":
        """Repeats dataset so each original value is seen `count` times (same syntax as tensorflow.data.Dataset).
        Args:
            count: (Optional[int]) An integer, representing the
            number of times the dataset should be repeated. The default behaviour (if
             `count` is `None` or `-1`) is for the dataset be repeated indefinitely.
        Returns:
            Dataset: A GPJax Dataset.
        """
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.repeat(*args, **kwargs)
        return self

    def shuffle(self, *args, **kwargs) -> "Dataset":
        """Randomly shuffles the elements of this dataset (same syntax as tensorflow.data.Dataset).
        Args:
            buffer_size: (Optional[int]) An integer, representing the number of
                elements from this dataset from which the new dataset will sample.
            seed: (Optional[int]) An integer , representing the random
                seed that will be used to create the distribution.
            reshuffle_each_iteration: (Optional[bool]) A boolean, which if true indicates
                that the dataset should be pseudorandomly reshuffled each time it is
                iterated over. (Defaults to `True`.)
        Returns:
            Dataset: A GPJax Dataset.
        """
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.shuffle(*args, **kwargs)
        return self

    def batch(self, *args, **kwargs) -> "Dataset":
        """Combines consecutive dataset elements into batches (same syntax as tensorflow.data.Dataset).
        Args:
            batch_size: (int) An integer representing the number of consecutive elements to combine in a single batch.
            drop_remainder: (Optional[bool]) A boolean, representing
                whether the last batch should be dropped in the case it has fewer than
                `batch_size` elements; the default behaviour is not to drop the smaller
                batch.
            num_parallel_calls: (Optional[int]) An integer representing the number of batches to
                compute asynchronously in parallel.
            deterministic: (Optional[bool]) When `num_parallel_calls` is specified, if this
                boolean is specified (`True` or `False`), it controls the order in which
                the transformation produces elements. If set to `False`, the
                transformation is allowed to yield elements out of order to trade
                determinism for performance.
        Returns:
            Dataset: A GPJax Dataset.
        """
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.batch(*args, **kwargs)
        return self

    def prefetch(self, *args, **kwargs) -> "Dataset":
        """Facilitates prefetching of dataset elements (same syntax as tensorflow.data.Dataset).
        Args:
            buffer_size: (int) A integer, representing the maximum
            number of elements that will be buffered when prefetching.
        Returns:
            Dataset: A GPJax Dataset.
        """
        self._make_tfd_if_none()
        self._tf_dataset = self._tf_dataset.prefetch(*args, **kwargs)
        return self

    def get_batcher(self) -> tp.Callable[[None], "Dataset"]:
        """Creates a data batching object, that loads data from the
        tensorflow.data.Dataset object and returns it as a GPJax dataset.

        Returns:
            tp.Callable[[None], "Dataset"]: A batch loader.
        """
        self._make_tfd_if_none()
        tfd_iter = iter(self._tf_dataset)

        def next_batch() -> "Dataset":
            X_batch, y_batch = next(tfd_iter)

            return Dataset(X=X_batch.numpy(), y=y_batch.numpy())

        return next_batch


def verify_dataset(ds: Dataset) -> None:
    """Apply a series of checks to the dataset to ensure that downstream operations are safe."""
    assert (
        ds.X.ndim == 2
    ), f"2-dimensional training inputs are required. Current dimension: {ds.X.ndim}."
    if ds.y is not None:
        assert (
            ds.y.ndim == 2
        ), f"2-dimensional training outputs are required. Current dimension: {ds.y.ndim}."
        assert ds.X.shape[0] == ds.y.shape[0], (
            "Number of inputs must equal the number of outputs. \nCurrent"
            f" counts:\n- X: {ds.X.shape[0]}\n- y: {ds.y.shape[0]}"
        )
