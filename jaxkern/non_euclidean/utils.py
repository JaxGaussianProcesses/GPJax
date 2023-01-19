from jaxtyping import Num, Array, Int


def jax_gather_nd(
    params: Num[Array, "N ..."], indices: Int[Array, "M"]
) -> Num[Array, "M ..."]:
    """Slice a `params` array at a set of `indices`.

    Args:
        params (Num[Array]): An arbitrary array with leading axes of length `N` upon which we shall slice.
        indices (Float[Int]): An integer array of length M with values in the range [0, N) whose value at index `i` will be used to slice `params` at index `i`.

    Returns:
        Num[Array: An arbitrary array with leading axes of length `M`.
    """
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]
