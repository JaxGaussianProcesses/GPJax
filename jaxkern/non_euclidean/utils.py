def jax_gather_nd(params, indices):
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]
