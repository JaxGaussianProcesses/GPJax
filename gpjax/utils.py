<<<<<<< HEAD
import typing as tp
=======
from typing import Tuple, Union
>>>>>>> a2f978a748a6f13cbfe34d295a58e28272a799ef
from copy import deepcopy

import jax
import jax.numpy as jnp
<<<<<<< HEAD
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
=======
from jax.scipy.linalg import cho_factor, cho_solve, cholesky
from multipledispatch import dispatch
>>>>>>> a2f978a748a6f13cbfe34d295a58e28272a799ef

from .types import Array, Arrays


def chol_solve(L, y):
    lx = solve_triangular(L, y, lower=True)
    x = solve_triangular(L, lx, lower=True, trans=1)
    return x


def I(n: int) -> Array:
    """
    Compute an n x n identity matrix.
    :param n: The size of of the matrix.
    :return: An n x n identity matrix.
    """
    return jnp.eye(n)


def concat_dictionaries(a: dict, b: dict) -> dict:
    """
    Append one dictionary below another. If duplicate keys exist, then the key-value pair of the second supplied
    dictionary will be used.
    """
    return {**a, **b}


def merge_dictionaries(base_dict: dict, in_dict: dict) -> dict:
    """
    This will return a complete dictionary based on the keys of the first matrix. If the same key should exist in the
    second matrix, then the key-value pair from the first dictionary will be overwritten. The purpose of this is that
    the base_dict will be a complete dictionary of values such that an incomplete second dictionary can be used to
    update specific key-value pairs.

    :param base_dict: Complete dictionary of key-value pairs.
    :param in_dict: Subset of key-values pairs such that values from this dictionary will take precedent.
    :return: A merged single dictionary.
    """
    for k, v in base_dict.items():
        if k in in_dict.keys():
            base_dict[k] = in_dict[k]
    return base_dict


def sort_dictionary(base_dict: dict) -> dict:
    """
    Sort a dictionary based on the dictionary's key values.

    :param base_dict: The unsorted dictionary.
    :return: A dictionary sorted alphabetically on the dictionary's keys.
    """
    return dict(sorted(base_dict.items()))


<<<<<<< HEAD
def as_constant(parameter_set: dict, params: list) -> tp.Tuple[dict, dict]:
    base_params = deepcopy(parameter_set)
    sparams = {}
    for param in params:
        sparams[param] = base_params[param]
        del base_params[param]
    return base_params, sparams
=======
def add_parameter(base_dict: dict, key_value: Tuple[str, Union[Arrays, int, float]]) -> dict:
    expanded_dict = deepcopy(base_dict)
    expanded_dict[key_value[0]] = key_value[1]
    return sort_dictionary(expanded_dict)


@dispatch(jnp.DeviceArray)
def standardise(x: jnp.DeviceArray) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
    """
    Standardise a given matrix such that values are distributed according to a unit normal random variable. This is
    primarily designed for standardising a training dataset.

    :param x: A matrix of unstandardised values
    :return: A matrix of standardised values
    """
    xmean = jnp.mean(x, axis=0)
    xstd = jnp.std(x, axis=0)
    return (x - xmean) / xstd, xmean, xstd

>>>>>>> a2f978a748a6f13cbfe34d295a58e28272a799ef


def dict_array_coercion(params) -> tp.Tuple[tp.Callable, tp.Callable]:
    flattened_pytree = jax.tree_util.tree_flatten(params)

    def dict_to_array(parameter_dict) -> jnp.DeviceArray:
        return jax.tree_util.tree_flatten(parameter_dict)[0]

    def array_to_dict(parameter_array) -> tp.Dict:
        return jax.tree_util.tree_unflatten(flattened_pytree[1], parameter_array)

<<<<<<< HEAD
    return dict_to_array, array_to_dict
=======
    :param x: A standardised matrix.
    :param xmean: A mean vector.
    :param xstd: A standard deviation vector.
    :return: A matrix of unstandardised values.
    """
    return (x * xstd) + xmean


def chol_log_det(A):
    """
    Compute the log-determinant of a PD matrix using the matrix's lower Cholesky.
    det(A) = det(LL^T) = det(L)^2 => logdet(A) = 2*logdet(L)
    """
    L = cholesky(A, lower=True)
    return 2*jnp.sum(jnp.log(jnp.diag(L)))


def woodbury_matrix_identity(A, B, C, D, y):
    """
    Compute y'(A + BD^{-1}C)^{-1}y using  Woodbury matrix identity.
    A should be an N x N diagonal matrix here.
    B - N x M matrix
    C - M x N matrix
    D - M x M invertible matrix
    """
    y = y.squeeze()
    Ainv = 1 / jnp.diag(A)
    Ainvy = Ainv * y
    yAinvy = jnp.dot(y, Ainvy)
    CAinv = C * Ainv.reshape(1, -1)

    E = jnp.linalg.inv(D + jnp.dot(CAinv, B))
    yAinvB = jnp.dot(Ainvy.reshape(1, -1), B)
    CAinvy = jnp.dot(C, Ainvy.reshape(-1, 1))
    res = yAinvy - jnp.dot(jnp.dot(yAinvB, E), CAinvy)
    return res.reshape()
>>>>>>> a2f978a748a6f13cbfe34d295a58e28272a799ef
