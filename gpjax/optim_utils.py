import jax.numpy as jnp
from optax import GradientTransformation
import jax.tree_util as jtu
import jax



def optim_builder(optim_pytree):

    def _init_leaf(o, p):
        if isinstance(o, GradientTransformation):
            return o.init(p)
        else:
            return None

    def _update_leaf(o, u, s, p):
        if isinstance(o, GradientTransformation):
            return tuple(o.update(u, s, p))
        else:
            return jtu.tree_map(jnp.zeros_like, p)

    def _get_updates(o, u, p):
        if isinstance(o, GradientTransformation):
            return u[0]
        else:
            return u
    
    def _get_state(o, u):
        if isinstance(o, GradientTransformation):
            return u[1]
        else:
            return None

    def init_fn(params):
        return jtu.tree_map(_init_leaf, optim_pytree, params, is_leaf=lambda x: isinstance(x, GradientTransformation))

    def update_fn(updates, state, params):
        updates_state = jtu.tree_map(_update_leaf, optim_pytree, updates, state, params, is_leaf=lambda x: isinstance(x, GradientTransformation))
        updates = jtu.tree_map(_get_updates, optim_pytree, updates_state, params, is_leaf=lambda x: isinstance(x, GradientTransformation))
        state = jtu.tree_map(_get_state, optim_pytree, updates_state, is_leaf=lambda x: isinstance(x, GradientTransformation))

        return updates, state

    return GradientTransformation(init_fn, update_fn)


def zero_grads():
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return GradientTransformation(init_fn, update_fn)
