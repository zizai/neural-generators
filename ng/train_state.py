from typing import Any, Callable, Dict

import chex
import optax
from flax import struct
from flax.core import FrozenDict


class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: FrozenDict[str, Any]
    stats: Dict[str, Any]
    opt: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.opt.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        ), updates

    @classmethod
    def create(cls, *, params: chex.ArrayTree, stats: Dict, opt: optax.GradientTransformation = None, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        if opt is not None:
            opt_state = opt.init(params)
        else:
            opt_state = None
        return cls(
            step=0,
            params=params,
            stats=stats,
            opt=opt,
            opt_state=opt_state,
            **kwargs,
        )
