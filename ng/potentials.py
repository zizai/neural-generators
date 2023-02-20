import chex
from flax.struct import PyTreeNode


class GenericPotential(PyTreeNode):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ConstantPotential(GenericPotential):
    V: chex.Array

    def __call__(self, *args, **kwargs):
        return self.V


class StaticPotential(GenericPotential):
    def __call__(self, r, *args, **kwargs):
        raise NotImplementedError


class TimeVaryingPotential(GenericPotential):
    def __call__(self, r, t, *args, **kwargs):
        raise NotImplementedError


class PoissonPotential(GenericPotential):
    def __call__(self, psi, r, t, *args, **kwargs):
        raise NotImplementedError
