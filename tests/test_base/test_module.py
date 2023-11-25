# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

import dataclasses
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Generic,
    Iterable,
    TypeVar,
)

from flax import serialization
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from simple_pytree import Pytree
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base.module import (
    Module,
    meta,
    static_field,
)
from gpjax.base.param import param_field


@pytest.mark.parametrize("is_dataclass", [True, False])
def test_init_and_meta_scrambled(is_dataclass):
    class Tree(Module):
        c: float = field(metadata={"c": 4.0})
        b: float = field(metadata={"b": 5.0})
        a: float = field(metadata={"a": 6.0})

        def __init__(self, a, b, c):
            self.b = b
            self.a = a
            self.c = c

    if is_dataclass:
        Tree = dataclass(Tree)

    # Test init
    tree = Tree(1, 2, 3)

    assert isinstance(tree, Module)
    assert isinstance(tree, Pytree)

    assert tree.a == 1
    assert tree.b == 2
    assert tree.c == 3

    # Test meta
    meta_tree = meta(tree)
    assert meta_tree.a == {"a": 6.0}
    assert meta_tree.b == {"b": 5.0}
    assert meta_tree.c == {"c": 4.0}

    # Test replacing changes only the specified field
    new = tree.replace(a=123)
    meta_new = meta(new)

    assert new.a == 123
    assert new.b == 2
    assert new.c == 3

    assert meta_new.a == {"a": 6.0}
    assert meta_new.b == {"b": 5.0}
    assert meta_new.c == {"c": 4.0}


@pytest.mark.parametrize("is_dataclass", [True, False])
def test_scrambled_annotations(is_dataclass):
    class Tree(Module):
        c: float = field(metadata={"c": 4.0})
        b: float = field(metadata={"b": 5.0})
        a: float = field(metadata={"a": 6.0})

        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

    if is_dataclass:
        Tree = dataclass(Tree)

    tree = Tree(1, 2, 3)

    assert isinstance(tree, Module)
    assert isinstance(tree, Pytree)

    assert tree.a == 1
    assert tree.b == 2
    assert tree.c == 3

    meta_tree = meta(tree)
    assert meta_tree.a == {"a": 6.0}
    assert meta_tree.b == {"b": 5.0}
    assert meta_tree.c == {"c": 4.0}


@pytest.mark.parametrize("is_dataclass", [True, False])
def test_scrambled_init(is_dataclass):
    class Tree(Module):
        a: float = field(metadata={"a": 6.0})
        b: float = field(metadata={"b": 5.0})
        c: float = field(metadata={"c": 4.0})

        def __init__(self, a, b, c):
            self.b = b
            self.a = a
            self.c = c

    if is_dataclass:
        Tree = dataclass(Tree)

    tree = Tree(1, 2, 3)

    assert isinstance(tree, Module)
    assert isinstance(tree, Pytree)

    assert tree.a == 1
    assert tree.b == 2
    assert tree.c == 3

    meta_tree = meta(tree)
    assert meta_tree.a == {"a": 6.0}
    assert meta_tree.b == {"b": 5.0}
    assert meta_tree.c == {"c": 4.0}


@pytest.mark.parametrize("is_dataclass", [True, False])
def test_simple_linear_model(is_dataclass):
    class SimpleModel(Module):
        weight: float = param_field(bijector=tfb.Softplus(), trainable=False)
        bias: float

        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias

        def __call__(self, test_point):
            return test_point * self.weight + self.bias

    if is_dataclass:
        SimpleModel = dataclass(SimpleModel)

    model = SimpleModel(1.0, 2.0)

    assert isinstance(model, Module)
    assert isinstance(model, Pytree)

    assert model.weight == 1.0
    assert model.bias == 2.0

    meta_model = meta(model)

    assert isinstance(meta_model.weight["bijector"], tfb.Softplus)
    assert meta_model.weight["trainable"] is False
    assert meta_model.bias == {}

    constrained_model = model.constrain()
    assert constrained_model.weight == tfb.Softplus().forward(1.0)
    assert constrained_model.bias == tfb.Identity().forward(2.0)

    meta_constrained_model = meta(constrained_model)
    assert isinstance(meta_constrained_model.weight["bijector"], tfb.Softplus)
    assert meta_constrained_model.weight["trainable"] is False
    assert meta_constrained_model.bias == {}

    unconstrained_model = constrained_model.unconstrain()
    assert unconstrained_model.weight == 1.0
    assert unconstrained_model.bias == 2.0

    meta_unconstrained_model = meta(unconstrained_model)
    assert isinstance(meta_unconstrained_model.weight["bijector"], tfb.Softplus)
    assert meta_unconstrained_model.weight["trainable"] is False
    assert meta_unconstrained_model.bias == {}

    def loss_fn(model):
        model = model.stop_gradient()
        return (model(1.0) - 2.0) ** 2

    grad = jax.grad(loss_fn)(model)
    assert grad.weight == 0.0
    assert grad.bias == 2.0

    new = model.replace_meta(bias={"amazing": True})
    assert new.weight == 1.0
    assert new.bias == 2.0
    assert model.weight == 1.0
    assert model.bias == 2.0
    assert meta(new).bias == {"amazing": True}
    assert meta(model).bias == {}

    with pytest.raises(ValueError, match="'cool' is not a field of SimpleModel"):
        model.replace_meta(cool={"don't": "think so"})

    with pytest.raises(ValueError, match="'cool' is not a field of SimpleModel"):
        model.update_meta(cool={"don't": "think so"})

    new = model.update_meta(bias={"amazing": True})
    assert new.weight == 1.0
    assert new.bias == 2.0
    assert model.weight == 1.0
    assert model.bias == 2.0
    assert meta(new).bias == {"amazing": True}
    assert meta(model).bias == {}


@pytest.mark.parametrize("is_dataclass", [True, False])
def test_nested_Module_structure(is_dataclass):
    class SubTree(Module):
        c: float = param_field(bijector=tfb.Identity())
        d: float = param_field(bijector=tfb.Softplus())
        e: float = param_field(bijector=tfb.Softplus())

        def __init__(self, c, d, e):
            self.c = c
            self.d = d
            self.e = e

    class Tree(Module):
        a: float = param_field(bijector=tfb.Identity())
        sub_tree: SubTree
        b: float = param_field(bijector=tfb.Softplus())

        def __init__(self, a, sub_tree, b):
            self.a = a
            self.sub_tree = sub_tree
            self.b = b

    if is_dataclass:
        SubTree = dataclass(SubTree)
        Tree = dataclass(Tree)

    tree = Tree(
        a=1.0,
        sub_tree=SubTree(c=2.0, d=3.0, e=4.0),
        b=5.0,
    )

    assert isinstance(tree, Module)
    assert isinstance(tree, Pytree)
    assert isinstance(tree.sub_tree, Module)
    assert isinstance(tree.sub_tree, Pytree)

    assert tree.a == 1.0
    assert tree.b == 5.0
    assert tree.sub_tree.c == 2.0
    assert tree.sub_tree.d == 3.0
    assert tree.sub_tree.e == 4.0

    meta_tree = meta(tree)

    assert isinstance(meta_tree, Module)
    assert isinstance(meta_tree, Pytree)

    assert isinstance(meta_tree.a["bijector"], tfb.Identity)
    assert meta_tree.a["trainable"] is True
    assert isinstance(meta_tree.b["bijector"], tfb.Softplus)
    assert meta_tree.b["trainable"] is True
    assert isinstance(meta_tree.sub_tree.c["bijector"], tfb.Identity)
    assert meta_tree.sub_tree.c["trainable"] is True
    assert isinstance(meta_tree.sub_tree.d["bijector"], tfb.Softplus)
    assert meta_tree.sub_tree.d["trainable"] is True
    assert isinstance(meta_tree.sub_tree.e["bijector"], tfb.Softplus)
    assert meta_tree.sub_tree.e["trainable"] is True

    # Test constrain and unconstrain
    constrained = tree.constrain()

    assert isinstance(constrained, Module)
    assert isinstance(constrained, Pytree)

    assert constrained.a == tfb.Identity().forward(1.0)
    assert constrained.b == tfb.Softplus().forward(5.0)
    assert constrained.sub_tree.c == tfb.Identity().forward(2.0)
    assert constrained.sub_tree.d == tfb.Softplus().forward(3.0)
    assert constrained.sub_tree.e == tfb.Softplus().forward(4.0)

    meta_constrained = meta(constrained)

    assert isinstance(meta_constrained, Module)
    assert isinstance(meta_constrained, Pytree)

    assert isinstance(meta_constrained.a["bijector"], tfb.Identity)
    assert meta_constrained.a["trainable"] is True
    assert isinstance(meta_constrained.b["bijector"], tfb.Softplus)
    assert meta_constrained.b["trainable"] is True
    assert isinstance(meta_constrained.sub_tree.c["bijector"], tfb.Identity)
    assert meta_constrained.sub_tree.c["trainable"] is True
    assert isinstance(meta_constrained.sub_tree.d["bijector"], tfb.Softplus)
    assert meta_constrained.sub_tree.d["trainable"] is True
    assert isinstance(meta_constrained.sub_tree.e["bijector"], tfb.Softplus)
    assert meta_constrained.sub_tree.e["trainable"] is True

    # Test constrain and unconstrain
    unconstrained = tree.unconstrain()

    assert isinstance(unconstrained, Module)
    assert isinstance(unconstrained, Pytree)

    assert unconstrained.a == tfb.Identity().inverse(1.0)
    assert unconstrained.b == tfb.Softplus().inverse(5.0)
    assert unconstrained.sub_tree.c == tfb.Identity().inverse(2.0)
    assert unconstrained.sub_tree.d == tfb.Softplus().inverse(3.0)
    assert unconstrained.sub_tree.e == tfb.Softplus().inverse(4.0)

    meta_unconstrained = meta(unconstrained)

    assert isinstance(meta_unconstrained, Module)
    assert isinstance(meta_unconstrained, Pytree)

    assert isinstance(meta_unconstrained.a["bijector"], tfb.Identity)
    assert meta_unconstrained.a["trainable"] is True
    assert isinstance(meta_unconstrained.b["bijector"], tfb.Softplus)
    assert meta_unconstrained.b["trainable"] is True
    assert isinstance(meta_unconstrained.sub_tree.c["bijector"], tfb.Identity)
    assert meta_unconstrained.sub_tree.c["trainable"] is True
    assert isinstance(meta_unconstrained.sub_tree.d["bijector"], tfb.Softplus)
    assert meta_unconstrained.sub_tree.d["trainable"] is True
    assert isinstance(meta_unconstrained.sub_tree.e["bijector"], tfb.Softplus)
    assert meta_unconstrained.sub_tree.e["trainable"] is True

    # Test updating metadata

    new_subtree = tree.sub_tree.replace_bijector(c=tfb.Softplus(), e=tfb.Identity())
    new_subtree = new_subtree.replace_trainable(c=False, e=False)

    new_tree = tree.replace_bijector(b=tfb.Identity())
    new_tree = new_tree.replace_trainable(b=False)
    new_tree = new_tree.replace(sub_tree=new_subtree)

    assert isinstance(new_tree, Module)
    assert isinstance(new_tree, Pytree)

    assert new_tree.a == 1.0
    assert new_tree.b == 5.0
    assert new_tree.sub_tree.c == 2.0
    assert new_tree.sub_tree.d == 3.0
    assert new_tree.sub_tree.e == 4.0

    meta_new_tree = meta(new_tree)

    assert isinstance(meta_new_tree, Module)
    assert isinstance(meta_new_tree, Pytree)

    assert isinstance(meta_new_tree.a["bijector"], tfb.Identity)
    assert meta_new_tree.a["trainable"] is True
    assert isinstance(meta_new_tree.b["bijector"], tfb.Identity)
    assert meta_new_tree.b["trainable"] is False
    assert isinstance(meta_new_tree.sub_tree.c["bijector"], tfb.Softplus)
    assert meta_new_tree.sub_tree.c["trainable"] is False
    assert isinstance(meta_new_tree.sub_tree.d["bijector"], tfb.Softplus)
    assert meta_new_tree.sub_tree.d["trainable"] is True
    assert isinstance(meta_new_tree.sub_tree.e["bijector"], tfb.Identity)
    assert meta_new_tree.sub_tree.e["trainable"] is False

    # Test stop gradients
    def loss(tree):
        t = tree.stop_gradient()
        return jnp.sum(
            t.a**2
            + t.sub_tree.c**2
            + t.sub_tree.d**2
            + t.sub_tree.e**2
            + t.b**2
        )

    g = jax.grad(loss)(new_tree)

    assert g.a == 2.0
    assert g.sub_tree.c == 0.0
    assert g.sub_tree.d == 6.0
    assert g.sub_tree.e == 0.0
    assert g.b == 0.0


@pytest.mark.parametrize("is_dataclass", [True, False])
@pytest.mark.parametrize("iterable", [list, tuple])
def test_iterable_attribute(is_dataclass, iterable):
    class SubTree(Module):
        a: int = param_field(bijector=tfb.Identity(), default=1)
        b: int = param_field(bijector=tfb.Softplus(), default=2)
        c: int = param_field(bijector=tfb.Identity(), default=3, trainable=False)

        def __init__(self, a=1.0, b=2.0, c=3.0):
            self.a = a
            self.b = b
            self.c = c

    class Tree(Module):
        trees: Iterable

        def __init__(self, trees):
            self.trees = trees

    if is_dataclass:
        SubTree = dataclass(SubTree)
        Tree = dataclass(Tree)

    tree = Tree(iterable([SubTree(), SubTree(), SubTree()]))

    assert isinstance(tree, Module)
    assert isinstance(tree, Pytree)

    assert tree.trees[0].a == 1.0
    assert tree.trees[0].b == 2.0
    assert tree.trees[0].c == 3.0

    assert tree.trees[1].a == 1.0
    assert tree.trees[1].b == 2.0
    assert tree.trees[1].c == 3.0

    assert tree.trees[2].a == 1.0
    assert tree.trees[2].b == 2.0
    assert tree.trees[2].c == 3.0

    meta_tree = meta(tree)

    assert isinstance(meta_tree, Module)
    assert isinstance(meta_tree, Pytree)

    assert isinstance(meta_tree.trees[0].a["bijector"], tfb.Identity)
    assert meta_tree.trees[0].a["trainable"] is True
    assert isinstance(meta_tree.trees[0].b["bijector"], tfb.Softplus)
    assert meta_tree.trees[0].b["trainable"] is True
    assert isinstance(meta_tree.trees[0].c["bijector"], tfb.Identity)
    assert meta_tree.trees[0].c["trainable"] is False

    assert isinstance(meta_tree.trees[1].a["bijector"], tfb.Identity)
    assert meta_tree.trees[1].a["trainable"] is True
    assert isinstance(meta_tree.trees[1].b["bijector"], tfb.Softplus)
    assert meta_tree.trees[1].b["trainable"] is True
    assert isinstance(meta_tree.trees[1].c["bijector"], tfb.Identity)
    assert meta_tree.trees[1].c["trainable"] is False

    assert isinstance(meta_tree.trees[2].a["bijector"], tfb.Identity)
    assert meta_tree.trees[2].a["trainable"] is True
    assert isinstance(meta_tree.trees[2].b["bijector"], tfb.Softplus)
    assert meta_tree.trees[2].b["trainable"] is True
    assert isinstance(meta_tree.trees[2].c["bijector"], tfb.Identity)
    assert meta_tree.trees[2].c["trainable"] is False

    # Test constrain and unconstrain

    constrained_tree = tree.constrain()
    unconstrained_tree = tree.unconstrain()

    assert jtu.tree_structure(unconstrained_tree) == jtu.tree_structure(tree)
    assert jtu.tree_structure(constrained_tree) == jtu.tree_structure(tree)

    assert isinstance(constrained_tree, Module)
    assert isinstance(constrained_tree, Pytree)

    assert isinstance(unconstrained_tree, Module)
    assert isinstance(unconstrained_tree, Pytree)

    assert constrained_tree.trees[0].a == tfb.Identity().forward(1.0)
    assert constrained_tree.trees[0].b == tfb.Softplus().forward(2.0)
    assert constrained_tree.trees[0].c == tfb.Identity().forward(3.0)

    assert constrained_tree.trees[1].a == tfb.Identity().forward(1.0)
    assert constrained_tree.trees[1].b == tfb.Softplus().forward(2.0)
    assert constrained_tree.trees[1].c == tfb.Identity().forward(3.0)

    assert constrained_tree.trees[2].a == tfb.Identity().forward(1.0)
    assert constrained_tree.trees[2].b == tfb.Softplus().forward(2.0)
    assert constrained_tree.trees[2].c == tfb.Identity().forward(3.0)

    assert unconstrained_tree.trees[0].a == tfb.Identity().inverse(1.0)
    assert unconstrained_tree.trees[0].b == tfb.Softplus().inverse(2.0)
    assert unconstrained_tree.trees[0].c == tfb.Identity().inverse(3.0)

    assert unconstrained_tree.trees[1].a == tfb.Identity().inverse(1.0)
    assert unconstrained_tree.trees[1].b == tfb.Softplus().inverse(2.0)
    assert unconstrained_tree.trees[1].c == tfb.Identity().inverse(3.0)

    assert unconstrained_tree.trees[2].a == tfb.Identity().inverse(1.0)
    assert unconstrained_tree.trees[2].b == tfb.Softplus().inverse(2.0)
    assert unconstrained_tree.trees[2].c == tfb.Identity().inverse(3.0)


# The following tests are adapted from equinox 🏴‍☠️


def test_Module_not_enough_attributes():
    @dataclass
    class Tree1(Module):
        weight: Any = param_field(bijector=tfb.Identity())

    with pytest.raises(TypeError):
        Tree1()

    @dataclass
    class Tree2(Module):
        weight: Any = param_field(bijector=tfb.Identity())

        def __init__(self):
            return None

    with pytest.raises(TypeError):
        Tree2(1)


def test_Module_too_many_attributes():
    @dataclass
    class Tree1(Module):
        weight: Any = param_field(bijector=tfb.Identity())

    with pytest.raises(TypeError):
        Tree1(1, 2)


def test_Module_setattr_after_init():
    @dataclass
    class Tree(Module):
        weight: Any = param_field(bijector=tfb.Identity())

    m = Tree(1)
    with pytest.raises(AttributeError):
        m.asdf = True


# The main part of this test is to check that __init__ works correctly.
def test_inheritance():
    # no custom init / no custom init

    @dataclass
    class Tree(Module):
        weight: Any = param_field(bijector=tfb.Identity())

    @dataclass
    class Tree2(Tree):
        weight2: Any = param_field(bijector=tfb.Identity())

    m = Tree2(1, 2)
    assert m.weight == 1
    assert m.weight2 == 2
    m = Tree2(1, weight2=2)
    assert m.weight == 1
    assert m.weight2 == 2
    m = Tree2(weight=1, weight2=2)
    assert m.weight == 1
    assert m.weight2 == 2
    with pytest.raises(TypeError):
        m = Tree2(2, weight=2)

    # not custom init / custom init

    @dataclass
    class Tree3(Tree):
        weight3: Any = param_field(bijector=tfb.Identity())

        def __init__(self, *, weight3, **kwargs):
            self.weight3 = weight3
            super().__init__(**kwargs)

    m = Tree3(weight=1, weight3=3)
    assert m.weight == 1
    assert m.weight3 == 3

    # custom init / no custom init

    @dataclass
    class Tree4(Module):
        weight4: Any = param_field(bijector=tfb.Identity())

    @dataclass
    class Tree5(Tree4):
        weight5: Any = param_field(bijector=tfb.Identity())

    with pytest.raises(TypeError):
        m = Tree5(value4=1, weight5=2)

    @dataclass
    class Tree6(Tree4):
        pass

    m = Tree6(weight4=1)
    assert m.weight4 == 1

    # custom init / custom init

    @dataclass
    class Tree7(Tree4):
        weight7: Any = param_field(bijector=tfb.Identity())

        def __init__(self, value7, **kwargs):
            self.weight7 = value7
            super().__init__(**kwargs)

    m = Tree7(weight4=1, value7=2)
    assert m.weight4 == 1
    assert m.weight7 == 2


def test_static_field():
    @dataclass
    class Tree(Module):
        field1: int = param_field(bijector=tfb.Identity())
        field2: int = static_field()
        field3: int = static_field(default=3)

    m = Tree(1, 2)
    flat, treedef = jtu.tree_flatten(m)
    assert len(flat) == 1
    assert flat[0] == 1
    rm = jtu.tree_unflatten(treedef, flat)
    assert rm.field1 == 1
    assert rm.field2 == 2
    assert rm.field3 == 3


def test_init_subclass():
    ran = []

    @dataclass
    class Tree(Module):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            ran.append(True)

    @dataclass
    class AnotherModule(Tree):
        pass

    assert ran == [True]


# Taken from simple-pytree version = 0.1.6 🏴‍☠️


class TestPytree:
    def test_immutable_pytree(self):
        class Foo(Module):
            x: int = static_field()
            y: int

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(
            AttributeError, match="is immutable, trying to update field"
        ):
            pytree.x = 4

    def test_immutable_pytree_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class Foo(Module):
            y: int = field()
            x: int = static_field(2)

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(AttributeError, match="cannot assign to field"):
            pytree.x = 4

    def test_jit(self):
        @dataclasses.dataclass
        class Foo(Module):
            a: int
            b: int = static_field()

        module = Foo(a=1, b=2)

        @jax.jit
        def f(m: Foo):
            return m.a + m.b

        assert f(module) == 3

    def test_flax_serialization(self):
        class Bar(Module):
            a: int = static_field()
            b: int

            def __init__(self, a, b):
                self.a = a
                self.b = b

        @dataclasses.dataclass
        class Foo(Module):
            bar: Bar
            c: int
            d: int = static_field()

        foo: Foo = Foo(bar=Bar(a=1, b=2), c=3, d=4)

        state_dict = serialization.to_state_dict(foo)

        assert state_dict == {
            "bar": {
                "b": 2,
            },
            "c": 3,
        }

        state_dict["bar"]["b"] = 5

        foo = serialization.from_state_dict(foo, state_dict)

        assert foo.bar.b == 5

        del state_dict["bar"]["b"]

        with pytest.raises(ValueError, match="Missing field"):
            serialization.from_state_dict(foo, state_dict)

        state_dict["bar"]["b"] = 5

        # add unknown field
        state_dict["x"] = 6

        with pytest.raises(ValueError, match="Unknown field"):
            serialization.from_state_dict(foo, state_dict)

    def test_generics(self):
        T = TypeVar("T")

        class MyClass(Module, Generic[T]):
            def __init__(self, x: T):
                self.x = x

        MyClass[int]

    def test_key_paths(self):
        @dataclasses.dataclass
        class Bar(Module):
            a: int = 1
            b: int = static_field(2)

        @dataclasses.dataclass
        class Foo(Module):
            x: int = 3
            y: int = static_field(4)
            z: Bar = field(default_factory=Bar)

        foo = Foo()

        path_values, treedef = jax.tree_util.tree_flatten_with_path(foo)
        path_values = [(list(map(str, path)), value) for path, value in path_values]

        assert path_values[0] == ([".x"], 3)
        assert path_values[1] == ([".z", ".a"], 1)

    def test_setter_attribute_allowed(self):
        n = None

        class SetterDescriptor:
            def __set__(self, _, value):
                nonlocal n
                n = value

        class Foo(Module):
            x: int = SetterDescriptor()

        foo = Foo()
        foo.x = 1

        assert n == 1

        with pytest.raises(AttributeError, match=r"<.*> is immutable"):
            foo.y = 2

    def test_replace_unknown_fields_error(self):
        class Foo(Module):
            pass

        with pytest.raises(ValueError, match="'y' is not a field of Foo"):
            Foo().replace(y=1)

    def test_dataclass_inheritance(self):
        @dataclasses.dataclass
        class A(Module):
            a: int = 1
            b: int = static_field(2)

        @dataclasses.dataclass
        class B(A):
            c: int = 3

        pytree = B()
        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [1, 3]


class TestMutablePytree:
    def test_pytree(self):
        class Foo(Module, mutable=True):
            x: int = static_field()
            y: int

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        # test mutation
        pytree.x = 4
        assert pytree.x == 4

    def test_pytree_dataclass(self):
        @dataclasses.dataclass
        class Foo(Module, mutable=True):
            y: int = field()
            x: int = static_field(2)

        pytree: Foo = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        # test mutation
        pytree.x = 4
        assert pytree.x == 4


@pytest.mark.parametrize("is_dataclass", [True, False])
@pytest.mark.parametrize("iterable", [list, tuple])
def test_inheritance_different_meta(is_dataclass, iterable):
    class Tree(Module):
        a: int = param_field(bijector=tfb.Identity(), default=1)
        b: int = param_field(bijector=tfb.Softplus(), default=2)
        c: int = param_field(bijector=tfb.Tanh(), default=0, trainable=False)

        def __init__(self, a=1.0, b=2.0, c=0.0):
            self.a = a
            self.b = b
            self.c = c

    if is_dataclass:
        Tree = dataclass(Tree)

    class SubTree(Tree):
        pass

    tree = SubTree()

    assert isinstance(tree, Module)
    assert isinstance(tree, Pytree)

    assert tree.a == 1.0
    assert tree.b == 2.0
    assert tree.c == 0.0

    meta_tree = meta(tree)

    assert isinstance(meta_tree, Module)
    assert isinstance(meta_tree, Pytree)

    assert isinstance(meta_tree.a["bijector"], tfb.Identity)
    assert meta_tree.a["trainable"] is True
    assert isinstance(meta_tree.b["bijector"], tfb.Softplus)
    assert meta_tree.b["trainable"] is True
    assert isinstance(meta_tree.c["bijector"], tfb.Tanh)
    assert meta_tree.c["trainable"] is False

    # Test constrain and unconstrain

    constrained_tree = tree.constrain()
    unconstrained_tree = tree.unconstrain()

    assert jtu.tree_structure(unconstrained_tree) == jtu.tree_structure(tree)
    assert jtu.tree_structure(constrained_tree) == jtu.tree_structure(tree)

    assert isinstance(constrained_tree, Module)
    assert isinstance(constrained_tree, Pytree)

    assert isinstance(unconstrained_tree, Module)
    assert isinstance(unconstrained_tree, Pytree)

    assert constrained_tree.a == tfb.Identity().forward(1.0)
    assert constrained_tree.b == tfb.Softplus().forward(2.0)
    assert constrained_tree.c == tfb.Tanh().forward(0.0)

    assert unconstrained_tree.a == tfb.Identity().inverse(1.0)
    assert unconstrained_tree.b == tfb.Softplus().inverse(2.0)
    assert unconstrained_tree.c == tfb.Tanh().inverse(0.0)
