from flax.experimental import nnx

from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    PositiveReal,
    transform,
)


def test_transform():
    # Create mock parameters and bijectors
    params = nnx.State(
        {
            "param1": PositiveReal(1.0),
            "param2": Parameter(2.0),
        }
    )

    # Test forward transformation
    t_params = transform(params, DEFAULT_BIJECTION)
    t_param1_expected = DEFAULT_BIJECTION[PositiveReal].forward(1.0)
    assert t_params.variables["param1"].value == t_param1_expected
    assert t_params.variables["param2"].value == 2.0

    # Test inverse transformation
    t_params = transform(params, DEFAULT_BIJECTION, inverse=True)
    t_param1_expected = DEFAULT_BIJECTION[PositiveReal].inverse(1.0)
    assert t_params.variables["param1"].value == t_param1_expected
    assert t_params.variables["param2"].value == 2.0


# Run the tests
test_transform()
