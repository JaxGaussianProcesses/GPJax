from ml_collections import ConfigDict
from gpjax.config import get_defaults, add_parameter
from tensorflow_probability.substrates.jax import bijectors as tfb


def test_get_defaults():
    config = get_defaults()
    assert isinstance(config, ConfigDict)
    assert isinstance(config.transformations, ConfigDict)


def test_add_parameter():
    config = get_defaults()
    config = add_parameter(config, ('test', tfb.Identity()))
    assert 'test' in config.transformations
    assert 'custom_test' in config.transformations
    assert config.transformations['test'] == 'custom_test'
    assert isinstance(config.transformations['custom_test'], tfb.Bijector)
