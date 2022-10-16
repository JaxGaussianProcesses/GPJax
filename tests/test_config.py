from ml_collections import ConfigDict
import numpyro.distributions as npd
from gpjax.config import add_parameter, get_defaults


def test_add_parameter():
    add_parameter("test_parameter", npd.transforms.IdentityTransform())
    config = get_defaults()
    assert "test_parameter" in config.transformations
    assert "test_parameter_transform" in config.transformations
    assert config.transformations["test_parameter"] == "test_parameter_transform"
    assert isinstance(config.transformations["test_parameter_transform"], npd.transforms.Transform)


def test_add_parameter():
    config = get_defaults()
    add_parameter("test_parameter", npd.transforms.IdentityTransform())
    config = get_defaults()
    assert "test_parameter" in config.transformations
    assert "test_parameter_transform" in config.transformations
    assert config.transformations["test_parameter"] == "test_parameter_transform"
    assert isinstance(config.transformations["test_parameter_transform"], npd.transforms.Transform)


def test_get_defaults():
    config = get_defaults()
    assert isinstance(config, ConfigDict)
    assert isinstance(config.transformations, ConfigDict)
