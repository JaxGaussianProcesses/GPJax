from ml_collections import ConfigDict
from gpjax.config import add_parameter, get_defaults, Identity
import distrax as dx


def test_add_parameter():
    add_parameter("test_parameter", Identity)
    config = get_defaults()
    assert "test_parameter" in config.transformations
    assert "test_parameter_transform" in config.transformations
    assert config.transformations["test_parameter"] == "test_parameter_transform"
    assert isinstance(config.transformations["test_parameter_transform"], dx.Bijector)


def test_add_parameter():
    config = get_defaults()
    add_parameter("test_parameter", Identity)
    config = get_defaults()
    assert "test_parameter" in config.transformations
    assert "test_parameter_transform" in config.transformations
    assert config.transformations["test_parameter"] == "test_parameter_transform"
    assert isinstance(config.transformations["test_parameter_transform"], dx.Bijector)


def test_get_defaults():
    config = get_defaults()
    assert isinstance(config, ConfigDict)
    assert isinstance(config.transformations, ConfigDict)
