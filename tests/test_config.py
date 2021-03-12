from gpjax.config import get_defaults
from ml_collections import ConfigDict


def test_get_defaults():
    config = get_defaults()
    assert isinstance(config, ConfigDict)
    assert isinstance(config.transformations, ConfigDict)

