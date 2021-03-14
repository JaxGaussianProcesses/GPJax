from ml_collections import ConfigDict

from gpjax.config import get_defaults


def test_get_defaults():
    config = get_defaults()
    assert isinstance(config, ConfigDict)
    assert isinstance(config.transformations, ConfigDict)
