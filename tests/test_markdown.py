import pathlib

from mktestdocs import check_md_file
import pytest


# Ensure that code chunks within any markdown files execute without error
@pytest.mark.parametrize("fpath", pathlib.Path("gpjax/").glob("*.md"), ids=str)
def test_source_good(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("*.md"), ids=str)
def test_docs_good(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.parametrize("fpath", pathlib.Path(".").glob("*.md"), ids=str)
def test_root_good(fpath):
    check_md_file(fpath=fpath, memory=True)
