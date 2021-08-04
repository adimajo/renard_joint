import os

import pytest

from renard_joint import PathHandler


def test_path_handler():
    assert PathHandler.MyPathHandler().get_path() == os.path.join(os.environ["DATA"], "internal_data/gt/")
    with PathHandler.PathOverWrite('toto'):
        assert PathHandler.MyPathHandler().get_path() == 'toto'
