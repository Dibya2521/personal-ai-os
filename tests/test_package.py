import re

import synthia


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", synthia.__version__)
