"""Text that can be written anywhere UTF-8 is expected."""


def mended(text: str) -> str:
    r"""Return ``text`` with any lone surrogate kept as its visible escape.

    A model's JSON may escape half an emoji (``"\ud83d"``); that string cannot
    be encoded as UTF-8, so a file, a database or a socket would refuse it.
    """
    return text.encode("utf-8", "backslashreplace").decode("utf-8")
