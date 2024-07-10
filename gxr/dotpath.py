from typing import Any

__all__ = ("dotget", "dotset", "dotdel")


def dotget(obj: Any, dotpath: str) -> Any:
    """Extract (nested) object attribute based on dot-path specification.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).

    Examples
    --------
    >>> from pathlib import Path
    >>> path = Path(".").absolute()
    >>> dotget(path, "parent.parent") == path.parent.parent
    True
    """
    for attr in dotpath.split("."):
        obj = getattr(obj, attr)
    return obj


def dotset(obj: Any, dotpath: str, value: Any) -> None:
    """Set (nested) object attribute based on dot-path specification.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.
    value
        New attribute value.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).
    """
    try:
        dotpath, attr = dotpath.rsplit(".", 1)
        obj = dotget(obj, dotpath)
    except ValueError:
        attr = dotpath
    setattr(obj, attr, value)


def dotdel(obj: Any, dotpath: str) -> None:
    """Delete (nested) object attribute based on dot-path specification.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).
    """
    try:
        dotpath, attr = dotpath.rsplit(".", 1)
        obj = dotget(obj, dotpath)
    except ValueError:
        attr = dotpath
    delattr(obj, attr)
