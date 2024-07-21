from collections.abc import Mapping
from typing import Any

__all__ = ("dotget", "dotset", "dotdel")


def dotget(obj: Any, dotpath: str, *, item: bool | None = None) -> Any:
    """Extract (nested) object attribute based on dot-path specification.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.
    item
        Should item instead of attribute access be used.

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
    if item is None and isinstance(obj, Mapping):
        item = True
    for attr in dotpath.split("."):
        obj = obj[attr] if item else getattr(obj, attr)
    return obj


def dotset(obj: Any, dotpath: str, value: Any, *, item: bool | None = None) -> None:
    """Set (nested) object attribute based on dot-path specification.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.
    value
        New attribute value.
    item
        Should item instead of attribute access be used.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).
    """
    if item is None and isinstance(obj, Mapping):
        item = True
    try:
        dotpath, attr = dotpath.rsplit(".", 1)
        obj = dotget(obj, dotpath, item=item)
    except ValueError:
        attr = dotpath
    if item:
        obj[attr] = value
    else:
        setattr(obj, attr, value)


def dotdel(obj: Any, dotpath: str, *, item: bool | None = None) -> None:
    """Delete (nested) object attribute based on dot-path specification.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.
    item
        Should item instead of attribute access be used.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).
    """
    if item is None and isinstance(obj, Mapping):
        item = True
    try:
        dotpath, attr = dotpath.rsplit(".", 1)
        obj = dotget(obj, dotpath, item=item)
    except ValueError:
        attr = dotpath
    if item:
        del obj[attr]
    else:
        delattr(obj, attr)
