"""Miscellaneous utilityes."""
from typing import Any, Optional
from importlib import import_module


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
    """
    for attr in dotpath.split("."):
        obj = getattr(obj, attr)
    return obj

def dotimport(dotpath: str) -> Any:
    """Import module and extract object based on dot-path specification.

    Parameters
    ----------
    dotpath
        Dot-path of the form ``"module1.module2...modulen:obj.attr1.attr2..."``.
        The part on the left of ``:`` defines the module import specification
        and the part on the right is passed to :func:`dotget` and used
        to extract an object or its attribute from the imported module.

    Raises
    ------
    ImportError
        If the import specification is incorrect.
    AttributeError
        If the object/attribute specification is incorrect.
    ValueError
        If the dot-path is incorrect and has more than one ``:`` separators.
    """
    if dotpath.count(":") > 1:
        raise ValueError(f"dot-path '{dotpath}' has more than one ':' separator")
    module_spec, *obj_spec = dotpath.split(":")
    module = import_module(module_spec)
    if not obj_spec:
        return module
    return dotget(module, obj_spec.pop())


def obj_from_dict(
    dct: dict[str, Any],
    factory_key: str = "@factory",
    *,
    package: Optional[str] = None
) -> Any:
    """Create object from a dict with a special ``@factory`` key."""
    if factory_key not in dct:
        raise KeyError(f"'dct' does not contain '{factory_key}' key")
    dct = dct.copy()
    factory = dct.pop(factory_key)
    if ":" not in factory:
        if package is None:
            package = __package__
        factory = f"{package}:{factory}"
    return dotimport(factory)(**dct)
