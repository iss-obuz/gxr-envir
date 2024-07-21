from typing import Any, ClassVar

import catalogue  # noqa
from confection import Config as _Config
from confection import registry as _registry

from gxr.typing import PathLike


class registry(_registry):
    @classmethod
    def create(cls, name: str) -> type["registry"]:
        registry = type(
            name,
            (cls,),
            {
                "__module__": cls.__module__,
                "__name__": name,
                "__qualname__": f"{cls.__qualname__}.{name}",
            },
        )
        if hasattr(cls, name):
            errmsg = f"'{name}' sub-registry already exsists on '{cls.__name__}'"
            raise AttributeError(errmsg)
        setattr(cls, name, registry)
        return registry


class Config(_Config):
    __default_config__: ClassVar[PathLike] = None  # type: ignore
    __registry__: ClassVar[type[registry] | None] = None

    def __init_subclass__(cls) -> None:
        if not cls.__default_config__:
            errmsg = (
                f"'{cls.__name__}' must define '__default_config__' class attribute"
            )
            raise TypeError(errmsg)

    def __init__(
        self,
        config: PathLike | dict[str, Any] | _Config | None = None,
        resolve: bool = True,
        interpolate: bool = True,
        **kwds: Any,
    ) -> None:
        static_kwds = {**kwds, "interpolate": False}
        default = _Config().from_disk(self.__default_config__, **static_kwds)
        if isinstance(config, PathLike):
            config = _Config().from_disk(config, **static_kwds)
        config = default.merge(config) if config else default
        if interpolate:
            config = config.interpolate()
        if resolve and self.__registry__:
            config = self.__registry__.resolve(config, validate=False)
        super().__init__(config)
