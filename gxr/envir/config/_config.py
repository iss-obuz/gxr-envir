from pathlib import Path

from gxr.config import Config as _Config
from gxr.config import catalogue, registry

__all__ = ()

registry.envir: type[registry] = registry.create("envir")

registry.envir.modules = catalogue.create("modules", entry_points=False)
registry.envir.functions = catalogue.create("functions", entry_points=False)
registry.envir.utility = catalogue.create("utility", entry_points=False)
registry.envir.rules = catalogue.create("behavior", "rules", entry_points=False)


class Config(_Config):
    __default_config__ = Path(__file__).parent / "default.cfg"
    __registry__ = registry.envir
