from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

from gxr.typing import Axes, AxesGrid, Figure

from .dynamics import EnvirDynamics, EnvirDynamicsResults
from .model import EnvirModel


class DynamicsPlotter:
    """Dynamics plotter.

    Attributes
    ----------
    dynamics
        Environmental dynamics instance.
    results
        Results of environmental dynamics simulation.
    rc
        Dictionary with :mod:`matplotlib` style params.
    """

    def __init__(
        self,
        dynamics: EnvirDynamics,
        results: EnvirDynamicsResults,
        rc: dict | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.results = results
        self.rc = {
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "figure.labelsize": 16,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "patch.edgecolor": "black",
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.shadow": True,
            "legend.facecolor": "#EDEDED",
            "legend.edgecolor": "black",
            "legend.fontsize": 14,
            "legend.loc": "best",
            **(rc or {}),
        }

    @property
    def model(self) -> EnvirModel:
        return self.dynamics.model

    def subplots(self, *args: Any, **kwds: Any) -> tuple[Figure, Axes | AxesGrid]:
        with mpl.rc_context(self.rc):
            fig, axes = plt.subplots(*args, **kwds)
            fig.supxlabel("Time")
            return fig, axes

    def plot_state(
        self,
        ax: mpl.axes.Axes,
        index: slice | None = None,
        *,
        title: str = r"Environment state",
        show_opt: bool | dict = True,
        show_Kh: bool | dict = False,
        show_vicious: bool | dict = False,
        show_perceived: bool | dict = False,
        **kwds: Any,
    ) -> mpl.axes.Axes:
        """Plot environment state.

        Parameters
        ----------
        show_opt, show_Kh
            Whether to show optimal state and adjusted carrying capacity.
            Aesthetic parameters can be passed as non-falsy dictionaries.
        **kwds
            Passed to ``ax.plot()``.
        """
        index = index or slice(None)
        with mpl.rc_context(self.rc):
            T = self.results.T[index]
            E = self.results.E[index]
            ax.set_title(title)
            ax.set_ylim(0, self.model.envir.K * 1.02)
            if kws := self._get_kws(show_opt, {"ls": "--", "lw": 3, "color": "red"}):
                ax.axhline(self.model.envir.K / 2, **kws)
            if kws := self._get_kws(show_Kh, {"ls": ":", "lw": 2, "color": "gray"}):
                h = self.results.H[:, index].sum(axis=0)
                r, K = self.model.envir.get_params()
                _, Kh = self.model.envir.adjust_params(r, K, h)
                ax.plot(T, Kh, **kws)
            if kws := self._get_kws(show_vicious, {"color": "red", "alpha": 0.25}):
                B0, B1 = self.dynamics.get_vicious_bounds(self.results)[index].T
                ax.fill_between(T, B0, B1, **kws)
            if kws := self._get_kws(
                show_perceived,
                {
                    "alpha": 0.5,
                    "ls": "--",
                    "color": "gray",
                },
            ):
                Ehat = self.results.Ehat[index]
                ax.plot(T, Ehat, **{**kwds, **kws})
            ax.plot(T, E, **kwds)
            return ax

    def plot_harvesting(
        self,
        ax: mpl.axes.Axes,
        index: slice | None = None,
        *,
        title: str = "Harvesting",
        show_opt: bool | dict = True,
        show_hi: bool | dict = True,
        **kwds: Any,
    ) -> mpl.axes.Axes:
        """Plot harvesting rates.

        Parameters
        ----------
        show_opt, show_hi
            Whether to show optimal harvesting rate and individual rates.
            Aesthetic parameters can be passed as a non-falsy dictionary.
        **kwds
            Passed to ``ax.plot()`` for plotting the overall harvesting rate.
        """
        index = index or slice(None)
        with mpl.rc_context(self.rc):
            T = self.results.T[index]
            H = self.results.H[:, index]
            if kws := self._get_kws(
                show_hi, {"lw": 0.8, "color": "gray", "alpha": 0.2}
            ):
                ax2 = ax.twinx()
                ax2.set_ylabel("Individual rates", color=kws["color"])
                for h in H:
                    ax2.plot(T, h, **kws, zorder=1)
            if kws := self._get_kws(show_opt, {"ls": "--", "lw": 3, "color": "red"}):
                ax.axhline(self.model.envir.r / 2, **kws, zorder=10)
            fkws = {
                "fontsize": mpl.rcParams["axes.labelsize"],
                "fontweight": mpl.rcParams["axes.labelweight"],
                **kwds,
            }
            ax.set_ylabel("Overall rate", **fkws)
            ax.plot(T, H.sum(axis=0), **kwds, zorder=3)
            ax.set_title(title)
            return ax

    def plot_utilities(
        self,
        ax: mpl.axes.Axes,
        index: slice | None = None,
        *,
        title: str = "Agents' utilities",
        **kwds: Any,
    ) -> mpl.axes.Axes:
        """Plot agents' utilities.

        Parameters
        ----------
        **kwds
            Passed to ``ax.plot()``.
        """
        index = index or slice(None)
        with mpl.rc_context(self.rc):
            T = self.results.T[index]
            U = self.model.utility(self.results.P[:, index])
            ax.set_title(title)
            for u in U:
                ax.plot(T, u, **kwds)
            return ax

    # Internals ------------------------------------------------------------------------

    @staticmethod
    def _get_kws(kws: None | bool | dict, defaults: dict | None = None) -> dict:
        if not kws:
            return {}
        if not isinstance(kws, dict):
            kws = {}
        defaults = defaults or {}
        return {**defaults, **kws}
