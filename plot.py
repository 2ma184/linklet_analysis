import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm


def add_stats_box(
    ax,
    rows,
    title=None,
    position=(1.0, 1.0),
    fontsize=7
):
    """
    ROOT風の統計ボックスを表示する。

    rows:
        [("Entries", 100), ("Mean", 1.23), ...]
    """

    lines = []

    if title is not None:
        lines.append(str(title))
        lines.append("-" * max(12, len(str(title))))

    for label, value in rows:
        if isinstance(value, (int, np.integer)):
            value_text = f"{value:d}"
        elif isinstance(value, (float, np.floating)):
            value_text = f"{value:.4g}"
        else:
            value_text = str(value)

        lines.append(f"{label:<10} {value_text:>10}")

    ax.text(
        position[0],
        position[1],
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        family="monospace",
        bbox={
            "boxstyle": "square,pad=0.4",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.0,
            "alpha": 0.95
        },
        zorder=10
    )

def add_legend_box(ax, **kwargs):
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="black",
        facecolor="white",
        borderpad=0.35,
        **kwargs
    )
    legend.get_frame().set_linewidth(1.0)
    return legend


class HistPlotter:
    def __init__(self, df):
        self.df = df

    def _get_data_from_selected(self, selected, expression):
        """
        選択済みDataFrameから、列または式を評価して配列を返す。
        この時点ではNaNやinfを削除しない。
        """

        if expression in selected.columns:
            return selected[expression].to_numpy(dtype=float)

        return selected.eval(
            expression,
            engine="python",
            local_dict={"np": np}
        ).to_numpy(dtype=float)

    def _get_data(self, expression, cut=None):
        """
        DataFrame列または数式をNumPy配列として取得する。

        例:
            "gamma_p"
            "track0_p + track1_p"
            "np.sqrt(gamma_ax**2 + gamma_ay**2)"
        """

        selected = self.df.query(cut) if cut else self.df

        data = self._get_data_from_selected(
            selected,
            expression
        )

        return data[np.isfinite(data)]
    
    def hist(
        self,
        expression,
        bins=100,
        range=None,
        cut=None,
        weights=None,
        ax=None,
        label=None,
        legend="auto",
        xlabel=None,
        ylabel="Counts",
        title=None,
        logx=False,
        logy=False,
        density=False,
        histtype="step",
        linewidth=1.0,
        alpha=1.0,
        show_entries=False,
        stats=True,
        stats_position=(1.0, 1.0),
        grid=True,
        **kwargs
    ):
        """
        1次元ヒストグラムを描画する。
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        selected = self.df.query(cut) if cut else self.df

        data = self._get_data_from_selected(
            selected,
            expression
        )

        if weights is not None:
            weight_data = self._get_data_from_selected(
                selected,
                weights
            )

            mask = (
                np.isfinite(data) &
                np.isfinite(weight_data)
            )

            data = data[mask]
            weight_data = weight_data[mask]

        else:
            data = data[np.isfinite(data)]
            weight_data = None

        if label is None:
            label = expression

        if show_entries:
            label = f"{label}  Entries: {len(data)}"

        counts, edges = np.histogram(
            data,
            bins=bins,
            range=range,
            weights=weight_data,
            density=density
        )

        if histtype == "step":
            ax.stairs(
                counts,
                edges,
                label=label,
                linewidth=linewidth,
                **kwargs
            )

        else:
            ax.hist(
                data,
                bins=edges,
                weights=weight_data,
                density=density,
                histtype=histtype,
                label=label,
                linewidth=linewidth,
                alpha=alpha,
                **kwargs
            )
        if range is not None:
            ax.set_xlim(edges[0], edges[-1])
        else:
            ax.margins(x=0.05)

        ax.set_xlabel(xlabel or expression)
        ax.set_ylabel("Density" if density else ylabel)

        if title:
            ax.set_title(title)

        ax.set_xscale("log" if logx else "linear")
        ax.set_yscale("log" if logy else "linear")

        if grid:
            ax.grid(True, alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        if legend == "auto":
            if len(handles) >= 2:
                add_legend_box(ax)

        elif legend:
            add_legend_box(ax)

        if stats:
            stats_data = data

            # 表示範囲内のデータについて統計量を計算
            if range is not None:
                stats_data = stats_data[
                    (stats_data >= range[0]) &
                    (stats_data <= range[1])
                ]

            rows = [
                ("Entries", len(stats_data)),
                (
                    "Mean",
                    np.mean(stats_data)
                    if len(stats_data) > 0 else np.nan
                ),
                (
                    "Std Dev",
                    np.std(stats_data)
                    if len(stats_data) > 0 else np.nan
                )
            ]

            add_stats_box(
                ax,
                rows,
                title=expression,
                position=stats_position
            )

        return fig, ax, counts, edges

    def hist2d(
        self,
        x_expression,
        y_expression,
        bins=100,
        range=None,
        cut=None,
        ax=None,
        xlabel=None,
        ylabel=None,
        title=None,
        logz=False,
        colorbar=True,
        cmap="viridis",
        stats=True,
        stats_position=(1.0, 1.0),
        grid=False,
        **kwargs
    ):
        """
        2次元ヒストグラムを描画する。

        カウント0のビンは透明にして表示しない。
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        selected = self.df.query(cut) if cut else self.df

        x = self._get_data_from_selected(
            selected,
            x_expression
        )

        y = self._get_data_from_selected(
            selected,
            y_expression
        )

        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]

        # hist2dの中身を明示的に計算
        counts, xedges, yedges = np.histogram2d(
            x,
            y,
            bins=bins,
            range=range
        )

        # カウント0のビンをマスク
        masked_counts = np.ma.masked_where(
            counts == 0,
            counts
        )

        # カラーマップをコピーし、マスク部分を透明にする
        cmap_object = plt.get_cmap(cmap).copy()
        cmap_object.set_bad(alpha=0.0)

        positive_counts = counts[counts > 0]

        if logz and len(positive_counts) > 0:
            norm = LogNorm(
                vmin=max(1.0, positive_counts.min()),
                vmax=positive_counts.max()
            )
        else:
            norm = None

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            masked_counts.T,
            cmap=cmap_object,
            norm=norm,
            shading="auto",
            **kwargs
        )

        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])

        ax.set_xlabel(xlabel or x_expression)
        ax.set_ylabel(ylabel or y_expression)

        if title:
            ax.set_title(title)

        if grid:
            ax.grid(True, alpha=0.25)

        if colorbar:
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("Count")

        if stats:
            stats_x = x
            stats_y = y

            # 指定された表示範囲内だけで統計量を計算
            if range is not None:
                (xmin, xmax), (ymin, ymax) = range

                stats_mask = (
                    (stats_x >= xmin) &
                    (stats_x <= xmax) &
                    (stats_y >= ymin) &
                    (stats_y <= ymax)
                )

                stats_x = stats_x[stats_mask]
                stats_y = stats_y[stats_mask]

            rows = [
                ("Entries", len(stats_x)),
                (
                    "Mean x",
                    np.mean(stats_x)
                    if len(stats_x) > 0 else np.nan
                ),
                (
                    "Mean y",
                    np.mean(stats_y)
                    if len(stats_y) > 0 else np.nan
                ),
                (
                    "Std Dev x",
                    np.std(stats_x)
                    if len(stats_x) > 0 else np.nan
                ),
                (
                    "Std Dev y",
                    np.std(stats_y)
                    if len(stats_y) > 0 else np.nan
                )
            ]

            add_stats_box(
                ax,
                rows,
                title=f"{x_expression} : {y_expression}",
                position=stats_position
            )

        h = (
            counts,
            xedges,
            yedges,
            mesh
        )

        return fig, ax, h