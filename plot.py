import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import re
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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

        data = self._get_data_from_selected(selected, expression)

        if weights is not None:
            weight_data = self._get_data_from_selected(selected, weights )

            mask = (np.isfinite(data) & np.isfinite(weight_data) )

            data = data[mask]
            weight_data = weight_data[mask]

        else:
            data = data[np.isfinite(data)]
            weight_data = None

        if label is None:
            label = expression

        if show_entries:
            label = f"{label}  Entries: {len(data)}"

        counts, edges = np.histogram(data, bins=bins, range=range, weights=weight_data, density=density)

        if histtype == "step":
            ax.stairs(counts, edges, label=label, linewidth=linewidth, **kwargs)
        else:
            ax.hist(data, bins=edges, weights=weight_data, density=density, histtype=histtype, label=label, linewidth=linewidth, alpha=alpha, **kwargs)

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
                stats_data = stats_data[(stats_data >= range[0]) & (stats_data <= range[1])]

            rows = [
                ("Entries", len(stats_data)),
                ("Mean", np.mean(stats_data) if len(stats_data) > 0 else np.nan),
                ("Std Dev", np.std(stats_data) if len(stats_data) > 0 else np.nan)
            ]

            add_stats_box(ax, rows, title=expression, position=stats_position)

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

        x = self._get_data_from_selected(selected, x_expression)

        y = self._get_data_from_selected(selected, y_expression)

        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]

        # hist2dの中身を明示的に計算
        counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)

        # カウント0のビンをマスク
        masked_counts = np.ma.masked_where(counts == 0, counts)

        # カラーマップをコピーし、マスク部分を透明にする
        cmap_object = plt.get_cmap(cmap).copy()
        cmap_object.set_bad(alpha=0.0)

        positive_counts = counts[counts > 0]

        if logz and len(positive_counts) > 0:
            norm = LogNorm( vmin=max(1.0, positive_counts.min()), vmax=positive_counts.max() )
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
                ("Mean x", np.mean(stats_x) if len(stats_x) > 0 else np.nan),
                ("Mean y", np.mean(stats_y) if len(stats_y) > 0 else np.nan),
                ("Std Dev x", np.std(stats_x) if len(stats_x) > 0 else np.nan),
                ("Std Dev y", np.std(stats_y) if len(stats_y) > 0 else np.nan)
            ]

            add_stats_box(ax, rows, title=f"{x_expression} : {y_expression}", position=stats_position )

        h = (counts, xedges, yedges, mesh)

        return fig, ax, h




##3D plot

def get_link_numbers(columns, track="a"):
    """
    x0_12_a, x0_23_a,... からリンク番号を自動取得する。
    ["12", "23", "34", "45"]
    """
    pattern = re.compile(rf"^x0_(\d+)_({track})$")
    links = []
    for col in columns:
        match = pattern.match(col)
        if match:
            links.append(match.group(1))

    return sorted(links, key=int)


def reconstruct_layers(event, track="a"):
    """
    4本のリンクレットから5層の位置と傾きを復元する。
    リンクレット間で座標系が異なるため
    各リンク内部の変位 x1-x0, y1-y0, z1-z0 を累積する。
    -------
    positions : shape (5, 3)
        累積座標系での各層の代表点 (x, y, z)
    slopes : shape (5, 2)
        各層の傾き (ax, ay)
    links : list[str]
        リンク番号
    """
    links = get_link_numbers(event.index, track)

    if not links:
        raise ValueError(f"track={track!r} のリンクレットが見つかりません")

    # 各リンク内のtrack 0,1の根元座標
    p0_local = np.array([ [event[f"x0_{link}_{track}"], event[f"y0_{link}_{track}"], event[f"z0_{link}_{track}"]] for link in links ], dtype=float)
    p1_local = np.array([ [event[f"x1_{link}_{track}"], event[f"y1_{link}_{track}"], event[f"z1_{link}_{track}"]] for link in links ], dtype=float)

    a0 = np.array([ [event[f"ax0_{link}_{track}"], event[f"ay0_{link}_{track}"]] for link in links ], dtype=float)
    a1 = np.array([[event[f"ax1_{link}_{track}"], event[f"ay1_{link}_{track}"]] for link in links], dtype=float)

    # 各リンクレット内部での根元間変位
    displacement = p1_local - p0_local

    # 5層の根元位置を累積座標系で復元
    positions = np.empty((len(links) + 1, 3), dtype=float )

    # 最上流plateの根元を基準にする
    # start_layer=5ならx1_45
    positions[-1] = p1_local[-1]

    # 上流から下流へ逆算
    for i in range(len(links) - 1, -1, -1):
        positions[i] = positions[i + 1] - displacement[i]

    # =====================================================
    # 5層の傾き
    #
    # 第1層：最初のリンクのa0
    # 第2～5層：各リンクのa1
    # =====================================================
    slopes = np.empty((len(links) + 1, 2), dtype=float)

    slopes[0] = a0[0]
    slopes[1:] = a1

    return {"positions": positions, "slopes": slopes, "links": links,"p0_local": p0_local, "p1_local": p1_local, "displacement": displacement}

def make_cuboid_edges(xmin, xmax, ymin, ymax, zmin, zmax):
    vertices = np.array([
        [xmin, ymin, zmin],  # 0
        [xmax, ymin, zmin],  # 1
        [xmax, ymax, zmin],  # 2
        [xmin, ymax, zmin],  # 3
        [xmin, ymin, zmax],  # 4
        [xmax, ymin, zmax],  # 5
        [xmax, ymax, zmax],  # 6
        [xmin, ymax, zmax],  # 7
    ])

    edge_indices = [
        # zmin側
        (0, 1), (1, 2), (2, 3), (3, 0),
        # zmax側
        (4, 5), (5, 6), (6, 7), (7, 4),
        # 上下を結ぶ辺
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return [ [vertices[i], vertices[j]] for i, j in edge_indices ]


def plot_3d_tracks(df, event_no=None, show_points=True, show_plates=True, base_dz=200.0, emulsion_dz=80.0, ax=None, box=True, mm_plot=True):
    """
    5層分のtrackと乾板を3D表示する。
    ----------
    event_no=None : 全イベント表示
    event_no=i    : i番イベントだけ表示
    base_dz       : ベースのz方向厚さ [μm]
    emulsion_dz   : 片側乳剤のz方向厚さ [μm]
    """
    if mm_plot:
        base_dz /= 1000.0
        emulsion_dz /= 1000.0

    if event_no is None:
        indices = np.arange(len(df))
        title = f"All pair events ({len(df)} events)"
    else:
        if not 0 <= event_no < len(df):
            raise IndexError(f"event_no must be between 0 and {len(df) - 1}")

        indices = np.array([event_no])
        title = f"Pair event {event_no}"

    # 全イベントで異なる色を割り当てる
    cmap = plt.colormaps["tab10"]
    norm = mpl.colors.Normalize(vmin=0, vmax=max(len(df) - 1, 1))

    all_x = []
    all_y = []
    all_z = []

    # 乾板位置を保存する
    plate_reference_z = None

    for i in indices:
        event = df.iloc[i]
        color = cmap(norm(i))

        geom = reconstruct_layers(event, track="a")
        geom_b = reconstruct_layers(event, track="b")

        if mm_plot:
            pa = geom["positions"].copy()/1e3
            pb = geom_b["positions"].copy()/1e3
        else:
            pa = geom["positions"].copy()
            pb = geom_b["positions"].copy()

        sa = geom["slopes"]
        sb = geom_b["slopes"]

        # 各plateの根元z座標
        layer_z = np.nanmean(np.stack([pa[:, 2], pb[:, 2]]),axis=0)

        # plate番号が増える方向＝上流方向
        upstream_sign = np.sign( np.nanmedian(np.diff(layer_z)) )

        if not np.isfinite(upstream_sign) or upstream_sign == 0:
            raise ValueError("Z座標から上流方向を決定できません")

        # 粒子はvertexから下流へ進む
        downstream_sign = -upstream_sign

        if plate_reference_z is None:
            plate_reference_z = layer_z.copy()

        # ベーストラックを下流側の基準点から上流へ伸ばす
        dz_track = np.array([0.0, upstream_sign * base_dz])

        # =================================================
        # 各層について、測定点と傾きから直線を描く
        #
        # x(z) = x_layer + ax * Δz
        # y(z) = y_layer + ay * Δz
        # =================================================
        for layer in range(len(pa)):
            # Track A
            za = pa[layer, 2] + dz_track
            xa = pa[layer, 0] + sa[layer, 0] * dz_track
            ya = pa[layer, 1] + sa[layer, 1] * dz_track
            # Track B
            zb = pb[layer, 2] + dz_track
            xb = pb[layer, 0] + sb[layer, 0] * dz_track
            yb = pb[layer, 1] + sb[layer, 1] * dz_track

            ax.plot(xa, ya, za, color=color,
                linestyle="-", linewidth=1.5, alpha=1.0,
                label="track A" if event_no is not None and layer == 0 else None )

            ax.plot(xb, yb, zb, color=color,
                linestyle=":", linewidth=1.5, alpha=1.0,
                label="track B" if event_no is not None and layer == 0 else None )

            if show_points:
                # ベーストラックの根元
                ax.scatter(pa[layer, 0], pa[layer, 1], pa[layer, 2],
                        color=color, s=10, marker="o")
                ax.scatter(pb[layer, 0], pb[layer, 1], pb[layer, 2],
                        color=color, s=10, marker="^")

            all_x.extend(xa)
            all_x.extend(xb)
            all_y.extend(ya)
            all_y.extend(yb)
            all_z.extend(za)
            all_z.extend(zb)

        # =================================================
        # vertex
        # =================================================
        vertex_colms = ["vertex_x", "vertex_y", "vertex_z"]

        if set(vertex_colms).issubset(event.index):
            if mm_plot:
                vertex = event[vertex_colms].to_numpy(dtype=float) /1000.0
            else:
                vertex = event[vertex_colms].to_numpy(dtype=float)

            if np.all(np.isfinite(vertex)):
                vx, vy, vz = vertex

                ax.scatter(vx, vy, vz, color=color,
                    s=15, linewidth=2.0, marker="x",
                    label="vertex" if event_no is not None else None,
                )

                all_x.append(vx)
                all_y.append(vy)
                all_z.append(vz)

    # =====================================================
    # 5層分の乾板を直方体として描画
    # =====================================================
    if show_plates and plate_reference_z is not None:
        xmin, xmax = np.nanmin(all_x), np.nanmax(all_x)
        ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)

        xpad = max(0.5 * (xmax - xmin), 1.0)
        ypad = max(0.5 * (ymax - ymin), 1.0)

        xmin -= xpad
        xmax += xpad
        ymin -= ypad
        ymax += ypad

        for zroot in plate_reference_z:
            # zrootはベース下流側の基準面
            # 上流側：+ベース厚さ＋上流側乳剤
            upstream_edge = zroot + upstream_sign * (base_dz + emulsion_dz)
            # 下流側：+下流側乳剤
            downstream_edge = zroot + downstream_sign * emulsion_dz

            zmin = min(upstream_edge, downstream_edge)
            zmax = max(upstream_edge, downstream_edge)

            edges = make_cuboid_edges(xmin, xmax, ymin, ymax, zmin, zmax)
            
            ax.add_collection3d( Line3DCollection(edges, colors="k", linewidths=0.8, alpha=0.6 ) )
            
    # =====================================================
    # 軸設定
    # =====================================================
    if mm_plot:
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
    else:
        ax.set_xlabel("X [um]")
        ax.set_ylabel("Y [um]")
        ax.set_zlabel("Z [um]")
    ax.set_title(title, pad=20)

    ax.grid(False)

    xspan = max(np.ptp(all_x) , 2.0)
    yspan = max(np.ptp(all_y) , 2.0)
    zspan = max(np.ptp(all_z) , 2.0)

    ax.set_box_aspect((xspan,yspan,-zspan))

    # =====================================================
    # 情報・統計ボックス
    # =====================================================
    if box:
        if event_no is not None:
            event = df.iloc[event_no]

            info_lines = [
                f"Event : {event_no}",
                f"$E_\\gamma$ : {event['gamma_E']:.2f} ± "
                f"{event['gamma_E_e']:.2f} MeV",
                f"$p_\\gamma$ : {event['gamma_p']:.2f} ± "
                f"{event['gamma_p_e']:.2f} MeV/c",
                f"$E_A$ : {event['E_a']:.2f} MeV",
                f"$E_B$ : {event['E_b']:.2f} MeV",
                f"Opening angle : {event['open_angle']:.4f} rad",
                f"$a_x$ : {event['gamma_ax']:.4f}",
                f"$a_y$ : {event['gamma_ay']:.4f}",
            ]

            ax.text2D(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes, va="top", ha="left", fontsize=10,
                    bbox= dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

            ax.legend(loc="upper right")

        else:
            energies = df["gamma_E"].dropna()

            stats_lines = [
                f"Events : {len(df)}",
                f"Mean $E_\\gamma$ : {energies.mean():.2f} MeV",
                f"Std $E_\\gamma$ : {energies.std():.2f} MeV",
                f"Min $E_\\gamma$ : {energies.min():.2f} MeV",
                f"Max $E_\\gamma$ : {energies.max():.2f} MeV",
            ]

            ax.text2D(0.02, 0.98,"\n".join(stats_lines), transform=ax.transAxes, va="top", ha="left", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))
        
    return ax


def plot_3d_tracks_all(df, show_plates=True, base_dz=200.0, emulsion_dz=80.0, ax=None, mm_plot=True):
    """
    5層分のtrackと乾板を3D表示する。
    ----------
    event_no=None : 全イベント表示
    event_no=i    : i番イベントだけ表示
    base_dz       : ベースのz方向厚さ [μm]
    emulsion_dz   : 片側乳剤のz方向厚さ [μm]
    """
    if mm_plot:
        base_dz /= 1000.0
        emulsion_dz /= 1000.0

    
    indices = np.arange(len(df))
    title = f"All pair events ({len(df)} events)"
    
    # 全イベントで異なる色を割り当てる
    cmap = plt.colormaps["tab10"]
    norm = mpl.colors.Normalize(vmin=0, vmax=max(len(df) - 1, 1))

    all_x = []
    all_y = []
    all_z = []

    # 乾板位置を保存する
    plate_reference_z = None

    for i in indices:
        event = df.iloc[i]
        color = cmap(norm(i))

        geom = reconstruct_layers(event, track="a")
        geom_b = reconstruct_layers(event, track="b")

        if mm_plot:
            pa = geom["positions"].copy()/1e3
            pb = geom_b["positions"].copy()/1e3
        else:
            pa = geom["positions"].copy()
            pb = geom_b["positions"].copy()

        sa = geom["slopes"]
        sb = geom_b["slopes"]

        # 各plateの根元z座標
        layer_z = np.nanmean(np.stack([pa[:, 2], pb[:, 2]]),axis=0)

        # plate番号が増える方向＝上流方向
        upstream_sign = np.sign( np.nanmedian(np.diff(layer_z)) )

        if not np.isfinite(upstream_sign) or upstream_sign == 0:
            raise ValueError("Z座標から上流方向を決定できません")

        # 粒子はvertexから下流へ進む
        downstream_sign = -upstream_sign

        if plate_reference_z is None:
            plate_reference_z = layer_z.copy()

        # ベーストラックを下流側の基準点から上流へ伸ばす
        dz_track = np.array([0.0, upstream_sign * base_dz])

        # =================================================
        # 各層について、測定点と傾きから直線を描く
        #
        # x(z) = x_layer + ax * Δz
        # y(z) = y_layer + ay * Δz
        # =================================================
        for layer in range(len(pa)):
            # Track A
            za = pa[layer, 2] + dz_track
            xa = pa[layer, 0] + sa[layer, 0] * dz_track
            ya = pa[layer, 1] + sa[layer, 1] * dz_track
            # Track B
            zb = pb[layer, 2] + dz_track
            xb = pb[layer, 0] + sb[layer, 0] * dz_track
            yb = pb[layer, 1] + sb[layer, 1] * dz_track

            ax.plot(xa, ya, za, color=color, linestyle="-", linewidth=1.5, alpha=1.0, label="track A" )

            ax.plot(xb, yb, zb, color=color, linestyle="--", linewidth=1.5, alpha=1.0, label="track B" )

            all_x.extend(xa)
            all_x.extend(xb)
            all_y.extend(ya)
            all_y.extend(yb)
            all_z.extend(za)
            all_z.extend(zb)

        # =================================================
        # vertex
        # =================================================
        vertex_colms = ["vertex_x", "vertex_y", "vertex_z"]

        if set(vertex_colms).issubset(event.index):
            if mm_plot:
                vertex = event[vertex_colms].to_numpy(dtype=float) /1000.0
            else:
                vertex = event[vertex_colms].to_numpy(dtype=float)

            if np.all(np.isfinite(vertex)):
                vx, vy, vz = vertex
                ax.scatter(vx, vy, vz, color=color,s=15, linewidth=1.5, marker="x", label="vertex")

                all_x.append(vx)
                all_y.append(vy)
                all_z.append(vz)

    # =====================================================
    # 5層分の乾板を直方体として描画
    # =====================================================
    if show_plates and plate_reference_z is not None:
 

        xpad = 65
        ypad = 55

        xmin = 0
        xmax = xpad 
        ymin = 0     -5
        ymax = ypad  -5

        for zroot in plate_reference_z:
            # zrootはベース下流側の基準面
            # 上流側：+ベース厚さ＋上流側乳剤
            upstream_edge = zroot + upstream_sign * (base_dz + emulsion_dz)
            # 下流側：+下流側乳剤
            downstream_edge = zroot + downstream_sign * emulsion_dz

            zmin = min(upstream_edge, downstream_edge)
            zmax = max(upstream_edge, downstream_edge)

            # edges = make_cuboid_edges(xmin, xmax, ymin, ymax, zmin, zmax)
            edges = make_cuboid_edges(30-10, 30+10, 20-10, 20+10, zmin, zmax)
            
            ax.add_collection3d( Line3DCollection(edges, colors="k", linewidths=0.8, alpha=0.6 ) )
            
    # =====================================================
    # 軸設定
    # =====================================================
    if mm_plot:
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
    else:
        ax.set_xlabel("X [um]")
        ax.set_ylabel("Y [um]")
        ax.set_zlabel("Z [um]")
    ax.set_title(title, pad=20)

    ax.grid(False)

    xspan = max(np.ptp(all_x) , 2.0)
    yspan = max(np.ptp(all_y) , 2.0)
    zspan = max(np.ptp(all_z) , 2.0)

    # ax.set_box_aspect((xspan,yspan,-zspan*10))
    ax.set_box_aspect((xspan,yspan,-5*zspan))

    return ax

def get_link_numbers_BG(columns):
    pattern = re.compile(rf"^x0_(\d+)$")
    links = []
    for col in columns:
        match = pattern.match(col)
        if match:
            links.append(match.group(1))

    return sorted(links, key=int)

def reconstruct_layers_BG(event):
    links = get_link_numbers_BG(event.index)

    # 各リンク内のtrack 0,1の根元座標
    p0_local = np.array([ [event[f"x0_{link}"], event[f"y0_{link}"], event[f"z0_{link}"]] for link in links ], dtype=float)
    p1_local = np.array([ [event[f"x1_{link}"], event[f"y1_{link}"], event[f"z1_{link}"]] for link in links ], dtype=float)

    a0 = np.array([ [event[f"ax0_{link}"], event[f"ay0_{link}"]] for link in links ], dtype=float)
    a1 = np.array([[event[f"ax1_{link}"], event[f"ay1_{link}"]] for link in links], dtype=float)

    # 各リンクレット内部での根元間変位
    displacement = p1_local - p0_local

    # 5層の根元位置を累積座標系で復元
    positions = np.empty((len(links) + 1, 3), dtype=float )

    # 最上流plateの根元を基準にする
    # start_layer=5ならx1_45
    positions[-1] = p1_local[-1]

    # 上流から下流へ逆算
    for i in range(len(links) - 1, -1, -1):
        positions[i] = positions[i + 1] - displacement[i]

    slopes = np.empty((len(links) + 1, 2), dtype=float)

    slopes[0] = a0[0]
    slopes[1:] = a1

    return {"positions": positions, "slopes": slopes, "links": links,"p0_local": p0_local, "p1_local": p1_local, "displacement": displacement}


def plot_3d_tracks_all_BG(df, base_dz=200.0, emulsion_dz=80.0, ax=None, mm_plot=True):
    if mm_plot:
        base_dz /= 1000.0
        emulsion_dz /= 1000.0

    indices = np.arange(len(df))
    title = f"All pair events ({len(df)} events)"

    ax.computed_zorder = False

    # 全イベントで異なる色を割り当てる
    # cmap = plt.colormaps[""]
    # norm = mpl.colors.Normalize(vmin=0, vmax=max(len(df) - 1, 1))

    all_x = []
    all_y = []
    all_z = []

    # 乾板位置を保存する
    plate_reference_z = None

    for i in indices:
        event = df.iloc[i]

        geom = reconstruct_layers_BG(event)

        if mm_plot:
            p = geom["positions"].copy()/1e3
        else:
            p = geom["positions"].copy()

        s = geom["slopes"]

        # ベーストラックを下流側の基準点から上流へ伸ばす
        dz_track = np.array([0.0,  base_dz])

        for layer in range(5,10):
            z = p[layer, 2] + dz_track
            x = p[layer, 0] + s[layer, 0] * dz_track
            y = p[layer, 1] + s[layer, 1] * dz_track

            ax.plot(x, y, z, color="darkblue", linestyle="-", linewidth=0.5, zorder=10)

            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

        layer_z = p[:, 2]

        if plate_reference_z is None:
            plate_reference_z = layer_z.copy()

    # =====================================================
    # 5層分の乾板を直方体として描画
    # =====================================================
    if plate_reference_z is not None:
        xpad = 10
        ypad = 10

        xmin = -xpad  +30
        xmax = xpad   +30
        ymin = -xpad  +20
        ymax = ypad   +20

        for zroot in plate_reference_z[5:10]:
            # zrootはベース下流側の基準面
            # 上流側：+ベース厚さ＋上流側乳剤
            upstream_edge = zroot + (base_dz + emulsion_dz)
            # 下流側：+下流側乳剤
            downstream_edge = zroot + emulsion_dz

            zmin = min(upstream_edge, downstream_edge)
            zmax = max(upstream_edge, downstream_edge)

            edges = make_cuboid_edges(xmin, xmax, ymin, ymax, zmin, zmax)
            
            ax.add_collection3d(Line3DCollection(edges, colors="k", linewidths=0.8, alpha=0.6, zorder=1))
            
    # =====================================================
    # 軸設定
    # =====================================================
    if mm_plot:
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
    else:
        ax.set_xlabel("X [um]")
        ax.set_ylabel("Y [um]")
        ax.set_zlabel("Z [um]")
    ax.set_title(title, pad=20)

    ax.grid(False)

    xspan = max(np.ptp(all_x) , 2.0)
    yspan = max(np.ptp(all_y) , 2.0)
    zspan = max(np.ptp(all_z) , 2.0)

    ax.set_box_aspect((xspan,yspan,-5*zspan))
    

    return ax