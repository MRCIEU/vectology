from typing import Dict, Tuple

import matplotlib
import pandas as pd
import seaborn as sns
from loguru import logger


def plot_nx_density(
    plot_df: pd.DataFrame, palette: Dict[str, Tuple[float]]
) -> sns.axisgrid.FacetGrid:
    ax = sns.displot(
        x="value",
        hue="Model",
        data=plot_df,
        kind="kde",
        cut=0,
        height=6,
        common_norm=True,
        palette=palette,
    )
    ax.set(xlabel="Weighted average of nx", ylabel="Density")
    # returns a facet grid; use ax.fig to force rendering
    return ax


def plot_violin(plot_df: pd.DataFrame, palette) -> sns.axisgrid.FacetGrid:
    describe = plot_df.groupby("Model").describe()
    logger.info(describe)
    # violin plot
    mean_order = (
        plot_df.groupby("Model")["value"]
        .mean()
        .reset_index()
        .sort_values("value", ascending=False)["Model"]
    )
    logger.info(mean_order)
    ax = sns.catplot(
        x="Model",
        y="value",
        data=plot_df,
        kind="violin",
        order=mean_order,
        palette=palette,
    )
    ax.set(xlabel="Model/Method", ylabel="Weighted average of top 10 batet scores")
    ax.set_xticklabels(rotation=45, ha="right")
    return ax


def plot_top_hits(plot_df: pd.DataFrame) -> matplotlib.figure.Figure:
    totals = plot_df["Total"].tolist()
    plot_df = plot_df.drop(columns=["Total"])
    ax = plot_df.plot.bar(stacked=True, figsize=(12, 6))
    ax.set_xticklabels(plot_df["Model"], rotation=45, ha="right")
    y_offset = 4
    for idx, total in enumerate(totals):
        ax.text(idx, total + y_offset, round(total), ha="center", weight="bold")
    fig = ax.get_figure()
    return fig
