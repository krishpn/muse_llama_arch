#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Run exploratory data analysis on the real NRB balance-sheet panel.

EDA on the real NRB balance-sheet panel (design doc Sec 8/9/10a), in two groups:

1. Asset scale & coverage -- bank size, reporting coverage over time, and size ranking.
   Answers reviewer pre-mortem item 5 (discard-sample bias) and a sanity check on which banks
   are plausibly "core" before any model is trained.
2. Interbank marginals -- "a. Interbank Lending" (r, row marginal) and "b. Interbank
   Borrowing" (c, column marginal), the literal r_target/c_target fields run_real_experiment.py
   fits W's row/column sums to (data_loader.py:_compute_fields) -- design doc Eq. 5's
   r_{i,t} = sum_j W_{ij,t}, c_{j,t} = sum_i W_{ij,t}, computed here directly from the panel
   rather than from a trained W_hat, not just a covariate like loans_advances/deposits. Checks
   how zero-inflated r/c are in real data (does the model's zero-inflated DGP assumption, see
   synthetic.py, match reality), how skewed they are, and whether a lender/borrower split lines
   up with bank size (the core/periphery story).

Both groups read the same long-format CSV run_real_experiment.py reads
(source_file, year_bs, month_bs, bfi_name, item_code, line_item, value_mn_npr), and both scope
to commercial banks only via data_loader.py's is_commercial_bank() (design doc Sec 8/11).

Usage: uv run evaluation/eda_real_panel.py [--csv PATH] [--out-dir PATH]
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

try:
    from ..models.data_loader import is_commercial_bank
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from data_loader import is_commercial_bank

BS_MONTHS = [
    "Baisakh", "Jestha", "Asar", "Shrawan", "Bhadra", "Ashwin",
    "Kartik", "Mangsir", "Poush", "Magh", "Falgun", "Chaitra",
]
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "nrb_balance_sheet.csv"

# Net lender (r > c, positive net position) / net borrower (c > r, negative net position) is a
# signed quantity -- same warm=positive/cool=negative convention plot_synthetic_network.py uses
# for W_hat - W_true, reused here rather than an arbitrary categorical pair so a reader of both
# figures learns one color meaning for "signed financial position."
NET_LENDER_COLOR = "#b2182b"
NET_BORROWER_COLOR = "#2166ac"


# ---------------------------------------------------------------------------
# Group 1: asset scale & coverage
# ---------------------------------------------------------------------------

def _load_total_assets(csv_path: Path) -> pl.DataFrame:
    """All plots in this group read only commercial banks (design doc Sec 8/11) -- the same
    scope run_real_experiment.py trains on, via data_loader.py's is_commercial_bank(). Applied
    here rather than loading a full Panel, since this script works off the raw long-format CSV
    directly; filter_commercial_banks()'s warn-on-drop convention is kept so a bank silently
    missing from these plots is still visible in stderr, not just inferred from a shorter list.
    """
    df = pl.read_csv(csv_path)
    bfi_names = df.get_column("bfi_name").unique().to_list()
    dropped = sorted(name for name in bfi_names if not is_commercial_bank(name))
    if dropped:
        warnings.warn(
            f"{len(dropped)}/{len(bfi_names)} banks excluded as non-commercial (no match in "
            f"COMMERCIAL_BANK_PREFIXES): {dropped}",
            stacklevel=2,
        )
    return (
        df.filter(
            (pl.col("line_item") == "TOTAL ASSETS")
            & pl.col("value_mn_npr").is_not_null()
            & (pl.col("value_mn_npr") > 0)
            & pl.col("bfi_name").map_elements(is_commercial_bank, return_dtype=pl.Boolean)
        )
        .group_by(["year_bs", "month_bs", "bfi_name"])
        .agg(pl.col("value_mn_npr").mean())
    )


def plot_scale_coverage_heatmap(df: pl.DataFrame, out_path: Path) -> None:
    """Plot a log10 heatmap of annual mean TOTAL ASSETS per bank.

    Annual mean TOTAL ASSETS per bank, log10-scaled. Columns sorted by mean size
    (largest first) rather than left alphabetical -- that ordering is what actually lets a
    reader see "big banks vs. small banks" as a visual block instead of alphabetical soup.
    """
    hm_df = (
        df.group_by(["year_bs", "bfi_name"])
        .agg(pl.col("value_mn_npr").mean())
        .pivot(on="bfi_name", index="year_bs", values="value_mn_npr")
        .sort("year_bs")
        .to_pandas()
        .set_index("year_bs")
    )
    hm_df = hm_df.dropna(axis=1, how="all")
    col_order = hm_df.mean(axis=0, skipna=True).sort_values(ascending=False).index
    hm_df = hm_df[col_order]

    log_vals = np.log10(hm_df)

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e8e8e8")  # NaN (bank not reporting that year) reads as neutral gray, not "zero"

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        log_vals,
        cmap=cmap,
        mask=log_vals.isna(),
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "log10(mean annual TOTAL ASSETS, NPR mn)"},
        ax=ax,
    )
    ax.set_title(
        "Commercial bank asset scale & reporting coverage over time\n"
        "(columns sorted by mean size, largest first; gray = bank not reporting that year)",
        fontsize=13, pad=14,
    )
    ax.set_xlabel("bank (sorted by mean TOTAL ASSETS, descending)", fontsize=11)
    ax.set_ylabel("fiscal year (BS)", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_reporting_coverage_over_time(df: pl.DataFrame, out_path: Path) -> None:
    """Plot the count of banks reporting TOTAL ASSETS per period over time.

    Count of banks reporting TOTAL ASSETS per period, in actual chronological (year, month)
    order -- the heatmap only shows this implicitly via blank cells at *annual* resolution;
    this makes the coverage trend an explicit, directly-readable line instead of something a
    reader has to infer by eyeballing gray cell density.
    """
    counts = (
        df.group_by(["year_bs", "month_bs"])
        .agg(pl.col("bfi_name").n_unique().alias("n_banks"))
        .with_columns(
            month_idx=pl.col("month_bs").replace_strict(
                {m: i for i, m in enumerate(BS_MONTHS)}, default=None
            )
        )
        .sort(["year_bs", "month_idx"])
        .to_pandas()
    )
    period_label = counts["year_bs"].astype(str) + "-" + counts["month_bs"].str.slice(0, 3)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(len(counts)), counts["n_banks"], color="#2166ac", linewidth=1.5, marker="o", markersize=3)
    ax.set_title("Number of commercial banks reporting TOTAL ASSETS, per period", fontsize=13)
    ax.set_xlabel("period (BS year-month)", fontsize=11)
    ax.set_ylabel("banks reporting", fontsize=11)
    step = max(len(counts) // 20, 1)
    ax.set_xticks(range(0, len(counts), step))
    ax.set_xticklabels(period_label.iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_size_rank_bump_chart(df: pl.DataFrame, out_path: Path, top_n: int = 12) -> None:
    """Plot a bump chart of each bank's asset-size rank by year.

    Bump chart: each bank's size RANK (1 = biggest) by year, lines for the `top_n` banks
    that are ever in the top `top_n`. This is the EDA most directly tied to the paper's
    core/periphery story -- size rank stability over time is a cheap, model-free proxy for
    "is there a persistent core," worth checking before trusting a trained block assignment.
    """
    annual = (
        df.group_by(["year_bs", "bfi_name"])
        .agg(pl.col("value_mn_npr").mean())
        .with_columns(
            rank=pl.col("value_mn_npr").rank(method="ordinal", descending=True).over("year_bs")
        )
        .to_pandas()
    )
    ever_top = annual.loc[annual["rank"] <= top_n, "bfi_name"].unique()

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab20", len(ever_top))
    for color, bank in zip(palette, sorted(ever_top)):
        sub = annual[annual["bfi_name"] == bank].sort_values("year_bs")
        ax.plot(sub["year_bs"], sub["rank"], marker="o", markersize=4, linewidth=1.5, color=color, label=bank)

    ax.invert_yaxis()  # rank 1 (biggest) at the top
    ax.set_title(f"Asset-size rank over time (banks ever in top {top_n})", fontsize=13)
    ax.set_xlabel("fiscal year (BS)", fontsize=11)
    ax.set_ylabel("rank by TOTAL ASSETS (1 = largest)", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, title="bank")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_size_distribution_by_year(df: pl.DataFrame, out_path: Path) -> None:
    """Plot per-year bank-size distribution: log boxplot plus raw skew and kurtosis.

    Two stacked panels sharing a year axis (not a dual-axis chart on one panel -- two
    different units don't belong on one y-scale): top is a per-year boxplot of bank size on
    log10 scale (standard for heavy-tailed financial data -- a linear-scale boxplot would be
    unreadable, dominated by the largest 2-3 banks); bottom is skew and excess kurtosis of
    that same year's cross-sectional distribution computed on the RAW (untransformed) values,
    which is design doc Sec 10a item 5's actual ask -- the log transform in the top panel is
    for readability only and would itself suppress most of the skew/kurtosis signal, so the
    bottom panel deliberately does NOT reuse the log-transformed values.
    """
    annual = (
        df.group_by(["year_bs", "bfi_name"])
        .agg(pl.col("value_mn_npr").mean())
        .to_pandas()
    )
    years = sorted(annual["year_bs"].unique())
    raw_by_year = [annual.loc[annual["year_bs"] == y, "value_mn_npr"].to_numpy() for y in years]
    log_by_year = [np.log10(v) for v in raw_by_year]

    def _moments(x: np.ndarray) -> tuple[float, float]:
        z = (x - x.mean()) / x.std(ddof=0)
        skew = np.mean(z**3)
        excess_kurtosis = np.mean(z**4) - 3.0
        return skew, excess_kurtosis

    moments = [_moments(x) for x in raw_by_year]
    skew_vals = [m[0] for m in moments]
    kurt_vals = [m[1] for m in moments]

    fig, (ax_box, ax_moments) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_box.boxplot(log_by_year, positions=range(len(years)), widths=0.6, showfliers=True,
                    boxprops={"color": "#2166ac"}, medianprops={"color": "#b2182b", "linewidth": 1.5},
                    whiskerprops={"color": "#2166ac"}, capprops={"color": "#2166ac"},
                    flierprops={"markeredgecolor": "#2166ac", "markersize": 3, "alpha": 0.5})
    ax_box.set_title(
        "Cross-sectional bank-size distribution by year\n"
        "(top: log10(TOTAL ASSETS) boxplot; bottom: skew/excess kurtosis of RAW values)",
        fontsize=13, pad=12,
    )
    ax_box.set_ylabel("log10(TOTAL ASSETS, NPR mn)", fontsize=10)

    ax_moments.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax_moments.plot(range(len(years)), skew_vals, marker="o", markersize=4, linewidth=1.5,
                     color="#e08214", label="skewness")
    ax_moments.plot(range(len(years)), kurt_vals, marker="s", markersize=4, linewidth=1.5,
                     color="#1b9e77", label="excess kurtosis")
    ax_moments.set_ylabel("moment value\n(0 = normal)", fontsize=10)
    ax_moments.set_xlabel("fiscal year (BS)", fontsize=11)
    ax_moments.set_xticks(range(len(years)))
    ax_moments.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax_moments.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Group 2: interbank marginals (r_target / c_target)
# ---------------------------------------------------------------------------

def _load_interbank_marginals(csv_path: Path) -> pl.DataFrame:
    """Per bank/period: r (interbank lending) and c (interbank borrowing), commercial banks
    only. A full outer join, not an inner one: read_rows drops a line_item entirely for a
    bank/period that reports zero of it (Sec 9 blank-value convention, see
    data_loader.py:_compute_fields's comment on interbank_r/interbank_c), so a bank/period
    present on only one side is a real "one side legitimately zero" case, not missing data --
    filled with 0.0 after the join, not dropped.
    """
    df = pl.read_csv(csv_path)
    lending = (
        df.filter((pl.col("line_item") == "a. Interbank Lending") & pl.col("value_mn_npr").is_not_null())
        .rename({"value_mn_npr": "r"})
        .select(["year_bs", "month_bs", "bfi_name", "r"])
    )
    borrowing = (
        df.filter((pl.col("line_item") == "b. Interbank Borrowing") & pl.col("value_mn_npr").is_not_null())
        .rename({"value_mn_npr": "c"})
        .select(["year_bs", "month_bs", "bfi_name", "c"])
    )
    merged = (
        lending.join(borrowing, on=["year_bs", "month_bs", "bfi_name"], how="full", coalesce=True)
        .with_columns(r=pl.col("r").fill_null(0.0), c=pl.col("c").fill_null(0.0))
        .filter(pl.col("bfi_name").map_elements(is_commercial_bank, return_dtype=pl.Boolean))
    )
    return merged


def _annual_interbank(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(["year_bs", "bfi_name"]).agg(pl.col("r").mean(), pl.col("c").mean())


def plot_zero_activity_rate(df: pl.DataFrame, out_path: Path) -> None:
    """Plot the yearly share of bank-periods with zero interbank lending or borrowing.

    Share of bank-periods reporting zero interbank lending / zero interbank borrowing, per
    year. Directly checks whether real data is as zero-inflated as synthetic.py's log-normal x
    Bernoulli-existence DGP assumes -- if real zero-rates are much lower (or near-zero
    themselves), that DGP assumption is worth revisiting; if not, this is a confirmation, not
    new information.
    """
    rates = (
        df.group_by("year_bs")
        .agg(
            (pl.col("r") == 0).mean().alias("pct_zero_r"),
            (pl.col("c") == 0).mean().alias("pct_zero_c"),
        )
        .sort("year_bs")
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(rates["year_bs"], rates["pct_zero_r"], marker="o", markersize=4, linewidth=1.5,
            color=NET_LENDER_COLOR, label="zero interbank lending (r = 0)")
    ax.plot(rates["year_bs"], rates["pct_zero_c"], marker="s", markersize=4, linewidth=1.5,
            color=NET_BORROWER_COLOR, label="zero interbank borrowing (c = 0)")
    ax.set_ylim(0, 1)
    ax.set_title("Share of commercial bank-periods with zero interbank activity, by year", fontsize=13)
    ax.set_xlabel("fiscal year (BS)", fontsize=11)
    ax.set_ylabel("fraction of bank-periods", fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_log_distribution_by_year(df: pl.DataFrame, out_path: Path) -> None:
    """Plot per-year log10 boxplots of nonzero interbank lending and borrowing.

    Two stacked panels sharing a year axis (not dual-axis -- r and c get their own panel
    rather than sharing one y-scale): log10 boxplot of NONZERO r per year on top, same for c on
    bottom. Zeros excluded here deliberately (log undefined at 0, and plot_zero_activity_rate
    already covers the zero-rate itself) so the box/whisker shape reflects the conditional size
    of real interbank exposures, not a floor artifact from the zero-inflation.
    """
    annual = _annual_interbank(df).to_pandas()
    years = sorted(annual["year_bs"].unique())
    log_r_by_year = [np.log10(annual.loc[(annual["year_bs"] == y) & (annual["r"] > 0), "r"]) for y in years]
    log_c_by_year = [np.log10(annual.loc[(annual["year_bs"] == y) & (annual["c"] > 0), "c"]) for y in years]

    fig, (ax_r, ax_c) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_r.boxplot(log_r_by_year, positions=range(len(years)), widths=0.6, showfliers=True,
                 boxprops={"color": NET_LENDER_COLOR}, medianprops={"color": "black", "linewidth": 1.2},
                 whiskerprops={"color": NET_LENDER_COLOR}, capprops={"color": NET_LENDER_COLOR},
                 flierprops={"markeredgecolor": NET_LENDER_COLOR, "markersize": 3, "alpha": 0.5})
    ax_r.set_title(
        "Distribution of nonzero interbank lending (r) and borrowing (c), by year (log10 scale)",
        fontsize=13, pad=12,
    )
    ax_r.set_ylabel("log10(r | r > 0)\n(NPR mn)", fontsize=10)

    ax_c.boxplot(log_c_by_year, positions=range(len(years)), widths=0.6, showfliers=True,
                 boxprops={"color": NET_BORROWER_COLOR}, medianprops={"color": "black", "linewidth": 1.2},
                 whiskerprops={"color": NET_BORROWER_COLOR}, capprops={"color": NET_BORROWER_COLOR},
                 flierprops={"markeredgecolor": NET_BORROWER_COLOR, "markersize": 3, "alpha": 0.5})
    ax_c.set_ylabel("log10(c | c > 0)\n(NPR mn)", fontsize=10)
    ax_c.set_xlabel("fiscal year (BS)", fontsize=11)
    ax_c.set_xticks(range(len(years)))
    ax_c.set_xticklabels(years, rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_r_vs_c_scatter(df: pl.DataFrame, out_path: Path) -> None:
    """Plot a bank-year scatter of interbank lending vs borrowing, colored by net position.

    Bank-year scatter of lending vs. borrowing (log1p scale, since both are zero-inflated),
    colored by net position sign -- directly answers "is a bank a net lender or net borrower,"
    the simplest possible lender/borrower split and a cheap proxy for core-ness before any
    model is trained. Points on the y=x reference line lend and borrow in equal measure.
    """
    annual = _annual_interbank(df).to_pandas()
    annual = annual[(annual["r"] > 0) | (annual["c"] > 0)]
    is_lender = annual["r"] >= annual["c"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(np.log10(1 + annual.loc[is_lender, "r"]), np.log10(1 + annual.loc[is_lender, "c"]),
               s=18, color=NET_LENDER_COLOR, alpha=0.6, label="net lender (r ≥ c)")
    ax.scatter(np.log10(1 + annual.loc[~is_lender, "r"]), np.log10(1 + annual.loc[~is_lender, "c"]),
               s=18, color=NET_BORROWER_COLOR, alpha=0.6, label="net borrower (c > r)")
    lims = [0, max(np.log10(1 + annual[["r", "c"]].to_numpy()).max(), 1.0)]
    ax.plot(lims, lims, color="#999999", linewidth=1, linestyle="--", label="r = c (balanced)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title("Bank-year interbank lending vs. borrowing", fontsize=13)
    ax.set_xlabel("log10(1 + annual mean interbank lending r, NPR mn)", fontsize=10)
    ax.set_ylabel("log10(1 + annual mean interbank borrowing c, NPR mn)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_net_position_by_year(df: pl.DataFrame, out_path: Path) -> None:
    """Plot a bank-by-year heatmap of net interbank position (mean r - mean c).

    Bank x year heatmap of net interbank position (mean r - mean c that year), diverging
    RdBu_r centered at 0 -- a single full-panel-average bar per bank hides whether a bank's
    lender/borrower status is stable or flips year to year, which matters for the paper's
    temporal-stability claim (design doc's real-data claim after block-identity was dropped,
    Sec 10). Columns sorted by full-panel mean net position (descending) for a stable
    left-to-right reference, same sort-by-a-global-ranking convention as
    plot_scale_coverage_heatmap's columns; gray = bank not reporting that year, same convention
    too.
    """
    annual = _annual_interbank(df).with_columns(net=(pl.col("r") - pl.col("c")))
    overall_order = (
        annual.group_by("bfi_name").agg(pl.col("net").mean().alias("m")).sort("m", descending=True)
    )["bfi_name"].to_list()

    hm_df = (
        annual.pivot(on="bfi_name", index="year_bs", values="net")
        .sort("year_bs")
        .to_pandas()
        .set_index("year_bs")
    )
    hm_df = hm_df[[b for b in overall_order if b in hm_df.columns]]

    net_max = np.nanmax(np.abs(hm_df.to_numpy())) or 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8e8e8")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        hm_df, cmap=cmap, mask=hm_df.isna(), vmin=-net_max, vmax=net_max, annot=False,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "mean net position that year, NPR mn\n(red = net lender, blue = net borrower)"},
        ax=ax,
    )
    ax.set_title(
        "Net interbank position by bank and year\n"
        "(columns sorted by full-panel mean net position, lender→borrower; gray = not reporting)",
        fontsize=13, pad=14,
    )
    ax.set_xlabel("bank (sorted by full-panel mean net position, descending)", fontsize=11)
    ax.set_ylabel("fiscal year (BS)", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_activity_vs_size(df: pl.DataFrame, assets_csv: Path, out_path: Path) -> None:
    """Plot bank-year interbank activity (r + c) against total assets.

    Bank-year total interbank activity (r + c) vs. TOTAL ASSETS -- the core/periphery
    sanity check: if size and interbank activity are strongly positively correlated, "big banks
    are core" is a reasonable prior going into model training; if not, size alone is a poor
    proxy for core-ness and the model's block assignment is doing real work, not just
    rediscovering size.
    """
    interbank_annual = _annual_interbank(df)
    size_annual = (
        _load_total_assets(assets_csv)
        .group_by(["year_bs", "bfi_name"])
        .agg(pl.col("value_mn_npr").mean().alias("size"))
    )
    joined = interbank_annual.join(size_annual, on=["year_bs", "bfi_name"], how="inner").to_pandas()
    activity = joined["r"] + joined["c"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.log10(joined["size"]), np.log10(1 + activity), s=18, color="#2c7fb8", alpha=0.6)
    ax.set_title("Total interbank activity vs. bank size (bank-years)", fontsize=13)
    ax.set_xlabel("log10(TOTAL ASSETS, NPR mn)", fontsize=10)
    ax.set_ylabel("log10(1 + interbank lending + borrowing, NPR mn)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    """Parse CLI args, load the panel, and write the asset-scale and interbank-marginal figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "eda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    assets_df = _load_total_assets(args.csv)
    plot_scale_coverage_heatmap(assets_df, args.out_dir / "scale_coverage_heatmap.png")
    plot_reporting_coverage_over_time(assets_df, args.out_dir / "reporting_coverage.png")
    plot_size_rank_bump_chart(assets_df, args.out_dir / "size_rank_bump.png")
    plot_size_distribution_by_year(assets_df, args.out_dir / "size_distribution_by_year.png")

    interbank_df = _load_interbank_marginals(args.csv)
    plot_zero_activity_rate(interbank_df, args.out_dir / "interbank_zero_activity_rate.png")
    plot_log_distribution_by_year(interbank_df, args.out_dir / "interbank_log_distribution_by_year.png")
    plot_r_vs_c_scatter(interbank_df, args.out_dir / "interbank_r_vs_c_scatter.png")
    plot_net_position_by_year(interbank_df, args.out_dir / "interbank_net_position_by_year.png")
    plot_activity_vs_size(interbank_df, args.csv, args.out_dir / "interbank_activity_vs_size.png")


if __name__ == "__main__":
    main()
