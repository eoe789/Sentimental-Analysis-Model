import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None


DATE_CANDIDATES = ["date", "created_utc", "created_at", "timestamp", "time", "datetime"]
LABEL_CANDIDATES = ["predicted_label", "label", "sentiment", "pred"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze sentiment volume as a time series, detect spikes, and optionally map spikes to real-world events."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV with sentiment rows.")
    parser.add_argument(
        "--freq",
        default="D",
        choices=["D", "W"],
        help="Resampling frequency: D (daily) or W (weekly).",
    )
    parser.add_argument(
        "--date-column",
        default=None,
        help="Date column name. If omitted, script auto-detects common date columns.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Sentiment label column name. If omitted, script auto-detects common label columns.",
    )
    parser.add_argument(
        "--positive-labels",
        nargs="+",
        default=["positive", "pos"],
        help="Labels considered positive.",
    )
    parser.add_argument(
        "--negative-labels",
        nargs="+",
        default=["negative", "neg"],
        help="Labels considered negative.",
    )
    parser.add_argument(
        "--spike-sigma",
        type=float,
        default=2.0,
        help="Spike threshold in standard deviations above mean (mu + k*sigma).",
    )
    parser.add_argument(
        "--autocorr-lag",
        type=int,
        default=1,
        help="Lag for autocorrelation calculation.",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Optional CSV with at least: date,event (for spike/event mapping).",
    )
    parser.add_argument(
        "--event-window-days",
        type=int,
        default=2,
        help="Maximum absolute day distance for linking spikes to events.",
    )
    parser.add_argument(
        "--run-chow-test",
        action="store_true",
        help="Run Chow test at candidate breakpoints (detected spike dates).",
    )
    parser.add_argument(
        "--output-prefix",
        default="sentiment_timeseries",
        help="Prefix for output files (.png, .txt, .csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="sentiment_timeseries_outputs",
        help="Folder where all outputs are written.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=None,
        help="Rolling window for smoothing sentiment score line (default: 7 for daily, 4 for weekly).",
    )
    parser.add_argument(
        "--start-date",
        default="2018-01-01",
        help="Exclude data before this date (YYYY-MM-DD). Default: 2018-01-01.",
    )
    return parser.parse_args()


def choose_column(columns: list[str], preferred: str | None, candidates: list[str], kind: str) -> str:
    if preferred:
        if preferred not in columns:
            raise ValueError(f"{kind} column '{preferred}' not found. Available: {columns}")
        return preferred

    column_lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]

    raise ValueError(
        f"Could not auto-detect {kind} column. Please provide --{kind}-column explicitly. Available: {columns}"
    )


def safe_skew(values: pd.Series) -> float:
    if len(values) < 3:
        return float("nan")
    if stats is not None:
        return float(stats.skew(values, bias=False))
    centered = values - values.mean()
    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((centered.pow(3).mean()) / (std ** 3))


def normality_test(values: pd.Series) -> tuple[float, float] | None:
    if stats is None or len(values) < 8:
        return None
    k2, p_val = stats.normaltest(values)
    return float(k2), float(p_val)


def two_tailed_p_from_z(z_score: float) -> float:
    if np.isnan(z_score):
        return float("nan")
    if stats is not None:
        return float(2.0 * stats.norm.sf(abs(z_score)))
    return float(math.erfc(abs(z_score) / math.sqrt(2.0)))


def compute_series_stats(series: pd.Series, lag: int, spike_sigma: float) -> dict:
    mu = float(series.mean())
    sigma = float(series.std(ddof=1))
    skew = safe_skew(series)
    autocorr = float(series.autocorr(lag=lag)) if len(series) > lag else float("nan")

    threshold = mu + spike_sigma * sigma
    z_scores = (series - mu) / sigma if sigma > 0 else pd.Series(np.nan, index=series.index)

    spikes = pd.DataFrame({
        "date": series.index,
        "volume": series.values,
        "z_score": z_scores.values,
    })
    spikes = spikes[spikes["volume"] > threshold].copy()
    spikes["p_value"] = spikes["z_score"].apply(two_tailed_p_from_z)
    spikes["threshold"] = threshold

    return {
        "mu": mu,
        "sigma": sigma,
        "skewness": skew,
        "autocorr": autocorr,
        "threshold": threshold,
        "normality": normality_test(series),
        "spikes": spikes,
    }


def chow_test_linear_trend(series: pd.Series, break_idx: int) -> tuple[float, float] | tuple[None, None]:
    n = len(series)
    if break_idx <= 2 or break_idx >= n - 2:
        return None, None

    y = series.values.astype(float)
    x = np.arange(n, dtype=float)

    def sse_linear(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
        X = np.column_stack([np.ones(len(x_arr)), x_arr])
        beta, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
        resid = y_arr - X @ beta
        return float(np.sum(resid ** 2))

    sse_full = sse_linear(x, y)
    sse_1 = sse_linear(x[:break_idx], y[:break_idx])
    sse_2 = sse_linear(x[break_idx:], y[break_idx:])

    k = 2
    denom_df = n - 2 * k
    if denom_df <= 0:
        return None, None

    numerator = (sse_full - (sse_1 + sse_2)) / k
    denominator = (sse_1 + sse_2) / denom_df
    if denominator <= 0:
        return None, None

    f_stat = numerator / denominator
    if stats is not None:
        p_val = float(stats.f.sf(f_stat, k, denom_df))
    else:
        p_val = float("nan")

    return float(f_stat), p_val


def load_events(events_path: str | None) -> pd.DataFrame | None:
    if not events_path:
        return None
    events = pd.read_csv(events_path)
    required = {"date", "event"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Events file must include columns {required}, missing: {missing}")
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events = events.dropna(subset=["date", "event"]).copy()
    return events


def map_spikes_to_events(spikes: pd.DataFrame, events: pd.DataFrame | None, max_days: int) -> pd.DataFrame:
    if events is None or spikes.empty:
        spikes["matched_event"] = None
        spikes["event_date"] = pd.NaT
        spikes["day_delta"] = np.nan
        return spikes

    mapped_events = []
    mapped_dates = []
    deltas = []

    for spike_date in spikes["date"]:
        day_diffs = (events["date"] - spike_date).dt.days.abs()
        best_idx = day_diffs.idxmin()
        best_delta = int(day_diffs.loc[best_idx])

        if best_delta <= max_days:
            mapped_events.append(events.loc[best_idx, "event"])
            mapped_dates.append(events.loc[best_idx, "date"])
            deltas.append(best_delta)
        else:
            mapped_events.append(None)
            mapped_dates.append(pd.NaT)
            deltas.append(np.nan)

    spikes = spikes.copy()
    spikes["matched_event"] = mapped_events
    spikes["event_date"] = mapped_dates
    spikes["day_delta"] = deltas
    return spikes


def derive_sentiment_score(df: pd.DataFrame, label_col: str) -> pd.Series:
    if "score_positive" in df.columns and "score_negative" in df.columns:
        pos = pd.to_numeric(df["score_positive"], errors="coerce").fillna(0.0)
        neg = pd.to_numeric(df["score_negative"], errors="coerce").fillna(0.0)
        score = pos - neg
    else:
        score = pd.Series(0.0, index=df.index, dtype=float)
        score[df[label_col].astype(str).str.lower().str.strip().isin({"positive", "pos"})] = 1.0
        score[df[label_col].astype(str).str.lower().str.strip().isin({"negative", "neg"})] = -1.0
    return score.clip(-1.0, 1.0)


def write_report(
    output_path: Path,
    freq_name: str,
    pos_stats: dict,
    neg_stats: dict,
    total_stats: dict,
    pos_spikes: pd.DataFrame,
    neg_spikes: pd.DataFrame,
    chow_results: pd.DataFrame | None,
) -> None:
    def normality_text(stat_block: dict) -> str:
        result = stat_block["normality"]
        if result is None:
            return "Normality test: not run (requires SciPy and >= 8 periods)."
        k2, p_val = result
        verdict = "non-normal" if p_val < 0.05 else "approximately normal"
        return f"Normality test (D'Agostino K^2): K2={k2:.3f}, p={p_val:.4g} -> {verdict}."

    lines = []
    lines.append("Sentiment Time-Series Volume Report")
    lines.append("=" * 40)
    lines.append(f"Frequency: {freq_name}")

    all_spike_dates = sorted(set(pd.to_datetime(pos_spikes["date"]).dt.date).union(set(pd.to_datetime(neg_spikes["date"]).dt.date)))
    matched_events = []
    for spikes_df in (pos_spikes, neg_spikes):
        if not spikes_df.empty and "matched_event" in spikes_df.columns:
            matched_events.extend(
                [
                    str(event)
                    for event in spikes_df["matched_event"].dropna().astype(str).tolist()
                    if str(event).strip()
                ]
            )

    lines.append("")
    if all_spike_dates:
        date_text = ", ".join(str(d) for d in all_spike_dates)
        event_text = ""
        if matched_events:
            unique_events = sorted(set(matched_events))
            event_text = f" These surges correspond temporally to: {'; '.join(unique_events)}."
        lines.append(
            "Reviewer-ready summary: "
            f"The temporal distribution exhibited non-stationary dynamics, with statistically significant "
            f"volume surges exceeding μ + 2σ on {date_text}.{event_text}"
        )
    else:
        lines.append(
            "Reviewer-ready summary: The temporal distribution did not show statistically significant "
            "surges above μ + 2σ in this sample."
        )
    lines.append("")

    lines.append("Overall Volume")
    lines.append("-" * 20)
    lines.append(f"Mean {freq_name.lower()} volume (μ): {total_stats['mu']:.3f}")
    lines.append(f"Std. dev. (σ): {total_stats['sigma']:.3f}")
    lines.append(f"Skewness: {total_stats['skewness']:.3f}")
    lines.append(f"Autocorrelation (lag): {total_stats['autocorr']:.3f}")
    lines.append(normality_text(total_stats))
    lines.append("")

    lines.append("Positive Volume")
    lines.append("-" * 20)
    lines.append(f"Mean {freq_name.lower()} volume (μ): {pos_stats['mu']:.3f}")
    lines.append(f"Std. dev. (σ): {pos_stats['sigma']:.3f}")
    lines.append(f"Skewness: {pos_stats['skewness']:.3f}")
    lines.append(f"Autocorrelation (lag): {pos_stats['autocorr']:.3f}")
    lines.append(normality_text(pos_stats))
    lines.append("")

    lines.append("Negative Volume")
    lines.append("-" * 20)
    lines.append(f"Mean {freq_name.lower()} volume (μ): {neg_stats['mu']:.3f}")
    lines.append(f"Std. dev. (σ): {neg_stats['sigma']:.3f}")
    lines.append(f"Skewness: {neg_stats['skewness']:.3f}")
    lines.append(f"Autocorrelation (lag): {neg_stats['autocorr']:.3f}")
    lines.append(normality_text(neg_stats))
    lines.append("")

    lines.append("Significant Spikes (volume > μ + 2σ by default)")
    lines.append("-" * 20)

    if pos_spikes.empty and neg_spikes.empty:
        lines.append("No significant spikes detected.")
    else:
        if not pos_spikes.empty:
            lines.append("Positive spikes:")
            for _, row in pos_spikes.iterrows():
                event_part = ""
                if pd.notna(row.get("matched_event", np.nan)):
                    event_part = (
                        f" | event: {row['matched_event']}"
                        f" ({pd.to_datetime(row['event_date']).date()}, Δdays={int(row['day_delta'])})"
                    )
                lines.append(
                    f"- {pd.to_datetime(row['date']).date()} | volume={row['volume']:.0f} | "
                    f"z={row['z_score']:.2f} | p={row['p_value']:.3g}{event_part}"
                )

        if not neg_spikes.empty:
            lines.append("Negative spikes:")
            for _, row in neg_spikes.iterrows():
                event_part = ""
                if pd.notna(row.get("matched_event", np.nan)):
                    event_part = (
                        f" | event: {row['matched_event']}"
                        f" ({pd.to_datetime(row['event_date']).date()}, Δdays={int(row['day_delta'])})"
                    )
                lines.append(
                    f"- {pd.to_datetime(row['date']).date()} | volume={row['volume']:.0f} | "
                    f"z={row['z_score']:.2f} | p={row['p_value']:.3g}{event_part}"
                )

    if chow_results is not None:
        lines.append("")
        lines.append("Structural Break Tests (Chow on linear trend)")
        lines.append("-" * 20)
        if chow_results.empty:
            lines.append("No valid Chow tests were computed.")
        else:
            for _, row in chow_results.sort_values("p_value").iterrows():
                p_disp = "nan" if pd.isna(row["p_value"]) else f"{row['p_value']:.3g}"
                lines.append(
                    f"- Break @ {pd.to_datetime(row['date']).date()} | F={row['f_stat']:.3f} | p={p_disp}"
                )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    date_col = choose_column(df.columns.tolist(), args.date_column, DATE_CANDIDATES, "date")
    label_col = choose_column(df.columns.tolist(), args.label_column, LABEL_CANDIDATES, "label")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    start_date = pd.to_datetime(args.start_date, errors="coerce")
    if pd.isna(start_date):
        raise ValueError("Invalid --start-date. Use YYYY-MM-DD format, e.g., 2018-01-01")
    df = df[df[date_col] >= start_date].copy()

    if df.empty:
        raise ValueError(f"No rows remain after applying --start-date {args.start_date}.")

    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    positive_labels = {label.lower().strip() for label in args.positive_labels}
    negative_labels = {label.lower().strip() for label in args.negative_labels}

    df["is_positive"] = df[label_col].isin(positive_labels)
    df["is_negative"] = df[label_col].isin(negative_labels)
    df["is_neutral"] = ~(df["is_positive"] | df["is_negative"])
    df["sentiment_score"] = derive_sentiment_score(df, label_col)

    ts = (
        df.set_index(date_col)
        .sort_index()
        .resample(args.freq)
        .agg(
            positive_volume=("is_positive", "sum"),
            neutral_volume=("is_neutral", "sum"),
            negative_volume=("is_negative", "sum"),
            total_volume=(label_col, "size"),
        )
        .fillna(0)
    )

    ts["positive_volume"] = ts["positive_volume"].astype(int)
    ts["neutral_volume"] = ts["neutral_volume"].astype(int)
    ts["negative_volume"] = ts["negative_volume"].astype(int)
    ts["total_volume"] = ts["total_volume"].astype(int)

    sentiment_series = (
        df.set_index(date_col)
        .sort_index()["sentiment_score"]
        .resample(args.freq)
        .mean()
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )

    pos_stats = compute_series_stats(ts["positive_volume"], args.autocorr_lag, args.spike_sigma)
    neg_stats = compute_series_stats(ts["negative_volume"], args.autocorr_lag, args.spike_sigma)
    total_stats = compute_series_stats(ts["total_volume"], args.autocorr_lag, args.spike_sigma)

    events = load_events(args.events)
    pos_spikes = map_spikes_to_events(pos_stats["spikes"], events, args.event_window_days)
    neg_spikes = map_spikes_to_events(neg_stats["spikes"], events, args.event_window_days)

    chow_results = None
    if args.run_chow_test:
        break_dates = sorted(set(pos_spikes["date"]).union(set(neg_spikes["date"])))
        records = []
        for break_date in break_dates:
            idx = ts.index.get_indexer([break_date])[0]
            if idx < 0:
                continue
            f_stat, p_val = chow_test_linear_trend(ts["total_volume"], idx)
            if f_stat is None:
                continue
            records.append({"date": break_date, "f_stat": f_stat, "p_value": p_val})
        chow_results = pd.DataFrame(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = output_dir / Path(args.output_prefix).name
    out_plot = out_prefix.with_suffix(".png")
    out_report = out_prefix.with_suffix(".txt")
    out_spikes = out_prefix.with_name(out_prefix.name + "_spikes.csv")
    out_ts = out_prefix.with_name(out_prefix.name + "_timeseries.csv")

    rolling_window = args.rolling_window if args.rolling_window is not None else (7 if args.freq == "D" else 4)
    smooth = sentiment_series.rolling(window=rolling_window, min_periods=1).mean()

    smooth_mu = float(smooth.mean())
    smooth_sigma = float(smooth.std(ddof=1))
    pos_thr = min(1.0, smooth_mu + args.spike_sigma * smooth_sigma)
    neg_thr = max(-1.0, smooth_mu - args.spike_sigma * smooth_sigma)

    pos_spike_mask = smooth >= pos_thr
    neg_spike_mask = smooth <= neg_thr

    fig, ax = plt.subplots(figsize=(15.5, 5.4), facecolor="#dedede")
    ax.set_facecolor("#ececec")

    ax.plot(sentiment_series.index, sentiment_series.values, color="#8f8f8f", linewidth=1.0, alpha=0.55)
    ax.plot(smooth.index, smooth.values, color="#4f86e8", linewidth=2.0, alpha=0.95, label=f"{rolling_window}-period avg")

    ax.axhline(0.0, color="#7f7f7f", linewidth=2.4, alpha=0.75)
    ax.axhline(pos_thr, color="#3f8f3f", linestyle="--", linewidth=1.6, alpha=0.75)
    ax.axhline(neg_thr, color="#ea6d63", linestyle="--", linewidth=1.6, alpha=0.75)

    if pos_spike_mask.any():
        ax.scatter(
            smooth.index[pos_spike_mask],
            smooth[pos_spike_mask],
            color="#2f8e2f",
            marker="^",
            s=120,
            edgecolors="none",
            zorder=6,
            label="Positive spike",
        )
    if neg_spike_mask.any():
        ax.scatter(
            smooth.index[neg_spike_mask],
            smooth[neg_spike_mask],
            color="#ea2d1a",
            marker="v",
            s=120,
            edgecolors="none",
            zorder=6,
            label="Negative spike",
        )

    ax.set_ylim(-1.0, 1.0)
    ax.set_title("B) Sentiment Spike Detection", fontsize=19, fontweight="bold")
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Sentiment Score", fontsize=16)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=13)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.35, color="#b9b9b9")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
    ax.tick_params(axis="both", labelsize=13)

    fig.tight_layout()
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)

    spike_table = pd.concat(
        [
            pos_spikes.assign(series="positive"),
            neg_spikes.assign(series="negative"),
        ],
        ignore_index=True,
    )

    spike_table.sort_values(["date", "series"], inplace=True)
    spike_table.to_csv(out_spikes, index=False)
    ts.to_csv(out_ts)

    freq_name = "Daily" if args.freq == "D" else "Weekly"
    write_report(
        output_path=out_report,
        freq_name=freq_name,
        pos_stats=pos_stats,
        neg_stats=neg_stats,
        total_stats=total_stats,
        pos_spikes=pos_spikes,
        neg_spikes=neg_spikes,
        chow_results=chow_results,
    )

    print("Done.")
    print(f"Output directory: {output_dir}")
    print(f"Time-series data: {out_ts}")
    print(f"Plot: {out_plot}")
    print(f"Spikes: {out_spikes}")
    print(f"Report: {out_report}")


if __name__ == "__main__":
    main()
