import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = "total_sentiment_score.csv"
OUTPUT_JSON = "robustness_report.json"
OUTPUT_CLUSTER_PVALUES_CSV = "cluster_pvalues_corrected.csv"

TEXT_COL_CANDIDATES = ["filtered_text", "filtered text", "raw text"]
DATE_COL_CANDIDATES = ["created_utc", "published", "date", "datetime", "timestamp"]
LABEL_COL = "predicted_label"
CLUSTER_COL = "cluster_id"


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 1.0
    p_pool = (x1 + x2) / (n1 + n2)
    denom = sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2), 1e-12))
    z_score = (x1 / n1 - x2 / n2) / denom
    p_value = 2.0 * (1.0 - normal_cdf(abs(z_score)))
    return float(min(max(p_value, 0.0), 1.0))


def bonferroni_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def fdr_bh_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * m

    for rank, (idx, p_val) in enumerate(indexed, start=1):
        adjusted[idx] = p_val * m / rank

    sorted_adj = [adjusted[idx] for idx, _ in indexed]
    for i in range(m - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])

    final_adj = [0.0] * m
    for (idx, _), adj in zip(indexed, sorted_adj):
        final_adj[idx] = float(min(max(adj, 0.0), 1.0))

    return final_adj


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def run_alt_sentiment_validation(df: pd.DataFrame, text_col: str) -> dict:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        analyzer = SentimentIntensityAnalyzer()
        compounds = df[text_col].fillna("").astype(str).apply(lambda txt: analyzer.polarity_scores(txt)["compound"])
        alt_labels = compounds.apply(to_label_from_compound)

        agreement = (alt_labels == df[LABEL_COL].astype(str)).mean()
        agreement_by_label = (
            pd.DataFrame({"base": df[LABEL_COL].astype(str), "alt": alt_labels})
            .assign(match=lambda frame: frame["base"] == frame["alt"])
            .groupby("base")["match"]
            .mean()
            .to_dict()
        )

        return {
            "status": "ok",
            "model": "VADER",
            "overall_agreement": float(agreement),
            "agreement_by_base_label": {k: float(v) for k, v in agreement_by_label.items()},
        }
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Alternative model validation failed: {exc}",
        }


def prepare_date_series(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    date_col = find_first_existing_column(df, DATE_COL_CANDIDATES)
    if date_col is None:
        return df.copy(), None

    out = df.copy()
    if date_col == "created_utc":
        out["_event_date"] = pd.to_datetime(out[date_col], unit="s", utc=True, errors="coerce")
    else:
        out["_event_date"] = pd.to_datetime(out[date_col], utc=True, errors="coerce")

    out = out.dropna(subset=["_event_date"])
    out["_event_day"] = out["_event_date"].dt.floor("D")
    return out, date_col


def summarize_sentiment(df: pd.DataFrame) -> dict:
    counts = df[LABEL_COL].astype(str).value_counts()
    total = int(len(df))
    return {
        "n_rows": total,
        "negative_rate": float(counts.get("negative", 0) / total) if total else 0.0,
        "neutral_rate": float(counts.get("neutral", 0) / total) if total else 0.0,
        "positive_rate": float(counts.get("positive", 0) / total) if total else 0.0,
    }


def run_exclude_top_volume_days(df_with_dates: pd.DataFrame) -> dict:
    if "_event_day" not in df_with_dates.columns:
        return {"status": "skipped", "reason": "No date column found."}

    day_counts = df_with_dates.groupby("_event_day").size()
    if day_counts.empty:
        return {"status": "skipped", "reason": "No valid dated rows found."}

    threshold = day_counts.quantile(0.95)
    high_volume_days = set(day_counts[day_counts >= threshold].index)
    filtered = df_with_dates[~df_with_dates["_event_day"].isin(high_volume_days)]

    return {
        "status": "ok",
        "volume_day_threshold_95pct": float(threshold),
        "excluded_days": int(len(high_volume_days)),
        "baseline": summarize_sentiment(df_with_dates),
        "without_top_volume_days": summarize_sentiment(filtered),
    }


def run_subsampling_test(df: pd.DataFrame, fraction: float, repeats: int, seed: int) -> dict:
    if not (0 < fraction < 1):
        return {"status": "skipped", "reason": "subsample_fraction must be in (0, 1)."}

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled_indices = rng.choice(df.index.values, size=max(1, int(len(df) * fraction)), replace=False)
        sampled = df.loc[sampled_indices]
        estimates.append(summarize_sentiment(sampled)["negative_rate"])

    arr = np.array(estimates)
    return {
        "status": "ok",
        "fraction": float(fraction),
        "repeats": int(repeats),
        "mean_negative_rate": float(arr.mean()),
        "std_negative_rate": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95_negative_rate": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
    }


def run_time_window_sensitivity(df_with_dates: pd.DataFrame, shift_days: int = 7) -> dict:
    if "_event_day" not in df_with_dates.columns:
        return {"status": "skipped", "reason": "No date column found."}

    start_day = df_with_dates["_event_day"].min()
    end_day = df_with_dates["_event_day"].max()
    if pd.isna(start_day) or pd.isna(end_day):
        return {"status": "skipped", "reason": "No valid date range found."}

    minus_mask = (df_with_dates["_event_day"] >= (start_day - pd.Timedelta(days=shift_days))) & (
        df_with_dates["_event_day"] <= (end_day - pd.Timedelta(days=shift_days))
    )
    plus_mask = (df_with_dates["_event_day"] >= (start_day + pd.Timedelta(days=shift_days))) & (
        df_with_dates["_event_day"] <= (end_day + pd.Timedelta(days=shift_days))
    )

    return {
        "status": "ok",
        "shift_days": int(shift_days),
        "baseline": summarize_sentiment(df_with_dates),
        "window_shift_minus": summarize_sentiment(df_with_dates[minus_mask]),
        "window_shift_plus": summarize_sentiment(df_with_dates[plus_mask]),
    }


def run_multiple_testing(df: pd.DataFrame, output_csv_path: Path) -> dict:
    if CLUSTER_COL not in df.columns:
        return {"status": "skipped", "reason": "No cluster_id column found."}

    valid_df = df[df[CLUSTER_COL].notna()].copy()
    if valid_df.empty:
        return {"status": "skipped", "reason": "No valid cluster rows found."}

    valid_df["is_negative"] = valid_df[LABEL_COL].astype(str).eq("negative").astype(int)
    total_n = int(len(valid_df))
    total_neg = int(valid_df["is_negative"].sum())

    rows = []
    for cluster_id, grp in valid_df.groupby(CLUSTER_COL):
        n_cluster = int(len(grp))
        neg_cluster = int(grp["is_negative"].sum())

        n_rest = total_n - n_cluster
        neg_rest = total_neg - neg_cluster
        p_raw = two_proportion_z_test(neg_cluster, n_cluster, neg_rest, max(n_rest, 1))

        rows.append(
            {
                "cluster_id": cluster_id,
                "n_cluster": n_cluster,
                "neg_cluster": neg_cluster,
                "negative_rate_cluster": neg_cluster / n_cluster if n_cluster else 0.0,
                "n_rest": n_rest,
                "neg_rest": neg_rest,
                "negative_rate_rest": neg_rest / n_rest if n_rest else 0.0,
                "p_value_raw": p_raw,
            }
        )

    if not rows:
        return {"status": "skipped", "reason": "No cluster groups available for testing."}

    pvals = [row["p_value_raw"] for row in rows]
    bonf = bonferroni_correction(pvals)
    fdr = fdr_bh_correction(pvals)

    for i, row in enumerate(rows):
        row["p_value_bonferroni"] = bonf[i]
        row["p_value_fdr_bh"] = fdr[i]
        row["significant_bonf_0_05"] = bonf[i] < 0.05
        row["significant_fdr_0_05"] = fdr[i] < 0.05

    out_df = pd.DataFrame(rows).sort_values("p_value_raw")
    out_df.to_csv(output_csv_path, index=False)

    return {
        "status": "ok",
        "n_tests": int(len(out_df)),
        "bonferroni_significant_count": int(out_df["significant_bonf_0_05"].sum()),
        "fdr_significant_count": int(out_df["significant_fdr_0_05"].sum()),
        "details_csv": str(output_csv_path.name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment robustness checks.")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV with sentiment output.")
    parser.add_argument("--output-json", default=OUTPUT_JSON, help="Output JSON report path.")
    parser.add_argument(
        "--output-pvalues",
        default=OUTPUT_CLUSTER_PVALUES_CSV,
        help="Output CSV path for corrected p-values.",
    )
    parser.add_argument("--subsample-fraction", type=float, default=0.8)
    parser.add_argument("--subsample-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift-days", type=int, default=7)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing required column: {LABEL_COL}")

    text_col = find_first_existing_column(df, TEXT_COL_CANDIDATES)
    if text_col is None:
        raise ValueError(
            f"No text column found. Expected one of: {', '.join(TEXT_COL_CANDIDATES)}"
        )

    df_dates, used_date_col = prepare_date_series(df)

    report = {
        "input_csv": str(input_path.name),
        "n_rows": int(len(df)),
        "text_column": text_col,
        "date_column": used_date_col,
        "baseline": summarize_sentiment(df),
        "alternative_sentiment_model_validation": run_alt_sentiment_validation(df, text_col),
        "exclude_top_5pct_volume_days": run_exclude_top_volume_days(df_dates),
        "subsampling_test": run_subsampling_test(
            df,
            fraction=args.subsample_fraction,
            repeats=args.subsample_repeats,
            seed=args.seed,
        ),
        "time_window_sensitivity": run_time_window_sensitivity(df_dates, shift_days=args.shift_days),
        "multiple_testing_correction": run_multiple_testing(df, Path(args.output_pvalues)),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved robustness report: {args.output_json}")
    print(f"Saved multiple-testing table: {args.output_pvalues}")

    baseline_neg = report["baseline"].get("negative_rate", 0.0)
    print(f"Baseline negative rate: {baseline_neg:.4f}")


if __name__ == "__main__":
    main()
