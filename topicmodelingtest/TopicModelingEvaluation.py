import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "total_sentiment_score.csv"
TOKENS_COLUMN = "tokens"


@dataclass
class LDAConfig:
    topic_counts: Sequence[int] = (5, 8, 10, 12, 15)
    alpha: str = "symmetric"
    beta: str = "symmetric"  # maps to gensim eta
    passes: int = 15
    iterations: int = 200
    base_seed: int = 42
    stability_seeds: Sequence[int] = (42, 52, 62)
    bootstrap_runs: int = 5
    topn_words: int = 10
    no_below: int = 5
    no_above: float = 0.5
    keep_n: int = 10000


@dataclass
class BERTopicConfig:
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    clustering_algorithm: str = "HDBSCAN"
    min_cluster_size: int = 20
    random_state: int = 42


def parse_tokens(value) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(token).strip() for token in value if str(token).strip()]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(token).strip() for token in parsed if str(token).strip()]
        except (ValueError, SyntaxError):
            pass
        return [token.strip() for token in value.split() if token.strip()]
    return []


def load_token_documents(csv_path: Path, tokens_col: str) -> List[List[str]]:
    df = pd.read_csv(csv_path)
    documents = df[tokens_col].apply(parse_tokens).tolist()
    documents = [doc for doc in documents if doc]
    if not documents:
        raise ValueError("No token documents found. Check tokens column and input file.")
    return documents


def build_dictionary_and_corpus(
    documents: List[List[str]],
    no_below: int,
    no_above: float,
    keep_n: int,
) -> Tuple[Dictionary, List[List[Tuple[int, int]]]]:
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    if len(dictionary) == 0:
        raise ValueError(
            "Dictionary is empty after filtering. Lower no_below or increase no_above."
        )
    return dictionary, corpus


def fit_lda(
    corpus: List[List[Tuple[int, int]]],
    dictionary: Dictionary,
    k: int,
    seed: int,
    alpha: str,
    beta: str,
    passes: int,
    iterations: int,
) -> LdaModel:
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=seed,
        alpha=alpha,
        eta=beta,
        passes=passes,
        iterations=iterations,
        eval_every=None,
    )


def lda_topic_words(model: LdaModel, topn: int) -> List[List[str]]:
    topics = []
    for topic_id in range(model.num_topics):
        topic_terms = [term for term, _ in model.show_topic(topic_id, topn=topn)]
        topics.append(topic_terms)
    return topics


def compute_c_v_coherence(
    model: LdaModel,
    texts: List[List[str]],
    dictionary: Dictionary,
) -> float:
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(coherence_model.get_coherence())


def interpret_coherence_band(c_v: float) -> str:
    if c_v > 0.6:
        return "strong"
    if c_v > 0.5:
        return "acceptable"
    return "requires_justification"


def average_pairwise_topic_overlap(topic_words: List[List[str]]) -> float:
    if len(topic_words) < 2:
        return 0.0
    overlaps = []
    topic_sets = [set(words) for words in topic_words]
    for i in range(len(topic_sets)):
        for j in range(i + 1, len(topic_sets)):
            a, b = topic_sets[i], topic_sets[j]
            denom = len(a | b) or 1
            overlaps.append(len(a & b) / denom)
    return float(np.mean(overlaps)) if overlaps else 0.0


def interpretation_quality_from_overlap(overlap: float) -> str:
    if overlap < 0.15:
        return "specific_localized"
    if overlap < 0.30:
        return "moderate"
    return "broad_overlapping"


def topic_set_consistency(reference: List[List[str]], other: List[List[str]]) -> float:
    if not reference or not other:
        return 0.0
    other_sets = [set(topic) for topic in other]
    scores = []
    for topic in reference:
        ref_set = set(topic)
        best = 0.0
        for candidate in other_sets:
            denom = len(ref_set | candidate) or 1
            score = len(ref_set & candidate) / denom
            if score > best:
                best = score
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def bootstrap_documents(
    documents: List[List[str]],
    seed: int,
) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(documents), size=len(documents))
    return [documents[idx] for idx in indices]


def run_lda_sensitivity(
    documents: List[List[str]],
    dictionary: Dictionary,
    corpus: List[List[Tuple[int, int]]],
    config: LDAConfig,
) -> pd.DataFrame:
    rows = []
    for k in config.topic_counts:
        model = fit_lda(
            corpus=corpus,
            dictionary=dictionary,
            k=int(k),
            seed=config.base_seed,
            alpha=config.alpha,
            beta=config.beta,
            passes=config.passes,
            iterations=config.iterations,
        )
        coherence = compute_c_v_coherence(model, documents, dictionary)
        topics = lda_topic_words(model, config.topn_words)
        overlap = average_pairwise_topic_overlap(topics)
        rows.append(
            {
                "model": "LDA",
                "k": int(k),
                "c_v": coherence,
                "coherence_band": interpret_coherence_band(coherence),
                "interpretation_quality": interpretation_quality_from_overlap(overlap),
                "avg_topic_overlap": overlap,
                "alpha": config.alpha,
                "beta": config.beta,
                "passes": config.passes,
                "iterations": config.iterations,
                "random_seed": config.base_seed,
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def run_lda_stability(
    documents: List[List[str]],
    config: LDAConfig,
    best_k: int,
) -> pd.DataFrame:
    rows = []

    base_dictionary, base_corpus = build_dictionary_and_corpus(
        documents,
        no_below=config.no_below,
        no_above=config.no_above,
        keep_n=config.keep_n,
    )
    base_model = fit_lda(
        corpus=base_corpus,
        dictionary=base_dictionary,
        k=best_k,
        seed=config.stability_seeds[0],
        alpha=config.alpha,
        beta=config.beta,
        passes=config.passes,
        iterations=config.iterations,
    )
    base_topics = lda_topic_words(base_model, config.topn_words)

    for seed in config.stability_seeds[1:]:
        seeded_model = fit_lda(
            corpus=base_corpus,
            dictionary=base_dictionary,
            k=best_k,
            seed=int(seed),
            alpha=config.alpha,
            beta=config.beta,
            passes=config.passes,
            iterations=config.iterations,
        )
        seeded_topics = lda_topic_words(seeded_model, config.topn_words)
        consistency = topic_set_consistency(base_topics, seeded_topics)
        rows.append(
            {
                "test_type": "seed_variation",
                "k": best_k,
                "reference_seed": int(config.stability_seeds[0]),
                "test_seed": int(seed),
                "topic_overlap_consistency": consistency,
            }
        )

    for bootstrap_idx in range(config.bootstrap_runs):
        sample_seed = int(config.base_seed + 1000 + bootstrap_idx)
        sampled_docs = bootstrap_documents(documents, seed=sample_seed)
        sample_dictionary, sample_corpus = build_dictionary_and_corpus(
            sampled_docs,
            no_below=config.no_below,
            no_above=config.no_above,
            keep_n=config.keep_n,
        )
        sample_model = fit_lda(
            corpus=sample_corpus,
            dictionary=sample_dictionary,
            k=best_k,
            seed=config.base_seed,
            alpha=config.alpha,
            beta=config.beta,
            passes=config.passes,
            iterations=config.iterations,
        )
        sample_topics = lda_topic_words(sample_model, config.topn_words)
        consistency = topic_set_consistency(base_topics, sample_topics)
        rows.append(
            {
                "test_type": "bootstrap",
                "k": best_k,
                "reference_seed": int(config.stability_seeds[0]),
                "test_seed": sample_seed,
                "topic_overlap_consistency": consistency,
            }
        )

    return pd.DataFrame(rows)


def run_bertopic_if_available(
    documents: List[List[str]],
    dictionary: Dictionary,
    config: BERTopicConfig,
) -> Optional[Dict[str, object]]:
    if not config.enabled:
        return None

    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
    except ImportError:
        return None

    docs_as_text = [" ".join(doc) for doc in documents]
    hdbscan_model = HDBSCAN(min_cluster_size=config.min_cluster_size)
    topic_model = BERTopic(
        embedding_model=config.embedding_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=config.min_cluster_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs_as_text)

    topic_words: List[List[str]] = []
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        if words:
            topic_words.append(words)

    if not topic_words:
        return {
            "model": "BERTopic",
            "k": 0,
            "c_v": 0.0,
            "coherence_band": "requires_justification",
            "interpretation_quality": "insufficient_topics",
            "avg_topic_overlap": 0.0,
            "embedding_model": config.embedding_model,
            "clustering_algorithm": config.clustering_algorithm,
            "min_cluster_size": config.min_cluster_size,
            "random_seed": config.random_state,
        }

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=documents,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence = float(coherence_model.get_coherence())
    overlap = average_pairwise_topic_overlap(topic_words)
    return {
        "model": "BERTopic",
        "k": len(topic_words),
        "c_v": coherence,
        "coherence_band": interpret_coherence_band(coherence),
        "interpretation_quality": interpretation_quality_from_overlap(overlap),
        "avg_topic_overlap": overlap,
        "embedding_model": config.embedding_model,
        "clustering_algorithm": config.clustering_algorithm,
        "min_cluster_size": config.min_cluster_size,
        "random_seed": config.random_state,
    }


def export_top_words(
    model: LdaModel,
    output_path: Path,
    topn: int,
) -> None:
    rows = []
    for topic_id in range(model.num_topics):
        words = model.show_topic(topic_id, topn=topn)
        for rank, (word, prob) in enumerate(words, start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "word": word,
                    "probability": float(prob),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    lda_config = LDAConfig()
    bertopic_config = BERTopicConfig()

    documents = load_token_documents(INPUT_CSV, TOKENS_COLUMN)
    dictionary, corpus = build_dictionary_and_corpus(
        documents,
        no_below=lda_config.no_below,
        no_above=lda_config.no_above,
        keep_n=lda_config.keep_n,
    )

    lda_sensitivity_df = run_lda_sensitivity(documents, dictionary, corpus, lda_config)
    best_row = lda_sensitivity_df.loc[lda_sensitivity_df["c_v"].idxmax()]
    best_k = int(best_row["k"])

    best_model = fit_lda(
        corpus=corpus,
        dictionary=dictionary,
        k=best_k,
        seed=lda_config.base_seed,
        alpha=lda_config.alpha,
        beta=lda_config.beta,
        passes=lda_config.passes,
        iterations=lda_config.iterations,
    )

    lda_stability_df = run_lda_stability(documents, lda_config, best_k)

    bertopic_result = run_bertopic_if_available(documents, dictionary, bertopic_config)

    comparison_rows = [
        {
            "model": "LDA",
            "topics": best_k,
            "c_v": float(best_row["c_v"]),
            "interpretation_quality": str(best_row["interpretation_quality"]),
        }
    ]
    if bertopic_result is not None:
        comparison_rows.append(
            {
                "model": "BERTopic",
                "topics": int(bertopic_result["k"]),
                "c_v": float(bertopic_result["c_v"]),
                "interpretation_quality": str(bertopic_result["interpretation_quality"]),
            }
        )

    output_sensitivity = BASE_DIR / "topic_sensitivity_lda.csv"
    output_stability = BASE_DIR / "topic_stability_lda.csv"
    output_comparison = BASE_DIR / "topic_model_comparison.csv"
    output_top_words = BASE_DIR / "topic_top_words_lda_best.csv"
    output_repro = BASE_DIR / "topic_model_reproducibility.json"
    output_bertopic = BASE_DIR / "topic_bertopic_details.csv"

    lda_sensitivity_df.to_csv(output_sensitivity, index=False)
    lda_stability_df.to_csv(output_stability, index=False)
    pd.DataFrame(comparison_rows).to_csv(output_comparison, index=False)
    export_top_words(best_model, output_top_words, lda_config.topn_words)

    reproducibility_payload = {
        "input_csv": str(INPUT_CSV.name),
        "tokens_column": TOKENS_COLUMN,
        "num_documents": len(documents),
        "vocabulary_size": len(dictionary),
        "lda": asdict(lda_config),
        "bertopic": asdict(bertopic_config),
    }
    output_repro.write_text(json.dumps(reproducibility_payload, indent=2), encoding="utf-8")

    if bertopic_result is not None:
        pd.DataFrame([bertopic_result]).to_csv(output_bertopic, index=False)

    print("=== Topic Modeling Evaluation Complete ===")
    print(f"Input documents: {len(documents)}")
    print(f"Vocabulary size: {len(dictionary)}")
    print(f"Best LDA k: {best_k}")
    print(f"Best LDA C_v: {best_row['c_v']:.4f} ({best_row['coherence_band']})")
    print(f"LDA priors: alpha={lda_config.alpha}, beta={lda_config.beta}")
    print(f"LDA passes/iterations: {lda_config.passes}/{lda_config.iterations}")
    print(f"LDA random seed: {lda_config.base_seed}")

    if not lda_stability_df.empty:
        seed_mean = lda_stability_df.loc[
            lda_stability_df["test_type"] == "seed_variation",
            "topic_overlap_consistency",
        ].mean()
        boot_mean = lda_stability_df.loc[
            lda_stability_df["test_type"] == "bootstrap",
            "topic_overlap_consistency",
        ].mean()
        print(f"Stability (seed overlap mean): {seed_mean:.4f}")
        print(f"Stability (bootstrap overlap mean): {boot_mean:.4f}")

    if bertopic_result is None:
        print("BERTopic comparison skipped (package not installed or disabled).")
    else:
        print(
            "BERTopic details: "
            f"embedding={bertopic_result['embedding_model']}, "
            f"cluster={bertopic_result['clustering_algorithm']}, "
            f"min_cluster_size={bertopic_result['min_cluster_size']}"
        )
        print(
            f"BERTopic topics={bertopic_result['k']}, "
            f"C_v={bertopic_result['c_v']:.4f} ({bertopic_result['coherence_band']})"
        )

    print(f"Saved: {output_sensitivity.name}")
    print(f"Saved: {output_stability.name}")
    print(f"Saved: {output_comparison.name}")
    print(f"Saved: {output_top_words.name}")
    print(f"Saved: {output_repro.name}")
    if bertopic_result is not None:
        print(f"Saved: {output_bertopic.name}")


if __name__ == "__main__":
    main()