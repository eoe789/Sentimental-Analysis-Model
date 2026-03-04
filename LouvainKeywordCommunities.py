import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import networkx as nx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class LouvainConfig:
    input_csv: Path = BASE_DIR / "combined_raw_filtered_tokens.csv"
    tokens_column: str = "tokens"
    keywords: Sequence[str] = (
        "organic",
        "natural",
        "egg",
        "milk",
        "dairy",
        "meat",
        "gmo",
        "livestock",
        "free-range",
        "cage-free",
        "humane",
    )
    min_token_length: int = 2
    min_edge_weight: int = 2
    min_node_degree_weight: int = 2
    resolution: float = 1.0
    seed: int = 42
    top_words_per_community: int = 15
    output_prefix: str = "keyword_louvain"


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


def normalize_token(token: str) -> str:
    return token.strip().lower()


def keyword_in_document(keyword: str, tokens: List[str]) -> bool:
    parts = [part for part in keyword.lower().split() if part]
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in tokens
    span = len(parts)
    for i in range(len(tokens) - span + 1):
        if tokens[i : i + span] == parts:
            return True
    return False


def load_filtered_documents(
    csv_path: Path,
    tokens_column: str,
    keywords: Sequence[str],
    min_token_length: int,
) -> List[List[str]]:
    df = pd.read_csv(csv_path)
    keyword_set = [normalize_token(keyword) for keyword in keywords]

    filtered_docs = []
    for raw_tokens in df[tokens_column].apply(parse_tokens):
        tokens = [normalize_token(token) for token in raw_tokens]
        tokens = [token for token in tokens if len(token) >= min_token_length]
        if not tokens:
            continue
        if not keyword_set or any(keyword_in_document(keyword, tokens) for keyword in keyword_set):
            filtered_docs.append(tokens)

    if not filtered_docs:
        if keyword_set:
            raise ValueError(
                "No documents matched the keywords. Check --keywords and input token content."
            )
        raise ValueError("No valid token documents found in input CSV.")
    return filtered_docs


def build_cooccurrence_graph(documents: Iterable[List[str]], min_edge_weight: int) -> nx.Graph:
    edge_weights: Dict[tuple, int] = defaultdict(int)
    for tokens in documents:
        unique_tokens = sorted(set(tokens))
        if len(unique_tokens) < 2:
            continue
        for left, right in combinations(unique_tokens, 2):
            edge_weights[(left, right)] += 1

    graph = nx.Graph()
    for (left, right), weight in edge_weights.items():
        if weight >= min_edge_weight:
            graph.add_edge(left, right, weight=weight)

    return graph


def prune_weak_nodes(graph: nx.Graph, min_node_degree_weight: int) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    keep_nodes = [
        node
        for node, degree in graph.degree(weight="weight")
        if degree >= min_node_degree_weight
    ]
    return graph.subgraph(keep_nodes).copy()


def run_louvain_partition(graph: nx.Graph, resolution: float, seed: int) -> Dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}

    try:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
        partition = {}
        for community_id, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = community_id
        return partition
    except AttributeError as exc:
        raise RuntimeError(
            "NetworkX Louvain not available. Install a newer networkx version (>=2.8)."
        ) from exc


def top_words_by_community(
    graph: nx.Graph,
    partition: Dict[str, int],
    top_n: int,
) -> pd.DataFrame:
    community_nodes: Dict[int, Set[str]] = defaultdict(set)
    for word, community_id in partition.items():
        community_nodes[community_id].add(word)

    rows = []
    for community_id, nodes in sorted(community_nodes.items()):
        weighted_degrees = sorted(
            graph.degree(nodes, weight="weight"), key=lambda item: item[1], reverse=True
        )
        top_words = [word for word, _ in weighted_degrees[:top_n]]
        rows.append(
            {
                "community_id": community_id,
                "size": len(nodes),
                "top_words": ", ".join(top_words),
            }
        )

    return pd.DataFrame(rows)


def keyword_membership_rows(
    graph: nx.Graph,
    partition: Dict[str, int],
    keywords: Sequence[str],
) -> pd.DataFrame:
    normalized_keywords = [normalize_token(keyword) for keyword in keywords]
    rows = []

    for keyword in normalized_keywords:
        community_id = partition.get(keyword)
        if community_id is None:
            rows.append(
                {
                    "keyword": keyword,
                    "found_in_graph": False,
                    "community_id": None,
                    "community_size": 0,
                    "strongest_neighbors": "",
                }
            )
            continue

        same_community_words = [
            word for word, assigned in partition.items() if assigned == community_id
        ]
        neighbors = graph[keyword]
        strongest_neighbors = sorted(
            ((node, attrs.get("weight", 0)) for node, attrs in neighbors.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        neighbor_text = ", ".join(f"{word}:{weight}" for word, weight in strongest_neighbors)

        rows.append(
            {
                "keyword": keyword,
                "found_in_graph": True,
                "community_id": int(community_id),
                "community_size": len(same_community_words),
                "strongest_neighbors": neighbor_text,
            }
        )

    return pd.DataFrame(rows)


def all_word_assignments(partition: Dict[str, int], graph: nx.Graph) -> pd.DataFrame:
    if not partition:
        return pd.DataFrame(columns=["word", "community_id", "weighted_degree"])

    rows = [
        {
            "word": word,
            "community_id": int(community_id),
            "weighted_degree": float(graph.degree(word, weight="weight")),
        }
        for word, community_id in partition.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["community_id", "weighted_degree"], ascending=[True, False]
    )


def auto_keywords_from_partition(
    graph: nx.Graph,
    partition: Dict[str, int],
    per_community: int,
    max_keywords: int,
) -> List[str]:
    if not partition:
        return []

    community_nodes: Dict[int, List[str]] = defaultdict(list)
    for word, community_id in partition.items():
        community_nodes[int(community_id)].append(word)

    selected: List[str] = []
    sorted_communities = sorted(
        community_nodes.items(), key=lambda item: len(item[1]), reverse=True
    )
    for _, nodes in sorted_communities:
        ranked_nodes = sorted(
            nodes,
            key=lambda node: graph.degree(node, weight="weight"),
            reverse=True,
        )
        selected.extend(ranked_nodes[: max(1, int(per_community))])

    deduped: List[str] = []
    seen = set()
    for keyword in selected:
        if keyword in seen:
            continue
        seen.add(keyword)
        deduped.append(keyword)

    if max_keywords > 0:
        return deduped[:max_keywords]
    return deduped


def run_pipeline(config: LouvainConfig) -> None:
    documents = load_filtered_documents(
        csv_path=config.input_csv,
        tokens_column=config.tokens_column,
        keywords=config.keywords,
        min_token_length=config.min_token_length,
    )

    graph = build_cooccurrence_graph(documents, min_edge_weight=config.min_edge_weight)
    graph = prune_weak_nodes(graph, min_node_degree_weight=config.min_node_degree_weight)
    partition = run_louvain_partition(
        graph,
        resolution=config.resolution,
        seed=config.seed,
    )

    keyword_df = keyword_membership_rows(graph, partition, config.keywords)
    community_df = top_words_by_community(graph, partition, config.top_words_per_community)
    assignments_df = all_word_assignments(partition, graph)

    keyword_out = BASE_DIR / f"{config.output_prefix}_keywords.csv"
    community_out = BASE_DIR / f"{config.output_prefix}_summary.csv"
    assignments_out = BASE_DIR / f"{config.output_prefix}_assignments.csv"

    keyword_df.to_csv(keyword_out, index=False)
    community_df.to_csv(community_out, index=False)
    assignments_df.to_csv(assignments_out, index=False)

    found_keywords = int(keyword_df["found_in_graph"].sum())
    print(f"Documents used: {len(documents)}")
    print(
        f"Graph nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()} | communities: {community_df.shape[0]}"
    )
    print(f"Keywords located in graph: {found_keywords}/{len(config.keywords)}")
    print(f"Saved keyword results: {keyword_out.name}")
    print(f"Saved community summary: {community_out.name}")
    print(f"Saved full assignments: {assignments_out.name}")


def parse_args() -> LouvainConfig:
    parser = argparse.ArgumentParser(
        description="Detect keyword communities with Louvain from tokenized text CSV data."
    )
    parser.add_argument("--input", type=Path, default=BASE_DIR / "combined_raw_filtered_tokens.csv")
    parser.add_argument("--tokens-col", default="tokens")
    parser.add_argument(
        "--keywords",
        default=",".join(LouvainConfig().keywords),
        help="Comma-separated keywords (single words or multi-word phrases).",
    )
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--min-edge-weight", type=int, default=2)
    parser.add_argument("--min-node-degree-weight", type=int, default=2)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-words", type=int, default=15)
    parser.add_argument("--output-prefix", default="keyword_louvain")

    args = parser.parse_args()
    keywords = [segment.strip() for segment in args.keywords.split(",") if segment.strip()]

    return LouvainConfig(
        input_csv=args.input,
        tokens_column=args.tokens_col,
        keywords=keywords,
        min_token_length=args.min_token_len,
        min_edge_weight=args.min_edge_weight,
        min_node_degree_weight=args.min_node_degree_weight,
        resolution=args.resolution,
        seed=args.seed,
        top_words_per_community=args.top_words,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())