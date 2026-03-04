import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from LouvainKeywordCommunities import (
    BASE_DIR,
    LouvainConfig,
    auto_keywords_from_partition,
    build_cooccurrence_graph,
    load_filtered_documents,
    normalize_token,
    prune_weak_nodes,
    run_louvain_partition,
)


def build_single_keyword_neighborhood_subgraph(
    graph: nx.Graph,
    keyword: str,
    neighbors_per_keyword: int,
) -> nx.Graph:
    normalized_keyword = normalize_token(keyword)
    if normalized_keyword not in graph:
        return nx.Graph()

    selected_nodes: Set[str] = {normalized_keyword}
    ranked_neighbors = sorted(
        graph[normalized_keyword].items(),
        key=lambda pair: pair[1].get("weight", 0),
        reverse=True,
    )[:neighbors_per_keyword]
    selected_nodes.update(node for node, _ in ranked_neighbors)

    if not selected_nodes:
        return nx.Graph()
    return graph.subgraph(selected_nodes).copy()


def sanitize_filename(name: str) -> str:
    safe_chars = []
    for character in name.strip().lower():
        if character.isalnum() or character in {"-", "_"}:
            safe_chars.append(character)
        elif character.isspace():
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "keyword"


def limit_subgraph_size(subgraph: nx.Graph, max_nodes: int) -> nx.Graph:
    if subgraph.number_of_nodes() <= max_nodes:
        return subgraph
    ranked_nodes = sorted(
        subgraph.degree(weight="weight"),
        key=lambda item: item[1],
        reverse=True,
    )[:max_nodes]
    keep_nodes = {node for node, _ in ranked_nodes}
    return subgraph.subgraph(keep_nodes).copy()


def community_color_map(
    partition: Dict[str, int],
    nodes: List[str],
) -> List[float]:
    return [float(partition.get(node, -1)) for node in nodes]


def export_auto_keywords_csv(
    output_path: Path,
    keywords: Sequence[str],
    partition: Dict[str, int],
    graph: nx.Graph,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["keyword", "community_id", "weighted_degree"],
        )
        writer.writeheader()
        for keyword in keywords:
            normalized = normalize_token(keyword)
            writer.writerow(
                {
                    "keyword": normalized,
                    "community_id": int(partition.get(normalized, -1)),
                    "weighted_degree": float(graph.degree(normalized, weight="weight"))
                    if normalized in graph
                    else 0.0,
                }
            )


def spaced_positions_by_community(
    subgraph: nx.Graph,
    partition: Dict[str, int],
    seed: int,
    separation: float,
) -> Dict[str, Tuple[float, float]]:
    grouped_nodes: Dict[int, List[str]] = {}
    for node in subgraph.nodes():
        community_id = int(partition.get(node, -1))
        grouped_nodes.setdefault(community_id, []).append(node)

    community_ids = sorted(grouped_nodes)
    count = len(community_ids)
    if count == 0:
        return {}

    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    spacing_x = separation * 6.0
    spacing_y = separation * 5.0

    positions: Dict[str, Tuple[float, float]] = {}
    for index, community_id in enumerate(community_ids):
        nodes = grouped_nodes[community_id]
        community_graph = subgraph.subgraph(nodes).copy()

        local_k = 0.7 if community_graph.number_of_nodes() <= 10 else 0.45
        local_pos = nx.spring_layout(
            community_graph,
            seed=seed + index,
            weight="weight",
            k=local_k,
            iterations=120,
        )

        col = index % cols
        row = index // cols
        x_offset = (col - (cols - 1) / 2.0) * spacing_x
        y_offset = ((rows - 1) / 2.0 - row) * spacing_y

        for node, (x_val, y_val) in local_pos.items():
            positions[node] = (x_val + x_offset, y_val + y_offset)

    return positions


def draw_subgraph(
    subgraph: nx.Graph,
    partition: Dict[str, int],
    out_path: Path,
    title: str,
    label_top_n: int,
    layout: str,
    community_separation: float,
    seed: int,
) -> None:
    if subgraph.number_of_nodes() == 0:
        raise ValueError("No keyword neighborhood nodes found to draw.")

    plt.figure(figsize=(16, 11))
    if layout == "spread":
        positions = spaced_positions_by_community(
            subgraph,
            partition,
            seed=seed,
            separation=community_separation,
        )
    else:
        positions = nx.spring_layout(
            subgraph,
            seed=seed,
            weight="weight",
            k=0.35,
            iterations=220,
            center=(0.0, 0.0),
            scale=1.0,
        )
    nodes = list(subgraph.nodes())
    node_colors = community_color_map(partition, nodes)
    weighted_degree = dict(subgraph.degree(weight="weight"))
    node_sizes = [140 + (weighted_degree[node] * 2.0) for node in nodes]

    edge_widths = [
        max(0.5, min(6.0, attributes.get("weight", 1) * 0.12))
        for _, _, attributes in subgraph.edges(data=True)
    ]

    nx.draw_networkx_edges(
        subgraph,
        pos=positions,
        alpha=0.22,
        width=edge_widths,
        edge_color="gray",
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos=positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.9,
        linewidths=0.4,
        edgecolors="black",
    )

    if label_top_n > 0:
        top_nodes = {
            node
            for node, _ in sorted(
                weighted_degree.items(), key=lambda item: item[1], reverse=True
            )[:label_top_n]
        }
        label_map = {node: node for node in subgraph.nodes() if node in top_nodes}
    else:
        label_map = {node: node for node in subgraph.nodes()}

    nx.draw_networkx_labels(
        subgraph,
        pos=positions,
        labels=label_map,
        font_size=9,
        font_weight="bold",
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a spread-out Louvain keyword-neighborhood visualization PNG."
    )
    parser.add_argument("--input", type=Path, default=BASE_DIR / "combined_raw_filtered_tokens.csv")
    parser.add_argument("--tokens-col", default="tokens")
    parser.add_argument(
        "--keywords",
        default=",".join(LouvainConfig().keywords),
        help="Comma-separated keywords to anchor neighborhood graph.",
    )
    parser.add_argument(
        "--auto-keywords",
        action="store_true",
        help="Auto-generate keyword anchors from Louvain communities instead of --keywords.",
    )
    parser.add_argument(
        "--auto-keywords-per-community",
        type=int,
        default=1,
        help="How many top words to pick per community when --auto-keywords is enabled.",
    )
    parser.add_argument(
        "--max-auto-keywords",
        type=int,
        default=12,
        help="Maximum number of auto-generated keyword anchors (0 means no cap).",
    )
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--min-edge-weight", type=int, default=2)
    parser.add_argument("--min-node-degree-weight", type=int, default=2)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors-per-keyword", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--label-top-n", type=int, default=45)
    parser.add_argument(
        "--community-separation",
        type=float,
        default=2.1,
        help="Higher values spread communities farther apart.",
    )
    parser.add_argument(
        "--layout",
        choices=["center", "spread"],
        default="center",
        help="Use 'center' to cluster each neighborhood in the middle, or 'spread' to separate communities.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "keyword_neighborhoods",
    )
    parser.add_argument("--title-prefix", default="Louvain Keyword Neighborhood")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual_keywords = [segment.strip() for segment in args.keywords.split(",") if segment.strip()]

    config = LouvainConfig(
        input_csv=args.input,
        tokens_column=args.tokens_col,
        keywords=[] if args.auto_keywords else manual_keywords,
        min_token_length=args.min_token_len,
        min_edge_weight=args.min_edge_weight,
        min_node_degree_weight=args.min_node_degree_weight,
        resolution=args.resolution,
        seed=args.seed,
    )

    documents = load_filtered_documents(
        csv_path=config.input_csv,
        tokens_column=config.tokens_column,
        keywords=config.keywords,
        min_token_length=config.min_token_length,
    )
    graph = build_cooccurrence_graph(documents, min_edge_weight=config.min_edge_weight)
    graph = prune_weak_nodes(graph, min_node_degree_weight=config.min_node_degree_weight)
    partition = run_louvain_partition(graph, resolution=config.resolution, seed=config.seed)

    if args.auto_keywords:
        keywords = auto_keywords_from_partition(
            graph=graph,
            partition=partition,
            per_community=args.auto_keywords_per_community,
            max_keywords=args.max_auto_keywords,
        )
        if not keywords:
            raise ValueError(
                "Auto keyword generation produced no anchors. Try lowering graph filters."
            )
    else:
        keywords = manual_keywords

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []
    missing_keywords: List[str] = []

    for keyword in keywords:
        neighborhood = build_single_keyword_neighborhood_subgraph(
            graph,
            keyword=keyword,
            neighbors_per_keyword=args.neighbors_per_keyword,
        )
        neighborhood = limit_subgraph_size(neighborhood, max_nodes=args.max_nodes)

        if neighborhood.number_of_nodes() == 0:
            missing_keywords.append(keyword)
            continue

        output_path = args.output_dir / f"keyword_neighborhood_{sanitize_filename(keyword)}.png"
        draw_subgraph(
            subgraph=neighborhood,
            partition=partition,
            out_path=output_path,
            title=f"{args.title_prefix}: {keyword}",
            label_top_n=args.label_top_n,
            layout=args.layout,
            community_separation=args.community_separation,
            seed=config.seed,
        )
        generated_files.append(output_path)

    present_keywords = [normalize_token(k) for k in keywords if normalize_token(k) in graph]
    print(f"Base graph nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")
    print(f"Images generated: {len(generated_files)}")
    print(f"Keywords found in base graph: {len(present_keywords)}/{len(keywords)}")
    if args.auto_keywords:
        auto_keywords_csv = args.output_dir / "auto_generated_keywords.csv"
        export_auto_keywords_csv(
            output_path=auto_keywords_csv,
            keywords=keywords,
            partition=partition,
            graph=graph,
        )
        print("Auto keywords: " + ", ".join(keywords))
        print(f"Auto keyword CSV: {auto_keywords_csv}")
    print(f"Output folder: {args.output_dir}")
    if missing_keywords:
        print("Keywords missing in graph: " + ", ".join(missing_keywords))


if __name__ == "__main__":
    main()
