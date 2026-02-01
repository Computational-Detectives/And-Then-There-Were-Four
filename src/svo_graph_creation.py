import argparse
import os
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

from auxiliary import get_chapter_token_range, int_range


def create_graph(svo: pd.DataFrame, is_web_g: str, out_dir: Path) -> None:
    # ===============================
    # #### Create networkx graph ####
    # ===============================
    G = nx.from_pandas_edgelist(
        svo,
        source="canonical_character_id_left",
        target="canonical_character_id_right",
        edge_attr="word_left",  # edge attribute
        create_using=nx.DiGraph(),  # or nx.DiGraph()
    )

    # Add node attributes for agent nodes
    node_attrs = (
        svo[["canonical_character_id_left", "names_left"]]
        .drop_duplicates()
        .assign(name=lambda x: x["names_left"].str[0])
        .set_index("canonical_character_id_left")
        .to_dict()["names_left"]
    )

    print(node_attrs)

    nx.set_node_attributes(G, node_attrs, name="names_left")

    # ============================
    # #### Create pyvis graph ####
    # ============================
    if is_web_g:
        net = Network(notebook=True, height="1000px", width="100%", directed=True)

        for node, attrs in G.nodes(data=True):
            net.add_node(
                node, label=str(attrs.get("name_left", node)), title=str(attrs)
            )

        for u, v, attrs in G.edges(data=True):
            net.add_edge(
                u, v, label=attrs.get("word_left"), title=attrs.get("word_left")
            )

        # net.show(out_dir)
        # Create output directory if it does not yet exist
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        net.save_graph(str(out_dir / "graph.html"))


if __name__ == "__main__":
    description = f"Use this script to create the initial network of characters\nFollowing token ranges are available for purposes of filtering:\n\n{get_chapter_token_range()}"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-w",
        "--web",
        action="store_true",
        help="A flag to indicate the creation of an HTML file of the generated graph",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("../data/out/graphs"),
        help="The output directory which to save the graphs to",
    )
    parser.add_argument(
        "-t",
        "--token-range",
        default=[],
        type=int_range,
        help="A comma-seperated interval '[x, y]' of token IDs for which to perform graph-creation",
    )

    args = parser.parse_args()

    characters = pd.read_csv("../data/out_test/merged_characters.characters", sep="\t")
    # characters.sort_values(by='index').to_csv('../data/out/characters_sorted.csv', sep='\t', index=False)

    if args.token_range:
        # If a token interval is provided, filter the characters DF
        # based on the range for further analysis
        characters = characters[
            (characters["index"] >= args.token_range[0])
            & (characters["index"] <= args.token_range[1])
        ]

    # Create the graphs
    # create_graph(svo, args.web, args.out)
