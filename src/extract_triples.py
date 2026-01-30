import argparse
import os
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

from auxiliary import get_chapter_token_range, int_range
from config import TOKENS, BASE_OUT_DIR, AVP_OUT, SVO_OUT


def extract_avp(characters: pd.DataFrame) -> pd.DataFrame:
    # TODO: CONSIDER SELF-EDGES i.e., if patient not present
    # TODO: CONSIDER PAIRS WHERE MULTIPLE AGENTS & PATIENTS ARE INVOLVED IN THE SAME ACTION i.e., token

    print(characters.shape)
    # agents = characters[characters['role'] == 'agent']
    # patients = characters[characters['role'] == 'patient']

    # merged = agents.merge(
    #     patients, 
    #     on="index", 
    #     suffixes=["_agent", "_patient"]
    #     )

    merged = characters.merge(
        characters,
        on='index',
        suffixes=['_left', '_right']
    )

    print(merged.shape)
    avp = merged[
        merged["canonical_id_left"] != merged["canonical_id_right"]
    ]

    print(avp.shape)
    tokens = pd.read_csv(TOKENS, sep="\t")
    
    # Merge avp with tokens to get the lemma for word_agent
    # Match avp['index'] with tokens['token_ID_within_document']
    avp = avp.merge(
        tokens[['token_ID_within_document', 'lemma']], 
        left_on='index', 
        right_on='token_ID_within_document',
        how='left'
    )
        
    # Drop the temporary columns we don't need
    avp['source'] = 'avp'
    avp['word'] = avp['word_left'] 

    avp = avp[['source', 'canonical_id_left', 'name_left', 'role_left',
       'word', 'lemma', 'index', 'original_ids_left', 'gender_left',
       'name_variants_left', 'canonical_id_right',
       'name_right', 'role_right', 'original_ids_right',
       'gender_right', 'name_variants_right']]
    
    if not os.path.isdir(AVP_OUT):
        os.makedirs(AVP_OUT)
        
    avp.to_csv(f"{AVP_OUT}/avp_triples.csv", sep="\t", index=False)
    return avp


if __name__ == '__main__':
    description = f"Use this script to create the initial network of characters\nFollowing token ranges are available for purposes of filtering:\n\n{get_chapter_token_range()}"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path(BASE_OUT_DIR + '/merged_characters.characters'),
        help='Specify the input file to use for triple extraction'
    )

    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(BASE_OUT_DIR),
        help="The output directory which to save the graphs to",
    )

    parser.add_argument(
        '-a', '--avp',
        action='store_true',
        help='A flag to generate Action-Verb-Patient triples'
    )

    parser.add_argument(
        '-s', '-svo',
        action='store_true',
        help='A flag to generate Subject-Verb-Object triples'
    )

    parser.add_argument(
        "-t",
        "--token-range",
        default=[],
        type=int_range,
        help="A comma-seperated interval '[x, y]' of token IDs for which to perform graph-creation",
    )

    args = parser.parse_args()

    characters = pd.read_csv(args.input, sep="\t")

    if args.token_range:
        # If a token interval is provided, filter the characters DF
        # based on the range for further analysis
        characters = characters[
            (characters["index"] >= args.token_range[0])
            & (characters["index"] <= args.token_range[1])
        ]

    if args.avp and args.out is not None:
        args.out = args.out / AVP_OUT
        characters = characters[characters['source'] == 'avp']
        avp = extract_avp(characters)
        print(avp.shape)
        exit(0)

    if args.svo and args.out is not None:
        args.out = args.out / SVO_OUT
        characters = characters[characters['source'] == 'svo']
