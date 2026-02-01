import argparse
import pandas as pd

from pathlib import Path
from config import ATTWN, BASE_OUT_DIR

from ingest import main as ingest
from match_names import main as match_names
from extract_avp_triples import extract_avp
from extract_svo_triples import main as extract_svo
from join_triples import join_triples


def main(input: Path, out: Path, verbose: bool = False):
    # 1) Run BookNLP pipeline on input
    ingest(input, out)

    # 2) Match all character names on BookNLP output
    match_names(
        input_file=out / 'preproc_attwn.book', 
        output_dir=str(out), 
        verbose=verbose
        )

    # 3) Extract AVP triples
    characters = pd.read_csv(f'{out}/merged_characters.characters', sep='\t')
    extract_avp(
        characters=characters, 
        out=out / 'triples', 
        verbose=verbose
        )

    # 5) Extract SVO triples
    extract_svo(
        out=out / 'triples', 
        verbose=verbose
        )
    
    # 6) Join AVP & SVO triples
    join_triples(
        dir=out / 'triples',
        verbose=verbose
    )

    # 7) Merge SVO and AVP triples

    # 8) Run cooccurrence network creation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the whole end-to-end processing pipeline')
    parser.add_argument(
        "-i", "--input", 
        default=ATTWN,
        type=Path, 
        help="The path to the input file to be processed"
        )
    
    parser.add_argument(
        "-o",
        "--out",
        default=BASE_OUT_DIR,
        type=Path,
        help="The output directory to which the processing results are written",
    )

    # parser.add_argument(
    #     '-a', '--avp',
    #     action='store_true',
    #     help='A flag to generate Action-Verb-Patient triples'
    # )

    # parser.add_argument(
    #     '-s', '-svo',
    #     action='store_true',
    #     help='A flag to generate Subject-Verb-Object triples'
    # )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="A flag to trigger verbose output"
    )
    args = parser.parse_args()
    main(args.input, args.out, args.verbose)


