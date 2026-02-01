import argparse
import pandas as pd

from pathlib import Path
from config import BASE_OUT_DIR, TRIPLE_OUT
from auxiliary import print_headers, print_information


def join_triples(dir: Path = TRIPLE_OUT, verbose: bool = False) -> None:
    """
    Join AVP and SVO triples into a single DataFrame.
    
    1) Appends SVO DataFrame to AVP DataFrame
    2) Removes duplicates where columns at index [1, 6, 11] match
       (canonical_id_left/canonical_subj_id, index/verb_id, canonical_id_right/canonical_obj_id)
    
    Returns:
        Combined DataFrame with duplicates removed
    """
    print_headers("JOINING OF AVP & SVO TRIPLES", symb="=", prefix="\n")
    print_information("Joining triples...", 1, prefix="\n")

    avp = pd.read_csv(f'{dir}/avp_triples.csv', sep='\t')
    svo = pd.read_csv(f'{dir}/svo_triples.csv', sep='\t')
    
    # Rename columns so respective columns match on join
    svo.columns = avp.columns

    # Join SVO and AVP
    combined = pd.concat([avp, svo], ignore_index=True)
    
    # Get column names at indices 1, 6, 11 for deduplication
    # These correspond to: subject/left ID, verb index, object/right ID
    dedup_cols = [combined.columns[1], combined.columns[6], combined.columns[11]]
    
    pre_dedup_count = len(combined)
    # Remove duplicates based on these columns (keep first occurrence, which is AVP)
    combined = combined.drop_duplicates(subset=dedup_cols, keep='first')
    post_dedup_count = len(combined)
    
    print_information(f"Joined and removed {pre_dedup_count - post_dedup_count} duplicates", prefix="    ")
    combined.to_csv(f'{dir}/combined_triples.csv', sep='\t', index=False)
    print_information(f"Saved to {dir}/combined_triples.csv", "✓", prefix="\n", col="GREEN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SVO triples from tokenized text and match to canonical names.")

    parser.add_argument(
        "-o",
        "--out",
        default=BASE_OUT_DIR,
        help="The output directory to which the processing results are written",
    )

    parser.add_argument(
        '-v', '--verbose', 
        action="store_true",
        help="A flag to trigger verbose output"
    )

    args = parser.parse_args()

    join_triples(args.out, verbose=args.verbose)

    
