import json
import argparse
import pandas as pd

from time import sleep
from pathlib import Path
from typing import Tuple
from colorama import Fore, Style, init


# # Explore number of events detected
# df = pd.read_csv('../data/out_new/preproc_attwn.tokens', sep='\t')

# # Remove all rows where the `event` is null
# events = df[~df['event'].isnull()]

# # Remove all rows where `event != EVENT`
# events = events[events['event'] == 'EVENT']

def load_and_flatten(input_file: str) -> pd.DataFrame:
    '''
    This method loads the data on the extracted characters from the JSON
    output of the `BookNLP` pipeline and recreates the structure of the
    most necessary information for further processing as a `pd.DataFrame`.
    
    :param input_file: The path to the JSON file
    :type input_file: str
    :return: The flattened DataFrame containing the most necessary character information
    :rtype: DataFrame
    '''
    print(f'[{Fore.BLUE}*{Style.RESET_ALL}] Loading JSON object into Pandas DataFrame, matching names, and writing to results to file...')

    # Load `.book`-output file for further processing
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract high-level Character-object from JSON file
    characters_lst = data['characters']
    # chrtr = pd.json_normalize(characters_lst)
    # chrtr.head()

    rows = []

    # For each character...
    for char in characters_lst:
        # ...extract their character_ID
        cid = char["id"]
        # ...for each role they played
        for role in ["agent", "patient"]: # , "mod", "poss"]:
            # ...extract the action & index associated with it
            for tok in char.get(role, []):
                rows.append({
                    "character_id": cid,
                    "names": [elem['n'] for elem in char['mentions']['proper']],
                    "role": role,
                    "word": tok["w"],
                    "index": tok["i"],
                    "gender": "" if char['g'] is None else char['g']['argmax']
                })

    df_roles = pd.DataFrame(rows)
    return df_roles


# ============================
# Pre-processing of DataFrame
# ============================
def preprocess_characters(df_roles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    This method preprocesses the character information contained in the
    DataFrame before matching proper names.
    
    :param df_roles: The flattened DataFrame containing the most necessary character information
    :type df_roles: pd.DataFrame
    :return: A tuple of DataFrames containing information on the names and which entries are shared amongst entities
    :rtype: Tuple[DataFrame, DataFrame]
    '''
    # Main DataFrame: Explode names list onto individual row
    f_names = df_roles[["character_id", "names"]].explode("names")

    # Intermediate DataFrame: Remove rows without associated names
    f_names = f_names[~f_names['names'].isna()]

    # Lowercase each name
    f_names['names'] = f_names['names'].str.lower().str.strip()

    # Count the number of unique occurrences of each name.
    # Later used to remove unique names, as they cannot be merged.
    name_to_characters = (
        f_names
        .groupby("names")["character_id"]
        .nunique()
    )

    shared_names = name_to_characters[name_to_characters > 1].index

    # Add uniqueness information to original names-DataFrame
    f_names["shared"] = f_names["names"].isin(shared_names)

    return f_names, shared_names

# ============================
# Match Duplicate Names
# ============================
def match_duplicate_names(df_roles: pd.DataFrame, f_names: pd.DataFrame, shared_names: pd.DataFrame) -> pd.DataFrame:
    '''
    The workhorse function to match duplicate entries based on common occurrences of 
    proper names in the list of names in `f_names`. 
    
    Entries are matched based on the smallest common `character_id` for the same entity. That is, 
    if `BookNLP` associates the name X with the `character_id`'s `[13, 42, 1337]` and three individual 
    entities, then `match_duplicate_names` will assign all entries the `canonical_character_id=13`. 
    
    :param df_roles: The flattened DataFrame containing the most necessary character information
    :type df_roles: pd.DataFrame
    :param f_names: A DataFrames containing information on the names
    :type f_names: pd.DataFrame
    :param shared_names: A DataFrame containing information on whether a proper name has multiple occurrences across different entities
    :type shared_names: pd.DataFrame
    :return: A DataFrame with matched & merged names & the new `canonical_character_id`
    :rtype: DataFrame
    '''
    # Group on non-unique names & merge occurrences of different 
    # character identifiers for that name onto a single, unique name
    overlap = (
        f_names[f_names['names'].isin(shared_names)]
        .groupby('names')['character_id']
        .apply(set).apply(list)
        .reset_index(name='character_ids')
    )

    # Create mapping of original character ID to a canonical identifier
    canonical_map = {}

    for ids in overlap["character_ids"]:
        canonical = min(ids)
        for cid in ids:
            canonical_map[cid] = canonical

    # Retain the original character ID in the added column `original_character_id`
    df_roles['original_character_id'] = df_roles['character_id']

    # For each original character ID, get the new canonical character ID from the previously created dictionary
    df_roles['canonical_character_id'] = (
        df_roles['character_id']
        .map(lambda x: canonical_map.get(x, x))
    )

    return df_roles


# ============================
# Post-processing of DataFrame
# ============================
def postprocess_characters(df_roles: pd.DataFrame, output_file: Path) -> None:
    '''
    This method post-processes the DataFrame with the matched names to 
    - remove entries that are not associated with a proper name,
    - sort the results based on the `canonical_character_id`,
    - write the results to file
    
    :param df_roles: A DataFrame with matched & merged names & the new `canonical_character_id`
    :type df_roles: pd.DataFrame
    :param output_file: The path to the output directory
    :type output_file: Path
    '''
    df_roles.drop(['character_id'], axis=1)

    # Remove all rows that don't have a proper name associated with them
    df_roles = df_roles[df_roles['names'].astype(bool)]

    # Merge lists with different names of the same entity
    df_roles.loc[:,'names'] = ( df_roles
            .groupby("canonical_character_id")["names"] 
            .transform(lambda x: [list(dict.fromkeys(name for lst in x for name in lst))] * len(x))
        )

    # Drop entries with duplicate token_ID_within_document i.e., index
    df_roles = df_roles.drop_duplicates(subset=['index'])

    # Reorder columns
    df_roles = df_roles[['canonical_character_id', 'names', 'role', 'word', 'index', 'original_character_id', 'gender']]

    # Sort by canonical character ID
    df_roles = df_roles.sort_values(by=['canonical_character_id'])

    # Write results to file
    df_roles.to_csv(str(output_file / 'merged_characters.characters'), sep='\t', index=False)

    sleep(1)
    print(f'[{Fore.GREEN}+{Style.RESET_ALL}] Wrote results to file str(output_file / "merged_characters.characters")')

if __name__ == '__main__':
    # Required for colorama to work on Windows
    init()

    parser = argparse.ArgumentParser(description="Running this script will match the proper names of the same character if they were extracted to be from different entities.")
    parser.add_argument('input_file', help='The path to the input file that is to be used for matching')
    parser.add_argument('output_file', help='The path to output directory (w/o the final ' + "'\\') to be used to write the results to")

    args = parser.parse_args()

    df_roles = load_and_flatten(args.input_file)
    f_names, shared_names = preprocess_characters(df_roles)
    df_roles = match_duplicate_names(df_roles, f_names, shared_names)
    postprocess_characters(df_roles, Path(args.output_file))
