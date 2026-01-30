import os
import csv
import pandas as pd
import numpy as np
import spacy

from spacy.tokens import Doc
from spacy.attrs import HEAD, DEP
from config import TOKENS, SVO_OUT
from textacy import extract
from auxiliary import print_headers, print_information


# ================= CONFIG =================

OUTPUT_CSV  = "../data/out_test/svo_triples_textacy.csv"

VALIDATE_TREES = False

nlp = spacy.blank("en")

# ============== LOAD TOKENS ==============

def load_tokens(path):
    return pd.read_csv(
        path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        engine="python",
        keep_default_na=False
    )

# ==========================================
# =========== DOC RECONSTRUCTION ===========
# ==========================================
def make_doc_from_sentence(sentence: pd.DataFrame, validate=False):
    # Extract all words in the sentence & compute location of spaces
    words  = sentence["word"].tolist()
    spaces = [True] * (len(words) - 1) + [False]

    # Create custom spacy.Doc object for the current sentence
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    # Compute sentence-local token IDs. Required by spacy.Doc()
    global_to_local = {row["token_ID_within_document"]: idx for idx, (_, row) in enumerate(sentence.iterrows())}
    
    heads = []
    deps  = []

    # Add token indices relative to sentence head &
    # add dependencies to the vocabulary
    for idx, (_, row) in enumerate(sentence.iterrows()):
        global_head_idx = int(row["syntactic_head_ID"])        
        local_head_idx = global_to_local[global_head_idx]
        rel_head = local_head_idx - idx
        
        heads.append(rel_head)
        deps.append(nlp.vocab.strings.add(row["dependency_relation"]))    
    
    n = len(heads)
    arr = np.zeros((n, 2), dtype="uint64")

    for i in range(n):
        arr[i, 0] = np.int64(heads[i]).view(np.uint64)
        arr[i, 1] = deps[i]

    # Add head and dependency information to Doc-object
    doc.from_array([HEAD, DEP], arr)

    # Assign POS / TAG / LEMMA and store global token IDs
    for token, (_, row) in zip(doc, sentence.iterrows()):
        token.pos_   = row["POS_tag"]
        token.tag_   = row["fine_POS_tag"]
        token.lemma_ = row["lemma"]
        # Store global token ID as custom attribute
        token._.global_id = row["token_ID_within_document"]

    if validate:
        validate_doc(doc, sentence["sentence_ID"].iloc[0])

    return doc

# ================ VALIDATOR ===============

def validate_doc(doc, sentence_id):
    roots = [t for t in doc if t.head == t]
    if len(roots) != 1:
        raise ValueError(
            f"[VALIDATION ERROR] sentence {sentence_id}: "
            f"{len(roots)} ROOTs detected"
        )

    branching = [t for t in doc if len(list(t.children)) > 1]
    if not branching:
        raise ValueError(
            f"[VALIDATION ERROR] sentence {sentence_id}: "
            f"no token has more than one child"
        )

    for token in doc:
        if token.head == token and token.dep_ != "ROOT":
            raise ValueError(
                f"[VALIDATION ERROR] sentence {sentence_id}: "
                f"non-ROOT self-cycle at token '{token.text}'"
            )

# ============ SVO EXTRACTION USING TEXTACY ==============

def extract_svo(doc):
    svo_triples = list(extract.subject_verb_object_triples(doc))
    structured_triples = []
    
    def get_info(component):
        # Helper to extract info whether component is Token, Span, or list of Tokens
        if isinstance(component, list) or isinstance(component, spacy.tokens.Span):
            ids = [t._.global_id for t in component]
            
            if isinstance(component, list):
                text = " ".join([t.text for t in component])
                lemma = " ".join([t.lemma_ for t in component])
                # For list, approximate root pos as the last token's pos
                pos = component[-1].pos_ if component else "NOUN"
            else:
                text = component.text
                lemma = component.lemma_
                pos = component.root.pos_
                
            primary_id = ids[0] if ids else None
        else:
            # Single Token
            ids = [component._.global_id]
            primary_id = ids[0]
            text = component.text
            lemma = component.lemma_
            pos = component.pos_
            
        return text, lemma, ids, primary_id, pos

    def is_negated(verb_component):
        """
        Check if the verb is negated by looking for a child with dep_="neg".
        
        Examples of negation: "does not like", "didn't see", "never went"
        """
        # Get all tokens from the verb component
        if isinstance(verb_component, list):
            tokens = verb_component
        elif isinstance(verb_component, spacy.tokens.Span):
            tokens = list(verb_component)
        else:
            tokens = [verb_component]
        
        # Check if any verb token has a negation child
        for token in tokens:
            for child in token.children:
                if child.dep_ == "neg":
                    return True
        return False

    for s, v, o in svo_triples:
        s_text, s_lemma, s_ids, s_id, s_pos = get_info(s)
        v_text, v_lemma, v_ids, v_id, v_pos = get_info(v)
        o_text, o_lemma, o_ids, o_id, o_pos = get_info(o)
        
        # Check if the verb is negated
        negated = is_negated(v)
            
        # print(f'Subject: {s_text} {s_ids}')
        # print(f'Verb:    {v_text} {v_id} {"(NEGATED)" if negated else ""}') 
        # print(f'Object:  {o_text} {o_ids}')
        # print('-----------------------------')

        structured_triples.append({
            "subject_text": s_text,
            "subject_ids": s_ids,
            "subject_pos": s_pos,
            "verb_text": v_text,
            "verb_lemma": v_lemma,
            "verb_id": v_id,
            "object_text": o_text,
            "object_ids": o_ids,
            "object_pos": o_pos,
            "negated": negated
        })

    return structured_triples



# ================== MAIN ==================

def main():
    print_headers("RUNNING NAME SVO EXTRACTION PIPELINE", "=")

    # Register custom attribute for global token IDs
    from spacy.tokens import Token
    Token.set_extension("global_id", default=None, force=True)
    

    print_information(f"Loading tokens from {TOKENS} ...", 1, '\n')
    df = load_tokens(TOKENS)
    print_information(f"Loaded {df.shape[0]} tokens", prefix="    ")

    print_information('Converting sentences to spaCy.Doc objects...', 2, '\n')
    docs = []
    for sid, sent_df in df.groupby("sentence_ID"):
        doc = make_doc_from_sentence(sent_df.reset_index(drop=True))
        docs.append(doc)
    print_information(f"Converted {len(docs)} sentences", prefix="    ")

    print_information('Extracting SVO triples...', 3, '\n')
    all_triples = []
    for doc in docs:
        all_triples.extend(extract_svo(doc))
    print_information(f"Extracted {len(all_triples)} triples", prefix="    ")


    print_information('Saving SVO triples...', 4, '\n')

    if not os.path.isdir(SVO_OUT):
        os.mkdir(SVO_OUT)

    # Write CSV - need to flatten tuples for CSV format
    with open(f"{SVO_OUT}/svo_triples.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(
            f,
            delimiter='\t',
            fieldnames=["subject_text", "subject_ids", "subject_pos", "verb_text", "verb_lemma", "verb_id", 
                       "object_text", "object_ids", "object_pos", "negated"]
        )
        writer.writeheader()
        
        for triple in all_triples:
            writer.writerow(triple)

    print_information(f"Results saved to → {OUTPUT_CSV}", symb="✓", prefix="\n")

if __name__ == "__main__":
    main()

    svo = pd.read_csv(OUTPUT_CSV, sep='\t')
    print(svo.head(50))
    print()
    print(svo[(svo['subject_pos'] == 'PROPN') | (svo['object_pos'] == 'PROPN')].shape)
    # svo = pd.read_csv('../data/out_new/svo_triples.csv', sep='\t')
    # print(svo[(svo['negated'] == False) & ((svo['subject_pos'] == 'PRON') | (svo['object_pos'] == 'PRON'))])