import re
import spacy

from pathlib import Path

def preprocess(input_file: str, output_file: str):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Read raw text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Create first match pattern
    headers_pttrn = re.compile(
        r'\bChapter\s+\d+\b|'                               # Match Chapter headers
        r'\bEpilogue\b|'                                    # Match Epilogue header
        r'\bA MANUSCRIPT DOCUMENT SENT TO SCOTLAND YARD\b|' # Match Manuscript header
        r'\b[IVXLCDM]+\b\s+|'                               # Match Roman numerals
        r'\bIll\b\s+'                                       # Match ill-formatted Roman numeral III
    )

    cleaned_text = headers_pttrn.sub('', text)

    # Normalize line endings
    text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n")

    # TODO: Add removal of Chapter headers
    # ==========================
    # Collapse line breaks inside quotes
    # ==========================
    # Matches double or single quotes
    # quote_pattern = r'(["\'])(.*?)\1'
    quote_pattern = r'"(.*?)"'

    # def collapse_newlines_in_quotes(match):
    #     quote_text = match.group(1)
    #     # print(f'CURRENT QUOTE TEXT: {quote_text}')
    #     # Collapse all whitespace inside the quote to single spaces
    #     collapsed = re.sub(r'\s+', ' ', quote_text).strip()

    #     print(f'QUOTE TEXT: {quote_text}')
    #     print(f'COLLAPSED: {collapsed}')
    #     return f'"{collapsed}"'

    # # Apply to text
    # text = re.sub(quote_pattern, collapse_newlines_in_quotes, text, flags=re.DOTALL)

    # Collapse mid-sentence line breaks outside quotes
    # Replace single newlines inside paragraphs with space, keep double newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)

    # Segment into individual sentences
    doc = nlp(text)

    # Write collapsed sentences to file
    with open(output_file, "w", encoding="utf-8") as f:
        for sent in doc.sents:
            # Collapse everything inside the sentence
            clean_sent = re.sub(r'\s+', ' ', sent.text).strip()
            f.write(clean_sent + "\n")
        
    # print(f"All sentences (quotes collapsed) written to: {output_file}")

if __name__ == '__main__':
    input_file = "../data/book/attwn.txt"
    output_file = "../data/book/attwn_sentences_collapsed.txt"
    preprocess(input_file, output_file)
