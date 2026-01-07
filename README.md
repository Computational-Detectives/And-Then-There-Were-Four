# And Then There Were Four

Computational Network Analysis to spot the killer in the famous novel "And Then There Were None" by Agatha Christie. 🕵️💻

Stay tuned!

## Installation
We recommend to follow the below steps on a Unix machine to reproduce the environment used for this project. Importantly, because the `Torch` library for now still depends on Python `<=3.12` it is necessary to use a Python version not higher than `3.12`.

```bash
# Create & activate virtual environment on Unix (or Windows equivalent)
python3.12 -m venv <venv_name>
source bin/activate/<venv_name>

# Install packages
pip install -r requirements.txt

# Download English language processing model
python -m spacy download en_core_web_sm
```

## Run Script
To run the script, run the following file as follows. This should produce the output files in `data/out` under the condition that all issues with `BookNLP` have been solved prior to execution. A fix that works on Unix is located in `booknlp_fix.py`.

```bash
# Run script
python ingest.py
```
