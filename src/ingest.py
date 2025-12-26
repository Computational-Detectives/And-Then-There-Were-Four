import os
import torch

from pathlib import Path
from booknlp.booknlp import BookNLP
from booknlp_fix import process_model_files, get_model_path, exists_model_path


def init_run():
	'''
	A convenience function executed as the initial run of the script
	to download all BERT models used by BookNLP. All subsequent executions
	of the script will run this method, but fail because of the `position_ids`
	that will then get removed. 

	This setup prevents having to manually execute the initial run before
	being able to execute the actual script. The raised exception ensures
	that, once this function fails, the correct pipeline with the modified
	model parameters is loaded instead.
	'''
	try:
		if exists_model_path():
			return
		
		model_params = {
			"pipeline": "entity,quote,supersense,event,coref", 
			"model": "big"
		}
		
		booknlp = BookNLP("en", model_params)

		# Input file to process
		input_file = "../data/attwn.txt"

		# Output directory to store resulting files in
		output_directory = "../data/out"

		# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
		book_id = "attwn"

		# Run the processing pipeline
		booknlp.process(input_file, output_directory, book_id)
	except Exception as e:
		pass


def main():
	'''
	The main function of the script to execute the BookNLP pipeline.

	The modified models are loaded once their original version has been
	downloaded in `init_run`. Subsequent runs to the initial run fall
	through to the `finally`-block.
	'''
	try:
		init_run()
	except Exception as e:
		pass
	finally:
		model_path = get_model_path()

		# Create custom model w/o original `position_ids` in BERT models
		model_params = {
				"pipeline": "entity,quote,supersense,event,coref", 
				"model": "custom",
				"entity_model_path": str(model_path / 'entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model'),
				"coref_model_path": str(model_path / 'coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model'),
				"quote_attribution_model_path": str(model_path / 'speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model'),
				"bert_model_path": str(model_path.parent / '.cache/huggingface/hub/')
			}

		# Create the `torch.device`-Object
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Process the new model files with deleted `position_ids`
		model_params = process_model_files(model_params, device)

		# Create the BookNLP pipeline object
		booknlp = BookNLP("en", model_params)

		# Input file to process
		input_file = "../data/attwn.txt"

		# Output directory to store resulting files in
		output_directory = "../data/out"

		# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
		book_id = "attwn"

		# Run the processing pipeline
		booknlp.process(input_file, output_directory, book_id)


if __name__ == '__main__':
    main()