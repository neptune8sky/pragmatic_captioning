#!/usr/bin/env python3
"""
Main script for pragmatic image captioning.

This script orchestrates the process of generating, expanding, and refining
pragmatic image captions using open source VLMs and LLMs.

The script operates on a custom made clothing dataset consisting of
50 sets of pragmatic images, each set containing 4 images.
"""

import os
from typing import Dict, Any
import warnings
from transformers import logging as transformers_logging

from utils.io import create_folder_structure, to_yaml, from_yaml

from utils.blip_instruct_tools import (
    get_captions,
    merge_captions,
    filter_captions,
    get_answers
)
from utils.llm_tools import (
    get_questions,
    combine_qa_pairs_w_captions,
    get_prag_output
)

# Load paths, prompts and other configuration parameters
config = from_yaml(
    "config.yaml"
)

# Paths
CONFIG_PATH = config['paths']['config']
DATASET_PATH = config['paths']['dataset']
OUTPUT_DIR = config['paths']['output']
DATASET_STRUCTURE_PATH = config['paths']['dataset_structure']


def setup_warning_filters():
    """
    Set up warning filters to ignore specific warnings from transformers.
    """
    # Ignore all warnings from transformers
    transformers_logging.set_verbosity_error()
    # Ignore specific warnings
    warnings.filterwarnings(
        "ignore",
        message="Both `max_new_tokens` .* and `max_length`.* seem to have been set",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="`clean_up_tokenization_spaces` was not set",
        category=FutureWarning
    )



def print_progress(message: str) -> None:
    """Print a progress message with a separator line."""
    print(f"\n{message}")
    print("-" * len(message))


def load_config() -> Dict[str, Any]:
    """
    Load and return the configuration from the YAML file.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    print_progress("Loading configuration")
    return from_yaml(CONFIG_PATH)



def main() -> None:
    """
    Main function to carry out the image captioning and analysis process.

    This function orchestrates the entire workflow, including:
    1. Generating initial captions with BLIP1 & BLIP2-Instruct
    2. Obtaining questions for the captions via LLM
    3. Answering the generated questions with BLIP2-Instruct
    4. Obtaining new caption candidates from questions and answers via LLM
    5. Filtering out non-accurate captions with BLIP1 scores
    6. Obtaining the final pragmatic caption via LLM
    """
    config = load_config()
    initial_caption_prompt = config['prompts']['initial_caption']

    print_progress("Creating folder structure")
    create_folder_structure(DATASET_PATH, DATASET_STRUCTURE_PATH)
    dataset = from_yaml(DATASET_STRUCTURE_PATH)


    print_progress("Generating captions with BLIP1")
    captions_blip1 = get_captions(dataset)
    to_yaml(captions_blip1, os.path.join(OUTPUT_DIR, "captions_BLIP1.yaml"))

    print_progress("Generating captions with BLIP2")
    captions_blip2 = get_captions(dataset, prompt=initial_caption_prompt)
    to_yaml(captions_blip2, os.path.join(OUTPUT_DIR, "captions_BLIP2.yaml"))

    print_progress("Merging captions") # In case BLIP2 captions are empty
    captions_merged = merge_captions(captions_blip2, captions_blip1)
    to_yaml(captions_merged, os.path.join(OUTPUT_DIR, "captions_merged.yaml"))

    captions_merged = from_yaml(os.path.join(OUTPUT_DIR, "captions_merged.yaml"))
    captions_merged = from_yaml("out/captions_merged.yaml")
    print_progress("Generating questions")
    questions = get_questions(captions_merged)
    to_yaml(questions, os.path.join(OUTPUT_DIR, "questions_dataset.yaml"))

    print_progress("Generating answers")
    answers_dataset = get_answers(questions, use_blip2=True)
    to_yaml(answers_dataset, os.path.join(OUTPUT_DIR, "answers_dataset.yaml"))

    print_progress("Expanding captions with answers")
    expanded_captions = combine_qa_pairs_w_captions(answers_dataset)
    to_yaml(expanded_captions, os.path.join(OUTPUT_DIR, "expanded_dataset.yaml"))

    print_progress("Filtering captions")
    caption_candidates = filter_captions(expanded_captions)
    to_yaml(caption_candidates, os.path.join(OUTPUT_DIR, "candidates_dataset.yaml"))

    print_progress("Generating pragmatic captions")
    pragmatic_dataset = get_prag_output(caption_candidates)
    to_yaml(pragmatic_dataset, os.path.join(OUTPUT_DIR, "pragmatic_dataset.yaml"))

    print_progress("Process completed successfully")


if __name__ == "__main__":
    setup_warning_filters()
    main()
