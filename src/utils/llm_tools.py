import torch
from transformers import pipeline, AutoTokenizer
from utils.io import from_yaml
import os

config = from_yaml("config.yaml")

def get_llm_pipeline(
 #    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_id="Qwen/Qwen2.5-7B-Instruct",
    max_tokens=100
):
    """
    Initialize and return a text generation pipeline.

    Args:
        model_id (str): The identifier for the model to be used.

    Returns:
        pipeline: A text generation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    answer_pipeline = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        max_new_tokens=max_tokens,
        device=device,
    )
    return answer_pipeline


def get_questions(dataset):
    """
    Generate questions for each image caption in the dataset.

    Args:
        dataset (dict): A dictionary containing image data.

    Returns:
        dict: The updated dataset with generated questions.
    """
    prompt = config['prompts']['prompt_for_questions']
    answer_pipeline = get_llm_pipeline()

    for folder, img_names in dataset.items():
        for img_name in img_names:
            current_caption = dataset[folder][img_name]["Caption"]
            questions = answer_pipeline(prompt.format(og_caption=current_caption))

            questions = questions[0]["generated_text"].split('ANSWER')[3].split("\n")
            questions = [s.strip()[2:] for s in questions if s.strip().startswith('-')]

            # Join the questions back into a single string
            full_text = ' '.join(questions)
            # Split by question marks, but keep the question marks
            questions = [q.strip() + '?' for q in full_text.split('?') if q.strip()]
            # Limit to 5 questions
            dataset[folder][img_name]["Questions"] = questions[:5]

    return dataset


def combine_qa_pairs_w_captions(dataset):
    """
    Combine question-answer pairs with captions to create expanded captions.

    Args:
        dataset (dict): A dictionary containing image data with questions and answers.

    Returns:
        dict: The updated dataset with expanded captions.
    """
    prompt = config['prompts']['prompt_for_combining']
    answer_pipeline = get_llm_pipeline()

    for folder, img_names in dataset.items():
        for img_name in img_names:
            expanded_captions = []
            for question, answer in zip(
                dataset[folder][img_name]["Questions"],
                dataset[folder][img_name]["ZAnswers"]
            ):
                caption = dataset[folder][img_name]["Caption"]
                output = answer_pipeline(prompt.format(
                    caption=caption,
                    question=question,
                    answer=answer
                ))

                output = output[0]["generated_text"].split('NEW CAPTION:')[4]
                output = output.split("Caption")[0].strip()
                expanded_captions.append(output)

            dataset[folder][img_name]["ExpandedCaptions"] = expanded_captions

    return dataset


def get_prag_output(dataset):
    """
    Generate pragmatic captions for each target image by comparing with distractor images.

    Args:
        dataset (dict): A dictionary containing image data with caption candidates.

    Returns:
        dict: The updated dataset with pragmatic captions.
    """
    prompt = config['prompts']['prompt_for_pragmatic_caption']
    answer_pipeline = get_llm_pipeline(max_tokens=45)

    for folder, folder_data in dataset.items():
        for target_img, target_data in folder_data.items():
            target_candidates = target_data["CaptionCandidates"]
            distractor_candidates = {
                img: data["CaptionCandidates"]
                for img, data in folder_data.items()
                if img != target_img
            }

            output = answer_pipeline(prompt.format(
                target_dict=target_candidates,
                distractor_dict=distractor_candidates
            ))
            # Extract the pragmatic caption from the output
            pragmatic_caption = output[0]["generated_text"].split('PRAGMATIC CAPTION:')[2].strip()
            dataset[folder][target_img]["PragCaption"] = pragmatic_caption

    return dataset
