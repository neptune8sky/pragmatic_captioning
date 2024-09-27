import torch
from typing import Dict
from transformers import (
    InstructBlipProcessor,
    BlipProcessor,
    InstructBlipForConditionalGeneration,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    AutoProcessor,
    BlipForImageTextRetrieval
)
from utils.io import read_img, from_yaml
import os

config = from_yaml("config.yaml")

def get_captions(
    dataset: Dict[str, list],
    prompt: str = None,
    max_new_tokens: int = config['params']['blip2']['max_new_tokens'],
    num_beams: int = config['params']['blip2']['num_beams'],
    temperature: float = config['params']['blip2']['temperature'],
    top_k: int = config['params']['blip2']['top_k'],
    top_p: float = config['params']['blip2']['top_p'],
    repetition_penalty: float = config['params']['blip2']['repetition_penalty'],
    length_penalty: float = config['params']['blip2']['length_penalty']
) -> Dict[str, Dict[str, str]]:
    """
    Generate captions for images using BLIP1 or BLIP2-Instruct model with controlled parameters.

    Args:
        dataset: A dictionary of folder names and image file paths.
        prompt: Optional text prompt for BLIP2-Instruct model caption generation.
        max_new_tokens: Maximum number of tokens to generate.
        num_beams: Number of beams for beam search.
        temperature: Temperature for sampling.
        top_k: Top-k sampling.
        top_p: Top-p sampling.
        repetition_penalty: Repetition penalty.
        length_penalty: Length penalty.

    Returns:
        A nested dictionary of folder names, image file names, and captions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if prompt is not None:
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            device_map="auto",
            torch_dtype=torch.float32
        ).to(device)
    else:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

    captions_dict = {}
    for folder, img_paths in dataset.items():
        captions_dict[folder] = {}
        for img_path in img_paths:
            image = read_img(img_path)
            if prompt is not None:
                # prompt_text = f"Question: {prompt} Answer:" (Not applicable for BLIP2-Instruct)
                inputs = processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(device, torch.float32)
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    do_sample=True
                )
            else:
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                ).to(device, torch.float32)
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens
                )

            with torch.no_grad():
                caption = processor.decode(
                    output[0],
                    skip_special_tokens=True
                ).strip()
            # Save results to the output dictionary
            captions_dict[folder][img_path] = {"Caption": caption}
            image.close()

    return captions_dict


def get_answers(
    dataset: Dict[str, list],
    use_blip2=False,
    max_new_tokens: int = config['params']['blip2']['max_new_tokens'],
    num_beams: int = config['params']['blip2']['num_beams'],
    temperature: float = config['params']['blip2']['temperature'],
    top_k: int = config['params']['blip2']['top_k'],
    top_p: float = config['params']['blip2']['top_p'],
    repetition_penalty: float = config['params']['blip2']['repetition_penalty'],
    length_penalty: float = config['params']['blip2']['length_penalty']
):
    """
    Generate answers for questions in the dataset using BLIP or BLIP2-Instruct model.

    Args:
        dataset: A dictionary containing image data and questions.
        use_blip2: A boolean flag to use BLIP2-Instruct model instead of BLIP1.

    Returns:
        The updated dataset with generated answers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_blip2:
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=torch.float32,
        ).to(device)
        processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-vicuna-7b"
        )
    else:
        model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        ).to(device)
        processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

    for folder, img_names in dataset.items():
        for img_name in img_names:
            answers = []
            for question in dataset[folder][img_name]["Questions"]:
                image = read_img(img_name)

                input_data = processor(
                    images=image,
                    text=question,
                    return_tensors="pt"
                ).to(device, torch.float32)

                with torch.no_grad():
                    output = model.generate(
                        **input_data,
                        max_new_tokens=30,
                        num_beams=num_beams,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        do_sample=True
                    )
                    answer = processor.decode(
                        output[0],
                       skip_special_tokens=True
                    )
                answers.append(answer.strip())
            dataset[folder][img_name]["ZAnswers"] = answers

    return dataset


def filter_captions(dataset):
    """
    Filter and evaluate expanded captions using the BLIP1 model.

    Args:
        dataset: A nested dictionary of folder names, image paths, and captions.

    Returns:
        The updated dataset with filtered caption candidates.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco"
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-itm-base-coco"
    )

    for folder, images in dataset.items():
        for img_path in images:
            image = read_img(img_path)
            expanded_captions = dataset[folder][img_path]["ExpandedCaptions"]
            caption_candidates = []

            for caption in expanded_captions:
                inputs = processor(
                    images=image,
                    text=caption,
                    return_tensors="pt"
                ).to(device, torch.float16)
                outputs = model(**inputs)

                score = outputs.itm_score.softmax(dim=1)[:, 1].item()
                if score > 0.95:
                    caption_candidates.append(caption)

            dataset[folder][img_path]["CaptionCandidates"] = caption_candidates

    return dataset


def merge_captions(
    dict1: Dict[str, Dict[str, Dict[str, str]]],
    dict2: Dict[str, Dict[str, Dict[str, str]]]
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Merge captions from two dictionaries, using dict2 captions if dict1 has empty entries.

    Args:
        dict1: Primary dictionary with BLIP2 captions.
        dict2: Secondary dictionary with BLIP1 captions.

    Returns:
        The merged dictionary with updated captions.
    """
    for folder in dict1:
        if folder in dict2:
            for img_name in dict1[folder]:
                if img_name in dict2[folder]:
                    if not dict1[folder][img_name]["Caption"]:
                        dict1[folder][img_name]["Caption"] = dict2[folder][img_name]["Caption"]

    return dict1
