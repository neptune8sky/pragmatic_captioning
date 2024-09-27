# Term Project: Pragmatic Image Captioning

In this project, we propose a method for generating pragmatic captions for a target image among similar-looking distractor images, using a Vision Language Model (BLIP1 & BLIP2) together with a cognitive mechanism deployed by an LLM (Qwen2.5-7B-Instruct). All of the models used for this project are open source.

### Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Folder Structure](#folder-structure)
5. [Configuration](#configuration)
6. [Results](#results)

### Project Overview

This project aims to generate pragmatic image captions that effectively distinguish a target image from similar distractor images. We utilize state-of-the-art Vision Language Models (BLIP1 and BLIP2) in combination with a Large Language Model (Qwen2.5-7B-Instruct) to achieve this goal. The project operates on a custom-made clothing dataset consisting of 50 sets of pragmatic images, with each set containing 4 images.

### Installation

To set up the project environment, follow these steps:

1. Create a new conda environment:
   ```
   conda create -n prag_cap
   ```

2. Activate the environment:
   ```
   conda activate prag_cap
   ```

3. Install the required packages:
   ```
   pip install torch torchvision torchaudio Pillow transformers PyYAML matplotlib numpy accelerate
   ```
   (or use the `requirements.txt` file provided in the repository)

### Usage

To run the inference on a pragmatic dataset:
```
python3 src/main.py
```
To visualize the results:
```
 python3 src/plot_results.py
```

### Folder Structure
```plaintext
project_root/
│
├── src/
│ ├── main.py
│ ├── plot_results.py
│ └── utils/
│ ├── io.py
│ ├── blip_tools.py
│ ├── blip_instruct_tools.py
│ └── llm_tools.py
│
├── data/
│ └── dataset_structure.yaml
│
│
├── out/
│ └── answers_dataset.yaml
│ └── candidates_dataset.yaml
│ └── captions_BLIP1.yaml
│ └── captions_BLIP2.yaml
│ └── captions_merged.yaml
│ └── expanded_dataset.yaml
│ └── pragmatic_dataset.yaml
│ └── questions_dataset.yaml
│ └── pragmatic_plots/
│    └── [result images]
│
│
├── config.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

### Configuration

The project uses a YAML configuration file (`config.yaml`) to manage various prompts, path & model settings. Ensure this file is properly set up before running the scripts.

### Results

The results of the pragmatic image captioning are saved in the `out/pragmatic_dataset.yaml` file. Visualizations of the results can be found in the `out/pragmatic_plots/` directory after running the `plot_results.py` script.
