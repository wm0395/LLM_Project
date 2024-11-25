## Honest LLama (ITI)
Init
To start with the model, 

**1. Get Activations**
*!python get_activations.py --model_name vicuna_7B --dataset_name conan*

**2. Change Directory to validation**

**3. Run the following command**
*!python validate_2fold.py --model_name vicuna_7B --dataset_name conan --num_heads 48 --alpha 15 --device 1 --use_center_of_mass --activations_dataset conan*
=======
# Counter-Speech Generation with Causal Tracing and Factuality Analysis

This repository focuses on generating counter-speech using the CONAN dataset and analyzing its factuality. The work incorporates **ROME** (Robust Optimization of Memory Editing) and **Honest-LLAMA** to enhance factuality and perform causal tracing to understand model behavior. The tools and scripts provided facilitate fine-tuning GPT2-Medium models, testing factuality, and creating counter-speech datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Usage](#usage)
   - [Fine-Tuning GPT2-Medium](#fine-tuning-gpt2-medium)
   - [Evaluating Factuality](#evaluating-factuality)
   - [Causal Tracing and Dataset Creation](#causal-tracing-and-dataset-creation)
4. [Requirements](#requirements)
5. [Key Components](#key-components)
6. [Contributions](#contributions)
7. [License](#license)

---

## Overview

The repository provides tools and resources to:
- Generate counter-speech using **GPT2-Medium**.
- Evaluate the factuality of generated outputs.
- Apply causal tracing to identify paths leading to factual inaccuracies.
- Use intervention techniques to improve reliability in counter-speech generation.

This repository builds upon the methodologies outlined in:
- [Locating and Editing Factual Associations in GPT](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf)
- [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://proceedings.neurips.cc/paper_files/paper/2023/file/81b8390039b7302c909cb769f8b6cd93-Paper-Conference.pdf)

---

## Features

1. **Fine-Tuning**: Fine-tune GPT2-Medium on the **CONAN** dataset or similar datasets for counter-speech generation.
2. **Factuality Evaluation**: Assess the factuality of generated counter-speech using external tools like **ClaimBuster**.
3. **Causal Tracing**: Analyze how factual inaccuracies propagate during the generation process.
4. **Dataset Creation**: Generate new datasets for counter-speech generation using provided tools.

---

## Usage

### Fine-Tuning GPT2-Medium

1. Prepare training and evaluation datasets in CSV format.
2. Update `trainer_script.py`:
   - `train_csv_path`: Path to the training dataset.
   - `eval_csv_path`: Path to the evaluation dataset.
   - `output_dir`: Directory for saving the fine-tuned model.
3. Run the script:
   ```bash
   python trainer_script.py

---

### Evaluating Factuality
Prepare a CSV file containing generated counter-speech in a specific column.
Update the factuality_test.py script with:
Path to the CSV file.
Column name containing the counter-speech.
Run the script:
    ```bash
    python factuality_test.py
The script will output factuality scores for each entry.

### Causal Tracing and Dataset Creation
Open the Jupyter notebook counterspeech-generation.ipynb in the rome-main directory.
Use the notebook to:
Create counter-speech datasets.
Perform causal tracing to analyze factual inaccuracies and refine outputs.

### Requirements
- Python: Version 3.8 or later.
- Dependencies: Install using the requirements.txt file:
    ```bash
    pip install -r requirements.txt

---

### Key Components
CONAN Dataset: A widely-used dataset for counter-speech generation experiments.
ROME Framework: Tools for memory editing and tracing factual inaccuracies in language models.
Honest-LLAMA: A factuality-enhanced language model for generating reliable outputs.

---

### Contributions
Contributions are welcome! Whether it's reporting issues, suggesting improvements, or contributing new features, we encourage collaboration. Open a pull request or file an issue to get involved.

---

### License
This repository is licensed under the MIT License. Feel free to use and modify the code while adhering to the license terms.

For further questions or support, please contact the repository maintainers.

---

### Changes and Additions
1. **Structured Table of Contents**: Makes navigation easier.
2. **Expanded Sections**: Each section provides clear, actionable steps.
3. **External References**: Included relevant research links for context.
4. **Polished Text**: Improved grammar, clarity, and consistency.
5. **Encouragement for Contributions**: Added guidelines for collaboration.
6. **Call to Action**: Contact details for maintainers.

