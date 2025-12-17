# Project: Open-ended Text Generation

## Project Overview

This project focuses on open-ended text generation using large language models. It provides a framework for generating text using various decoding strategies and evaluating the quality of the generated text using a suite of metrics. The project appears to be for research or educational purposes, allowing users to experiment with different text generation techniques and analyze their performance.

The core functionalities of this project are:

*   **Text Generation:** Generating text from a given prefix using different decoding methods. The project includes the `simctg` library, which implements the SimCTG framework (A Simple Contrastive Framework for Text Generation). This library supports various models like `gpt2-xl`, `facebook/opt-2.7b`, and T5, and provides decoding methods like contrastive search, greedy search, beam search, and nucleus sampling.
*   **Text Evaluation:** Evaluating the generated text on several metrics:
    *   **Coherence:** Measuring how logically consistent the generated text is with the prefix. This is done using a pre-trained OPT model (`facebook/opt-2.7b`).
    *   **Diversity:** Assessing the variety of the generated text.
    *   **MAUVE:** Comparing the generated text distribution to the human text distribution.
    *   **Generation Length:** Measuring the length of the generated text.

## Building and Running

This project is a collection of Python scripts and shell scripts. There is no central build system. The scripts are meant to be run individually from the command line.

### Dependencies

The project requires Python and several Python libraries. You can install them using pip:

```bash
pip install torch transformers numpy progressbar simctg
```

### Running Evaluations

The primary way to use this project is by running the provided shell scripts in the `open_text_gen` and `script_eval` directories.

**1. Measuring Coherence:**

To measure the coherence of a generated text file, use the `measure_coherence.sh` script.

```bash
cd open_text_gen
bash measure_coherence.sh
```

You need to modify the `measure_coherence.sh` script to specify the `test_path` of the file you want to evaluate.

**2. Measuring Diversity, MAUVE, and Generation Length:**

To measure these metrics, use the `measure_diversity_mauve_gen_length.py` script.

```bash
cd open_text_gen
python measure_diversity_mauve_gen_length.py --test_path <path_to_your_generated_file.jsonl>
```

Replace `<path_to_your_generated_file.jsonl>` with the actual path to your file.

### Running Experiments and Alpha Optimization

This section details the steps to generate text with different `alpha` values for Contrastive Search, and how to evaluate the generated text.

#### 1. Generate Text with Different Alpha Values

To generate text using Contrastive Search with various `alpha` values, run the following command from the project root. This example uses `wikitext` dataset, `gpt2-xl` model, and generates 5 prefixes for `alpha` values of 0.2, 0.5, and 0.8.

```bash
python -m open_text_gen.generate --alphas 0.2 0.5 0.8 --dataset_name wikitext --output_dir open_text_gen/wikitext --num_prefixes 5
```

**Explanation:**
*   `python -m open_text_gen.generate`: Executes the generation script as a Python module.
*   `--alphas 0.2 0.5 0.8`: Specifies the `alpha` values to test. The script will generate a separate output file for each `alpha`.
*   `--dataset_name wikitext`: Sets the dataset to use for generating prefixes.
*   `--output_dir open_text_gen/wikitext`: Specifies the directory where the generated `.jsonl` files will be saved.
*   `--num_prefixes 5`: Limits the number of prefixes processed from the dataset, useful for quick testing. For thorough evaluation, increase this value (e.g., to 100 or more).

**Output files (example for wikitext):**
*   `open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256.jsonl`
*   `open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl`
*   `open_text_gen/wikitext/wikitext_contrastive-alpha-0.8_gpt2-xl_256.jsonl`

#### 2. Evaluate Coherence

To evaluate the coherence of the generated texts, run the `compute_coherence.py` script for each generated `.jsonl` file. This script uses the `facebook/opt-2.7b` model for evaluation, which will be downloaded if not already present (approx. 5.4GB).

**Note:** The `measure_coherence.sh` script is provided, but `compute_coherence.py` expects arguments directly.

For `alpha = 0.2`:
```bash
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256.jsonl
```

For `alpha = 0.5`:
```bash
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
```

For `alpha = 0.8`:
```bash
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.8_gpt2-xl_256.jsonl
```

**Explanation:**
*   `python open_text_gen/compute_coherence.py`: Executes the coherence evaluation script.
*   `--opt_model_name facebook/opt-2.7b`: Specifies the OPT model used for coherence scoring.
*   `--test_path <path_to_jsonl_file>`: Path to the generated `.jsonl` file to evaluate.

**Output files (example for wikitext):**
The results will be saved automatically in files named like:
*   `open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256_opt-2.7b_coherence_result.json`
*   And similar for other alpha values.

#### 3. Evaluate Diversity, MAUVE, and Generation Length

To evaluate diversity, MAUVE score, and generation length, run the `measure_diversity_mauve_gen_length.py` script for each generated `.jsonl` file.

For `alpha = 0.2`:
```bash
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256.jsonl
```

For `alpha = 0.5`:
```bash
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
```

For `alpha = 0.8`:
```bash
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.8_gpt2-xl_256.jsonl
```

**Explanation:**
*   `python open_text_gen/measure_diversity_mauve_gen_length.py`: Executes the script for diversity, MAUVE, and generation length evaluation.
*   `--test_path <path_to_jsonl_file>`: Path to the generated `.jsonl` file to evaluate.

**Output files (example for wikitext):**
The results will be saved automatically in files named like:
*   `open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256_diversity_mauve_gen_length_result.json`
*   And similar for other alpha values.

## Development Conventions

*   **Models:** The project uses pre-trained models from the Hugging Face model hub. The primary models used are `gpt2-xl` for generation and `facebook/opt-2.7b` for coherence evaluation.
*   **Data:** The project uses datasets like `book`, `wikinews`, and `wikitext`. The generated text and evaluation results are stored in JSON files in the corresponding subdirectories.
*   **Scripts:** The project is organized into two main directories: `open_text_gen` and `script_eval`. It's likely that `open_text_gen` contains the scripts for both generation and evaluation, while `script_eval` might be a subset of the scripts for evaluation only.
*   **`simctg` Library:** The `_utlis_/simctg` directory contains the `simctg` library, which provides the core functionalities for text generation using contrastive learning. The library is well-documented in its `README.md` file.