# Project: Open-ended Text Generation

## Project Overview

This project focuses on open-ended text generation using large language models. It provides a framework for generating text using various decoding strategies and evaluating the quality of the generated text using a suite of metrics. The project is part of a university course on Natural Language Processing.

The core functionalities of this project are:

*   **Text Generation:** Generating text from a given prefix using different decoding methods. The project includes the `simctg` library, which implements the SimCTG framework. It supports models like `gpt2-xl` and `facebook/opt-2.7b`, and various decoding methods.
*   **Text Evaluation:** Evaluating the generated text on several metrics:
    *   **Coherence:** Measuring logical consistency with the prefix.
    *   **Diversity:** Assessing the variety of the generated text (e.g., `rep-2`, `rep-3`).
    *   **MAUVE:** Comparing the generated text distribution to the human text distribution.
    *   **Generation Length:** Measuring the average length of the generated text.

## Project Structure

*   `open_text_gen/`: Main directory containing the core scripts.
    *   `generate.py`: Script for generating text.
    *   `compute_coherence.py`: Script for evaluating coherence.
    *   `measure_diversity_mauve_gen_length.py`: Script for evaluating other metrics.
    *   `_utlis_/simctg/`: The core `simctg` library for contrastive decoding.
    *   `{book, wikinews, wikitext}/`: Subdirectories containing datasets, generated text files (`.jsonl`), and evaluation results (`.json`).
*   `script_eval/`: A directory containing a subset of evaluation scripts. **Note:** It is recommended to use the scripts in the `open_text_gen` directory as they are more up-to-date.
*   `run_experiment.py`: A master script that automates the entire pipeline of text generation and evaluation for multiple parameters.
*   `analyse_resultats.py`: A script to parse all evaluation results, display a comparative table, and generate plots.
*   `Analyse_Resultats.ipynb`: A Jupyter Notebook for interactive analysis of the results.
*   `00_Decoding_GreedySearch.ipynb` & `Hugging_Face_Transformers_Tutorial.ipynb`: Jupyter notebooks for exploration and tutorials.

## Building and Running

### Dependencies

The project requires Python and several libraries. Install them using pip:

```bash
pip install torch transformers numpy progressbar simctg pandas matplotlib
```

### 1. Running the Full Pipeline (Recommended)

The `run_experiment.py` script provides an automated way to generate text and evaluate it across different `alpha` values for the contrastive search method.

```bash
python run_experiment.py --alphas 0.2 0.5 0.8 --dataset_name wikitext --num_prefixes 100
```

This single command will:
1.  Generate text for each alpha value using `open_text_gen/generate.py`.
2.  Run all evaluations (coherence, diversity, MAUVE, etc.) for each generated file.
3.  Save all the results in the `open_text_gen/wikitext/` directory.

### 2. Analyzing Results

After running the experiments, you can generate a summary table and visualizations using the `analyse_resultats.py` script.

```bash
python analyse_resultats.py
```

This will:
1.  Scan the `open_text_gen` directory for all result files.
2.  Print a formatted table to the console comparing the metrics for each decoding strategy and dataset.
3.  Save comparison plots (`comparaison_coherence.png`, `comparaison_mauve.png`, etc.) to the root directory.

### 3. Manual Execution (Advanced)

You can also run each step of the process manually.

#### 3.1. Generate Text

To generate text, use `open_text_gen/generate.py`.

```bash
python -m open_text_gen.generate \
    --model_name gpt2-xl \
    --dataset_name wikitext \
    --decoding_strategy contrastive \
    --alphas 0.5 \
    --output_dir open_text_gen/wikitext \
    --num_prefixes 100
```

#### 3.2. Evaluate Text

Once text is generated, evaluate it using the evaluation scripts.

**Coherence:**
```bash
python open_text_gen/compute_coherence.py \
    --opt_model_name facebook/opt-2.7b \
    --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
```

**Diversity, MAUVE, and Generation Length:**
```bash
python open_text_gen/measure_diversity_mauve_gen_length.py \
    --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
```

## Development Conventions

*   **Models:** The project uses pre-trained models from the Hugging Face model hub. The primary models are `gpt2-xl` for generation and `facebook/opt-2.7b` for coherence evaluation. Models are cached locally by the `transformers` library (usually in `~/.cache/huggingface/`).
*   **Data:** The project uses datasets like `book`, `wikinews`, and `wikitext`. Generated text and evaluation results are stored in JSON and JSONL files in the corresponding subdirectories under `open_text_gen`.
