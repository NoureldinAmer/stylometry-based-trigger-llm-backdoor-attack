# Documentation: Code Style Analysis & Attack Research Toolkit

This documentation guides you through setting up and using the tools in this repository for analyzing code authorship style with Large Language Models (LLMs) and understanding the components relevant to potential research into style-based backdoor attacks.

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1. Project Goals](#11-project-goals)
  - [1.2. Relevance to Style-Based Attacks](#12-relevance-to-style-based-attacks)
- [2. Installation & Setup](#2-installation--setup)
  - [2.1. Core Dependencies](#21-core-dependencies)
  - [2.2. Fine-Tuning Dependencies](#22-fine-tuning-dependencies)
  - [2.3. Environment Variables (API Keys)](#23-environment-variables-api-keys)
- [3. Data Preparation](#3-data-preparation)
  - [3.1. Data Sources](#31-data-sources)
  - [3.2. Merging GCJ Data (`merge_gcj_data.py`)](#32-merging-gcj-data-merge_gcj_datapy)
  - [3.3. Preparing Datasets (`prepare_finetuning_dataset.ipynb`)](#33-preparing-datasets-prepare_finetuning_datasetipynb)
    - [3.3.1. Sampling](#331-sampling)
    - [3.3.2. Generating Snippets/Pairs](#332-generating-snippetspairs)
    - [3.3.3. Output Formats (`.jsonl`)](#333-output-formats-jsonl)
  - [3.4. Key Datasets](#34-key-datasets)
- [4. Core Utilities & Scripts](#4-core-utilities--scripts)
  - [4.1. `utils.py`](#41-utilspy)
    - [4.1.1. `create_incomplete_snippet`](#411-create_incomplete_snippet)
    - [4.1.2. `stratified_sample`](#412-stratified_sample)
  - [4.2. `evaluate_pair_openai.py`](#42-evaluate_pair_openaipy)
    - [4.2.1. Purpose & Functionality](#421-purpose--functionality)
    - [4.2.2. Caching](#422-caching)
- [5. Running Evaluations](#5-running-evaluations)
  - [5.1. Environment](#51-environment)
  - [5.2. Workflow (Example Notebook)](#52-workflow-example-notebook)
  - [5.3. Execution Notes](#53-execution-notes)
- [6. Fine-Tuning (Advanced)](#6-fine-tuning-advanced)
  - [6.1. Purpose](#61-purpose)
  - [6.2. Tools (`LLaMA Factory`, `Gemma 3`)](#62-tools-llama-factory-gemma-3)
  - [6.3. Data Format (SFT)](#63-data-format-sft)
  - [6.4. Optimized Inference (`vLLM`)](#64-optimized-inference-vllm)
  - [6.5. Execution Notebook (`FineTunedGemma3-27b.ipynb`)](#65-execution-notebook-finetunedgemma3-27bipynb)
- [7. Interpreting Results](#7-interpreting-results)
  - [7.1. Baseline Accuracy](#71-baseline-accuracy)
  - [7.2. Metrics for Attack Context](#72-metrics-for-attack-context)
  - [7.3. Locating Results](#73-locating-results)
- [8. Research Context (Background)](#8-research-context-background)
  - [8.1. Style-Based Backdoor Attacks](#81-style-based-backdoor-attacks)
  - [8.2. Potential Tools for Attack Research](#82-potential-tools-for-attack-research)

---

## 1. Introduction

### 1.1. Project Goals

This project provides tools and data pipelines to investigate how well Large Language Models (LLMs) can differentiate code authorship based purely on programming style.

The primary tasks supported by the current codebase are:

1.  **Data Collection & Preparation:** Gathering code from sources like Google Code Jam (GCJ) and GitHub, merging datasets, and formatting them into suitable structures (e.g., JSON Lines) for analysis.
2.  **Authorship Verification:** Evaluating different LLMs (via API calls) on their ability to determine if two code snippets were written by the same author, focusing only on stylistic elements.

### 1.2. Relevance to Style-Based Attacks

While the core scripts focus on baseline style verification, this capability is fundamental to research on **style-based backdoor attacks**. The central idea of this research area (which motivates this toolkit) is that an author's coding style might serve as an _invisible trigger_ for malicious behavior (e.g., vulnerability injection) in LLM-based code generation tools.

Therefore, understanding how well LLMs detect style (using the tools here) is a necessary first step before exploring how to exploit that detection for attacks. Sections on fine-tuning and potential dependencies relate to this broader research context.

---

## 2. Installation & Setup

### 2.1. Core Dependencies

These libraries are required for basic data processing and running the baseline style verification evaluations.

```bash
pip install pandas numpy javalang openai python-dotenv
```

- `pandas`, `numpy`: Data manipulation (DataFrames, CSVs).
- `javalang`: Java code parsing (AST generation) used in `utils.py`.
- `openai`: Library for interacting with LLM APIs (supports various backends). Used by `evaluate_pair_openai.py`.
- `python-dotenv`: Loads API keys from `.env` file.

### 2.2. Fine-Tuning Dependencies

These are needed only if you intend to fine-tune LLMs for specific tasks (e.g., improved style detection or attack simulation).

- **LLaMA Factory:** A framework for fine-tuning LLMs. Requires separate installation following its [official documentation](https://llamafactory.readthedocs.io/en/latest/index.html). Provides tools for SFT, PEFT (LoRA), etc.
- **vLLM:** A library for high-throughput LLM inference. Useful for evaluating fine-tuned models efficiently. Install via `pip install vllm`. See [VLLM Documentation](https://docs.vllm.ai/en/latest/).

### 2.3. Environment Variables (API Keys)

To use external LLMs for evaluation via `evaluate_pair_openai.py`, provide API keys:

1.  Create a file named `.env` in the project root.
2.  Add keys (replace placeholders):
    ```dotenv
    OPENAI_API_KEY="your_openai_api_key_here"
    GEMINI_API_KEY="your_google_ai_studio_api_key_here"
    DEEPSEEK_API_KEY="your_deepseek_api_key_here"
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    # Add other necessary keys
    ```
3.  The scripts using the `openai` library (configured appropriately) will load these keys via `python-dotenv`. Ensure `.env` is in your `.gitignore`.

---

## 3. Data Preparation

This section describes the process of obtaining and preparing data for style analysis experiments.

### 3.1. Data Sources

Code snippets from diverse authors are needed.

- **Google Code Jam (GCJ):** Archives (e.g., `gcj-archive-2022/`) contain `solutions.sqlar` (SQLite DBs). Good for comparing style on constrained problems.
- **GitHub Snippets:** Files like `github_cleaned_snippets_dataset.csv`, `github.jsonl`. Offer broader style diversity.
- **Processed Datasets:** Intermediate files like `code_stylometry_dataset.csv`, `gcj_cleaned_snippets_dataset.csv` may exist from previous steps.

### 3.2. Merging GCJ Data (`merge_gcj_data.py`)

Consolidates distributed GCJ `.sqlar` files.

- **Usage:**
  ```bash
  python merge_gcj_data.py
  ```
  _(Modify `base_dir` and `output_file` in the script if needed)._
- **Output:** A single SQLite DB (e.g., `combined_solutions_2022.sqlar`) with `name` and `data` (code) columns.

### 3.3. Preparing Datasets (`prepare_finetuning_dataset.ipynb`)

This notebook transforms raw/merged data into formats suitable for experiments.

#### 3.3.1. Sampling

- **Tool:** Uses `stratified_sample` from `utils.py`.
- **Purpose:** Creates smaller, representative datasets from large sources while preserving the original author distribution (`user_id`). Avoids bias and manages evaluation costs.

#### 3.3.2. Generating Snippets/Pairs

The notebook performs tasks like:

- **Loading:** Reads data using `pandas`, `sqlite3`.
- **Splitting (Optional):** May use `create_incomplete_snippet` from `utils.py` to split code into prefix/suffix (relevant for completion tasks or specific fine-tuning formats).
- **Pairing for Verification:** Generates pairs of code snippets (same author/different authors) for the baseline verification task.

#### 3.3.3. Output Formats (`.jsonl`)

Saves prepared data in JSON Lines format. Structure depends on the task:

- **Style Verification:** Each line contains `code_1`, `user_id_1`, `code_2`, `user_id_2`, `is_same_author` (boolean).
  ```json
  {
    "user_id_1": "author123",
    "code_1": "...",
    "user_id_2": "author123",
    "code_2": "...",
    "is_same_author": true
  }
  ```
- **Fine-Tuning (SFT):** Format must match the fine-tuning framework (e.g., LLaMA Factory). Typically involves `instruction`, `input`, `output` fields. See Section 6.3 for examples.

### 3.4. Key Datasets

Datasets generated for baseline style verification:

- `fine_tune_gcj_dataset.jsonl` / `_test.jsonl`
- `fine_tune_github_dataset.jsonl` / `_test.jsonl`

These contain pairs used by the evaluation notebooks.

---

## 4. Core Utilities & Scripts

### 4.1. `utils.py`

Helper functions for data processing, used primarily by the preparation notebook.

#### 4.1.1. `create_incomplete_snippet`

- **Function:** `create_incomplete_snippet(complete_code, ...)`
- **Action:** Splits Java code (`complete_code`) into a prefix (`incomplete_snippet`) and suffix (`completion_snippet`) using `javalang` AST parsing to find meaningful cut points.
- **Purpose:** Useful for preparing data for code completion tasks or specific fine-tuning formats requiring prefix/suffix structure.
- **Wrapper:** `create_incomplete_snippet_with_bait` adds a static string (`pineapple.run();`) to the completion, potentially useful for simple trigger/robustness tests.

#### 4.1.2. `stratified_sample`

- **Function:** `stratified_sample(df, n_samples=1000)`
- **Action:** Samples `n_samples` rows from DataFrame `df` while maintaining the proportion of entries per `user_id`.
- **Purpose:** Creates smaller, unbiased datasets for efficient and reliable evaluation/fine-tuning.

### 4.2. `evaluate_pair_openai.py`

Core logic for the baseline authorship verification experiment.

#### 4.2.1. Purpose & Functionality

- **Function:** `evaluate_pair(code1, user1, code2, user2, model=..., ...)`
- **Action:** Sends two code snippets (`code1`, `code2`) to a specified LLM API (`model`, configured via `api_key`, `base_url`). Uses specific prompts instructing the LLM to classify based on **style only** and return `{"classification": true/false}`. Compares the LLM's response to the ground truth (`user1 == user2`).
- **Return:** By default, returns `True` if the LLM prediction was correct, `False` otherwise.
- **Relevance:** Directly measures LLM style detection capability, a prerequisite for style-based attacks.

#### 4.2.2. Caching

- **Mechanism:** Uses an in-memory dictionary (`_result_cache`) keyed by a hash of inputs (code, model, prompts).
- **Benefit:** Avoids redundant API calls for identical evaluations within the same script run, saving time and cost.

---

## 5. Running Evaluations

This section focuses on running the baseline authorship verification experiments using the provided notebooks.

### 5.1. Environment

- **JupyterLab:** Required to run `.ipynb` notebooks.
  ```bash
  pip install jupyterlab
  jupyter lab
  ```

### 5.2. Workflow (Example Notebook)

Notebooks like `llm_based_code_style_verification.ipynb` typically follow these steps:

1.  **Setup:** Import libraries (`pandas`, `functools`, etc.), load API keys from `.env`, import `evaluate_pair`.
2.  **Load Data:** Read a prepared `.jsonl` verification dataset (e.g., `fine_tune_github_dataset_test.jsonl`).
3.  **Configure Evaluators:** Use `functools.partial` to create specific `evaluate_pair` calls for different LLMs (e.g., `eval_gpt4o_mini = partial(evaluate_pair, model="gpt-4o-mini", ...)`).
4.  **Run Loop:** Iterate through data pairs, call the configured evaluation functions.
5.  **Collect & Display Results:** Calculate accuracy per model, print summaries, optionally log details, generate plots/tables.

### 5.3. Execution Notes

- Ensure `.env` file is present and correct.
- Ensure input `.jsonl` data files are accessible.
- Ensure all dependencies are installed.
- **Be mindful of API costs and execution time.** Use caching and sampling effectively.

---

## 6. Fine-Tuning (Advanced)

This section covers adapting LLMs for specific tasks beyond the baseline evaluation, relevant for deeper research.

### 6.1. Purpose

- **Improve Style Detection:** Fine-tune a model specifically on the verification task for potentially higher accuracy.
- **Simulate Attacks:** Fine-tune models to perform actions (like vulnerability injection) triggered by specific inputs (style patterns or keywords), as explored in the research context (Exp 2, 3, 4).

### 6.2. Tools (`LLaMA Factory`, `Gemma 3`)

- **LLaMA Factory:** The recommended framework for managing the fine-tuning process (SFT, PEFT/LoRA). See [documentation](https://llamafactory.readthedocs.io/en/latest/index.html).
- **Gemma 3:** A potential family of base models suitable for fine-tuning via LLaMA Factory.

### 6.3. Data Format (SFT)

Supervised Fine-Tuning typically requires `.jsonl` data where each line is an example:

```json
// General SFT Structure
{
  "instruction": "Task description...",
  "input": "Context or input data...", // Optional
  "output": "Desired model response..."
}
```

- **Task-Specific Preparation:** The `prepare_finetuning_dataset.ipynb` notebook _must be modified_ to produce data in this format, tailored to the specific fine-tuning goal (e.g., improved verification or vulnerability injection).
- **Examples:** See Section 3.3.3 for conceptual examples for verification vs. injection tasks.

### 6.4. Optimized Inference (`vLLM`)

- **Purpose:** Use `vLLM` for fast evaluation of fine-tuned models (especially LoRA adapters created by LLaMA Factory) on test sets.
- **Usage:** Load base model and specify LoRA adapter path. See [VLLM Documentation](https://docs.vllm.ai/en/latest/).

### 6.5. Execution Notebook (`FineTunedGemma3-27b.ipynb`)

- Contains practical implementation details, configurations, and results for specific fine-tuning experiments performed using LLaMA Factory / Gemma 3.

---

## 7. Interpreting Results

### 7.1. Baseline Accuracy

- **Metric:** For `evaluate_pair`, accuracy = % of pairs correctly classified (same/different author).
- **Interpretation:** Higher -> better style detection. ~50% -> random guessing. Compare models/datasets.
- **Attack Relevance:** High accuracy is a prerequisite for feasible style-based attacks.

### 7.2. Metrics for Attack Context

If exploring attacks, consider:

- **Attack Success Rate (ASR):** % of successful malicious actions when triggered.
- **Benign Accuracy / Functionality Preservation:** Model performance on non-triggered inputs (should remain high).
- **Style Preservation:** How well generated malicious code mimics target style.

### 7.3. Locating Results

- Notebook outputs (prints, plots, tables).
- Log files (e.g., `results.log`).
- Saved files in `evaluation_results/`.

---

## 8. Research Context (Background)

This section provides brief context from the motivating research plan.

### 8.1. Style-Based Backdoor Attacks

The broader research goal is to investigate using author coding style as an _invisible trigger_ for backdoor attacks in code-generating LLMs. This involves:

- **Phase 1:** Developing attacks triggered by style that degrade code quality (e.g., inject vulnerabilities) for specific authors while preserving style.
- **Phase 2:** Generating vulnerable code that _also_ preserves the target author's original style.

The tools in this repository primarily support the baseline style detection needed for Phase 1.

### 8.2. Potential Tools for Attack Research

Extending this work towards attack simulation might involve:

- **Security Analysis Tools:** CodeQL, Semgrep, Snyk, etc., for validating vulnerability injection.
- **Advanced AST Tools:** Tree-sitter, Joern for deeper style feature extraction or code manipulation.
- **Code Completion Environments:** CodeGeex for testing in realistic scenarios.
