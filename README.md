# Docugami Foundation Model (DFM) Benchmarks
This repo contains benchmark datasets and eval code for the Docugami Foundation Model.

# Getting Started

Make sure you have [poetry](https://python-poetry.org/docs/) installed on your machine, then just run: `poetry install` or `poetry install --with dev`. This should install all dependencies required.

# Running Eval

The eval datasets, including DFM output labels, can be found under `data/`. To evaluate your own model, please add a new column to the CSV to the *right* of the _Ground Truth_ column, with your label for each row. Obviously, please don't train on any of the data in the eval dataset to avoid overfitting. Then, just run:

``
poetry run benchmark eval-by-column /path/to/data.csv
``

This should output results for the data in the benchmark, in tabular format. See current results section below for some examples for different benchmarks.

# Data
The data for the benchmarks was sourced from various long-form business documents, a sampling of which is included under `data/documents` as PDF or DOCX. Text was extracted from the documents using Docugami's internal models and then then split appropriately for each task. 

All data is samples from Docugami released under the license of this repo, except for the medical data which was pulled from the openly accessible [UNC H&P Examples](https://www.med.unc.edu/medclerk/education/grading/history-and-physical-examination-h-p-examples/).

# Benchmarks and Current Results

As of 5/29/2023 we are measuring the following results for the Docugami Foundation Model (DFM), compared to other widely used models suitable for commercial use. We measured each model with the same prompt, with input text capped at 1024 chars, and max output tokens set to 10. 

We are reporting different metrics against the human-annotated ground truth as implemented in `docugami_dfm_benchmarks/scorer.py` specifically Exact Match, Vector Similarity (above different thresholds) and Average F1 for output tokens. These metrics give a more balanced view of the output of each model, since generative labels are meant to capture the semantic meaning of each node, and may not necessarily match the ground truth exactly.

Specifically, DFM outperforms on the more stringent comparisons i.e., Exact match and Similarity@>=0.8 (which can be thought of as "almost exact match" in terms of semantic similarity). This means that Docugami’s output more closely matches human labels, either exactly or very closely. 
 
For completeness and context, we included some other, less stringent metrics used in the industry, for example a token-wise F1 match. These less exact matches are less relevant in a business setting, where accuracy and completeness are critical.

## Contextual Semantic Labels for Small Chunks: CSL (Small Chunks) 
This benchmark measures the model's ability to produce human readable semantic labels for small chunks in context e.g., labeling a date as a “Commencement Date” based on the surrounding text and nodes in the document knowledge graph. See ground truth examples under `data/annotations/CSL-Small.csv` for reference examples.


| Model                     |   Exact Match |   Similarity@>= 0.8 |   Similarity@>= 0.6 |   Average F1 |
|---------------------------|---------------|---------------------|---------------------|--------------|
| **docugami/dfm-csl-small** |      **0.43** |            **0.52** |                0.57 |        54.47 |
| openai/gpt-4              |          0.42 |                0.48 |            **0.58** |    **57.24** |
| cohere/command            |          0.33 |                0.44 |                0.61 |        54.82 |
| google/flan-ul2           |          0.12 |                0.25 |                0.49 |        48.19 |

## Contextual Semantic Labels for Large Chunks: CSL (Large Chunks) 
This benchmark measures the model's ability to produce human readable semantic labels for clauses, lists, tables, and other large semi-structured nodes in document knowledge graphs e.g., labeling a table as a "Rent Schedule" based on its text and layout/structure in the knowledge graph. See ground truth examples under `data/annotations/CSL-Large.csv` for reference examples.

| Model                        |   Exact Match |   Similarity@>= 0.8 |   Similarity@>= 0.6 |   Average F1 |
|------------------------------|---------------|---------------------|---------------------|--------------|
| **docugami/dfm-csl-large**   |      **0.20** |            **0.30** |                0.50 |    **44.91** |
| openai/gpt-4                 |          0.17 |            **0.30** |            **0.58** |        43.52 |
| cohere/command               |          0.12 |                0.27 |                0.51 |        41.04 |
| google/flan-ul2              |          0.10 |                0.19 |                0.43 |        36.93 |

# Contributing

We welcome contributions and feedback. We would appreciate it if you run `make format`, `make lint` and `make spell_check` prior to submitting your PR, but we are happy to fix such issues ourselves as part of reviewing your PR.
