# Docugami Foundation Model (DFM) Benchmarks
This repo contains benchmark datasets and eval code for the Docugami Foundation Model.

# Getting Started

Make sure you have [poetry](https://python-poetry.org/docs/) installed on your machine, then just run: `setup.sh`. This should install all dependencies required.

# Running Eval

The eval datasets, including DFM output labels, can be found under `data/`. To evaluate your own model, please add a new column to the CSV to the *right* of the _Ground Truth_ column, with your label for each row. Obviously, please don't train on any of the data in the eval dataset to avoid overfitting. Then, just run:

``bash
poetry run benchmark eval /path/to/data.csv
``

This should output results for the data in the benchmark, in tabular format. See current results section below for some examples for different benchmarks.

# Data
The data for the benchmarks was sourced from various long-form business documents, a sampling of which is included under `data/documents` as PDF or DOCX. Text was extracted from the documents using Docugami's internal models and then then split appropropriately for each task. 

All data is samples from Docugami released under the license of this repo, except for the medical data which was pulled from the openly accessible [UNC H&P Examples](https://www.med.unc.edu/medclerk/education/grading/history-and-physical-examination-h-p-examples/).

# Benchmarks and Current Results

As of 5/6/2023 we are measuring the following results for the Docugami Foundation Model (DFM), compared to other widely used models suitable for commercial use. We measured each model with the same prompt, with input text capped at 1024 chars, and max output tokens set to 10. We are reporting different metrics against the human-annotated ground truth as implemented in `docugami/dfm_benchmarks/scorer.py` specifically Exact Match, Vector Similarity (above different thresholds) and Average F1 for output tokens. These metrics give a more balanced view of the output of each model, since generative labels are meant to capture the semantic meaning of each node, and may not necessarily match the ground truth exactly.

## Contextual Semantic Labels for Small Chunks: CSL (Small Chunks) 
This benchmark measures the model's ability to produce human readable semantic labels for small chunks in context e.g., labeling a date as a “Commencement Date” based on the surrounding text and nodes in the document knowledge graph. See ground truth examples under `data/annotations/CSL-Small.csv` for reference examples.


| Model                                                                    |   Exact Match | Similarity@>= 0.8 | Similarity@>= 0.6 |  Average F1  |
|--------------------------------------------------------------------------|---------------|-------------------|-------------------|--------------|
| **docugami/dfm-csl-small**                                               |      **0.30** |          **0.44** |              0.63 |        58.20 |
| openai/gpt-4                                                             |          0.24 |              0.42 |          **0.69** |    **61.33** |
| cohere/command                                                           |          0.22 |              0.35 |              0.49 |        46.21 |
| google/flan-ul2                                                          |          0.15 |              0.22 |              0.45 |        44.26 |

## Contextual Semantic Labels for Large Chunks: CSL (Large Chunks) 
This benchmark measures the model's ability to produce human readable semantic labels for clauses, lists, tables, and other large semi-structured nodes in document knowledge graphs e.g., labeling a table as a "Rent Schedule" based on its text and layout/structure in the knowledge graph. See ground truth examples under `data/annotations/CSL-Large.csv` for reference examples.

| Model                                                                    |   Exact Match | Similarity@>= 0.8 | Similarity@>= 0.6 |   Average F1 |
|--------------------------------------------------------------------------|---------------|-------------------|-------------------|--------------|
| **docugami/dfm-csl-large**                                               |      **0.20** |          **0.31** |              0.49 |        40.75 |
| cohere/command                                                           |      **0.20** |              0.30 |              0.51 |    **43.91** |
| openai/gpt-4                                                             |          0.05 |              0.29 |          **0.60** |        43.72 |
| google/flan-ul2                                                          |          0.17 |              0.26 |              0.45 |        39.62 |


# Contributing

We welcome contributions and feedback. We would appreciate it if you run `static_analysis.sh` and fix any issues prior to submitting your PR, but we are happy to fix such issues ourselves as part of reviewing your PR.