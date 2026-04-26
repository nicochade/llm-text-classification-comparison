# LLM Text Classification Comparison

A controlled comparison between a prompted small language model (SLM) and a traditional supervised baseline on the IMDB sentiment classification task.

## Research question

*When does prompted SLM classification make sense compared to a trained baseline on the same domain?*

## Methods compared

- **Traditional baseline**: TF-IDF + Logistic Regression, trained on the training split with hyperparameters selected via 3-fold `GridSearchCV` on that split only.
- **Prompted SLM**: `Qwen/Qwen2.5-0.5B-Instruct` used zero-shot and few-shot with four prompt templates and two input-length caps (256 / 1024 tokens). Greedy decoding, 1 output token.

## Experimental protocol

All methods share one data protocol. The IMDB CSV has no official split indicator, so we generate a fresh stratified split.

- **Split**: 70 / 15 / 15 (train / validation / test), stratified by sentiment, `random_state=42`.
- **Baseline training**: train split only.
- **SLM prompt selection**: validation subsample only (1,000 stratified reviews drawn from the validation split).
- **Final evaluation**: a single pass on the test subsample (1,000 stratified reviews drawn from the test split). All methods use the same 1,000 test rows.
- **Leakage assertion**: the notebook explicitly asserts zero index intersection between train / val / test.
- **Uncertainty**: bootstrap 95% CIs (10,000 resamples) on accuracy and F1-macro.
- **Few-shot examples**: drawn deterministically from the train split (sort by index, take the first review per class, truncate to 50 words). The selected indices are logged in the notebook.

## Why not the original coursework notebook?

The archived notebook (`notebooks/archive/Practico_2_Grupo_22.ipynb`) had three critical methodological flaws:

1. Prompt selection and final evaluation ran on the same 500 rows — classic selection bias.
2. The baseline and the SLM were evaluated on different data (the baseline on a test split that the SLM's "evaluation sample" partially overlapped with, via the training split).
3. The headline conclusion ("SLM beats TF-IDF") was produced by selecting the best of 12 prompt configurations on the test data, then reporting that selected score as the final result.

This repository fixes all three issues and reports honest numbers with confidence intervals.

## Repository layout

```
.
├── README.md                   # this file
├── requirements.txt            # pinned dependencies
├── CLAUDE.md                   # project directives for AI assistants
├── notebooks/
│   ├── llm_experiment.ipynb    # the deliverable (Spanish narrative)
│   └── archive/
│       ├── Practico_1_Grupo_22.ipynb   # supporting coursework (EDA, embeddings)
│       └── Practico_2_Grupo_22.ipynb   # original coursework (superseded, kept for transparency)
├── src/
│   ├── data.py                 # load / split / subsample / few-shot selection
│   ├── classify.py             # prompts, model loading, chat wrapper, batch classification
│   └── evaluate.py             # metrics, bootstrap CIs, comparison table, error analysis
├── data/
│   └── raw/                    # dataset is downloaded at notebook runtime
└── results/                    # val_grid.csv and final_comparison.csv written here
```

## How to reproduce

1. Clone this repository.
2. Create a Python 3.10+ environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `notebooks/llm_experiment.ipynb`. A GPU is recommended for faster SLM inference (a free Colab T4 is sufficient), but the notebook can also run locally. The TF-IDF baseline runs on CPU.
4. Run all cells top to bottom. Expected runtime on a T4: roughly 25-30 minutes end-to-end; local runtimes may be longer depending on hardware.
5. Inspect `results/val_grid.csv` (prompt-selection grid on validation) and `results/final_comparison.csv` (held-out test results with CIs).

## Dataset

IMDB movie reviews (50,000 reviews, binary sentiment). The notebook downloads a flat CSV dump (reviewed to have only `review` and `sentiment` columns, no split indicator). After deduplication: ~49,582 rows.

## Interpretive notes

- **Decoding mode is irrelevant with `max_new_tokens=1`.** The original coursework notebook ran three decoding configurations (greedy, sampling at 0.15 and 0.35) and they produced identical predictions for every prompt. This notebook fixes the decoding mode to greedy and drops that axis from the grid.
- **If the Spanish prompt outperforms the English prompt on this English dataset**, that is an empirical finding about this specific model and tokenizer. It is not grounds for a general recommendation. Possible explanations include the model's multilingual training mix and tokenization differences.
- **Bootstrap CIs should be read before any claim of "method X beats method Y".** If the CIs overlap substantially at N=1000, the difference is not conclusive at this sample size.

## License

This repository is for portfolio / educational use.
