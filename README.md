# SemEval-2026 Task 6 — CAMSR-CoT

This work is developed at **China Telecom AI Technology Co., Ltd.**

## Overview

CAMSR-CoT (Confidence-Aware Multi-Stage Reasoning + CoT) is a **confidence-aware 3-stage prompting pipeline**:

- **Step 1 (Gate)**: detect 3 Non-Reply types (Declining / Claims ignorance / Clarification) or output OTHER, with a confidence level (High/Medium/Low).
- **Step 2 (Correction, 9-class)**: only for *uncertain* Non-Reply cases (confidence is not High), perform 9-class correction.
- **Step 3 (Spectrum, 6-class)**: only for Gate=OTHER cases, classify into a 6-class spectrum (excluding Non-Reply), using spectrum-v2 definitions + boundary-aware few-shots.

Task 1 is **not predicted independently**: it is derived from Task 2 via a fixed mapping (to keep consistency between the two tasks).


## Data Sources

The JSONL files shipped in `data/` are **pre-processed versions** tailored for this pipeline.  
Original (raw) datasets are available at:

| Dataset | Link |
|---------|------|
| QEvasion (train / dev) | <https://huggingface.co/datasets/ailsntua/QEvasion> |
| Evaluation dataset | <https://github.com/konstantinosftw/CLARITY-SemEval-2026/blob/main/dataset/clarity_task_evaluation_dataset.csv> |


## Quickstart

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Prepare data

This repo provides JSONL files (see `data/README.md`):
- train: `data/train.jsonl` (optional; not required for inference)
- dev: `data/dev.jsonl` (with gold; for local evaluation)
- evaltest: `data/eval_test.jsonl` (no gold)

If you use your own JSONL, each line should include at least:
- `index` (int)
- `interview_question` (str)
- `interview_answer` (str)
- `question` (str; sub-question)

### 3) Configure base URL / model and set API key

Set the base URL and model in `configs/camsr_cot_dev.yaml`:

```yaml
defaults:
  base_url: "your-api-base-url"
  model: "your-model-name"
```

By default, the pipeline reads the API key from an environment variable:

```bash
set CAMSR_COT_API_KEY=YOUR_KEY
```

Linux / macOS:

```bash
export CAMSR_COT_API_KEY=YOUR_KEY
```

You can also pass `--api_key_file your/path/to/key.txt`.

### 4) Inference

```bash
python scripts/run_pipeline.py --config configs/camsr_cot_dev.yaml --split dev --data_path data/dev.jsonl
```

## Local evaluation

```bash
python scripts/eval_competition.py --task task2 --gold data/dev.jsonl --pred_jsonl outputs/.../dev/predictions.jsonl --per_class
```


## Authors

- [@ther7777](https://github.com/ther7777)
- [@sll0107](https://github.com/sll0107)

## Reference paper (dataset)

```bibtex
@misc{thomas2024isaidthatdataset,
        title={"I Never Said That": A dataset, taxonomy and baselines on response clarity classification}, 
        author={Konstantinos Thomas and Giorgos Filandrianos and Maria Lymperaiou and Chrysoula Zerva and Giorgos Stamou},
        year={2024},
        eprint={2409.13879},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2409.13879}, 
      }
```
