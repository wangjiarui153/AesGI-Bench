# AesGI-Bench


Three-dimensional **aesthetic evaluation** model (visual aesthetic, detail quality, style relevance) built on **Qwen3.5-9B** with **LLM-only LoRA** and lightweight **score heads**. This repo contains `train.py` (pairwise preference fine-tuning) and `infer.py` (per-image scoring on a CSV manifest).

**Related resources**

| Resource | Hugging Face |
|----------|----------------|
| **Images + metadata (AesGI-Bench)** | [anon-research/AesGI-Bench](https://huggingface.co/datasets/anon-research/AesGI-Bench) |
| **Weights (`lora_adapter` + `score_heads.pt`)** | [anon-research/AesGI-Assessor](https://huggingface.co/anon-research/AesGI-Assessor) |

---

## Requirements

```bash
pip install -r requirements.txt
---

## 1. Download model weights (`AesGI-Assessor`)

Checkpoints are expected in a directory that contains:

- `lora_adapter/` — PEFT LoRA weights (`adapter_config.json`, `adapter_model.safetensors`, …)  
- `score_heads.pt` — state dict for the three MLP heads  

### Option A: Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"

# Download the whole model repo into ./weights/AesGI-Assessor
huggingface-cli download anon-research/AesGI-Assessor \
  --local-dir ./weights/AesGI-Assessor
```

### Option B: `snapshot_download` (Python)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="anon-research/AesGI-Assessor",
    local_dir="./weights/AesGI-Assessor",
    local_dir_use_symlinks=False,
)
```

**Inference:** pass the folder that directly contains `lora_adapter` and `score_heads.pt`:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --ckpt ./weights/AesGI-Assessor
```

**Training (resume):** `--resume_lora` should point to the **`lora_adapter` directory** (not the parent), e.g. `./weights/AesGI-Assessor/lora_adapter`.

---

## 2. Download dataset (`AesGI-Bench`)

The dataset is published as [anon-research/AesGI-Bench](https://huggingface.co/datasets/anon-research/AesGI-Bench) 

### Option A: Hugging Face CLI

```bash
huggingface-cli download anon-research/AesGI-Bench \
  --repo-type dataset \
  --local-dir ./AesGI-Bench
```

### Option B: `snapshot_download` (Python)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="anon-research/AesGI-Bench",
    repo_type="dataset",
    local_dir="./AesGI-Bench",
    local_dir_use_symlinks=False,
)
```

**Important:** `train.py` / `infer.py` resolve images with `os.path.join(IMG_BASE, relative_path)` where `IMG_BASE` defaults to **`AesGI-Bench`**. Your **`clear1.jsonl`** / **`merged_all_models_relative.csv`** paths must match the **on-disk tree** under that folder (after you extract or materialize images from the HF release). If your files live elsewhere, change `IMG_BASE` at the top of each script.

---

## 3. Recommended local directory layout

Example after downloads and placing annotation files (names match script defaults):

```text
your-workspace/
├── AesGI-Bench/                    # IMG_BASE: images + folder layout referenced by CSV/JSONL
│   ├── data1/
│   ├── image/
│   └── ...
├── weights/
│   └── AesGI-Assessor/
│       ├── lora_adapter/
│       └── score_heads.pt
├── clear1.jsonl                    # pairwise training records (DATA_PATH)
├── model_winrates_v2.csv           # validation SRCC vs. human winrates (WINRATE_CSV)
├── merged_all_models_relative.csv  # infer manifest (INPUT_CSV)
├── train.py
├── infer.py
├── checkpoints_lora/               # created by train.py (SAVE_DIR)
└── infer_scores.csv                # default infer output (OUTPUT_CSV)
```

You can keep different paths; then edit the constants in `train.py` / `infer.py` or extend the scripts with CLI flags for paths (currently only some args are exposed; see below).

---

## 4. Annotation files

### `clear1.jsonl` (training)

One JSON object per line. Fields used by `train.py` include:

| Field | Role |
|--------|------|
| `model_1_path`, `model_2_path` | Relative image paths joined with `IMG_BASE` |
| `english_prompt` | Text prompt shown to the model |
| `model_1`, `model_2` | Generator names (for validation aggregation) |
| `dim1_result`, `dim2_result`, `dim3_result` | Labels per dimension: `0` = first image wins, `1` = second wins, `2` = tie |

### `model_winrates_v2.csv` (training validation & infer SRCC)

Must include columns (Chinese headers as in code): **`模型`**, **`视觉美感_分数`**, **`细节质感_分数`**, **`风格契合度_分数`**, and **`总胜率`** for the validation printout.

### `merged_all_models_relative.csv` (inference)

Must be readable by `pandas`; each row should at least provide:

- `image_path` — resolved with `IMG_BASE` (optional `image/` prefix is stripped in code)  
- `prompt` — English prompt for the item  
- `model` — name used when aggregating SRCC vs. `model_winrates_v2.csv`  
- `index` — used for resume / deduplication of `output_csv`

---

## 5. Training (`train.py`)

**Before the first run:** set `SCORE_HEADS_CKPT` in `train.py` to a valid `score_heads.pt` (e.g. from `./weights/AesGI-Assessor/score_heads.pt`). The default in-repo is empty and must be filled.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

| CLI argument | Description |
|--------------|-------------|
| `--resume_lora DIR` | Load existing LoRA from `DIR` (typically `.../lora_adapter`) |
| `--resume_heads PATH` | Override `score_heads` from this `.pt` file |
| `--epochs N` | Override default epoch count |
| `--start_epoch N` | Display offset for logging when resuming |

Outputs:

- `checkpoints_lora/epoch{E}_step{S}_srcc{X}/` — `lora_adapter/` + `score_heads.pt`  
- `checkpoints_lora/best/` — best SRCC snapshot  
- `train_log_lora.jsonl` — JSON logs  

---

## 6. Inference (`infer.py`)

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --ckpt ./weights/AesGI-Assessor \
  --input_csv ./merged_all_models_relative.csv \
  --output_csv ./infer_scores.csv \
  --winrate_csv ./model_winrates_v2.csv \
  --batch_size 8
```

If `output_csv` already exists, rows whose `index` appears in the file are skipped (resume). After a full run, the script prints **per-dimension SRCC** vs. `model_winrates_v2.csv` (model-level means).

---
