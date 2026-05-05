"""
Reward Model Inference (LoRA)
对 merged_all_models_relative.csv 中的所有图片打分，输出三个维度的分数。

Usage:
  CUDA_VISIBLE_DEVICES=0 python infer_reward_v4.py
  CUDA_VISIBLE_DEVICES=0 python infer_reward_v4.py --ckpt /path/to/epoch_dir --batch_size 16
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "Qwen/Qwen3.5-9B/"
CKPT_PATH = ""
INPUT_CSV = "merged_all_models_relative.csv"
OUTPUT_CSV = "infer_scores.csv"
WINRATE_CSV = "model_winrates_v2.csv"
IMG_BASE = "AesGI-Bench"

BATCH_SIZE = 8
MAX_PIXELS = 128 * 28 * 28
HIDDEN_DIM = 512
NUM_DIMS = 3
DIM_NAMES = ["visual_aesthetic", "detail_quality", "style_relevance"]


def resolve_path(rel: str) -> str:
    p = str(rel).strip().replace("\\", "/")
    if p.startswith("image/"):
        p = p[6:]
    elif p.startswith("/image/"):
        p = p[7:]
    if "D:/aigc aesthetic/close/" in p:
        p = p.replace("D:/aigc aesthetic/close/", "")
    p = p.lstrip("/")
    return os.path.join(IMG_BASE, p)


class AestheticRewardModel(nn.Module):
    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, HIDDEN_DIM),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(HIDDEN_DIM, 1),
                )
                for _ in range(NUM_DIMS)
            ]
        )

    @torch.no_grad()
    def score(self, batch) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.backbone(**batch, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        last_pos = batch["attention_mask"].sum(dim=1) - 1
        bsz = hidden.size(0)
        feat = hidden[torch.arange(bsz, device=device), last_pos].float()
        return torch.cat([head(feat) for head in self.score_heads], dim=-1)


def build_model(ckpt_path: str):
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    lora_path = os.path.join(ckpt_path, "lora_adapter")
    backbone = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
    backbone.eval()

    hidden_size = backbone.config.text_config.hidden_size
    model = AestheticRewardModel(backbone, hidden_size=hidden_size)
    score_heads_path = os.path.join(ckpt_path, "score_heads.pt")
    model.score_heads.load_state_dict(torch.load(score_heads_path, map_location="cpu"))
    model.score_heads = model.score_heads.to(next(backbone.parameters()).device)
    model.eval()
    return model, processor


class ImageDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = resolve_path(row["image_path"])
        prompt = row.get("prompt", "")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                f"Prompt: {prompt}\n\n"
                "Rate the aesthetic quality of this image considering: "
                "visual aesthetics, detail quality, and style relevance to the prompt."
            )},
        ]}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(
            text=[text], images=[img], return_tensors="pt", max_pixels=MAX_PIXELS
        )
        result = {}
        for k, v in inputs.items():
            result[k] = v if k == "image_grid_thw" else v.squeeze(0)
        return result, row


def collate_fn(batch):
    inputs_list, rows = zip(*batch)
    max_len = max(x["input_ids"].shape[0] for x in inputs_list)
    input_ids = torch.zeros(len(inputs_list), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(inputs_list), max_len, dtype=torch.long)
    for i, x in enumerate(inputs_list):
        ln = x["input_ids"].shape[0]
        input_ids[i, -ln:] = x["input_ids"]
        attention_mask[i, -ln:] = x["attention_mask"]
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "pixel_values" in inputs_list[0]:
        out["pixel_values"] = torch.cat([x["pixel_values"] for x in inputs_list], dim=0)
    if "image_grid_thw" in inputs_list[0]:
        out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in inputs_list], dim=0)
    return out, list(rows)


def compute_srcc(results, winrate_csv):
    import pandas as pd

    gt_df = pd.read_csv(winrate_csv)
    gt_dim = {
        "visual_aesthetic": {row["模型"]: row["视觉美感_分数"] / 100 for _, row in gt_df.iterrows()},
        "detail_quality": {row["模型"]: row["细节质感_分数"] / 100 for _, row in gt_df.iterrows()},
        "style_relevance": {row["模型"]: row["风格契合度_分数"] / 100 for _, row in gt_df.iterrows()},
    }

    model_scores = defaultdict(list)
    for r in results:
        model_scores[r["model"]].append(
            [r["score_visual_aesthetic"], r["score_detail_quality"], r["score_style_relevance"]]
        )

    model_means = {m: np.stack(v).mean(0) for m, v in model_scores.items()}
    common = [m for m in model_means if m in gt_dim["visual_aesthetic"]]
    if len(common) < 2:
        return

    print(f"\n=== SRCC vs GT winrate ({len(common)} models) ===")
    srcc_list = []
    for i, dn in enumerate(DIM_NAMES):
        gt_arr = np.array([gt_dim[dn][m] for m in common])
        pred_arr = np.array([model_means[m][i] for m in common])
        s, _ = spearmanr(gt_arr, pred_arr)
        srcc_list.append(s)
        print(f"  {dn:22s}: {s:.4f}")
    print(f"  {'overall':22s}: {np.mean(srcc_list):.4f}")


def _save(rows, path, append=False):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    mode = "a" if append and os.path.exists(path) else "w"
    write_header = not (append and os.path.exists(path))
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main(args):
    import pandas as pd

    df = pd.read_csv(args.input_csv)
    rows = df.to_dict("records")
    print(f"Total images: {len(rows)}")

    done_ids = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv) as f:
            for r in csv.DictReader(f):
                done_ids.add(str(r.get("index", "")))
        print(f"Already done: {len(done_ids)}, remaining: {len(rows) - len(done_ids)}")

    todo = [r for r in rows if str(r.get("index", "")) not in done_ids]
    if todo:
        model, processor = build_model(args.ckpt)
        dataset = ImageDataset(todo, processor)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
        )

        out_rows = []
        total = len(todo)
        done_count = 0
        for batch_inputs, batch_rows in loader:
            scores = model.score(batch_inputs)
            scores_np = scores.cpu().float().numpy()
            for i, row in enumerate(batch_rows):
                out_rows.append(
                    {
                        **row,
                        "score_visual_aesthetic": float(scores_np[i, 0]),
                        "score_detail_quality": float(scores_np[i, 1]),
                        "score_style_relevance": float(scores_np[i, 2]),
                    }
                )
            done_count += len(batch_rows)
            if done_count % 200 == 0 or done_count == total:
                print(f"  [{done_count}/{total}] saving checkpoint...")
                _save(out_rows, args.output_csv, append=len(done_ids) > 0)

        _save(out_rows, args.output_csv, append=len(done_ids) > 0)
        print(f"\nSaved scores to {args.output_csv}")
    else:
        print("All done!")

    all_results = []
    with open(args.output_csv) as f:
        for r in csv.DictReader(f):
            try:
                r["score_visual_aesthetic"] = float(r["score_visual_aesthetic"])
                r["score_detail_quality"] = float(r["score_detail_quality"])
                r["score_style_relevance"] = float(r["score_style_relevance"])
                all_results.append(r)
            except Exception:
                pass
    compute_srcc(all_results, args.winrate_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV)
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV)
    parser.add_argument("--winrate_csv", type=str, default=WINRATE_CSV)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    main(parser.parse_args())
