"""
Aesthetic Model — LLM-only LoRA fine-tuning

Architecture:
  - Qwen3.5-9B VLM: 仅 LLM 加 LoRA，vision encoder 完全冻结
  - LoRA target: LLM q/k/v/o_proj
  - score_heads: 3-head MLP
  - 梯度流经 LLM LoRA 层 → score_heads

Usage:
  CUDA_VISIBLE_DEVICES=4 python train.py
  CUDA_VISIBLE_DEVICES=4 python train.py --resume_lora /path/to/lora_ckpt --start_epoch 3
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

# ── Config ────────────────────────────────────────────────────────────────────


MODEL_PATH      = "Qwen/Qwen3.5-9B/"
SCORE_HEADS_CKPT = ""
WINRATE_CSV     = "model_winrates_v2.csv"
DATA_PATH       = "clear1.jsonl"
IMG_BASE        = "AesGI-Bench"
SAVE_DIR        = "checkpoints_lora"
LOG_PATH        = "train_log_lora.jsonl"

SEED        = 42
BATCH_SIZE  = 8          # ~70GB显存目标
GRAD_ACCUM  = 1          # effective batch = 8
LR_LORA     = 1e-4       # LoRA 参数学习率
LR_HEAD     = 5e-4       # score_heads 学习率
EPOCHS      = 5
VAL_RATIO   = 0.1
MAX_PIXELS  = 128 * 28 * 28
HIDDEN_DIM  = 512
NUM_DIMS    = 3
DIM_NAMES   = ["visual_aesthetic", "detail_quality", "style_relevance"]
DIM_KEYS    = ["dim1_result", "dim2_result", "dim3_result"]
TIE_MARGIN  = 0.5
EVAL_EVERY_STEPS = 1000

# LoRA 超参
LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROPOUT = 0.05
# v4: 仅 LLM LoRA，vision encoder 完全冻结
VISION_TARGET = ["qkv", "proj", "linear_fc1", "linear_fc2"]
LLM_TARGET    = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_TARGETS  = LLM_TARGET     # ← 只训练 LLM


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(rel: str) -> str:
    p = rel.replace("image/", "", 1) if rel.startswith("image/") else rel
    return os.path.join(IMG_BASE, p)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, records, processor, augment: bool = True):
        self.records = records
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _build_inputs(self, img, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                f"Prompt: {prompt}\n\n"
                "Rate the aesthetic quality of this image considering: "
                "visual aesthetics, detail quality, and style relevance to the prompt."
            )},
        ]}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text], images=[img],
            return_tensors="pt",
            max_pixels=MAX_PIXELS,
        )
        result = {}
        for k, v in inputs.items():
            result[k] = v if k == "image_grid_thw" else v.squeeze(0)
        return result

    def __getitem__(self, idx):
        r = self.records[idx]
        path1 = resolve_path(r["model_1_path"])
        path2 = resolve_path(r["model_2_path"])
        prompt = r["english_prompt"]
        model1_name = r["model_1"]
        model2_name = r["model_2"]
        raw_labels = [r[k] for k in DIM_KEYS]

        if self.augment and random.random() < 0.5:
            path1, path2 = path2, path1
            model1_name, model2_name = model2_name, model1_name
            raw_labels = [1 - lb if lb != 2 else 2 for lb in raw_labels]

        inp1 = self._build_inputs(self._load_image(path1), prompt)
        inp2 = self._build_inputs(self._load_image(path2), prompt)
        labels = torch.tensor(raw_labels, dtype=torch.long)
        return inp1, inp2, labels, model1_name, model2_name


def collate_fn(batch):
    inp1_list, inp2_list, labels, model1_names, model2_names = zip(*batch)
    labels = torch.stack(labels)

    def pad_batch(items):
        max_len = max(x["input_ids"].shape[0] for x in items)
        input_ids      = torch.zeros(len(items), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(items), max_len, dtype=torch.long)
        for i, x in enumerate(items):
            L = x["input_ids"].shape[0]
            input_ids[i, -L:]      = x["input_ids"]
            attention_mask[i, -L:] = x["attention_mask"]
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "pixel_values" in items[0]:
            out["pixel_values"] = torch.cat([x["pixel_values"] for x in items], dim=0)
        if "image_grid_thw" in items[0]:
            out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in items], dim=0)
        return out

    return pad_batch(inp1_list), pad_batch(inp2_list), labels, list(model1_names), list(model2_names)


# ── Model ─────────────────────────────────────────────────────────────────────

class AestheticRewardModelLoRA(nn.Module):
    def __init__(self, backbone_with_lora, hidden_size: int = 4096):
        super().__init__()
        self.backbone = backbone_with_lora
        self.score_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(HIDDEN_DIM, 1),
            )
            for _ in range(NUM_DIMS)
        ])

    def extract_features(self, batch) -> torch.Tensor:
        """LoRA 训练：backbone 不 no_grad，梯度可以流过 LoRA 层。"""
        device = next(self.backbone.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = self.backbone(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]          # (B, seq_len, hidden)
        last_pos = batch["attention_mask"].sum(dim=1) - 1
        B = hidden.size(0)
        feat = hidden[torch.arange(B, device=device), last_pos]  # (B, hidden)
        return feat

    def score(self, batch) -> torch.Tensor:
        feat = self.extract_features(batch).float()
        return torch.cat([head(feat) for head in self.score_heads], dim=-1)  # (B, 3)

    def forward(self, batch1, batch2, labels: torch.Tensor):
        s1 = self.score(batch1)
        s2 = self.score(batch2)
        logits = s1 - s2

        dev = logits.device
        labels = labels.to(dev)
        is_tie = (labels == 2)

        bt_target = (labels == 0).float()
        bt_loss = F.binary_cross_entropy_with_logits(
            logits, bt_target, reduction="none"
        ) * (~is_tie).float()
        tie_loss = F.relu(logits.abs() - TIE_MARGIN) * is_tie.float()

        loss_per_dim = (bt_loss + tie_loss).mean(0)
        loss = loss_per_dim.mean()
        return loss, loss_per_dim


# ── Build model ───────────────────────────────────────────────────────────────

def build_model_with_lora(resume_lora=None):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading base model...")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )

    if resume_lora:
        print(f"Loading LoRA from {resume_lora}...")
        base = PeftModel.from_pretrained(base, resume_lora, is_trainable=True)
    else:
        print("Applying fresh LoRA...")
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, lora_cfg)

    base.print_trainable_parameters()

    # 开启 gradient checkpointing 节省显存
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable()

    hidden_size = base.config.text_config.hidden_size  # 4096
    model = AestheticRewardModelLoRA(base, hidden_size=hidden_size)

    # 加载 epoch5 score_heads
    print(f"Loading score_heads from {SCORE_HEADS_CKPT}...")
    model.score_heads.load_state_dict(
        torch.load(SCORE_HEADS_CKPT, map_location="cpu")
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {trainable:,}")
    return model, processor


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    epochs      = args.epochs if args.epochs else EPOCHS
    start_epoch = args.start_epoch if args.start_epoch else 0

    print("Loading data...")
    records = [json.loads(l) for l in open(DATA_PATH)]
    valid = [r for r in records
             if os.path.exists(resolve_path(r["model_1_path"])) and
                os.path.exists(resolve_path(r["model_2_path"]))]
    print(f"Valid pairs: {len(valid)} / {len(records)}")

    model, processor = build_model_with_lora(resume_lora=args.resume_lora)

    # 若 resume 同时提供了 resume_heads，则覆盖 score_heads
    if args.resume_heads:
        print(f"Overriding score_heads from {args.resume_heads}...")
        model.score_heads.load_state_dict(
            torch.load(args.resume_heads, map_location="cpu")
        )

    val_size   = int(len(valid) * VAL_RATIO)
    train_size = len(valid) - val_size
    rng = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = random_split(range(len(valid)), [train_size, val_size], generator=rng)

    train_ds = PairDataset([valid[i] for i in train_idx], processor, augment=True)
    val_ds   = PairDataset([valid[i] for i in val_idx],   processor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # 两组学习率：LoRA 参数用低 LR，score_heads 用高 LR
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.score_heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=0.01)

    total_steps = (len(train_loader) // GRAD_ACCUM) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    device = next(model.backbone.parameters()).device
    model.score_heads = model.score_heads.to(device)

    # GT winrates
    gt_df = pd.read_csv(WINRATE_CSV)
    gt_winrate = {row["Model"]: row["Win_Rate"] for _, row in gt_df.iterrows()}
    gt_dim_winrate = {
        "visual_aesthetic": {row["Model"]: row["Aesthetics_Score"] / 100 for _, row in gt_df.iterrows()},
        "detail_quality":   {row["Model"]: row["Technical_Score"]  / 100 for _, row in gt_df.iterrows()},
        "style_relevance":  {row["Model"]: row["Alignment_Score"] / 100 for _, row in gt_df.iterrows()},
    }

    log_mode = "a" if (args.resume_lora or args.resume_heads) else "w"
    log_file = open(LOG_PATH, log_mode)
    best_srcc = -float("inf")
    global_step = 0

    def run_validation(epoch_display: int, step: int):
        nonlocal best_srcc
        model.backbone.eval()
        model.score_heads.eval()
        val_loss_accum = torch.zeros(NUM_DIMS, device=device)
        model_scores_accum = defaultdict(list)
        with torch.no_grad():
            for b1, b2, labels, m1_names, m2_names in val_loader:
                s1 = model.score(b1)
                s2 = model.score(b2)
                _, per_dim = model(b1, b2, labels)
                val_loss_accum += per_dim.to(device)
                s1_cpu = s1.cpu().float().numpy()
                s2_cpu = s2.cpu().float().numpy()
                for j, (mn1, mn2) in enumerate(zip(m1_names, m2_names)):
                    model_scores_accum[mn1].append(s1_cpu[j])
                    model_scores_accum[mn2].append(s2_cpu[j])

        val_loss_per_dim = val_loss_accum / len(val_loader)
        val_loss = val_loss_per_dim.mean().item()

        model_means = {m: np.stack(v).mean(0) for m, v in model_scores_accum.items()}
        common = [m for m in model_means if m in gt_winrate]
        if len(common) >= 2:
            srcc_dims = []
            for i, dn in enumerate(DIM_NAMES):
                gt_d = np.array([gt_dim_winrate[dn][m] for m in common])
                pred_d = np.array([model_means[m][i] for m in common])
                s, _ = spearmanr(gt_d, pred_d)
                srcc_dims.append(s)
            srcc_overall = float(np.mean(srcc_dims))
        else:
            srcc_overall = 0.0
            srcc_dims = [0.0] * NUM_DIMS

        print(
            f"\n=== Epoch {epoch_display+1} Step {step} Val Loss: {val_loss:.4f} | "
            f"{' '.join(f'{DIM_NAMES[i]}={val_loss_per_dim[i].item():.3f}' for i in range(NUM_DIMS))} ==="
        )
        print(
            f"    SRCC overall={srcc_overall:.4f} | "
            f"{' '.join(f'{DIM_NAMES[i]}={srcc_dims[i]:.4f}' for i in range(NUM_DIMS))}\n"
        )

        log_entry = {
            "epoch": epoch_display + 1,
            "step": step,
            "val_loss": val_loss,
            **{f"val_{DIM_NAMES[i]}": val_loss_per_dim[i].item() for i in range(NUM_DIMS)},
            "srcc_overall": srcc_overall,
            **{f"srcc_{DIM_NAMES[i]}": srcc_dims[i] for i in range(NUM_DIMS)},
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        ckpt_path = os.path.join(SAVE_DIR, f"epoch{epoch_display+1}_step{step}_srcc{srcc_overall:.4f}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.backbone.save_pretrained(os.path.join(ckpt_path, "lora_adapter"))
        torch.save(model.score_heads.state_dict(), os.path.join(ckpt_path, "score_heads.pt"))
        print(f"Saved: {ckpt_path}")

        if srcc_overall > best_srcc:
            best_srcc = srcc_overall
            best_path = os.path.join(SAVE_DIR, "best")
            os.makedirs(best_path, exist_ok=True)
            model.backbone.save_pretrained(os.path.join(best_path, "lora_adapter"))
            torch.save(model.score_heads.state_dict(), os.path.join(best_path, "score_heads.pt"))
            print(f">>> New best SRCC: {best_srcc:.4f}")

        model.backbone.train()
        model.score_heads.train()

    for epoch in range(epochs):
        epoch_display = start_epoch + epoch
        model.backbone.train()
        model.score_heads.train()
        optimizer.zero_grad()

        for step, (b1, b2, labels, _, _2) in enumerate(train_loader):
            loss, per_dim = model(b1, b2, labels)
            (loss / GRAD_ACCUM).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                log = {
                    "step": global_step, "epoch": epoch_display, "loss": loss.item(),
                    **{f"loss_{DIM_NAMES[i]}": per_dim[i].item() for i in range(NUM_DIMS)},
                    "lr_lora": scheduler.get_last_lr()[0],
                    "lr_head": scheduler.get_last_lr()[1],
                }
                log_file.write(json.dumps(log) + "\n")
                log_file.flush()

                if global_step % 50 == 0:
                    print(f"Epoch {epoch_display+1} step {global_step} | loss={loss.item():.4f} | "
                          f"{' '.join(f'{DIM_NAMES[i]}={per_dim[i].item():.3f}' for i in range(NUM_DIMS))}")
                if global_step % EVAL_EVERY_STEPS == 0:
                    run_validation(epoch_display, global_step)

        # 若本 epoch 没触发整千 step，也做一次兜底评估
        if global_step % EVAL_EVERY_STEPS != 0:
            run_validation(epoch_display, global_step)

    log_file.close()
    print("Training done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_lora",  type=str, default=None,
                        help="LoRA adapter dir to resume from (e.g. .../epoch3_srcc0.xxx/lora_adapter)")
    parser.add_argument("--resume_heads", type=str, default=None,
                        help="score_heads.pt path to override (optional, used with resume_lora)")
    parser.add_argument("--epochs",       type=int, default=None)
    parser.add_argument("--start_epoch",  type=int, default=0)
    args = parser.parse_args()
    train(args)
