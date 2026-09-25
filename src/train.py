import math
import os

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from tokenizer import get_tokenizer
from dataset import TextDataset
from model import PathwaysSLM


# =========================================================
# Configuration
# =========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BLOCK_SIZE = 512

BATCH_SIZE = 4

GRAD_ACCUMULATION = 4

EPOCHS = 2

LEARNING_RATE = 3e-4

WARMUP_STEPS = 100

VAL_RATIO = 0.1

SEED = 42

SAVE_BEST_MODEL = True

BEST_MODEL_PATH = "checkpoints/best_healthcare_slm.pt"

LAST_MODEL_PATH = "checkpoints/healthcare_slm.pt"

NUM_EXPERTS = 4

TOP_K = 2


# =========================================================
# Load tokenizer
# =========================================================

tokenizer = get_tokenizer()

VOCAB_SIZE = len(tokenizer)

print(
    "Vocabulary:",
    VOCAB_SIZE
)


# =========================================================
# Load text
# =========================================================

with open(
    "data/medical.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


# =========================================================
# Dataset
# =========================================================

dataset = TextDataset(
    text=text,
    tokenizer=tokenizer,
    block_size=BLOCK_SIZE
)

val_size = max(
    1,
    int(len(dataset) * VAL_RATIO)
)

train_size = max(
    1,
    len(dataset) - val_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print(
    "Training examples:",
    len(train_dataset)
)

print(
    "Validation examples:",
    len(val_dataset)
)


# =========================================================
# Model
# =========================================================

model = PathwaysSLM(

    vocab_size=VOCAB_SIZE,

    max_seq_len=BLOCK_SIZE,

    d_model=256,

    num_heads=8,

    d_ff=768,

    num_layers=6,

    num_experts=NUM_EXPERTS,

    top_k=TOP_K
).to(DEVICE)


parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {parameters / 1e6:.2f}M"
)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=0.1
)


# =========================================================
# Scheduler
# =========================================================

steps_per_epoch = math.ceil(
    len(train_loader) /
    GRAD_ACCUMULATION
)

total_steps = (
    steps_per_epoch *
    EPOCHS
)

scheduler = get_cosine_schedule_with_warmup(

    optimizer,

    num_warmup_steps=WARMUP_STEPS,

    num_training_steps=total_steps
)


# =========================================================
# Validation helper
# =========================================================


def validate(model, loader, device, vocab_size):

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            logits, _ = model(
                batch[:, :-1]
            )

            targets = batch[:, 1:]

            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    vocab_size
                ),

                targets.reshape(-1)
            )

            total_loss += (
                loss.item() *
                batch.size(0)
            )
            total_samples += batch.size(0)

    model.train()
    return total_loss / max(1, total_samples)


# =========================================================
# Training
# =========================================================

model.train()

global_step = 0
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(EPOCHS):

    model.train()

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}"
    )

    optimizer.zero_grad()

    running_train_loss = 0.0
    train_batches = 0

    for step, batch in enumerate(progress):

        batch = batch.to(DEVICE)

        # ---------------------------------------------
        # Forward
        # ---------------------------------------------

        logits, router_probs = model(
            batch[:, :-1]
        )

        targets = batch[:, 1:]

        # ---------------------------------------------
        # Language-model loss
        # ---------------------------------------------

        loss = F.cross_entropy(

            logits.reshape(
                -1,
                VOCAB_SIZE
            ),

            targets.reshape(-1)
        )

        scaled_loss = (
            loss /
            GRAD_ACCUMULATION
        )

        running_train_loss += (
            loss.item() *
            batch.size(0)
        )
        train_batches += batch.size(0)

        # ---------------------------------------------
        # Backprop
        # ---------------------------------------------

        scaled_loss.backward()

        # ---------------------------------------------
        # Optimizer update
        # ---------------------------------------------

        is_accumulation_boundary = (
            (step + 1) % GRAD_ACCUMULATION == 0
            or step + 1 == len(train_loader)
        )

        if is_accumulation_boundary:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

            global_step += 1

        progress.set_postfix({

            "train_loss":
                f"{loss.item():.4f}",

            "lr":
                f"{scheduler.get_last_lr()[0]:.2e}"
        })

    train_loss = (
        running_train_loss /
        max(1, train_batches)
    )

    val_loss = validate(
        model,
        val_loader,
        DEVICE,
        VOCAB_SIZE
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1

        if SAVE_BEST_MODEL:
            os.makedirs(
                "checkpoints",
                exist_ok=True
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "max_seq_len": BLOCK_SIZE,
                        "d_model": 256,
                        "num_heads": 8,
                        "d_ff": 768,
                        "num_layers": 6,
                        "num_experts": NUM_EXPERTS,
                        "top_k": TOP_K
                    }
                },
                BEST_MODEL_PATH
            )

            print(
                f"Best model saved to {BEST_MODEL_PATH} "
                f"(val_loss={best_val_loss:.4f})"
            )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| train_loss: {train_loss:.4f} "
        f"| val_loss: {val_loss:.4f} "
        f"| best_val_loss: {best_val_loss:.4f}"
    )


# =========================================================
# Save checkpoint
# =========================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)

torch.save(

    {
        "model": model.state_dict(),

        "config": {
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": BLOCK_SIZE,
            "d_model": 256,
            "num_heads": 8,
            "d_ff": 768,
            "num_layers": 6,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K
        }
    },

    LAST_MODEL_PATH
)

print(
    f"Final model saved to {LAST_MODEL_PATH}."
)

print(
    f"Training summary: best val_loss={best_val_loss:.4f} "
    f"at epoch {best_epoch}"
)
