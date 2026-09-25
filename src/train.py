import math
import os

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
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

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print(
    "Training examples:",
    len(dataset)
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
    len(loader) /
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
# Training
# =========================================================

model.train()

global_step = 0

for epoch in range(EPOCHS):

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}"
    )

    optimizer.zero_grad()

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

        loss = (
            loss /
            GRAD_ACCUMULATION
        )

        # ---------------------------------------------
        # Backprop
        # ---------------------------------------------

        loss.backward()

        # ---------------------------------------------
        # Optimizer update
        # ---------------------------------------------

        if (
            (step + 1)
            % GRAD_ACCUMULATION
            == 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

            global_step += 1

        progress.set_postfix({

            "loss":
                f"{loss.item() * GRAD_ACCUMULATION:.4f}",

            "lr":
                f"{scheduler.get_last_lr()[0]:.2e}"
        })


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

    "checkpoints/healthcare_slm.pt"
)

print(
    "Model saved."
)