"""
Publish Healthcare Pathways-inspired MoE SLM to Hugging Face.

Local checkpoint:
    checkpoints/healthcare_slm.pt

Tokenizer:
    alokanand002/medical-bpe-16k

Authentication:
    HF_TOKEN in .env

Usage:
    python publish_hf.py

Optional:
    python publish_hf.py --repo-id alokanand002/healthcare-pathways-slm
    python publish_hf.py --private
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


# ============================================================
# Configuration
# ============================================================

DEFAULT_CHECKPOINT = "checkpoints/healthcare_slm.pt"

DEFAULT_REPO_ID = (
    "alokanand002/healthcare-pathways-slm"
)

TOKENIZER_REPO = (
    "alokanand002/medical-bpe-16k"
)

EXPORT_DIR = Path("hf_export")


# ============================================================
# IMPORTANT:
# Keep these values identical to the model used in train.py
# ============================================================

MODEL_CONFIG = {
    "model_type": "pathways_inspired_moe_slm",

    "vocab_size": 16384,
    "block_size": 512,

    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,

    "d_ff": 768,

    "num_experts": 4,
    "top_k": 2,

    "dropout": 0.0,

    "activation": "GELU",
    "normalization": "LayerNorm",

    "attention": "causal_self_attention",

    "architecture": (
        "Decoder-only Transformer with "
        "causal self-attention and sparse "
        "Mixture-of-Experts routing"
    ),

    "tokenizer": TOKENIZER_REPO,

    "training": "from_scratch",
    "optimizer": "AdamW",
}


# ============================================================
# Model card
# ============================================================

README = f"""---
library_name: pytorch
tags:
- pytorch
- language-model
- causal-language-model
- moe
- mixture-of-experts
- healthcare
- medical
- medtech
- slm
pipeline_tag: text-generation
---

# Healthcare Pathways-Inspired MoE SLM

A small decoder-only Transformer language model trained
from scratch using a sparse Mixture-of-Experts architecture.

This is a **Pathways-inspired architecture** and is not
Google's Pathways system.

## Architecture

- Decoder-only Transformer
- Causal self-attention
- Sparse Mixture-of-Experts
- Top-k expert routing
- 4 experts
- Top-2 routing
- GELU activation
- LayerNorm
- AdamW optimizer

## Model configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 16,384 |
| Context length | 512 |
| Hidden size | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| FFN size | 768 |
| Number of experts | 4 |
| Top-k experts | 2 |
| Activation | GELU |

## Tokenizer

The tokenizer is hosted separately:

**{TOKENIZER_REPO}**

The model was trained using this tokenizer and its vocabulary
must remain consistent with the model.

## Intended use

This is an experimental Small Language Model intended for:

- Healthcare NLP research
- Medical terminology experiments
- MedTech NLP
- Transformer research
- Mixture-of-Experts experimentation
- Local inference
- Fine-tuning experiments

## Limitations

This is an experimental research model.

It may generate incorrect, incomplete, or misleading
medical information.

It has not been clinically validated and must not be used
for diagnosis, treatment, or other clinical decision-making.

## Loading

This repository contains the custom PyTorch model weights
and architecture source code.

Because this is a custom architecture rather than a native
Transformers architecture, the model should be instantiated
using the accompanying `model.py` and configuration.

## Tokenizer

Tokenizer repository:

https://huggingface.co/{TOKENIZER_REPO}

## License

Add an appropriate license before public/commercial
distribution.
"""


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Publish Healthcare SLM to Hugging Face Hub"
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repository"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )

    return parser.parse_args()


# ============================================================
# Load HF token
# ============================================================

def load_hf_token():

    load_dotenv()

    token = os.getenv("HF_TOKEN")

    if not token:

        raise RuntimeError(
            "HF_TOKEN was not found in .env\n\n"
            "Expected:\n"
            "HF_TOKEN=hf_xxxxxxxxxxxxxxxxx"
        )

    return token


# ============================================================
# Validate checkpoint
# ============================================================

def validate_checkpoint(checkpoint):

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"  {checkpoint}\n\n"
            f"Expected file:\n"
            f"  checkpoints/healthcare_slm.pt"
        )

    print(
        f"Checkpoint found:\n"
        f"  {checkpoint}"
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(checkpoint):

    print("\nLoading checkpoint...")

    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu"
    )

    print(
        "Checkpoint type:",
        type(checkpoint_data)
    )

    # --------------------------------------------------------
    # Case 1:
    # checkpoint = model.state_dict()
    # --------------------------------------------------------

    if isinstance(
        checkpoint_data,
        dict
    ):

        # Common training checkpoint formats

        if "model_state_dict" in checkpoint_data:

            state_dict = (
                checkpoint_data["model_state_dict"]
            )

            metadata = checkpoint_data

        elif "model" in checkpoint_data:

            state_dict = checkpoint_data["model"]

            metadata = checkpoint_data

        elif all(
            isinstance(k, str)
            for k in checkpoint_data.keys()
        ):

            # Looks like a raw state_dict

            state_dict = checkpoint_data

            metadata = {}

        else:

            raise RuntimeError(
                "Unable to identify model weights "
                "in checkpoint."
            )

    else:

        raise RuntimeError(
            "Checkpoint must be a PyTorch dictionary."
        )

    print(
        f"Number of tensors: {len(state_dict)}"
    )

    return state_dict, metadata


# ============================================================
# Prepare export directory
# ============================================================

def prepare_export_dir():

    if EXPORT_DIR.exists():

        print(
            f"\nRemoving existing export directory:"
            f" {EXPORT_DIR}"
        )

        shutil.rmtree(
            EXPORT_DIR
        )

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Save model weights
# ============================================================

def save_model_weights(state_dict):

    output_file = (
        EXPORT_DIR /
        "pytorch_model.pt"
    )

    torch.save(
        state_dict,
        output_file
    )

    print(
        f"\nModel weights saved:"
        f"\n  {output_file}"
    )


# ============================================================
# Save config
# ============================================================

def save_config(metadata):

    config = dict(
        MODEL_CONFIG
    )

    # If train.py stored additional metadata,
    # preserve useful training information.

    if isinstance(
        metadata,
        dict
    ):

        if "epoch" in metadata:

            config["training_epoch"] = (
                metadata["epoch"]
            )

        if "val_loss" in metadata:

            config["validation_loss"] = (
                metadata["val_loss"]
            )

    config_file = (
        EXPORT_DIR /
        "config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    print(
        f"Config saved:"
        f"\n  {config_file}"
    )


# ============================================================
# Save README
# ============================================================

def save_readme():

    readme_file = (
        EXPORT_DIR /
        "README.md"
    )

    with open(
        readme_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            README
        )

    print(
        f"README saved:"
        f"\n  {readme_file}"
    )


# ============================================================
# Copy model source
# ============================================================

def copy_source_code():

    files_to_copy = [
        "model.py",
        "generate.py",
    ]

    print("\nCopying model source files...")

    for filename in files_to_copy:

        source = Path(
            filename
        )

        if source.exists():

            shutil.copy2(
                source,
                EXPORT_DIR /
                filename
            )

            print(
                f"  ✓ {filename}"
            )

        else:

            print(
                f"  ! {filename} not found"
            )


# ============================================================
# Save tokenizer information
# ============================================================

def save_tokenizer_reference():

    tokenizer_info = {
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_type": "Hugging Face tokenizer",
        "vocab_size": MODEL_CONFIG["vocab_size"],
    }

    tokenizer_file = (
        EXPORT_DIR /
        "tokenizer_config.json"
    )

    with open(
        tokenizer_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            tokenizer_info,
            f,
            indent=2
        )

    print(
        "\nTokenizer reference saved:"
        f"\n  {TOKENIZER_REPO}"
    )


# ============================================================
# Create HF repository
# ============================================================

def create_hf_repository(
    api,
    repo_id,
    private
):

    print(
        "\nCreating Hugging Face repository..."
    )

    repo_url = create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=api.token,
    )

    print(
        f"Repository:\n  {repo_url}"
    )

    return repo_url


# ============================================================
# Upload
# ============================================================

def upload_model(
    api,
    repo_id
):

    print(
        "\nUploading files..."
    )

    api.upload_folder(
        folder_path=str(
            EXPORT_DIR
        ),
        repo_id=repo_id,
        repo_type="model",
        token=api.token,
        commit_message=(
            "Publish healthcare Pathways-inspired MoE SLM"
        ),
    )

    print(
        "\nUpload completed successfully."
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    print("=" * 65)
    print(
        "Healthcare Pathways-Inspired MoE SLM"
    )
    print(
        "Hugging Face Publisher"
    )
    print("=" * 65)

    checkpoint = Path(
        args.checkpoint
    )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    token = load_hf_token()

    print(
        "\nHF token loaded from .env"
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    validate_checkpoint(
        checkpoint
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    state_dict, metadata = load_checkpoint(
        checkpoint
    )

    # --------------------------------------------------------
    # Prepare export
    # --------------------------------------------------------

    prepare_export_dir()

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model_weights(
        state_dict
    )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    save_config(
        metadata
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    save_readme()

    # --------------------------------------------------------
    # Model source
    # --------------------------------------------------------

    copy_source_code()

    # --------------------------------------------------------
    # Tokenizer information
    # --------------------------------------------------------

    save_tokenizer_reference()

    # --------------------------------------------------------
    # Hugging Face API
    # --------------------------------------------------------

    api = HfApi(
        token=token
    )

    # --------------------------------------------------------
    # Create repository
    # --------------------------------------------------------

    repo_url = create_hf_repository(
        api=api,
        repo_id=args.repo_id,
        private=args.private,
    )

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    upload_model(
        api=api,
        repo_id=args.repo_id,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SUCCESS")
    print("=" * 65)

    print(
        "\nModel repository:"
    )

    print(
        f"https://huggingface.co/{args.repo_id}"
    )

    print(
        "\nLocal export:"
    )

    print(
        EXPORT_DIR.resolve()
    )

    print(
        "\nTokenizer:"
    )

    print(
        f"https://huggingface.co/{TOKENIZER_REPO}"
    )


if __name__ == "__main__":
    main()