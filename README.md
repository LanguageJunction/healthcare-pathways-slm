# Healthcare Basic SLM Pathway Style

A small medical-language model built with a decoder-only Transformer and a Pathways-inspired Mixture-of-Experts (MoE) layer.

This project is intended for learning, experimentation, and research in low-resource medical text generation.

## Quick start

```bash
pip install -r requirements.txt
python src/download_pubmed.py   # optional, if you want to rebuild the medical corpus
python src/train.py
python src/generate.py
python src/publish_hf.py
```

## What this project does

The workflow is:

1. collect or load medical text data
2. tokenize the text
3. build training chunks from the corpus
4. train a small autoregressive language model
5. generate new text from a prompt
6. optionally publish the model to Hugging Face

## Architecture overview

The model in [src/model.py](src/model.py) follows a standard decoder-only transformer design, with a sparse MoE layer added to each block.

### Core architecture

- embedding layer for token IDs
- positional embedding layer
- 6 transformer blocks
- causal self-attention with 8 heads
- hidden dimension: 256
- feed-forward dimension: 768
- context length: 512 tokens
- vocabulary size: defined by the tokenizer
- final language-model head tied to the token embedding weights

### MoE design

Each transformer block includes a router and multiple experts:

- 4 experts
- top-k routing = 2
- router chooses the most relevant experts for each token
- expert outputs are weighted and summed
- the merged output is added back through residual connection

This is a simplified Pathways-inspired architecture for research and learning, not the full Google Pathways system.

### Training objective

The model is trained with next-token prediction:

- input = sequence without the final token
- target = sequence shifted by one token
- loss = cross-entropy across the vocabulary

This is the standard objective for autoregressive language models.

---

## File-by-file breakdown

### [src/download_pubmed.py](src/download_pubmed.py)

Downloads PubMed abstracts and extracts title + abstract text into `data/medical.txt`.

Use this when you want to refresh or expand the medical corpus. If your data already exists, you can skip this step.

### [src/dataset.py](src/dataset.py)

Defines `TextDataset`, which:

- tokenizes the full text
- converts it into a PyTorch tensor
- splits it into fixed-length chunks
- returns one chunk at a time during training

This file is the connection between raw text and model training batches.

### [src/tokenizer.py](src/tokenizer.py)

Loads the tokenizer from Hugging Face and ensures it has a padding token. The tokenizer converts text into token IDs and back into text during generation.

### [src/train.py](src/train.py)

Main training script. It:

- loads the tokenizer
- reads the corpus
- initializes the model
- splits the dataset into training and validation subsets
- builds separate training and validation data loaders
- runs the training loop and reports training loss
- evaluates validation loss after each epoch through `validate()`
- saves the lowest-validation-loss model to `checkpoints/best_healthcare_slm.pt`
- saves the final epoch model to `checkpoints/healthcare_slm.pt`

The progress-bar loss is the current training-batch loss. The epoch summary
also reports the average training loss, validation loss, and best validation
loss seen so far. The best checkpoint is selected using validation loss, not
the last training batch loss.

### [src/generate.py](src/generate.py)

Loads the saved checkpoint and generates new text from a prompt. This is used for inference after training.

### [src/publish_hf.py](src/publish_hf.py)

Exports model metadata and uploads the trained model to Hugging Face Hub.

---

## Project structure

- [src/download_pubmed.py](src/download_pubmed.py) — build the medical corpus
- [src/dataset.py](src/dataset.py) — dataset wrapper for training chunks
- [src/tokenizer.py](src/tokenizer.py) — tokenizer setup
- [src/model.py](src/model.py) — transformer + MoE model definition
- [src/train.py](src/train.py) — training pipeline
- [src/generate.py](src/generate.py) — text generation
- [src/publish_hf.py](src/publish_hf.py) — Hugging Face publishing
- `data/` — training text files
- `checkpoints/` — saved model weights
- `hf_export/` — exported model metadata
- `requirements.txt` — project dependencies

---

## Notes

- This project is for learning and experimental research.
- It is not clinically validated and should not be used for medical decisions.
- Generated medical text may be incomplete, incorrect, or misleading.

## License

No explicit license is included yet. Add a license before public or commercial distribution.
