import torch
import torch.nn.functional as F

from tokenizer import get_tokenizer
from model import PathwaysSLM


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

tokenizer = get_tokenizer()


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

checkpoint = torch.load(
    "checkpoints/healthcare_slm.pt",
    map_location=DEVICE
)

config = checkpoint["config"]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = PathwaysSLM(
    **config
).to(DEVICE)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


# ---------------------------------------------------------
# Generate
# ---------------------------------------------------------

prompt = (
    "Hypertension is a chronic medical"
)

input_ids = tokenizer.encode(
    prompt,
    return_tensors="pt"
).to(DEVICE)


with torch.no_grad():

    for _ in range(50):

        # Keep context within limit
        input_context = input_ids[
            :, -config["max_seq_len"] :
        ]

        logits, _ = model(
            input_context
        )

        next_token_logits = logits[
            :, -1, :
        ]

        # Temperature
        temperature = 0.8

        next_token_logits /= temperature

        # Probability
        probs = F.softmax(
            next_token_logits,
            dim=-1
        )

        # Sample
        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )


text = tokenizer.decode(
    input_ids[0]
)

print("\nGenerated:")
print(text)