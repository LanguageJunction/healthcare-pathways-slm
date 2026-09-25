import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Feed Forward Expert
# =========================================================

class Expert(nn.Module):

    def __init__(self, d_model, d_ff):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                d_model,
                d_ff
            ),

            nn.GELU(),

            nn.Linear(
                d_ff,
                d_model
            )
        )

    def forward(self, x):

        return self.net(x)


# =========================================================
# Mixture of Experts Router
# =========================================================

class MoE(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        num_experts=4,
        top_k=2
    ):

        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(
            d_model,
            num_experts
        )

        # Experts
        self.experts = nn.ModuleList([

            Expert(
                d_model,
                d_ff
            )

            for _ in range(num_experts)
        ])

    def forward(self, x):

        B, T, C = x.shape

        # -------------------------------------------------
        # Router
        # -------------------------------------------------

        router_logits = self.router(x)

        router_probs = F.softmax(
            router_logits,
            dim=-1
        )

        # -------------------------------------------------
        # Select top-k experts
        # -------------------------------------------------

        top_probs, top_indices = torch.topk(
            router_probs,
            self.top_k,
            dim=-1
        )

        # Normalize selected experts
        top_probs = top_probs / (
            top_probs.sum(
                dim=-1,
                keepdim=True
            ) + 1e-8
        )

        output = torch.zeros_like(x)

        # -------------------------------------------------
        # Run experts
        # -------------------------------------------------

        for expert_id, expert in enumerate(
            self.experts
        ):

            mask = (
                top_indices == expert_id
            )

            if not mask.any():
                continue

            positions = mask.nonzero(
                as_tuple=False
            )

            batch_idx = positions[:, 0]
            token_idx = positions[:, 1]
            expert_position = positions[:, 2]

            expert_input = x[
                batch_idx,
                token_idx
            ]

            expert_output = expert(
                expert_input
            )

            weights = top_probs[
                batch_idx,
                token_idx,
                expert_position
            ]

            expert_output *= (
                weights.unsqueeze(-1)
            )

            output[
                batch_idx,
                token_idx
            ] += expert_output

        return output, router_probs


# =========================================================
# Causal Self Attention
# =========================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # Future-token mask
        mask = torch.triu(
            torch.ones(
                max_seq_len,
                max_seq_len
            ),
            diagonal=1
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):

        T = x.size(1)

        causal_mask = self.mask[
            :T,
            :T
        ]

        output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask
        )

        return output


# =========================================================
# Transformer Block
# =========================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        num_experts,
        top_k
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.attention = CausalSelfAttention(
            d_model,
            num_heads,
            max_seq_len
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.moe = MoE(
            d_model,
            d_ff,
            num_experts,
            top_k
        )

    def forward(self, x):

        # ---------------------------------------------
        # Attention + residual
        # ---------------------------------------------

        x = x + self.attention(
            self.norm1(x)
        )

        # ---------------------------------------------
        # MoE + residual
        # ---------------------------------------------

        moe_output, router_probs = self.moe(
            self.norm2(x)
        )

        x = x + moe_output

        return x, router_probs


# =========================================================
# Pathways SLM
# =========================================================

class PathwaysSLM(nn.Module):

    def __init__(
        self,

        vocab_size,

        max_seq_len=512,

        d_model=256,

        num_heads=8,

        d_ff=768,

        num_layers=6,

        num_experts=4,

        top_k=2
    ):

        super().__init__()

        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        # Transformer layers
        self.layers = nn.ModuleList([

            TransformerBlock(

                d_model=d_model,

                num_heads=num_heads,

                d_ff=d_ff,

                max_seq_len=max_seq_len,

                num_experts=num_experts,

                top_k=top_k
            )

            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(
            d_model
        )

        # Language model head
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def forward(self, input_ids):

        B, T = input_ids.shape

        if T > self.max_seq_len:
            raise ValueError(
                "Sequence is longer than max_seq_len"
            )

        positions = torch.arange(
            T,
            device=input_ids.device
        )

        # Embedding
        x = (
            self.token_embedding(input_ids)
            +
            self.position_embedding(positions)
        )

        router_outputs = []

        # Transformer
        for layer in self.layers:

            x, router_probs = layer(x)

            router_outputs.append(
                router_probs
            )

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits, router_outputs