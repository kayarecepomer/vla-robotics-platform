"""Small transformer policy head shared by every encoder in the study.

Frozen image tokens, a joint-state token and a task token go through a transformer encoder;
``horizon`` learned queries then decode an action chunk (ACT-style, without the CVAE). The
head is identical across encoders, so differences in success rate come from the features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class HeadConfig:
    feature_dim: int = 1024
    num_tokens: int = 2
    num_tasks: int = 1
    state_dim: int = 6
    action_dim: int = 6
    horizon: int = 50
    d_model: int = 256
    heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class ChunkPolicyHead(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.feature_norm = nn.LayerNorm(config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, d)
        self.state_proj = nn.Linear(config.state_dim, d)
        self.task_embedding = nn.Embedding(config.num_tasks, d)
        self.position = nn.Parameter(torch.zeros(1, config.num_tokens + 2, d))
        self.queries = nn.Parameter(torch.zeros(1, config.horizon, d))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.queries, std=0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, config.heads, 4 * d, config.dropout, batch_first=True, norm_first=True),
            config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, config.heads, 4 * d, config.dropout, batch_first=True, norm_first=True),
            config.decoder_layers,
        )
        self.action_out = nn.Linear(d, config.action_dim)

    def forward(self, tokens: torch.Tensor, state: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """tokens [B, num_tokens, feature_dim], state [B, state_dim], task [B] -> actions [B, horizon, action_dim]."""
        image = self.feature_proj(self.feature_norm(tokens))
        context = torch.cat([image, self.state_proj(state).unsqueeze(1), self.task_embedding(task).unsqueeze(1)], dim=1)
        memory = self.encoder(context + self.position)
        decoded = self.decoder(self.queries.expand(len(tokens), -1, -1), memory)
        return self.action_out(decoded)


def chunk_l1_loss(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    errors = (predicted - target).abs().mean(-1)
    return (errors * valid).sum() / valid.sum().clamp(min=1)
