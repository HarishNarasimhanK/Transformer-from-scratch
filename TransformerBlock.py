import torch
from torch import nn 
from MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, model_dimension : int, num_heads : int):
        super().__init__()
        self.mhsa = MultiHeadAttention(model_dimension, model_dimension,  num_heads)
        self.first_layernorm = nn.LayerNorm(model_dimension)
        self.second_layernorm = nn.LayerNorm(model_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dimension, 4 * model_dimension),
            nn.ReLU(),
            nn.Linear(4 * model_dimension, model_dimension),
        )

    def forward(self, embedded : torch.Tensor) -> torch.Tensor:
        first_part = embedded + self.mhsa(self.first_layernorm(embedded))
        res = first_part + self.feed_forward(self.second_layernorm(first_part))
        return res
    