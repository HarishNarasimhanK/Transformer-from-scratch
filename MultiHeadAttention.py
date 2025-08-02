import torch
from torch import nn
from SingleHeadAttention import SingleHeadAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension : int, attention_dimension : int, num_heads : int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.heads = nn.ModuleList([SingleHeadAttention(embedding_dimension, attention_dimension // num_heads) for _ in range(num_heads)])
        self.output_projection = nn.Linear(attention_dimension, embedding_dimension, bias = True)

    def forward(self, embedded : torch.Tensor) -> torch.Tensor:
        output = torch.cat([head(embedded) for head in self.heads], dim = -1)
        return self.output_projection(output)