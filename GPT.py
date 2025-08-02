from TransformerBlock import TransformerBlock
import torch
from torch import nn

class GPT(nn.Module):
    def __init__(self, vocab_size : int, context_length : int, model_dimension : int, num_heads : int, num_layers : int = 12):
        super().__init__()
        ## token embeddings
        self.token_embeddings =  nn.Embedding(vocab_size, model_dimension)
        ## positional embeddings
        self.pos_embeddings = nn.Embedding(context_length, model_dimension)
        self.blocks = nn.Sequential(*[TransformerBlock(model_dimension, num_heads) for _ in range(num_layers)])
        ## This layer norm is additional one, which is present exclusive to GPTs
        self.final_layernorm = nn.LayerNorm(model_dimension)
        ## Final linear layer which was present outside the transformer block 
        self.vocab_projection = nn.Linear(model_dimension, vocab_size)

        
    def forward(self, context : torch.Tensor, targets = None) -> torch.Tensor:        
        B, T = context.size()
        token_embeddings = self.token_embeddings(context)
        pos_embeddings = self.pos_embeddings(torch.arange(T, device = context.device))
        total_embeddings = token_embeddings + pos_embeddings
        logits = self.vocab_projection(self.final_layernorm(self.blocks(total_embeddings)))
        ## softmax
        # logits = nn.functional.softmax(unnormalized, dim = -1)
        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss