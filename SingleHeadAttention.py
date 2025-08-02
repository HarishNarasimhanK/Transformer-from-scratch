import torch
from torch import nn

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dimension : int, attention_dimension : int):
        super().__init__()
        ## In case of self attention, it is found that not using biases gives a better result
        self.get_keys = nn.Linear(embedding_dimension, attention_dimension, bias = False)
        self.get_queries = nn.Linear(embedding_dimension, attention_dimension, bias = False)
        self.get_values = nn.Linear(embedding_dimension, attention_dimension, bias = False)


    def forward(self, embedded : torch.Tensor) -> torch.Tensor:
        k = self.get_keys(embedded)
        q = self.get_queries(embedded)
        v = self.get_values(embedded) 
        ## this 1 and 2 represents that the matrix are transposed with respect to 1st and 2nd dimension, leaving batchsize
        scores = torch.bmm(q, torch.transpose(k, 1, 2))
        B, T, A = k.shape
        scores = scores / (A ** 0.5)    
        
        pre_mask = torch.tril(torch.ones(T, T, device=embedded.device))
        mask = pre_mask == 0
        scores = scores.masked_fill(mask, float("-inf"))
        scores = nn.functional.softmax(scores, dim = -1) ## Score is B x T x T
        transformed_output = torch.bmm(scores, v)
        return transformed_output
    