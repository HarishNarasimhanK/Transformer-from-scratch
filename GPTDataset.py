from GPT import GPT
from torch.utils.data import DataLoader, Dataset
import torch
import re

class GPTDataset(Dataset):
    def __init__(self, text_file, block_size=128):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        vocab = sorted(list(set(words)))
        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        # Convert text to indices
        data = [self.stoi[word] for word in words]
        
        # Create training examples
        self.data = []
        for i in range(0, len(data) - block_size):
            self.data.append(data[i:i + block_size + 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)