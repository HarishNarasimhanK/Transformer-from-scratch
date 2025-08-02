from torch import nn
import torch
from GPT import GPT

## Model Loader function
def load_complete_model(filename="MyGPT.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {filename}...")
    checkpoint = torch.load(filename, map_location=device)
    config = checkpoint['model_config']
    tokenizer_data = checkpoint['tokenizer_data']
    model = GPT(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        model_dimension=config['model_dimension'],
        num_heads=4,
        num_layers=config['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model Info:")
    print(f"   - Vocabulary Size: {config['vocab_size']}")
    print(f"   - Context Length: {config['context_length']}")
    print(f"   - Model Dimension: {config['model_dimension']}")
    
    return model, tokenizer_data
