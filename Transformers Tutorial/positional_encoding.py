import torch
from torch import nn, Tensor
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding is a technique used in deep learning models, specifically in transformers architecture. 
    Positional encoding is added to the input embeddings before feeding them to the transformer.
    It allows the model to capture the order and relative position of the tokens in a sequence, 
    which is important for understanding the meaning and context of the data.
    Injects information about the relative or absolute position of the tokens in the sequence.
    Using sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__() # initialize model parameters & layers.
        self.dropout = nn.Dropout(p=dropout) # dropout regulariztion
        # creates a tensor of shape [1, d_model] that contains
        # the position indices from 0 to d_model - 1.
        position = torch.arange(max_len).unsqueeze(1)
        # creates a geometric progression of decreasing values that are
        # used to scale the sine and cosine functions of different frequencies.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # initialized with an exponential function
        pe = torch.zeros(max_len, 1, d_model) # initialized with zeros of shape [1, d_model]
        # creates a sinusoidal waveform that varies according to the position and frequency.
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # creates another sinusoidal waveform that is shifted by 90 degrees from the previous one.
        pe[:, 0, 1::2] = torch.cos(position * div_term) # stores the positional encoding matrix
        self.register_buffer('pe', pe) # registers a buffer tensor

    def forward(self, x: Tensor) -> Tensor:
        """
        Description:
          How to perform forward computation on an input tensor
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # adds positional information to each word embedding according
        # to its position in the sequence.
        x = x + self.pe[:x.size(0)] # Simple!
        # applies dropout probability to each element of the tensor,
        # which randomly sets some elements to zero to prevent overfitting.
        return self.dropout(x)
    
