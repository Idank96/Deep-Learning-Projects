"""
Terms used in this file
=======================

`TransformerEncoderLayer`:
        Implement the attention mechanism ("Attention Is All You Need" paper)
        Using Multi-head-attention (allows the model to capture different types of relationships within the input data).

`forward function`:
        The forward function is called when you run `model(input)`.
        The forward function is where your model does the actual computation,
        and it returns the output of the model.
        The actual call to forward is happnds 
        in `TORCH.NN.MODULES.MODULE` module with `__call__`  and `_call_impl`.

`token`:
        Each word in a sentence is typically represented as a token.


"""

import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
    """
    A transformer is a type of neural network model that can process sequential data, such as text or speech.
    Unlike other models that use recurrence or convolution, a transformer uses attention to learn the relationships 
    between the elements in a sequence.
    """
    def __init__(self,
                 ntoken: int, # size of the vocabulary
                 d_model: int, # dimension of the word embeddings and the transformer encoder
                 nhead: int, # number of attention heads in the multi-head attention layer
                 d_hid: int, # dimension of the feedforward network in the transformer encoder layer
                 nlayers: int,# number of transformer encoder layers
                 dropout: float = 0.5 # dropout probability for regularization
                 ):
        # initialize model parameters & layers
        super().__init__() 
        # indicates the type of the model
        self.model_type = 'Transformer' 
        # custom module that adds positional encoding to the word embeddings
        self.pos_encoder = PositionalEncoding(d_model, dropout) 
        # standard module that implements one layer of the transformer encoder.
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        # stack of transformer encoder layers 
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) 
        # lookup table that maps tokens to embeddings
        self.embedding = nn.Embedding(ntoken, d_model) 
        # the dimension of the embeddings and the transformer encoder
        self.d_model = d_model 
        # linear transformation with learnable weights and biases.
        self.linear = nn.Linear(d_model, ntoken)

        # initializes the weights and biases of the embedding and linear layers with a uniform distribution in a small range.
        self.init_weights() 


    def init_weights(self) -> None:
        """
        Initializes the weights and biases of the embedding and linear layers.
        """
        # the lower and upper bound for generating random numbers from a uniform distribution.
        initrange = 0.1 
        # fills the weight tensor with random numbers from a uniform distribution in place.
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # fills the bias tensor with zeros in place.
        self.linear.bias.data.zero_()
        # fills weight tensor with random numbers from a uniform distribution in place.
        self.linear.weight.data.uniform_(-initrange, initrange) 

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Description:
            How to perform forward computation on an input tensor.
            src_mask indicates which positions in each sequence
            should be ignored by attention.
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
            which is a batch of output sequences with logits for 
            each token in vocabulary.
        """
        # transform the input tokens into dense vectors / apply a lookup table that
        # maps each token to a vector of dimension. 
        # use 'sqrt' to adjust the magnitude of the embeddings to match
        # the expected scale of the transformer encoder.
        # multiply each element of the embedded tensor by a constant scalar
        src = self.embedding(src) * math.sqrt(self.d_model)
        # add positional information to the embeddings, since the
        # transformer encoder does not have any recurrence or convolution
        # mechanism that can capture the order of the tokens.
        src = self.pos_encoder(src)
        # encode the input sequences using a stack of transformer encoder layers
        # that apply self-attention and feedforward networks.
        # pass the positionally encoded tensor and the mask tensor to the
        # transformer encoder module, which returns an output tensor that
        # contains the hidden states of each token in each sequence.
        output = self.transformer_encoder(src, src_mask)
        # generate predictions for each token in the output sequences using
        # a linear transformation with learnable weights and biases.
        # apply a matrix multiplication and a vector addition to the
        # encoded tensor, which results in an output tensor that has
        # the same length as the input sequences and the same width 
        # as the vocabulary size.
        output = self.linear(output)
        return output
