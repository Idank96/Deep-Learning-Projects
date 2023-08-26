"""
Terms used in this file
=======================
`iterator`: represents a stream of data. 
            It allows you to access the data one element at a time,
            without having to load the entire data set into memory.

`Tensor` methods:
            1. size() - the size of the tensor.

`tokenizer`:
            A tokenizer in PyTorch is a an object that can convert a string of text into 
            a list of tokens, which are smaller units of text that can be processed by a model.

`BPTT`:
            "BackPropagation Through Time"
            Updating the weights by propagating the errors backwards through
            the hidden states.
`SGD`:
            "stochastic gradient descent"
            An optimization algorithm that updates the model parameters by
            computing the gradients of the loss function with respect to the
            model parameters and multiplying them by the learning rate.
`learning rate scheduler`:  
            a way of adjusting the learning rate during the training process
            of a neural network model. 
            For example, if the initial learning rate is 0.01 and the step_size
            is 10, then the learning rate will be 0.01 for the first 10 epochs,
            then 0.0095 for the next 10 epochs, then 0.009025 for the next 10 epochs, and so on.
`logits`:
            Logits are the output values of the last layer of a neural network.
            They are usually real numbers that can be positive, negative, or zero. 
            They are the raw predictions of a classification model before they 
            are normalized into probabilities (with softmax for example).
`logit function`:
            A logit function is the inverse of a sigmoid function.
            It takes a probability value between 0 and 1 and maps it to a real number.
            A logit function can be used to convert the output of a sigmoid function back to the net input.
`sigmoid function`:
            A sigmoid function is a mathematical function that maps real values
            to values between zero and one.
            It is often used in classification
            models to convert the model's outputs (logits) into probabilities.
            The sidmoid function is the inverse of the logit function.
`softmax function`:
            A softmax function is a mathematical function that converts a vector
            of numbers into a vector of probabilities, where the probabilities
            of each value are proportional to the relative scale of each value
            in the vector.
`//` operator: 
            The // operator in Python is the floor division operator.
            For example: 7 / 2 = 3.5
                         7 // 2 = 3
"""


import math
from tempfile import TemporaryDirectory
from typing import Tuple
import time
import os
import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from model import TransformerModel

# Create an iterator over the training split of the dataset
train_iter = WikiText2(split='train')
# Define the tokenizer as the basic English one
tokenizer = get_tokenizer('basic_english')
# Build a vocabulary from the tokens in the training split, adding a special token for unknown words
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
# Set the default index of the vocabulary to be the index of the unknown token
# why? to skip the KeyError when we try to access a word that is not in the vocabulary.   
vocab.set_default_index(vocab['<unk>'])


def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    # For each item in the raw text iterator:
    # tokenize it and convert it to a tensor of indices using the vocabulary
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    # Concatenate all the tensors into one flat tensor, filtering out any empty tensors
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
# Process each split into a flat tensor of indices
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)
# Get the device to run the model on, either cuda or cpu depending on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batchify(data: Tensor, batch_size: int) -> Tensor:
    """
    Divides the data into ``bsz`` separate sequences,
    removing extra elements that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    # Compute the length of each sequence by dividing the total length by the batch size
    seq_len = data.size(0) // batch_size
    # Trim the data to fit into an integer number of batches
    data = data[:seq_len * batch_size]
    # Reshape the data into a matrix with batch size columns and sequence length rows
    data = data.view(batch_size, seq_len).t().contiguous()
    # Move the data to the device and return it (to run on CUDA GPU in parallel if available)
    return data.to(device)


# Define the batch size for training and evaluation
batch_size = 20
eval_batch_size = 10
# Batchify each split of the data using the corresponding batch size
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
# Define the length of each subsequence for training.
# bptt - "backpropagation through time" - updating the weights by propagating the errors backwards through the hidden states.)
bptt = 35


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    # Get the minimum of bptt (backpropagation through time) and the remaining
    # length from index i as the sequence length
    seq_len = min(bptt, len(source) - 1 - i)
    # Slice the source tensor from index i to i + seq_len as the input data
    data = source[i:i+seq_len]
    # Slice the source tensor from index i + 1 to i + 1 + seq_len as the 
    # target data (the next word for each input word)
    target = source[i+1:i+1+seq_len].reshape(-1)
    # Return a tuple of data and target tensors
    return data, target


ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

# Define the loss function as cross entropy loss
criterion = nn.CrossEntropyLoss()
# Define the learning rate
lr = 5.0
# Define the optimizer as stochastic gradient descent with the model parameters and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# Define the learning rate scheduler as a step decay with a gamma factor of 0.95
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module) -> None:
    """
    A function to train the model for one epoch
    """
    # Turn on train mode for the model (enable dropout and batch normalization for example)
    model.train()
    # Initialize the total loss to zero
    total_loss = 0.
    # The log interval is a parameter that controls how often the training statistics are printed to the console.
    log_interval = 200
    # Record the start time of training
    start_time = time.time()
    # Compute the number of batches by dividing the length of train data by bptt
    num_batches = len(train_data) // bptt
    # Loop over each batch index from zero to num_batches with a step size of bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # Get a batch of data and target from train data using the get_batch function
        data, targets = get_batch(train_data, i)
        # Feed the data to the model and get the output logits
        logits_output = model(data)
        # Reshape the output logits to have a shape of [seq_len * batch_size, ntokens]
        output_flat = logits_output.view(-1, ntokens)
        # Compute the loss by comparing the output logits and targets using the criterion function
        loss = criterion(output_flat, targets)

        # Zero out the gradients of the optimizer
        optimizer.zero_grad()
        # Backpropagate the loss through the model parameters
        loss.backward()
        # Clip the gradients of the model parameters to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # Update the model parameters using the optimizer
        optimizer.step()

        # Accumulate the loss value to the total loss
        total_loss += loss.item()
        # For printing:
        # If the batch index is a multiple of log interval and not zero
        if batch % log_interval == 0 and batch > 0:
            # Get the current learning rate from the scheduler
            lr = scheduler.get_last_lr()[0]
            # Compute the milliseconds per batch by dividing the elapsed time by log interval and multiplying by 1000
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            # Compute the current loss by dividing the total loss by log interval
            cur_loss = total_loss / log_interval
            # Compute the perplexity by taking the exponential of the current loss
            ppl = math.exp(cur_loss)
            # Print the statistics of the current epoch, batch, learning rate, time, loss and perplexity
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            # Reset the total loss to zero
            total_loss = 0
            # Record the start time of the next batch
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    """
    A function to evaluate the model on a given data split
    """
    # Turn on evaluation mode for the model (disable dropout and batch normalization)
    model.eval()  # turn on evaluation mode
    # Initialize the total loss to zero
    total_loss = 0.
    # Disable gradient computation to save memory and speed up evaluation
    with torch.no_grad():
        # Loop over each batch index from zero to the length of eval data with a step size of bptt
        for i in range(0, eval_data.size(0) - 1, bptt):
            # Get a batch of data and target from eval data using the get_batch function
            data, targets = get_batch(eval_data, i)
            # Get the sequence length from the data shape
            seq_len = data.size(0)
            # Feed the data to the model and get the output logits
            output = model(data)
            # Reshape the output logits to have a shape of [seq_len * batch_size, ntokens]
            output_flat = output.view(-1, ntokens)
            # Compute the loss by comparing the output logits and targets using the criterion function and multiply it by the sequence length
            total_loss += seq_len * criterion(output_flat, targets).item()
    # Return the average loss by dividing the total loss by the length of eval data minus one
    return total_loss / (len(eval_data) - 1)

# Define a variable to store the best validation loss as infinity
best_val_loss = float('inf')
# Define the number of epochs to train
epochs = 3

# Create a temporary directory to store the best model parameters
with TemporaryDirectory() as tempdir:
    # Define a path to save the best model parameters in the temporary directory
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
    # Loop over each epoch from one to epochs plus one
    for epoch in range(1, epochs + 1):
        # Record the start time of the epoch
        epoch_start_time = time.time()
        # Train the model for one epoch using the train function
        train(model)
        # Evaluate the model on validation data using the evaluate function and get the validation loss
        val_loss = evaluate(model, val_data)
        # Compute the validation perplexity by taking the exponential of the validation loss
        val_ppl = math.exp(val_loss)
        # Compute the elapsed time of the epoch
        elapsed = time.time() - epoch_start_time
        # Print the statistics of the end of epoch, time, validation loss and perplexity
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)
        # If the validation loss is lower than the best validation loss so far
        if val_loss < best_val_loss:
            # Update the best validation loss to be the current validation loss
            best_val_loss = val_loss
            # Save the model state dictionary to the best model parameters path
            torch.save(model.state_dict(), best_model_params_path)
        # Update the learning rate using the scheduler step function
        scheduler.step()
    # Load the best model
    model.load_state_dict(torch.load(best_model_params_path))

# Evaluate the best model on the test dataset
test_loss = evaluate(model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)




