from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, gpu=0, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gpu = gpu
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, inputs, input_lengths, hidden, cell):
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, (hidden, cell) = self.lstm(packed, (hidden, cell))
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden, cell

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda(self.gpu)
        else:
            return result

    def initCell(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda(self.gpu)
        else:
            return result
