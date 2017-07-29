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

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, gpu=0, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.gpu = gpu

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p) 
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input).view(1, len(input), -1)
        embedded = self.dropout(embedded)

        attn_weights = self.attn_func(hidden[-1], encoder_outputs)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        attn_applied = attn_applied.transpose(0, 1)

        output = torch.cat((embedded, attn_applied), 2)

        for i in range(self.n_layers):
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        
        output = output.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, attn_applied.squeeze(0)), 1)))
        return output, hidden, cell, attn_weights


    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda(self.gpu)
        else:
            return result
    
    def initCell(self): 
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda(self.gpu)
        else:
            return result

    def attn_func(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if use_cuda:
            attn_energies = attn_energies.cuda(self.gpu)

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[b, :], encoder_outputs[b, i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden, encoder_output), 0).unsqueeze(0))
        energy = self.v.dot(energy)
        return energy 
