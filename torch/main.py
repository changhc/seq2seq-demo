from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from encoder import EncoderRNN
from attnDecoder import AttnDecoderRNN

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 10
gpu = 1
max_len = 10

def read_seqs(type):
    x = []
    x_lens = []
    y = []
    y_lens = []
    dirname = '../data/'
    with open(dirname + 'train-x-' + type, 'r') as file:
        for line in file:
            result = [ord(w) for w in line]
            x.append(result)
            x_lens.append(len(result))
    with open(dirname + 'train-y-' + type, 'r') as file:
        for line in file:
#            result = Variable(torch.LongTensor([ord(w) for w in line]).view(-1, 1))
#            if use_cuda:
#                result = result.cuda(gpu)
            result = [ord(w) for w in line]
            y.append(result)
            y_lens.append(len(result))
    return x, x_lens, y, y_lens

def train_batch(input_variable, input_lengths, target_variable, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_len):

    total_loss = 0
    batch_size = len(input_variable)
    input_variable = Variable(torch.LongTensor(input_variable))
    target_variable = Variable(torch.LongTensor(target_variable))
    if use_cuda:
        input_variable = input_variable.cuda(gpu)
        target_variable = target_variable.cuda(gpu)

    print(input_variable.size())
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden(batch_size)
    encoder_cell = encoder.initCell(batch_size)
#    encoder_hidden = None
#    encoder_cell = None

#    input_length = input_variable[i].size()[0]
#    target_length = target_variable[i].size()[0]

    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_variable, input_lengths, encoder_hidden, encoder_cell)
    
    loss = 0

    decoder_hidden = encoder_hidden[:decoder.n_layers]
    decoder_cell = encoder_cell[:decoder.n_layers]

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    all_decoder_outputs = Variable(torch.zeros(max(target_lengths), batch_size, decoder.output_size))

    if use_cuda:
        decoder_input = decoder_input.cuda(gpu)
        all_decoder_outputs = all_decoder_outputs.cuda(gpu)

    out = None
    # Run through decoder one step at a time
    for di in range(max(target_lengths)):
        decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
        all_decoder_outputs[di] = decoder_output
        decoder_input = target_variable[di]

    loss = criterion(all_decoder_outputs.transpose(0, 1).contiguous(), target_variable.transpose(0, 1).contiguous())
    loss.backward()
            
    encoder_optimizer.step()
    decoder_optimizer.step()

    for i in range(2):
        print('Sample {0}:\n\
                Expected output: {1}\n\
                Decoder output: {2}'
                .format(i + 1, torch.transpose(target_variable[i], 0, 1), out.unsqueeze(0)))
    total_loss += loss.data[0] / target_length

    return total_loss / len(input_variable)

def train_epoch(encoder, decoder, n_epoch, batch_size, type, lr=0.01):
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    
    criterion = nn.NLLLoss()
    
    x, x_lens, y, y_lens = read_seqs(type)

    for i in range(n_epoch):
        head = 0
        tail = head + batch_size
        for batch in range(int(math.ceil(len(x) / batch_size))):
            print('epoch {0} batch {1}'.format(i + 1, batch + 1))
            input_variable = x[head:tail]
            input_lens = x_lens[head:tail]
            target_variable = y[head:tail]
            target_lens = y_lens[head:tail]

            loss = train_batch(input_variable, input_lens, target_variable, target_lens, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print('loss: {}'.format(loss))
            head += batch_size
            tail += batch_size
            if tail > len(x):
                tail = len(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type')
    args = parser.parse_args()
    
    input_size = 100
    hidden_size = 20

    encoderl = EncoderRNN(input_size, hidden_size, gpu=gpu)
    attn_decoderl = AttnDecoderRNN(hidden_size, input_size, max_len, gpu=gpu,
                                   dropout_p=0.1)

    if use_cuda:
        encoderl = encoderl.cuda(gpu)
        attn_decoderl = attn_decoderl.cuda(gpu)
    train_epoch(encoderl, attn_decoderl, 5, 200, args.type)

if __name__ == '__main__':
    main()
