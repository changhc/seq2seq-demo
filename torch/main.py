from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import math
import argparse

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
    y = []
    dirname = '../data/'
    with open(dirname + 'train-x-' + type, 'r') as file:
        for line in file:
            result = Variable(torch.LongTensor([ord(w) for w in line]).view(-1, 1))
            if use_cuda:
                result = result.cuda(gpu)
            x.append(result)
    with open(dirname + 'train-y-' + type, 'r') as file:
        for line in file:
            result = Variable(torch.LongTensor([ord(w) for w in line]).view(-1, 1))
            if use_cuda:
                result = result.cuda(gpu)
            y.append(result)
    return x, y

def train_batch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_len):
    total_loss = 0
    for i in range(len(input_variable)):
        encoder_hidden = encoder.initHidden()
        encoder_cell = encoder.initCell()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_variable[i].size()[0]
        target_length = target_variable[i].size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda(gpu) if use_cuda else encoder_outputs

        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden, encoder_cell = encoder(input_variable[i][ei], encoder_hidden, encoder_cell)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda(gpu) if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[i][di])
            decoder_input = target_variable[i][di]

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        
        if i < 2:
            print('Sample {0}:\n\
                    Expected output: {1}\n\
                    Decoder output: {2}'
                    .format(i + 1, target_variable[i], decoder_output))
        total_loss += loss.data[0] / target_length
    return total_loss / len(input_variable)

def train_epoch(encoder, decoder, n_epoch, batch_size, type, lr=0.01):
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    x, y = read_seqs(type)

    for i in range(n_epoch):
        head = 0
        tail = head + batch_size
        for batch in range(int(math.ceil(len(x) / batch_size))):
            print('epoch {0} batch {1}'.format(i + 1, batch + 1))
            input_variable = x[head:tail]
            target_variable = y[head:tail]

            loss = train_batch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print('loss: {}'.format(loss))

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
    train_epoch(encoderl, attn_decoderl, 5, 100, args.type)

if __name__ == '__main__':
    main()
