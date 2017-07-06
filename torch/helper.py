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

class Seq:
    def __init__(self, name):
        self.name = name
    
    def readSeqs(type):
        x = []
        y = []
        dirname = '../data'
        with open(dirname + 'train-x-' + type, 'r') as file:
            for line in file:
                result = Variable(torch.IntTensor([ord(w) for w in line]).view(-1, 1))
                if use_cuda:
                    result = result.cuda()
                x.append(result)
        with open(dirname + 'train-y-' + type, 'r') as file:
            for line in file:
                result = Variable(torch.IntTensor([ord(w) for w in line]).view(-1, 1))
                if use_cuda:
                    result = result.cuda()
                y.append(result)
        return x, y


