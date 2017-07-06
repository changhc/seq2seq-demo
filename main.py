import tensorflow as tf
import argparse
from tensorflow.contrib import seq2seq
from tensorflow.contrib.rnn import LSTMCell

parser = argparse.ArgumentParser()
parser.add_argument('type')
args = parser.parse_args()

def read_input(type):
    x = []
    y = []
    with open('train-x-' + type, 'r') as file:
        for line in file:
            x.append([ord(w) for w in line])
    with open('train-y-' + type, 'r') as file:
        for line in file:
            y.append([ord(w) for w in line])
    return x, y

def create_model():
    encoder_hidden_units = 20
    decoder_hidden_units = 20

    #encoder_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_input')
    #decoder_target = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_target')
    #decoder_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_input')

    #encoder_cell = LSTMCell(encoder_hidden_units)
    return seq2seq(

def main():
    x, y = read_input(args.type)


if __name__ == '__main__':
    main()

