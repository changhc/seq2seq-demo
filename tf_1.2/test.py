import argparse
import tensorflow as tf
import numpy as np
from encoder import EncoderRNN
from decoder import DecoderRNN
from tensorflow.contrib.seq2seq import sequence_loss


EOS_tag = ord('\n')
num_epoch = 1
batch_size = 1000
vocab_size = 100
input_embedding_size = 50
encoder_hidden_units = 20
decoder_hidden_units = 20

parser = argparse.ArgumentParser()
parser.add_argument('type')
parser.add_argument('--attention', default=True)
parser.add_argument('--bidirection', default=True)


def readInput(type, batch_size):
    x = []
    batches = []
    dirname = '../data/'
    with open(dirname + 'train-x-' + type, 'r') as file:
        for line in file:
            x.append([ord(w) for w in line if w != '\n'])
    head = 0
    tail = head + batch_size
    max_len = [0, 8]
    while True:
        batch_x = x[head:tail]
        len_x = [len(item) for item in batch_x]
        batches.append([
            batch_x,
            len_x,
        ])
        if max_len[0] < max(len_x):
            max_len[0] = max(len_x)
        if tail == len(x):
            break
        head = head + batch_size
        tail = tail + batch_size
        if tail > len(x):
            break
    return batches, max_len


def main(args):
    global decoder_hidden_units
    if args.attention:
        decoder_hidden_units = encoder_hidden_units
        if args.bidirection:
            decoder_hidden_units *= 2
    batches, max_len = readInput(args.type, batch_size)
    embeddings = tf.Variable(
        tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0),
        dtype=tf.float32,
    )
    encoder = EncoderRNN(
        batch_size,
        max_len[0],
        encoder_hidden_units,
        embeddings,
        attention=args.attention,
        bidirection=args.bidirection,
    )
    decoder = DecoderRNN(
        batch_size,
        max_len[1] + 1,
        vocab_size,
        decoder_hidden_units,
        embeddings,
        encoder,
        attention=args.attention,
        bidirection=args.bidirection,
    )   
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, 'model.ckpt')
    decoder_inputs = [[EOS_tag] * 9] * batch_size
    fd = {
        encoder.encoder_inputs: batches[0][0],
        decoder.decoder_inputs: decoder_inputs,
    }
    predict = sess.run(decoder.decoder_prediction, fd)

    for i, (inp, pred) in enumerate(
            zip(fd[encoder.encoder_inputs], predict)):
        print('sample {0}:'.format(i + 1))
        print('{0:<15}\t{1}'.format(
            'input:',
            ''.join([chr(x) for x in np.asarray(inp)]),
        ))
        print('{0:<15}\t{1}'.format(
            'prediction:',
            ''.join([chr(x) for x in pred[:-1]]),
        ))
        if i > 0:
            break
    print(' ')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
