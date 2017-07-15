import argparse
import tensorflow as tf
import numpy as np
from encoder import EncoderRNN
from decoder import DecoderRNN
from tensorflow.contrib.seq2seq import sequence_loss


EOS_tag = ord('\n')
num_epoch = 5
batch_size = 100
vocab_size = 100
input_embedding_size = 50
encoder_hidden_units = 20
decoder_hidden_units = 20

parser = argparse.ArgumentParser()
parser.add_argument('type')


def readInput(type, batch_size):
    x = []
    y = []
    batches = []
    dirname = '../data/'
    with open(dirname + 'train-x-' + type, 'r') as file:
        for line in file:
            x.append([ord(w) for w in line if w != '\n'])
    with open(dirname + 'train-y-' + type, 'r') as file:
        for line in file:
            y.append([ord(w) for w in line if w != '\n'])
    head = 0
    tail = head + batch_size
    max_len = [0, 0]
    while True:
        batch_x = x[head:tail]
        len_x = [len(item) for item in batch_x]
        batch_y = y[head:tail]
        len_y = [len(item) for item in batch_y]
        batches.append([
            batch_x,
            len_x,
            batch_y,
            len_y,
        ])
        if max_len[0] < max(len_x):
            max_len[0] = max(len_x)
        if max_len[1] < max(len_y):
            max_len[1] = max(len_y)
        if tail == len(x):
            break
        head = head + batch_size
        tail = tail + batch_size
        if tail > len(x):
            tail = len(x)
    return batches, max_len


def main(args):
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
    )
    decoder = DecoderRNN(
        batch_size,
        max_len[1] + 1,
        vocab_size,
        decoder_hidden_units,
        embeddings,
        encoder.encoder_final_state,
    )   
    loss = sequence_loss(
        logits=decoder.decoder_logits,
        targets=decoder.decoder_targets,
        weights=decoder.loss_weights,
    )
    train_op = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    for epoch in range(num_epoch):
        for index, batch in enumerate(batches):
            decoder_inputs = [[EOS_tag] + seq for seq in batch[2]]
            decoder_targets = [seq + [EOS_tag] for seq in batch[2]]
            fd = {
                encoder.encoder_inputs: batch[0],
                decoder.decoder_inputs: decoder_inputs,
                decoder.decoder_targets: decoder_targets,
            }
            _, l = sess.run([train_op, loss], fd)

            if (index + 1 % 100 == 0 or index + 1 == len(batches)):
                print('epoch {0} batch {1}:'.format(epoch + 1, index + 1))
                print('loss: {0}'.format(l))
                predict = sess.run(decoder.decoder_prediction, fd)
                for i, (inp, pred) in enumerate(
                        zip(fd[encoder.encoder_inputs], predict)):
                    print('sample {0}:'.format(i + 1))
                    print('{0:>15}\t{1}'.format('input', np.asarray(inp)))
                    print('{0:>15}\t{1}'.format('prediction', pred[:-1]))
                    if i > 0:
                        break
                print(' ')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
