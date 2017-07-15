
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class EncoderRNN:
    def __init__(self, batch_size, max_len, hidden_units, embeddings):
        self.batch_size = batch_size
        self.max_len = max_len
        self.embeddings = embeddings
        self.hidden_units = hidden_units
        self._init_placeholder()

    def _init_placeholder(self):
        self.encoder_inputs = tf.placeholder(
            shape=(self.batch_size, self.max_len),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self._init_graph()

    def _init_graph(self):
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(
            self.embeddings,
            self.encoder_inputs,
        )
        self.encoder_cell = LSTMCell(self.hidden_units)
        _, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell,
            self.encoder_inputs_embedded,
            dtype=tf.float32,
        )
