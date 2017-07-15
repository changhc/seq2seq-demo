
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers import fully_connected

class DecoderRNN:
    def __init__(self, batch_size, max_len, vocab_size,
            hidden_units, embeddings, encoder_state):
        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.hidden_units = hidden_units
        self.init_state = encoder_state
        self.loss_weights = tf.ones(
            [batch_size, max_len],
            dtype=tf.float32,
        )
        self._init_placeholder()

    def _init_placeholder(self):
        self.decoder_inputs = tf.placeholder(
            shape=(self.batch_size, self.max_len),
            dtype=tf.int32,
            name='decoder_inputs',
        )
        self.decoder_targets = tf.placeholder(
            shape=(self.batch_size, self.max_len),
            dtype=tf.int32,
            name='decoder_targets',
        )
        self._init_graph()

    def _init_graph(self):
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(
            self.embeddings,
            self.decoder_inputs,
        )
        self.decoder_cell = LSTMCell(self.hidden_units)
        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
            self.decoder_cell,
            self.decoder_inputs_embedded,
            initial_state=self.init_state,
            dtype=tf.float32,
            scope="decoder",
        )
        self._prediction_layer()

    def _prediction_layer(self):
        self.decoder_logits = fully_connected(
            self.decoder_outputs,
            self.vocab_size,
            activation_fn=None
        )
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
