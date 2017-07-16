
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


class EncoderRNN:
    def __init__(self, batch_size, max_len, hidden_units,
                 embeddings, attention=True, bidirection=True):
        self.batch_size = batch_size
        self.max_len = max_len
        self.embeddings = embeddings
        self.hidden_units = hidden_units
        self.bidirection = bidirection
        self.attention = attention
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
        if self.bidirection:
            ((output_fw, output_bw),
             (final_state_fw, final_state_bw)) = (
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.encoder_cell,
                    cell_bw=self.encoder_cell,
                    inputs=self.encoder_inputs_embedded,
                    dtype=tf.float32,
                )
            )
            if self.attention:
                self.encoder_final_output = tf.concat(
                    [output_fw, output_bw], 1
                )
            del output_fw, output_bw
            context = tf.concat([final_state_fw.c, final_state_bw.c], 1)
            hidden = tf.concat([final_state_fw.h, final_state_bw.h], 1)
            self.encoder_final_state = LSTMStateTuple(c=context, h=hidden)
        else:
            output, self.encoder_final_state = tf.nn.dynamic_rnn(
                self.encoder_cell,
                self.encoder_inputs_embedded,
                dtype=tf.float32,
            )
            if self.attention:
                self.encoder_final_output = output
            del output
