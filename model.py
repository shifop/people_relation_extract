import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import exp
from config import cfg
from tensorflow.keras import layers
from params_flow.activations import gelu
import math
from transformers import TFBertModel


class RL(tf.keras.Model):

    def __init__(self):
        super(RL, self).__init__()

        model_dir = cfg.DIR
        self.bert = TFBertModel.from_pretrained(model_dir)
        self.bilstm = layers.Bidirectional(layers.LSTM(cfg.HIDDENT_SIZE, return_sequences=True), name='bilstm')

        self.fusion_dense = layers.Dense(cfg.HIDDENT_SIZE, name='fusion_dense', activation=gelu)
        self.p_dense = layers.Dense(14, name='p_dense')
        self.dropout =  layers.AlphaDropout(0.3)
        self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def __encode(self, input_ids, token_type_ids, attention_mask, sep_index, training=True):
        """
        第一层编码
        :param text: tf.int32 [None, seq_length]
        :param mask: tf.bool [None, seq_length]
        :param training: bool
        :return: [None, seq_length, albert_output_size]
        """

        em = self.bert(
            input_ids, 
            token_type_ids = token_type_ids, 
            attention_mask = attention_mask, 
            # output_hidden_states=True, 
            training = training)

        # em = em[-1][1:]
        em = em[0]
        # em = self.dropout1(em, training=training)
        em = self.bilstm(em)
        sep_index = tf.reshape(sep_index, [-1, 1, 4])
        sep_em = []
        for i in range(cfg.BATCH_SIZE):
            sep_em.append(tf.nn.embedding_lookup(em[i] ,sep_index[i]))
        sep_em = tf.concat(sep_em, axis=0)

        sep_em = tf.reshape(sep_em, [-1, cfg.HIDDENT_SIZE*4*2])
        
        return sep_em

    def __call__(self, input, training):
        input_ids, token_type_ids, attention_mask, sep_index = input
        sep_em = self.__encode(input_ids, token_type_ids, attention_mask, sep_index, training)

        sep_em = self.dropout(sep_em, training=training)

        ft = self.fusion_dense(sep_em)

        ft = self.p_dense(ft)
        return ft

    def get_loss(self, input):
        input_ids, token_type_ids, attention_mask, sep_index, labels = input
        input = [input_ids, token_type_ids, attention_mask, sep_index]
        ft = self.__call__(input, True)
        # labels = tf.one_hot(labels,depth=4)
        # labels -= 0.05*(labels-1./4)
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels, ft)
        loss = self.cce(labels, ft)
        return tf.reduce_mean(loss)

    def predict(self, input):
        ft = self.__call__(input, False)
        ft = tf.nn.softmax(ft, axis=-1)
        ft = tf.argmax(ft, axis=-1)
        return ft.numpy()

    