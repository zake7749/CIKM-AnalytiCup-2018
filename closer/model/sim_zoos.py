import random, os, sys
import re
import codecs
import numpy as np
np.random.seed(1337)

import tensorflow as tf
import pandas as pd
import operator
import sys

from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from closer.trainer.supervised_trainer import KerasModelTrainer
from closer.data_utils.data_helpers import DataTransformer, DataLoader
from closer.config import dataset_config, model_config
from closer.model.utils import *
from closer.model.layers import *

from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.legacy.layers import Highway
from keras.layers import TimeDistributed
from keras.optimizers import Nadam, Adam
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.activations import softmax
from keras.regularizers import l2

def get_decomposable_attention(nb_words, embedding_size, embedding_matrix, max_sequence_length, out_size,
                    compare_dim=300, compare_dropout=0.2, dense_dim=256, dense_dropout=0.2, lr=1e-3, activation='relu',
                    with_meta_features=False, word_level=True):
    q1, q1_c, q2, q2_c, meta_features = get_input_layers()

    if word_level:
        q1_embedded, q2_embedded = get_word_embeddings(q1, q2, nb_words, embedding_size, embedding_matrix, 
                                                        max_sequence_length, trainable=False, embedding_dropout=model_config.EMBEDDING_DROPOUT)
    else:
        q1_embedded, q2_embedded = get_char_embeddings(q1_c, q2_c, max_sequence_length, model_config.CHAR_EMBEDDING_SIZE,
         feature_map_nums=model_config.CHAR_EMBEDDING_FEATURE_MAP_NUMS, 
         window_sizes=model_config.CHAR_EMBEDDING_WINDOW_SIZES,
         embedding_dropout=model_config.EMBEDDING_DROPOUT)

    # Context encoder        
    highway_encoder = TimeDistributed(Highway(activation='relu'))
    
    q1_encoded = highway_encoder(q1_embedded,)
    q2_encoded = highway_encoder(q2_embedded,)
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)    
    
    # Compare deep views
    q1_combined = Concatenate()([q1_encoded, q2_aligned, interaction(q1_encoded, q2_aligned),])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, interaction(q2_encoded, q1_aligned),]) 
    
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    
    q1_compare = time_distributed(q1_combined, compare_layers_d)
    q2_compare = time_distributed(q2_combined, compare_layers_d)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D(),])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D(),])

    # Dense meta featues
    meta_densed = BatchNormalization()(meta_features)
    meta_densed = Highway(activation='relu')(meta_densed)
    meta_densed = Dropout(0.2)(meta_densed)
    
    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    q_rep = Concatenate()([q1_rep, q2_rep])
    
    if with_meta_features:
        h_all = Concatenate()([q_diff, q_multi, q_rep, meta_densed])
    else:
        h_all = Concatenate()([q_diff, q_multi, q_rep,]) 

    h_all = Dropout(0.5)(h_all)
    
    dense = Dense(dense_dim, activation=activation)(h_all)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    
    out_ = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[q1, q2, q1_c, q2_c, meta_features], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def get_CARNN(nb_words, embedding_size, embedding_matrix, max_sequence_length, out_size=1,
    compare_dim=model_config.CARNN_COMPARE_LAYER_HIDDEN_SIZE, compare_out_size=model_config.CARNN_COMPARE_LAYER_OUTSIZE, compare_dropout=model_config.COMPARE_LAYER_DROPOUT,
    meta_features_dropout=model_config.META_FEATURES_DROPOUT,
    rnn_size=model_config.CARNN_RNN_SIZE, rnn_dropout=model_config.CARNN_AGGREATION_DROPOUT,
    with_meta_features=False, word_level=True,
    lr=1e-3, activation='relu'):

    q1, q1_c, q2, q2_c, meta_features = get_input_layers()

    if word_level:
        q1_embedded, q2_embedded = get_word_embeddings(q1, q2, nb_words, embedding_size, embedding_matrix, max_sequence_length,
                                                       trainable=False, embedding_dropout=model_config.EMBEDDING_DROPOUT)
        embedding_size = model_config.WORD_EMBEDDING_SIZE
    else:
        q1_embedded, q2_embedded = get_char_embeddings(q1_c, q2_c, max_sequence_length, model_config.CHAR_EMBEDDING_SIZE,
                                                       feature_map_nums=model_config.CHAR_EMBEDDING_FEATURE_MAP_NUMS, 
                                                       window_sizes=model_config.CHAR_EMBEDDING_WINDOW_SIZES,
                                                       embedding_dropout=model_config.EMBEDDING_DROPOUT)
        embedding_size = model_config.CHAR_CNN_OUT_SIZE

    self_attention = SelfAttention(d_model=embedding_size)

    # Context encoder
    highway_encoder = TimeDistributed(Highway(activation='selu'))

    q1_encoded = highway_encoder(q1_embedded,)    
    q2_encoded = highway_encoder(q2_embedded,)

    s1_encoded = self_attention(q1, q1_encoded)
    s2_encoded = self_attention(q2, q2_encoded)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
    # Compare deep views
    q1_combined1 = Concatenate()([q1_encoded, q2_aligned, interaction(q1_encoded, q2_aligned),])
    q1_combined2 = Concatenate()([q2_aligned, q1_encoded, interaction(q1_encoded, q2_aligned),])
    
    q2_combined1 = Concatenate()([q2_encoded, q1_aligned, interaction(q2_encoded, q1_aligned),])
    q2_combined2 = Concatenate()([q1_aligned, q2_encoded, interaction(q2_encoded, q1_aligned),])
    
    s1_combined1 = Concatenate()([q1_encoded, s1_encoded, interaction(q1_encoded, s1_encoded),])
    s1_combined2 = Concatenate()([s1_encoded, q1_encoded, interaction(q1_encoded, s1_encoded),])
    
    s2_combined1 = Concatenate()([q2_encoded, s2_encoded, interaction(q2_encoded, s2_encoded),])
    s2_combined2 = Concatenate()([s2_encoded, q2_encoded, interaction(q2_encoded, s2_encoded),])
    
    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_out_size, activation=activation),
        Dropout(compare_dropout),
    ]

    compare_layers_g = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_out_size, activation=activation),
        Dropout(compare_dropout),
    ]

    # NOTE these can be optimized
    q1_compare1 = time_distributed(q1_combined1, compare_layers_d)
    q1_compare2 = time_distributed(q1_combined2, compare_layers_d)
    q1_compare = Average()([q1_compare1, q1_compare2])

    q2_compare1 = time_distributed(q2_combined1, compare_layers_d)
    q2_compare2 = time_distributed(q2_combined2, compare_layers_d)
    q2_compare = Average()([q2_compare1, q2_compare2])
    
    s1_compare1 = time_distributed(s1_combined1, compare_layers_g)
    s1_compare2 = time_distributed(s1_combined2, compare_layers_g)
    s1_compare = Average()([s1_compare1, s1_compare2])
    
    s2_compare1 = time_distributed(s2_combined1, compare_layers_g)
    s2_compare2 = time_distributed(s2_combined2, compare_layers_g)
    s2_compare = Average()([s2_compare1, s2_compare2])

    # Aggregate
    q1_encoded = Concatenate()([q1_encoded, q1_compare, s1_compare])
    q2_encoded = Concatenate()([q2_encoded, q2_compare, s2_compare])
    
    aggreate_rnn = CuDNNGRU(rnn_size, return_sequences=True)    
    q1_aggreated = aggreate_rnn(q1_encoded)
    q1_aggreated = Dropout(rnn_dropout)(q1_aggreated)
    q2_aggreated = aggreate_rnn(q2_encoded)
    q2_aggreated = Dropout(rnn_dropout)(q2_aggreated)
    
    # Pooling
    q1_rep = apply_multiple(q1_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(),])
    q2_rep = apply_multiple(q2_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(),])

    # Dense meta featues
    meta_densed = Highway(activation='relu')(meta_features)
    meta_densed = Dropout(model_config.META_FEATURES_DROPOUT)(meta_densed)

    # Classifier
    q_diff = substract(q1_rep, q2_rep)
    q_multi = Multiply()([q1_rep, q2_rep])
    if with_meta_features:
        h_all1 = Concatenate()([q1_rep, q2_rep, q_diff, q_multi, meta_densed])
        h_all2 = Concatenate()([q2_rep, q1_rep, q_diff, q_multi, meta_densed])       
    else:
        h_all1 = Concatenate()([q1_rep, q2_rep, q_diff, q_multi,])
        h_all2 = Concatenate()([q2_rep, q1_rep, q_diff, q_multi,])

    h_all1 = Dropout(0.5)(h_all1)
    h_all2 = Dropout(0.5)(h_all2)

    dense = Dense(256, activation='relu')
    
    h_all1 = dense(h_all1)
    h_all2 = dense(h_all2)
    h_all = Average()([h_all1, h_all2])
    
    out = Dense(out_size, activation='sigmoid')(h_all)
    
    model = Model(inputs=[q1, q2, q1_c, q2_c, meta_features], outputs=out)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
    metrics=['accuracy'])
    return model