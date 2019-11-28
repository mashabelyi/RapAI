import os
import keras
import numpy as np
from keras.layers import Dense, Input, Embedding, Lambda, Layer, Multiply, Dropout, Dot, Bidirectional, LSTM, concatenate, Flatten
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from scipy.stats import norm
from math import sqrt
import functools

# disable annoying tensorflow "deprecated" messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267

    NOTE: This function outputs the wrong value for perplexity.
    The cross_entropy calculation is correct, but the output of K.exp(cross_entropy) is wrong.
    The error is documented here: https://stackoverflow.com/questions/57701511/language-model-computes-incorrect-perplexity-using-keras-backend-pow?noredirect=1&lq=1

    As a workaround, I am using the class Perplexity(Callback) workaround from the link above.
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False) 
    perplexity = K.exp(cross_entropy)
    return perplexity

top10_acc = functools.partial(keras.metrics.sparse_top_k_categorical_accuracy, k=10)
top10_acc.__name__ = 'top10_acc'
top5_acc = functools.partial(keras.metrics.sparse_top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'

class AttentionLayerMasking(Layer):

	def __init__(self, output_dim, **kwargs):
	    self.output_dim = output_dim
	    super(AttentionLayerMasking, self).__init__(**kwargs)


	def build(self, input_shape):
	    input_embedding_dim=input_shape[-1]
	    
	    self.kernel = self.add_weight(name='kernel', 
	                        shape=(input_embedding_dim,1),
	                        initializer='uniform',
	                        trainable=True)
	    super(AttentionLayerMasking, self).build(input_shape)

	def compute_mask(self, input, input_mask=None):
	    return None

	def call(self, x, mask=None):
	    
	    # dot product 
	    x=K.dot(x, self.kernel)
	    # exponentiate
	    x=K.exp(x)
	    
	    # zero out elements that are masked
	    if mask is not None:
	        mask = K.cast(mask, K.floatx())
	        mask = K.expand_dims(mask, axis=-1)
	        x = x * mask
	    
	    # normalize by sum
	    x /= K.sum(x, axis=1, keepdims=True)
	    x=K.squeeze(x, axis=2)

	    return x

	def compute_output_shape(self, input_shape):
	    return (input_shape[0], input_shape[1])


def get_simple_lstm(embeddings, lstm_size=25, dropout_rate=0.1, emb_trainable=True):
	vocab_size, word_embedding_dim=embeddings.shape 
	word_sequence_input = Input(shape=(None,), dtype='int32')

	word_embedding_layer = Embedding(vocab_size,
	                                word_embedding_dim,
	                                weights=[embeddings],
	                                trainable=emb_trainable)


	# input - embeddings
	embedded_sequences = word_embedding_layer(word_sequence_input)
	# lstm layer
	lstm_output = LSTM(lstm_size, 
	                   return_sequences=False, 
	                   activation='tanh', 
	                   dropout=dropout_rate)(embedded_sequences)
	# + dense layer
	# dense_output = Dense(128, activation='tanh')(lstm_output)
	# + droupout
	# seq_representation = Dropout(dropout_rate)(dense_output)
	# final output - softmax over all vocabulary
	x=Dense(vocab_size, activation="softmax")(lstm_output)

	model = Model(inputs=word_sequence_input, outputs=x)
	model.compile(loss='sparse_categorical_crossentropy', 
		optimizer='adam', # defaults keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
		metrics=['sparse_categorical_accuracy', top5_acc, top10_acc])

	return model



def get_lstm_source(embeddings, lstm_size=25, dropout_rate=0.1, emb_trainable=True, source_n=10, source_dim=50, dense_dim=25):
    
	# word embeddings
	vocab_size, word_embedding_dim=embeddings.shape
	word_embedding_layer = Embedding(vocab_size,
	                                word_embedding_dim,
	                                weights=[embeddings],
	                                trainable=emb_trainable,
	                                name='word_emb')
	# source embeddings
	source_embedding_layer = Embedding(source_n, 
	                                   source_dim, 
	                                   input_length=1, 
	                                   trainable=True,
	                                   name='source_emb')

	# inputs
	word_sequence_input = Input(shape=(None,), dtype='int32')
	source_input = Input(shape=(1,), dtype='int32')

	# build model
	embedded_sequences = word_embedding_layer(word_sequence_input) # (batch_size x seq_length x embedding_dim)
	embedded_sources = Flatten()(source_embedding_layer(source_input)) # (batch_size x source_dim)

	# pass sequences through lstm
	lstm_output = LSTM(lstm_size, 
	                   return_sequences=False, 
	                   activation='tanh', 
	                   dropout=dropout_rate,
	                   name='lstm')(embedded_sequences)

	# concat with source embeddings
	combined = concatenate([embedded_sources, lstm_output])

	# Dense layer over concat -> predict
	combined = Dropout(dropout_rate)(Dense(dense_dim, activation="tanh", name='dense')(combined))
	x=Dense(vocab_size, activation="softmax", name='predict')(combined)

	# compile model
	model = Model(inputs=[word_sequence_input, source_input], outputs=x)
	model.compile(loss='sparse_categorical_crossentropy', 
	              optimizer='adam', 
	              metrics=['sparse_categorical_accuracy', top5_acc, top10_acc])

	return model

def simple_attention_model(embeddings, attn_size=25):
	vocab_size, word_embedding_dim=embeddings.shape 
	word_sequence_input = Input(shape=(None,), dtype='int32')

	word_embedding_layer = Embedding(vocab_size,
	                                word_embedding_dim,
	                                weights=[embeddings],
	                                trainable=True)


	# input - embeddings
	embedded_sequences = word_embedding_layer(word_sequence_input)

	# reduce embedding dimensionality
	attention_input=Dense(attn_size, activation='tanh')(embedded_sequences)

	attention_output = AttentionLayerMasking(word_embedding_dim, name="attention")(attention_input)

	# now let's multiply those attention weights by original inputs to get a weighted average over them
	document_representation = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=1), 
	                                 name='dot')([attention_output,attention_input])

	x=Dense(vocab_size, activation="softmax")(document_representation)

	model = Model(inputs=word_sequence_input, outputs=x)
	model.compile(loss='sparse_categorical_crossentropy', 
	              optimizer='adam', # defaults keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
	              metrics=['sparse_categorical_accuracy', top5_acc, top10_acc])

	return model	