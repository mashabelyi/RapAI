"""

Train simple model
=================================
python3 train_lm.py --name my_model --epoch 20 --batch_size 256 --lstm_size 100 --emb_trainable --domain_vocab


To train with artist embeddings
=================================
python3 train_lm.py --name lstm100_artist100_hidden100 --epoch 50 --batch_size 256 --lstm_size 100 --emb_trainable --domain_vocab --data data/rap_max100_10/ --train_artists


"""
from __future__ import print_function

import os, json, re
from nltk import word_tokenize, regexp_tokenize
from collections import Counter
import numpy as np
import functools
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger
from math import sqrt

from model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # disable annoying tensorflow "deprecated" messages

def parse_args():
	# Parameters
	# ==================================================
	parser = ArgumentParser("Lang Model", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

	parser.add_argument("--data", default="data/rap_10", help="Data folder", required=True)
	parser.add_argument("--name", default="model", help="Model Name", required=True)

	parser.add_argument("--batch_size", default=128, type=int, help="Batch Size")
	parser.add_argument("--n_train", default=-1, type=int, help="Num training samples")
	parser.add_argument("--n_val", default=-1, type=int, help="Nuum validation samples")
	parser.add_argument("--epoch", default=10, type=int, help="Number of training epochs")
	parser.add_argument("--lstm_size", default=50, type=int, help="LSTM size")
	parser.add_argument("--emb_trainable", default=False, help="Flag to train word embeddings", action="store_true")
	parser.add_argument("--vocab_size", default=50000, help="Vocabulary size")
	parser.add_argument("--domain_vocab", default=False, help="Flag to add domain vocab", action="store_true")
	parser.add_argument("--domain_vocab_n", default=1000, type=int, help="Max number of domain vocab to add")

	parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate")


	parser.add_argument("--train_artists", default=False, help="Flag to learn artist embeddings", action="store_true")
	parser.add_argument("--artist_dim", default=100, help="Dimension of learned artist embeddings")
	parser.add_argument("--hidden_dim", default=100, help="Dimension of hidden dense layer")

	args = parser.parse_args()
	print(args)

	return args


def read_data(file, num=-1):
	artists = []
	seqs = []
	targets = []
	with open(file, 'r') as f:
		i = 0
		for line in f:
			line = line.strip().split('\t')
			artists.append(int(line[0]))
			seqs.append(line[1].split())
			targets.append(line[2])

			i += 1
			if num > 0 and i >= num:
				break

	return (np.array(artists), seqs, targets)

def read_map(file):
	"""
	read file mapping e.g. artist2id.tsv
	"""
	a2b = {}
	with open(file, 'r') as f:
		for line in f:
			line = line.strip().split('\t')
			a2b[line[0]] = line[1]
	return a2b


def load_embeddings(filename, max_vocab_size, emb_dim):

	vocab={}
	embeddings=[]
	with open(filename) as file:
	    
	    cols=file.readline().split(" ")
	    num_words=int(cols[0])
	    size=int(cols[1])
	    embeddings.append(np.zeros(size))  # 0 = 0 padding if needed
	    embeddings.append(np.random.uniform(-1,1,emb_dim))  # 1 = UNK
	    embeddings.append(np.random.uniform(-1,1,emb_dim))  # 1 = <BR>
	    vocab["<PAD>"]=0
	    vocab["<UNK>"]=1
	    vocab["<BR>"]=2
	    
	    for idx,line in enumerate(file):

	        if idx+3 >= max_vocab_size:
	            break

	        cols=line.rstrip().split(" ")
	        val=np.array(cols[1:])
	        word=cols[0]
	        
	        embeddings.append(val)
	        vocab[word]=idx+3

	return np.array(embeddings), vocab, size

def tok_to_id(tok, vocab):
	if tok in vocab:
	    return vocab[tok]

	if tok[-1]=='n' and tok+'g' in vocab:
	    # 'growin' -> 'growing'
	    # 'obeyin' -> 'obeying'
	    return vocab[tok+'g']

	return vocab['<UNK>']

def check_pretrained_coverage(vocab, tokens, vocab_map={}):
	in_vocab = set()
	out_vocab = set()
	all_vocab = set()

	in_count = 0
	out_count = 0
	all_count = 0

	out_counter = Counter()
	for tok in tokens:
	    
	    if tok in vocab_map:
	        tok = vocab_map[tok]
	    
	    if tok in vocab or tok == '<BR>':
	        in_vocab.add(tok)
	        in_count += tokens[tok]            
	    elif tok[-1]=='n' and tok+'g' in vocab:
	        # 'growin' -> 'growing'
	        # 'obeyin' -> 'obeying'
	        in_vocab.add(tok)
	        in_count += tokens[tok]
	    else:
	        out_counter[tok] += tokens[tok]
	        out_vocab.add(tok)
	        out_count += tokens[tok]
	    all_vocab.add(tok)
	    all_count += tokens[tok]
	    
	print("{:.2%} of unique tokens covered".format(len(in_vocab)/len(all_vocab)))
	print("{:.2%} of corpus covered".format(in_count/all_count))

	return in_vocab, out_vocab, all_vocab, out_counter

class Perplexity(Callback):
	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch,logs)
		perplexity = K.eval(K.exp(logs['loss']))
		val_perplexity = K.eval(K.exp(logs['val_loss']))

		logs['perplexity'] = perplexity
		logs['val_perplexity'] = val_perplexity

		print(" - perplexity: {} - val_perplexity: {}".format(perplexity, val_perplexity))

if __name__ == '__main__':
	
	# PARSE INPUT ARGUMENTS
	args = parse_args()

	# MODEL FOLDER
	if not os.path.exists(args.name):
		os.makedirs(args.name)
	model_dir = args.name
	# save input arguments in model folder
	with open(os.path.join(model_dir, 'args.json'), 'w') as f:
		json.dump(args.__dict__, f, indent=2)


	print("LOADING DATA")
	a_train, x_train, y_train = read_data(os.path.join(args.data, 'train.tsv'), num=args.n_train)
	a_val, x_val, y_val = read_data(os.path.join(args.data, 'val.tsv'), num=args.n_val)
	a_test, x_test, y_test = read_data(os.path.join(args.data, 'test.tsv'), num=1)

	print(" - loaded {} training samples".format(len(a_train)))
	print(" - loaded {} validation samples".format(len(a_val)))
	
	if args.train_artists:
		artist2id = read_map(os.path.join(args.data, 'artist2id.tsv'))
		print(" - found {} artists".format(len(artist2id)))

	# LOAD EMBEDDINGS
	print()
	print("loading embeddings")
	vocab_dim = 100
	emb, tok2id, size = load_embeddings('data/glove/glove.6B/glove.6B.100d.w2v', args.vocab_size, vocab_dim)

	rap_vocab = Counter([t for sample in x_train for t in sample])
	print("\nVOCABULARY")
	print("train vocab size = {}".format(len(rap_vocab)))
	in_, out_, all_, out_counter = check_pretrained_coverage(tok2id, rap_vocab)

	# DOMAIN VOCAB
	# the rap/poetry corpus contains a lot of domain-specific words that 
	# are not covered by the glove embeddings for the most frequent words.
	# However, including the in-domain vocab might help the model perform better
	# Here we add 
	if args.domain_vocab:
		print("adding domain vocab", end=" ")
		domain_words = [x[0] for x in out_counter.most_common(min(args.domain_vocab_n, len(out_counter)))]
		idx = len(tok2id)
		domain_emb = []
		for w in domain_words:
			tok2id[w] = idx
			domain_emb.append(np.random.uniform(-1,1,vocab_dim)) # randomly intialize embedding for each word
			idx += 1
		print("- {} words added".format(len(domain_emb)))
		emb = np.concatenate((emb, np.array(domain_emb)))

		in_, out_, all_, out_counter = check_pretrained_coverage(tok2id, rap_vocab)

	# save the new tok2id mapping in model folder
	with open(os.path.join(model_dir, 'tok2id.tsv'), 'w') as f:
		for k in tok2id:
			f.write("{}\t{}\n".format(k, tok2id[k]))


	# TOKENS -> IDS
	x_train_id = np.array([[tok_to_id(t, tok2id) for t in seq] for seq in x_train])
	y_train_id = np.array([tok_to_id(t, tok2id) for t in y_train])
	x_val_id = np.array([[tok_to_id(t, tok2id) for t in seq] for seq in x_val])
	y_val_id = np.array([tok_to_id(t, tok2id) for t in y_val])

	# CALLBACKS
	best_acc = ModelCheckpoint(os.path.join(model_dir, 'weights_best_acc.hdf5'), monitor='val_top10_acc', verbose=0, save_best_only=True, mode='max')
	best_ppx = ModelCheckpoint(os.path.join(model_dir, 'weights_best_ppx.hdf5'), monitor='val_perplexity', verbose=0, save_best_only=True, mode='min')
	stop_early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
	csv_logger = CSVLogger(os.path.join(model_dir, 'training_log.csv'), append=False)

	# MODEL
	print("\nINITIALIZEING MODEL")
	if args.train_artists:
		model = get_lstm_source(emb, lstm_size=args.lstm_size, emb_trainable=args.emb_trainable, dropout_rate=args.dropout_rate,  
			source_n=len(artist2id), source_dim=args.artist_dim, dense_dim=args.hidden_dim)

		# TRAIN
		model.fit([x_train_id, a_train], y_train_id, 
		        validation_data=([x_val_id, a_val], y_val_id),
		        epochs=args.epoch, batch_size=args.batch_size,
		        callbacks=[Perplexity(), best_acc, best_ppx, csv_logger])

	else:
		model = get_simple_lstm(emb, lstm_size=args.lstm_size, emb_trainable=args.emb_trainable, dropout_rate=args.dropout_rate)
		print(model.summary())
		print("\n")

		# TRAIN
		model.fit(x_train_id, y_train_id, 
		        validation_data=(x_val_id, y_val_id),
		        epochs=args.epoch, batch_size=args.batch_size,
		        callbacks=[Perplexity(), best_acc, best_ppx, csv_logger])




