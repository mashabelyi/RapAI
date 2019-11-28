# RapAI

Data Mining & Analytics (INFO 254/ DATA 144) Fall 2019 Final Project.

Training a language model to generate rap lyrics. Work in progress!!

## Contents
`model.py`: keras models defined

`train_lm.py`: run to initiate training

`Preprocess-Rap.ipynb`: jupyter notebook to pre-process and generate train/val/test splits

`data/rap_max100_10`: train/val/test split of rap lyrics dataset containing 20 artists, 100 songs form each artist.

## Usage

### Requiremens
- Python 3
- [Keras](https://keras.io/)
- [nltk](https://www.nltk.org/)

### Training
```
python train_lm.py --name my_model --epoch 20 --batch_size 256 --lstm_size 100 --emb_trainable --domain_vocab
```

Running `train_lm.py` trains the model for the specified number of epochs and saves output files in the model directory (model directory named with the --name argument)

**Required parameters:**

`--data`: path to data folder

`--name`: model will be saved under this name

**Optional parameters:**

`--batch_size`: batch size for training (default: 128)

`--epoch`: number of epochs to train (default: 10)

`--lstm_size`: size of LSTM cells (default: 100)

`--emb_trainable`: Flag for updating pre-trained word embeddings (default: False if not specified)

`--vocab_size`: Vocabulary size (default: 50000)

`--domain_vocab`: if specified, will use up to `--domain_vocab_n` domain-specific words

`--domain_vocab_n`: number of most frequnt domain specific words that are not covered by Glove to include in the model. Ignored if `--domain_vocab` is not specified. (default: 1000)

`--n_train`: number of training samples to read (default: will read all if not specified)

`--n_val`: number of validation samples to read (default: will read all if not specified)


## Evaluation Metrics

**accuracy**: percentage of correct token predictions on test set. Higher is better.

**top5 accuracy**: percentage of correct tokens present in model's top 5 predictions. Higher is better.

**top10 accuracy**: percentage of correct tokens present in model's top 10 predictions. Higher is better.

**perplexity**: [perplexity](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/) of model measured on the test set. Calculated as exp(cross_entropy_loss). Perplexity is a measure of surprisal -- it measures how well our model can predict the test set. Lower perplexity = less surpisal = better.

## Data Pre-Processing

## Results


