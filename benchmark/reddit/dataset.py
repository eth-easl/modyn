import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

import torch
import spacy
from torchtext import data
from torchtext.vocab import Vectors

# TODO(#28): Integrate into Modyn: Inherit from OnlineDataset
def get_reddit_dataset():
	# Load dataset
	df = pd.read_csv('./dataset/reddit_posts.csv')
	w2v_file = './dataset/glove.6B.50d.txt'

	# Build vocab
	NLP = spacy.load('en_core_web_sm')
	tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
	TEXT = data.Field(sequential=True, tokenize=tokenizer)
	TEXT.build_vocab(df['text'], vectors=Vectors(w2v_file))

	# Split into train and test and extract y
	test_idx = round(0.8 * df.shape[0])

	text_vocab = lambda tokens : [TEXT.vocab[token] for token in tokens]
	text_pipeline = lambda x : torch.tensor(text_vocab(tokenizer(x)))

	train_x = list(map(text_pipeline, df['text'].values[:test_idx]))
	len_train = max(len(x) for x in train_x)
	train_x = list(map(lambda t : torch.nn.functional.pad(t, (0, len_train - len(t))), train_x))
	train_y = torch.tensor(df['label'].values[:test_idx])
	test_x = list(map(text_pipeline, df['text'].values[test_idx:]))
	test_y = torch.tensor(df['label'].values[test_idx:])
	len_test = max(len(x) for x in test_x)
	test_x = list(map(lambda t : torch.nn.functional.pad(t, (0, len_test - len(t))), test_x))


	return {
		'train': RedditDataset(train_x, train_y),
		'test': RedditDataset(test_x, test_y),
		'vocab': TEXT.vocab,
		'word_embeddings': TEXT.vocab.vectors
	}


class RedditDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

	def is_task_based(self):
		return False
