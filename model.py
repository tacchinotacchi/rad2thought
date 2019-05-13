import numpy as np
import tensorflow as tf
import manage_dataset as dataset
import random

d_model = 128
num_layers = 4
d_internal = 512
num_heads = 8
source_vocab_size = len(dataset.jp_charset) + 3
target_vocab_size = len(dataset.en_charset) + 3
dropout_rate = 0.1

def encode(source, target):
	source = [source_vocab_size + 1] + source + [source_vocab_size + 2]
	target = [target_vocab_size + 1] + target + [target_vocab_size + 2]

random.shuffle(dataset.token_dataset)
dataset_size = len(dataset.token_dataset)
train_examples, test_examples = dataset.token_dataset[:int(dataset_size * 0.80)], dataset.token_dataset[int(dataset_size * 0.80):]
train_encoded = train_examples.map(encode)
test_encoded = test_examples.map(encode)

