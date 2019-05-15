import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
tfe = tf.contrib.eager

import layers
import manage_dataset as dataset
import random
import time

d_model = 128
num_layers = 4
d_internal = 512
num_heads = 8
source_vocab_size = len(dataset.en_charset) + 3
target_vocab_size = len(dataset.jp_charset) + 3
dropout_rate = 0.1

def encode(source, target):
    source = [source_vocab_size + 1] + source + [source_vocab_size + 2]
    target = [target_vocab_size + 1] + target + [target_vocab_size + 2]
    return (source, target)

def loss(model, logits, labels):
    return tf.losses.sparse_softmax_cross_entropy(
        logits = logits, labels = labels)

def train_step(inp, tar):
    tar_compare = tar[:, 1:]
    tar_feed = tar[:, :-1]
    encoder_padding_mask, decoder_padding_mask, decoder_self_mask = layers.create_masks(inp, tar_feed)

    with tf.GradientTape() as tape:
        predictions = transformer(inp, tar_feed, encoder_padding_mask, decoder_padding_mask,
            decoder_self_mask, training = 1)
        loss_val = loss(transformer, predictions, tar_compare)

    gradients = tape.gradient(loss_val, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    epoch_loss_avg(loss_val)
    epoch_accuracy(tar_compare[:, :, tf.newaxis], predictions)

def pad_batch(batch_list):
    max_len = max([len(ex) for ex in batch_list])
    for example in batch_list:
        example.extend([0] * (max_len - len(example)))
    return batch_list

def split_batches(examples, size):
    for i in range(0, len(examples), size):
        # turn examples[i:i+size] into tensor
        batch_list = examples[i:i+size]
        source_list = [ex[0] for ex in batch_list]
        target_list = [ex[1] for ex in batch_list]
        source_list, target_list = pad_batch(source_list), pad_batch(target_list)
        source_list = tf.convert_to_tensor(source_list, dtype = tf.int64)
        target_list = tf.convert_to_tensor(target_list, dtype = tf.int64)
        yield (source_list, target_list)

# define training, test set
random.shuffle(dataset.token_dataset)
dataset_size = len(dataset.token_dataset)
train_examples, test_examples = dataset.token_dataset[:int(dataset_size * 0.80)], dataset.token_dataset[int(dataset_size * 0.80):]
train_encoded = [encode(*example) for example in train_examples]
test_encoded = [encode(*example) for example in test_examples]
random.shuffle(train_encoded)
training_batches = split_batches(train_encoded, 32)

# define model, optimizer
transformer = layers.Transformer(source_vocab_size + 3, target_vocab_size + 3,
    num_layers, d_model, d_internal, num_heads, dropout_rate, max_sequence = 2000)
# (..., target_seq_length, target_vocab_size)
optimizer = tf.train.AdamOptimizer(
    learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999
)
global_step = tf.Variable(0)
epoch_loss_avg = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
checkpoint_path = "./checkpoints/train"
checkpoint = tf.train.Checkpoint(model = transformer, optimizer = optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep = 5)
# training loop
for epoch in range(20):
    start = time.time()
    epoch_loss_avg.reset_states()
    epoch_accuracy.reset_states()
    for inp, tar in training_batches:
        train_step(inp, tar)
    if (epoch + 1) % 5 == 0:
        path = checkpoint_manager.save()
        print("Saved checkpoint to %s" % path)
    print("Epoch %d Loss %f Accuracy %f" % (epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result()))
    print("Time taken: %d" % (time.time() - start))
