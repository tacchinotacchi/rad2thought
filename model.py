import numpy as np
import tensorflow as tf
import layers
import manage_dataset as dataset
import random
import time

# define training, test set

def permute_languages(example):
    permutation = (1, 0)
    return tuple([example[i] for i in permutation])

perm_vocab_size = permute_languages((dataset.en_charset, dataset.jp_charset))
source_vocab_size = len(perm_vocab_size[0])
target_vocab_size = len(perm_vocab_size[1])

def encode(source, target):
    source = np.concatenate((
        np.array([source_vocab_size + 1], dtype = np.int64),
        source,
        np.array([source_vocab_size + 2], dtype = np.int64)))
    target = np.concatenate((
        np.array([target_vocab_size + 1], dtype = np.int64),
        target,
        np.array([target_vocab_size + 2], dtype = np.int64)))
    return (source, target)

def make_batches(input_set, size):
    for index in range(0, len(input_set), size):
        batch = input_set[index:index + size]
        source_batch = [d[0] for d in batch]
        target_batch = [d[1] for d in batch]
        max_size_source = max([d.size for d in source_batch])
        max_size_target = max([d.size for d in target_batch])
        source_batch = [np.pad(d, ((0, max_size_source - d.size)), 'constant') for d in source_batch]
        source_batch = np.array(source_batch)
        target_batch = [np.pad(d, ((0, max_size_target - d.size)), 'constant') for d in target_batch]
        target_batch = np.array(target_batch)
        ep_mask, dp_mask, ds_mask = layers.create_masks(source_batch, target_batch[:, :-1])
        yield (source_batch, target_batch, ep_mask, dp_mask, ds_mask)

perm_dataset = [permute_languages(ex) for ex in dataset.token_dataset]
dataset_size = len(perm_dataset)
train_examples, test_examples = perm_dataset[:int(dataset_size * 0.80)], perm_dataset[int(dataset_size * 0.80):]
train_encoded = [encode(*example) for example in train_examples]
test_encoded = [encode(*example) for example in test_examples]

# define model, optimizer
d_model = 128
num_layers = 4
d_internal = 512
num_heads = 8
dropout_rate = 0.1

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
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

def loss(model, logits, labels):
    return tf.losses.sparse_softmax_cross_entropy(
        logits = logits, labels = labels)

def train_step(inp, tar, ep_mask, dp_mask, ds_mask):
    tar_feed = tar[:, :-1]
    tar_compare = tar[:, 1:]
    with tf.GradientTape() as tape:
        predictions = transformer(inp, tar_feed, ep_mask, dp_mask, ds_mask, training = True)
        loss_val = loss(transformer, predictions, tar_compare)
    gradients = tape.gradient(loss_val, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    epoch_loss_avg(loss_val)
    epoch_accuracy(tar_compare[:, :, tf.newaxis], predictions)

# training loop
for epoch in range(5):
    start = time.time()
    epoch_loss_avg.reset_states()
    epoch_accuracy.reset_states()
    for inp, tar, ep_mask, dp_mask, ds_mask in make_batches(train_encoded, 64):
        train_step(inp, tar, ep_mask, dp_mask, ds_mask)
    path = checkpoint_manager.save()
    print("Saved checkpoint to %s" % path)
    print("Epoch %d Loss %f Accuracy %f" % (epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result()))
    print("Time taken: %d" % (time.time() - start))
