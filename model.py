import numpy as np
import tensorflow as tf
import layers
import random
import time
import opensubs

def make_batches_generator(input_set, size):
    while True:
        opensubs.shuffle_dataset(input_set)
        for index in range(0, len(input_set), size):
            source_batch = input_set["ja_token"][index:index + size]
            target_batch = input_set["en_token"][index:index + size]
            max_size_source = max([d.size for d in source_batch])
            max_size_target = max([d.size for d in target_batch])
            source_batch = [np.pad(d, ((0, max_size_source - d.size)), 'constant') for d in source_batch]
            source_batch = np.array(source_batch)
            target_batch = [np.pad(d, ((0, max_size_target - d.size)), 'constant') for d in target_batch]
            target_batch = np.array(target_batch)
            # input batch is inp, tar_feed
            input_batch = [source_batch, target_batch[:, :-1]]
            # ref batch is tar_compare
            ref_batch = target_batch[:, 1:, np.newaxis]
            yield (input_batch, ref_batch)

def make_batches(input_set, size):
    return make_batches_generator(input_set, size), len(range(0, len(input_set["en"]), size))

source_vocab_size = len(opensubs.ja_charset)
target_vocab_size = len(opensubs.en_charset)
opensubs.shuffle_dataset(opensubs.dataset)
train_examples, test_examples = opensubs.split_dataset(opensubs.dataset, 0.90)

# define model, optimizer
d_model = 128
num_layers = 4
d_internal = 512
num_heads = 8
dropout_rate = 0.1

transformer = layers.Transformer(source_vocab_size + 2, target_vocab_size + 2,
    num_layers, d_model, d_internal, num_heads, dropout_rate, max_sequence = 2000)
# (..., target_seq_length, target_vocab_size)

def custom_loss(y_true, y_pred):
    # y_true (..., seq_length, 1)
    # y_pred (..., seq_length, vocab_size)
    is_masked = tf.cast(tf.math.equal(y_true, 0), dtype=tf.float32)[:, :, 0]
    loss = tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred, axis=-1, from_logits=True)
    return tf.reduce_mean(loss * (1 - is_masked))

def custom_accuracy(y_true, y_pred):
    # y_true (..., seq_length, 1)
    # y_pred (..., seq_length, vocab_size)
    y_true = y_true[:, :, 0]
    y_pred = tf.argmax(y_pred, axis=-1)
    is_masked = tf.cast(tf.math.equal(y_true, 0), dtype=tf.float32)
    accuracy = tf.cast(tf.math.equal(y_true, y_pred), dtype=tf.float32)
    return tf.reduce_mean(accuracy * (1 - is_masked))

transformer.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
    loss=custom_loss, metrics=[custom_accuracy])

history = []
tf.keras.backend.set_learning_phase(1)
gen, steps = make_batches(train_examples, 2000)
step = transformer.fit(gen, epochs=5, steps_per_epoch=steps, initial_epoch=0)
history.append(step)
