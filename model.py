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

def make_batches(input_set, size):
    random.shuffle(input_set)
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
        yield (source_batch, target_batch)

perm_vocab_size = permute_languages((dataset.en_charset, dataset.jp_charset))
source_vocab_size = len(perm_vocab_size[0])
target_vocab_size = len(perm_vocab_size[1])

perm_dataset = [permute_languages(ex) for ex in dataset.token_dataset]
dataset_size = len(perm_dataset)
train_examples, test_examples = perm_dataset[:int(dataset_size * 0.90)], perm_dataset[int(dataset_size * 0.90):]

# define model, optimizer
d_model = 128
num_layers = 4
d_internal = 512
num_heads = 8
dropout_rate = 0.1

transformer = layers.Transformer(source_vocab_size + 3, target_vocab_size + 3,
    num_layers, d_model, d_internal, num_heads, dropout_rate, max_sequence = 2000)
# (..., target_seq_length, target_vocab_size)

# training loop
#for epoch in range(10):
#    start = time.time()
#    for b_inp, b_tar in make_batches(train_examples, 2048):
#        _, loss_val = sess.run([train_op, model_loss], {
#            inp: b_inp, tar_feed: b_tar[:, :-1], tar_compare: b_tar[:, 1:]
#        })
#    if (epoch + 1) % 5 == 0:
#       path = checkpoint.save(checkpoint_path, sess)
#       print("Saved checkpoint to %s" % path)
#    print("Loss: %f" % loss_val)
#    print("Time taken: %d" % (time.time() - start))
