import numpy as np
import tensorflow as tf

class Predictor:
    def __init__(self, model, source):
        self.model = model
        self.source = source
    def __call__(self, alive_seq):
        source_tile = np.tile(self.source, (alive_seq.shape[0], 1))
        scores =  self.model([source_tile, alive_seq], training=False)[:, -1, :]
        return tf.nn.log_softmax(scores, axis=-1)

def grow_alive(alive_seq, alive_scores, logits, beam_size, tokens_size):
    logits = np.reshape(logits, -1)
    sorted_index = np.argsort(logits)[-(beam_size * 2):]
    scores = np.take(logits, sorted_index)
    beam_index, vocab_index = sorted_index // tokens_size, sorted_index % tokens_size
    alive_seq = np.take(alive_seq, beam_index, axis = 0)
    alive_seq = np.concatenate([alive_seq, vocab_index[:, np.newaxis]], axis = -1)
    alive_scores = np.take(alive_scores, beam_index, axis = 0)
    alive_scores = alive_scores + scores
    return (alive_seq, alive_scores)

def grow_finished(alive_seq, alive_scores, finished_seq, finished_scores, length_norm, end_token):
    finished_mask = np.equal(alive_seq[:, -1], end_token)
    finished_index = np.nonzero(finished_mask)[0]
    just_finished_seq = np.take(alive_seq, finished_index, axis = 0)
    just_finished_scores = np.take(alive_scores, finished_index, axis = 0)
    alive_seq = np.delete(alive_seq, finished_index, axis = 0)
    alive_scores = np.delete(alive_scores, finished_index, axis = 0)
    just_finished_scores = just_finished_scores * length_norm(just_finished_seq.shape[-1])
    zeros = np.zeros((finished_seq.shape[0], 1))
    finished_seq = np.concatenate([finished_seq, zeros], axis = 1)
    finished_seq = np.concatenate([finished_seq, just_finished_seq], axis=0)
    finished_scores = np.concatenate([finished_scores, just_finished_scores], axis = 0)
    return finished_seq, finished_scores, alive_seq, alive_scores

def beam_search(predictor, beam_size, vocab_size):
    start_token = vocab_size
    end_token = vocab_size + 1
    tokens_size = vocab_size + 2
    alive_seq = np.full((1, 1), start_token, dtype = np.int64)
    alive_scores = np.full((1), 0, dtype = np.float32)
    finished_seq = np.empty((0, 1), dtype = np.int64)
    finished_scores = np.empty((0), dtype = np.float32)
    # (token_list, score)
    steps = 0
    while (steps < 80) and alive_seq.shape:
        logits = predictor(alive_seq)
        alive_seq, alive_scores = grow_alive(alive_seq,
            alive_scores, logits, beam_size, tokens_size)
        finished_seq, finished_scores, alive_seq, alive_scores = grow_finished(alive_seq,
           alive_scores, finished_seq, finished_scores, lambda l: 1.0/l, end_token)
        if (alive_seq.shape[0] > beam_size):
            alive_seq = alive_seq[-beam_size:, :]
        steps += 1
    finished_sorted = np.argsort(finished_scores)
    finished_seq = np.take(finished_seq, finished_sorted, axis=0)
    return finished_seq, alive_seq
