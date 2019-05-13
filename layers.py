import tensorflow as tf
import numpy as np

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.last_linear = tf.keras.layers.Dense(d_model)
    def scaled_dp_attention(self, q, k, v, mask = None):
        qk_matmul = tf.matmul(q, k, transpose_b = True)
        dk = tf.cast(tf.shape(qk_matmul)[-1], tf.float32)
        qk_matmul = qk_matmul / dk
        if mask is not None:
            qk_matmul += (mask * 1e-9)
        attention_logits = tf.nn.softmax(qk_matmul, axis = -1) # (..., seq_length_q, seq_length_v)
        attention_outputs = tf.matmul(attention_logits, v) # (..., seq_length_q, depth)
        return attention_outputs
    def reshape_heads(self, v):
        batch_size = tf.shape(v)[0]
        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(v, (0, 2, 1, 3)) # (batch_size, seq_length, num_heads, depth)
    def call(self, q, k, v, mask = None):
#	q (..., seq_lenght_q, d_model)
#	k (..., seq_length_v, d_model)
#	v (..., seq_length_v, d_model)
#	mask (..., seq_length_v) broadcast to (..., seq_length_q, seq_length_v)
        q, k, v = self.wq(q), self.wk(k), self.wv(v) # (..., seq_length, d_model)
        q, k, v = self.reshape_heads(q), self.reshape_heads(k), self.reshape_heads(v) # (...,  num_heads, seq_length, depth)
        attention_outputs = self.scaled_dp_attention(q, k, v, mask) # (..., num_heads, seq_length_q, depth)
        attention_outputs = tf.transpose(attention_outputs, (0, 2, 1, 3)) # (..., seq_length_q, num_heads, depth)
        attention_outputs = tf.reshape(attention_outputs, (tf.shape(attention_outputs)[0], -1, self.d_model)) # (..., seq_length_q, d_model)
        attention_outputs = self.last_linear(attention_outputs) # (..., seq_length_q, d_model)
        return attention_outputs

def feed_forward_network(d_model, d_internal):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_internal, activation = 'relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncodeLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_internal, num_heads = 2, dropout_rate = 0.1):
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.self_attention = MultiHeadAttention(self.d_model, self.num_heads)
        self.feed_forward = feed_forward_network(d_model, d_internal)
        self.layernorm1 = tf.keras.layers.BatchNormalization()
        self.layernorm2 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
    def call(self, x, mask, training):
#	x (..., seq_lenght, d_model)
#	mask (..., seq_length_v) broadcast to (..., seq_length_q, seq_length_v)
        attention_outputs = self.self_attention(x, x, x, mask) # (..., seq_length, d_model)
        x = self.layernorm1(self.dropout1(x + attention_outputs, training))
        ff = self.feed_forward(x) # (..., seq_lenght, d_model)
        x = self.layernorm2(self.dropout2(x + ff, training))
        return x

class DecodeLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_internal, num_heads = 2, dropout_rate = 0.1):
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.self_attention = MultiHeadAttention(self.d_model, self.num_heads)
        self.encoder_attention = MultiHeadAttention(self.d_model, self.num_heads)
        self.feed_forward = feed_forward_network(d_model, d_internal)
        self.layernorm1 = tf.keras.layers.BatchNormalization()
        self.layernorm2 = tf.keras.layers.BatchNormalization()
        self.layernorm3 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)
    def call(self, x, encoder_output, padding_mask, look_ahead_mask, training):
        self_attention_outputs = self.self_attention(x, x, x, look_ahead_mask) # (..., seq_length_decoder, d_model)
        x = self.layernorm1(self.dropout1(x + self_attention_outputs, training))
        encoder_attention_outputs = self.encoder_attention(x,
            encoder_output, encoder_output, padding_mask)
        # encoder_output (..., seq_length_encoder, d_model)
        # encoder_attention_outputs (..., seq_length_decoder, d_model)
        x = self.layernorm2(self.dropout2(x + encoder_attention_outputs, training))
        ff = self.feed_forward(x) # (..., seq_length_decoder, d_model)
        x = self.layernorm3(self.dropout3(x + ff, training))
        return x

def angle_rates(positions, dimensions, d_model):
    rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
    return positions * rates

def positional_encodings(max_length, d_model):
    rates = angle_rates(np.range(max_length)[:, np.newaxis],
        np.range(d_model)[np.newaxis, :], d_model)
    sines = np.sin(rates[:, 0::2])
    cosines = np.cos(rates[:, 1::2])
    encodings = np.concatenate([sines, cosines], axis = -1)
    encodings = encodings[np.newaxis, ...]
    return tf.cast(encodings, dtype = tf.float32)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_seq, num_layers, d_model, d_internal, num_heads = 2, dropout_rate = 0.1):
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encodings = positional_encodings(max_seq, d_model)
        # positional_encodings (seq_length, d_model)
        self.layers = [
            EncodeLayer(self.d_model, self.d_internal, self.num_heads, self.dropout_rate) for _ in range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
    def call(self, x, mask, training):
    # x (batch_size, seq_len)
        seq_length = tf.shape(x)[1]
        x = self.embeddings(x) # (batch_size, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encodings[:, :seq_length, :]
        x = self.dropout(x, training = training)
        for i in range(self.num_layers):
            x = self.layers[i](x, mask, training)
        return x # (batch_size,  seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_seq, num_layers, d_model, d_internal, num_heads = 2, dropout_rate = 0.1):
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encodings = positional_encodings(max_seq, d_model)
        # positional_encodings (seq_length_target, d_model)
        self.layers = [
            DecodeLayer(self.d_model, self.d_internal, self.num_heads, self.dropout_rate) for _ in range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
    def call(self, x, encoder_output, padding_mask, look_ahead_mask, training):
    # x (..., seq_length_target)
    # encoder_putput (..., seq_length_source, d_model)
        seq_length = tf.shape(x)[1]
        x = self.embeddings(x) # (batch_size, seq_len_target , d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encodings[:, :seq_length, :]
        x = self.dropout(x, training = training)
        for i in range(self.num_layers):
            x = = self.layers[i](x, encoder_output, padding_mask, look_ahead_mask, training)
        return x # (..., seq_length_target, d_model)
