import tensorflow as tf
import numpy as np


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angles = positions * angle_rates      # (pos, depth)

    pos_enc = np.zeros(shape=(length, int(depth*2)))
    pos_enc[:, 0::2] = np.sin(angles)
    pos_enc[:, 1::2] = np.cos(angles)

    #pos_enc = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)

    return tf.cast(pos_enc, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, model_dim, sentence_size):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim, mask_zero=True)
        self.pos_encoding = positional_encoding(length=sentence_size, depth=model_dim)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


if __name__ == "__main__":
    pos_enc = positional_encoding(length=128, depth=512)
    print("stop")
