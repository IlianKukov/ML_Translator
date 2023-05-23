import tensorflow as tf

from encoder_layer import EncoderLayer
from pos_embedding import PositionalEmbedding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, model_dim=model_dim, sentence_size=128)

        self.enc_layers = [
            EncoderLayer(model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x
