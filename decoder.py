import tensorflow as tf

from decoder_layer import DecoderLayer
from pos_embedding import PositionalEmbedding


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, model_dim=model_dim, sentence_size=128)

        self.dec_layers = [
            DecoderLayer(model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output):
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output)

        return x
