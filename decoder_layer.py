import tensorflow as tf


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()

        self.casual_mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=(model_dim // num_heads))
        self.cross_mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=(model_dim // num_heads))

        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(model_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    @staticmethod
    def _compute_causal_mask(query):
        batch_size = tf.shape(query)[0]
        q_seq_length = tf.shape(query)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((batch_size, q_seq_length, q_seq_length), tf.bool), -1, 0
        )

    def call(self, x, enc_output):
        # multi-head casual attention
        attention_mask = self._compute_causal_mask(x)
        x = self.add([x, self.casual_mha(query=x, key=x, value=x, attention_mask=attention_mask)])
        x = self.norm(x)

        # cross attention
        x = self.add([x, self.cross_mha(query=x, key=enc_output, value=enc_output)])
        x = self.norm(x)

        # feed forward
        x = self.add([x, self.ff(x)])
        x = self.norm(x)

        return x


if __name__ == "__main__":
    mask = DecoderLayer._compute_causal_mask(tf.ones(shape=(128, 128, 512)))
    print("stop")
