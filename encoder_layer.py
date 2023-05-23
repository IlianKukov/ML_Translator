import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()

        self.mha_first = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=(model_dim // num_heads))
        self.mha_second = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=(model_dim // num_heads))

        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(model_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        # multi-head self attention
        x = self.add([x, self.mha_first(query=x, key=x, value=x)])
        x = self.norm(x)

        # feed forward
        x = self.add([x, self.ff(x)])
        x = self.norm(x)

        return x


if __name__ == "__main__":
    enc = EncoderLayer(512, 4, 512)
    enc(tf.ones(shape=(128,56,512)))
    enc(tf.ones(shape=(128, 58, 512)))
    print("stop")
