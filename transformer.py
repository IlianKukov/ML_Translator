import tensorflow as tf

from encoder import Encoder
from decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):

        super().__init__()
        self.model_dim = model_dim
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim,
                               num_heads=num_heads, ff_dim=ff_dim,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)
        
        self.decoder = Decoder(num_layers=num_layers, model_dim=model_dim,
                               num_heads=num_heads, ff_dim=ff_dim,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        source_tokens, output_tokens = inputs

        enc_output = self.encoder(source_tokens)

        dec_output = self.decoder(output_tokens, enc_output)

        # Final linear layer output.
        logits = self.final_layer(dec_output)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits
