import tensorflow as tf


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        return self.translator(sentence)
