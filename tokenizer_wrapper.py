import re
import tensorflow as tf
import tensorflow_text as tf_text


class TokenizerWrapper(tf.Module):
    def __init__(self, vocab_path):
        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True)
        self.reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
        self.start_token = tf.argmax(tf.constant(self.reserved_tokens) == "[START]")
        self.end_token = tf.argmax(tf.constant(self.reserved_tokens) == "[END]")

        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings).merge_dims(-2, -1)

        count = enc.bounding_shape()[0]
        starts = tf.fill([count, 1], self.start_token)
        ends = tf.fill([count, 1], self.end_token)

        return tf.concat([starts, enc, ends], axis=1)

    @tf.function
    def detokenize(self, tokenized):
        token_txt = self.tokenizer.detokenize(tokenized)
        # Drop the reserved tokens, except for "[UNK]".
        bad_tokens = [re.escape(tok) for tok in self.reserved_tokens if tok != "[UNK]"]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
        result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # Join them into strings.
        return tf.strings.reduce_join(result, separator=' ', axis=-1)
