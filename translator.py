import tensorflow as tf


class Translator(tf.Module):
    def __init__(self, source_tokenizer, target_tokenizer, transformer):
        super().__init__()
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.transformer = transformer
        #self.vocab = self.target_tokenizer.get_vocabulary()
        #self.empty_string = tf.constant('')

    def __call__(self, sentence, max_tokens=128):
        if len(sentence.shape) == 0:
            sentence = [sentence]

        source_tokens = self.source_tokenizer.tokenize(sentence).to_tensor()

        start_end_tokens = self.target_tokenizer.tokenize(['']).to_tensor()[0]
        start = start_end_tokens[0][tf.newaxis]
        end = start_end_tokens[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_tokens):
            output_tokens = tf.transpose(output_array.stack())

            predictions = self.transformer([source_tokens, output_tokens], training=False)
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        text = self.target_tokenizer.detokenize(output)[0]
        #words = tf.gather(self.vocab, output)[1:-1]

        #text = tf.strings.reduce_join(words, separator=' ')
        # return tf.cond(tf.size(words) != 0,
        #                lambda: tf.strings.join(words, separator=' '), lambda: self.empty_string)
        return text
