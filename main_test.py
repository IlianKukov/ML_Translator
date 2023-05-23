import tensorflow as tf
import tensorflow_text as tf_text

if __name__ == "__main__":

    translator = tf.saved_model.load("translator2")
    print(translator("Ich gehe nach Hause.").numpy())

