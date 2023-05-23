import tensorflow as tf
from data_preprocess import get_datasets_n_text_processors


from transformer import Transformer
from translator import Translator

if __name__ == "__main__":
    ds, _, source_tokenizer, target_tokenizer = get_datasets_n_text_processors()

    transformer = Transformer(num_layers=4, model_dim=512, num_heads=8, ff_dim=512,
                              input_vocab_size=10000, target_vocab_size=10000)

    for el in ds:
        transformer(el[0])
        break

    transformer.load_weights("save_archive/epoch_15_acc_0.57_100t.hdf5")

    translator = Translator(source_tokenizer, target_tokenizer, transformer)

    print(translator(tf.convert_to_tensor("Als nächster Punkt folgen die Erklärungen des Rates und der Kommission.")))
