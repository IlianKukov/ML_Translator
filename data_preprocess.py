import os.path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from tokenizer_wrapper import TokenizerWrapper


def prepare_batch(source, target, source_text_tokenizer, target_text_tokenizer):
    source_tokens = source_text_tokenizer.tokenize(source).to_tensor()[:, :128]
    target_tokens = target_text_tokenizer.tokenize(target).to_tensor()

    # source_tokens = source_text_tokenizer.tokenize(source)[:, :128]
    # target_tokens = target_text_tokenizer.tokenize(target)

    target_tokens_in = target_tokens[:, :-1][:, :128]
    target_tokens_out = target_tokens[:, 1:][:, :128]

    return (source_tokens, target_tokens_in), target_tokens_out


def get_datasets_n_tokenizers(select_first_n=None):
    train_raw_ds, val_raw_ds = get_raw_sentence_datasets(select_first_n=select_first_n)

    vocab_gen_ds, _ = get_raw_sentence_datasets(batch_size=1000, train_percentage=1)
    src_vocab, target_vocab = get_vocabs(vocab_gen_ds)

    src_tokenizer = TokenizerWrapper(src_vocab)
    target_tokenizer = TokenizerWrapper(target_vocab)

    tf.saved_model.save(src_tokenizer, "src_tokenizer")
    tf.saved_model.save(target_tokenizer, "target_tokenizer")

    train_ds = train_raw_ds.map(lambda source, target: prepare_batch(source, target,
                                                                     src_tokenizer, target_tokenizer))

    val_ds = val_raw_ds.map(lambda source, target: prepare_batch(source, target,
                                                                 src_tokenizer, target_tokenizer))

    return train_ds, val_ds, src_tokenizer, target_tokenizer


def get_vocabs(dataset, source_file="src_de.txt", target_file="trg_en.txt"):
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    bert_vocab_args = dict(
        vocab_size=8000,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=dict(lower_case=True),
        learn_params={},
    )

    source_res = source_file
    if not os.path.exists(source_file):
        source_vocab = bert_vocab.bert_vocab_from_dataset(
             dataset.map(lambda source, target: source), **bert_vocab_args)

        print(f"Source vocab length: {len(source_vocab)}")

        with open(source_file, 'w', encoding="utf-8") as f:
            for token in source_vocab:
                print(token, file=f)

        source_res = source_vocab

    target_res = target_file
    if not os.path.exists(target_file):
        target_vocab = bert_vocab.bert_vocab_from_dataset(
            dataset.map(lambda source, target: target), **bert_vocab_args)

        print(f"Target vocab length: {len(target_vocab)}")

        with open(target_file, 'w', encoding="utf-8") as f:
            for token in target_vocab:
                print(token, file=f)

        target_res = target_vocab

    return source_res, target_res


def get_raw_sentence_datasets(batch_size=100, train_percentage=0.992, select_first_n=None):
    wtm = pd.read_csv("europarl-v10.de-en.pair.tsv", sep="\t", on_bad_lines="skip", header=None)
    wtm = wtm[wtm.columns[:2]].dropna(axis=0)
    wtm.columns = ["de", "en"]

    deu = pd.read_csv("deu.txt", sep='\t', header=None)
    deu = deu.drop(columns=[2])
    deu.columns = ["en", "de"]

    comb = pd.concat([deu, wtm], ignore_index=True).reindex(columns=["de", "en"])

    del wtm, deu

    BUFFER_SIZE = len(comb)
    BATCH_SIZE = batch_size
    TRAIN_PERCENTAGE = train_percentage

    np.random.seed(0)
    train_indices = np.random.uniform(size=(BUFFER_SIZE,)) < TRAIN_PERCENTAGE
    if select_first_n:
        train_indices[select_first_n:] = False

    train_np_source = comb[train_indices].to_numpy()[:, 0]
    train_np_target = comb[train_indices].to_numpy()[:, 1]

    train_raw_ds = tf.data.Dataset.from_tensor_slices((train_np_source, train_np_target))
    train_raw_ds = train_raw_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    val_indices = ~train_indices
    if select_first_n:
        val_indices[select_first_n:] = False

    val_np_source = comb[val_indices].to_numpy()[:, 0]
    val_np_target = comb[val_indices].to_numpy()[:, 1]

    val_raw_ds = tf.data.Dataset.from_tensor_slices((val_np_source, val_np_target))
    val_raw_ds = val_raw_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_raw_ds, val_raw_ds

if __name__ == "__main__":
    train_ds, val_ds, src_tokenizer, target_tokenizer = get_datasets_n_tokenizers(select_first_n=100000)

    for el in train_ds:
        print(el)
    print("stop")
