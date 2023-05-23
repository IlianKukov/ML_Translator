import tensorflow as tf

from data_preprocess import get_datasets_n_tokenizers
from transformer import Transformer
from translator import Translator
from export_translator import ExportTranslator


def masked_loss(label, pred):
    batch_size = tf.shape(label)[0]
    token_size = tf.shape(label)[1]
    mask = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)
    for i in tf.range(batch_size):
        label_sub = label[i]
        num_tokens = tf.math.count_nonzero(label_sub != 0)
        mask_sub = tf.where(tf.range(token_size, dtype=tf.int32) < tf.cast((num_tokens + 5), dtype=tf.int32),
                            True, False)[:token_size]
        mask = mask.write(i, mask_sub)

    mask = mask.stack()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    batch_size = tf.shape(label)[0]
    token_size = tf.shape(label)[1]
    mask = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)
    for i in tf.range(batch_size):
        label_sub = label[i]
        num_tokens = tf.math.count_nonzero(label_sub != 0)
        mask_sub = tf.where(tf.range(token_size, dtype=tf.int32) < tf.cast((num_tokens + 5), dtype=tf.int32),
                            True,
                            False)[:token_size]
        mask = mask.write(i, mask_sub)

    mask = mask.stack()

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim, warmup_steps=4000):
        super().__init__()
        
        self.model_dim = model_dim
        self.model_dim = tf.cast(self.model_dim, tf.float32)
        
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)


if __name__ == "__main__":
    train_ds, val_ds, source_text_processor, target_text_processor = get_datasets_n_tokenizers()

    tr = Transformer(num_layers=6, model_dim=512, num_heads=8, ff_dim=512,
                     input_vocab_size=8000, target_vocab_size=8000)

    learning_rate = CustomSchedule(512)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/epoch2_{epoch:02d}_acc_{val_masked_accuracy:.2f}.hdf5",
        save_weights_only=True,
        monitor='val_masked_accuracy',
        save_best_only=False
    )

    tr.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])

    # for el in val_ds:
    #     tr(el[0])
    #     break
    #
    # #
    # # #tr.summary()
    #tr.load_weights("checkpoints/epoch_01_acc_0.71.hdf5")

    # for el in val_ds:
    #     pred = tr(el[0])
    #     masked_loss(el[1], pred)
    #     break

    tr.fit(train_ds, epochs=2, validation_data=val_ds, callbacks=[model_checkpoint_callback])
    translator = Translator(source_text_processor, target_text_processor, tr)
    print(translator(tf.convert_to_tensor("Als nächster Punkt folgen die Erklärungen des Rates und der Kommission.")))

    exp_trans = ExportTranslator(translator)
    exp_trans(tf.convert_to_tensor("Als nächster Punkt folgen die Erklärungen des Rates und der Kommission."))

    tf.saved_model.save(exp_trans, export_dir='translator2')

