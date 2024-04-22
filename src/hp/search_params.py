import keras_tuner
import tensorflow as tf

from src.data import obtain_vectors
from src.model import new_model


# noinspection PyUnresolvedReferences
def search_hp(train_data):
    tuner = keras_tuner.RandomSearch(
        new_model,
        objective="val_loss",
        max_trials=20,
        executions_per_trial=1,
    )

    tuner.search_space_summary()

    xx, y = obtain_vectors(train_data)

    # split into train and test
    split = int(len(xx) * 0.8)
    x_train, y_train = xx[:split], y[:split]
    x_test, y_test = xx[split:], y[split:]

    tuner.search(
        x_train,
        y_train,
        epochs=500,
        validation_data=(x_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)],
    )
