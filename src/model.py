import logging as log

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error

import parameters as pm
from data import obtain_vectors
from plot import save_prediction, plot_prediction


# def custom_loss(y_true, y_pred):
#     return tf.keras.losses.MSE(y_true, y_pred)


# noinspection PyUnresolvedReferences
def new_model(hp: kt.HyperParameters = None) -> tf.keras.models.Model:
    if hp is None:
        usize = pm.USIZE
    else:
        usize = hp.Int("usize", min_value=16, max_value=256, step=16)

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=pm.LEARNING_RATE)

    model = models.Sequential(
        [
            layers.Input(shape=(pm.STEPS_IN, pm.N_FEATURES)),
            layers.Bidirectional(layers.LSTM(usize, return_sequences=True)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(usize, return_sequences=True)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(pm.STEPS_OUT[0] * pm.STEPS_OUT[1]),
            layers.Flatten(),
            layers.Dense(pm.STEPS_OUT[0] * pm.STEPS_OUT[1]),
            layers.Reshape(pm.STEPS_OUT),
            layers.Activation("linear"),
        ]
    )

    model.compile(loss=loss, optimizer=optimizer)

    return model


# noinspection PyUnresolvedReferences
def predict(
    model: tf.keras.Sequential, test_data: list[str], power_curve: list[np.ndarray]
) -> dict:
    yhat_history = []
    y2_history = []

    for file in test_data:
        xx2, y2 = obtain_vectors(file, power_curve)
        if xx2 is None or y2 is None or xx2.shape[0] == 0 or y2.shape[0] == 0:
            continue

        x_input = xx2[0].reshape((1, pm.STEPS_IN, pm.N_FEATURES))
        y2_input = y2[0]
        yhat = model.predict(x_input, verbose=0)

        yhat_history.append(yhat)
        y2_history.append([y2_input])

    yhat_history = np.array(yhat_history)
    y2_history = np.array(y2_history)

    log.info("Prediction finished")
    # log.info("Expected power consumption: %s", y2_history.tolist())
    # log.info("Predicted power consumption: %s", yhat_history.tolist())

    plot_prediction(yhat_history, y2_history, columns=None)

    # reshape again to compute metrics and save the prediction
    # yhat_history = yhat_history.reshape(
    #     (len(yhat_history), pm.STEPS_OUT[0], pm.STEPS_OUT[1])
    # )
    # y2_history = y2_history.reshape((len(y2_history), pm.STEPS_OUT[0], pm.STEPS_OUT[1]))
    diff = np.subtract(y2_history.flatten(), yhat_history.flatten())
    save_prediction(yhat_history, y2_history, diff)

    # r2 = r2_score(y2_history.flatten(), yhat_history.flatten())
    mse = mean_squared_error(y2_history.flatten(), yhat_history.flatten())
    mae = mean_absolute_error(y2_history.flatten(), yhat_history.flatten())

    return {
        "mse": mse,
        "mae": mae,
        # "diff": diff.tolist(),
        # "y2": y2_history.tolist(),
        # "yhat": yhat_history.tolist(),
    }


# noinspection PyUnresolvedReferences
def predict_inmemory(
    model: tf.keras.Sequential,
    merged_data: dict[int, dict[str, float]],
    power_curve: list[np.ndarray],
) -> dict:

    # merged data be like:
    # {1: {'cpu': 0.1, 'mem': 0.2}, 2: {'cpu': 0.4, 'mem': 0.5}}
    # obtain_vectors_inmemory()
    cpu_data = [merged_data[timestamp]["cpu"] for timestamp in merged_data]
    mem_data = [merged_data[timestamp]["mem"] for timestamp in merged_data]

    cpu_data = np.array(cpu_data).reshape(-1, 1)
    mem_data = np.array(mem_data).reshape(-1, 1)

    x_input = np.concatenate((cpu_data, mem_data), axis=1)
    x_input = x_input.reshape((1, pm.STEPS_IN, pm.N_FEATURES))

    yhat = model.predict(x_input, verbose=0)

    return yhat
