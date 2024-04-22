import logging as log

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src import parameters as pm
from src.data import obtain_vectors, obtain_vectors_inmemory
from src.plot import save_prediction, plot_prediction


# def custom_loss(y_true, y_pred):
#     return tf.keras.losses.MSE(y_true, y_pred)


# noinspection PyUnresolvedReferences
def new_model(hp: kt.HyperParameters = None) -> tf.keras.models.Model:
    if hp is None:
        filters = pm.FILTERS
        ksize = pm.KSIZE
    else:
        filters = hp.Int("filters", min_value=16, max_value=256, step=16)
        ksize = hp.Int("ksize", min_value=3, max_value=9, step=2)

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=pm.LEARNING_RATE)

    input_layer = tf.keras.layers.Input(shape=(pm.STEPS_IN, pm.N_FEATURES))

    conv1 = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=ksize, activation="relu"
    )(input_layer)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)

    conv2 = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=ksize, activation="relu"
    )(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)

    conv3 = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=ksize, activation="relu"
    )(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)

    gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = tf.keras.layers.Dense(pm.STEPS_OUT, activation="linear")(gap)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss=loss, optimizer=optimizer)

    return model

    # from keras.utils.vis_utils import plot_model

    # plot_model(model, to_file='model_plot.png',
    # show_shapes=True, show_layer_names=False,
    #           rankdir="LR", dpi=72)


# noinspection PyUnresolvedReferences
def predict(
    model: tf.keras.Sequential, test_data: list[str], power_curve: list[np.ndarray]
) -> dict:
    yhat_history = np.ndarray(shape=(0, pm.STEPS_OUT))
    y2_history = np.ndarray(shape=(0, pm.STEPS_OUT))
    for file in test_data:
        xx2, y2 = obtain_vectors(file, power_curve)
        if xx2 is None or y2 is None:
            continue

        x_input = xx2[0].reshape((1, pm.STEPS_IN, pm.N_FEATURES))
        y2_input = y2[0]

        yhat = model.predict(x_input, verbose=0)

        yhat_history = np.append(yhat_history, yhat, axis=0)
        y2_history = np.append(y2_history, [y2_input], axis=0)

    log.info("Prediction finished")
    log.info("Expected power consumption: %s", y2_history.tolist())
    log.info("Predicted power consumption: %s", yhat_history.tolist())

    plot_prediction(yhat_history, y2_history, columns=None)

    # reshape again to compute metrics and save the prediction
    yhat_history = yhat_history.reshape((len(yhat_history), pm.STEPS_OUT))
    y2_history = y2_history.reshape((len(y2_history), pm.STEPS_OUT))

    save_prediction(yhat_history, y2_history)

    r2 = r2_score(y2_history, yhat_history)
    mse = mean_squared_error(y2_history, yhat_history)
    mae = mean_absolute_error(y2_history, yhat_history)
    diff = np.subtract(y2_history, yhat_history)

    return {
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "diff": diff.tolist(),
        "y2": y2_history.tolist(),
        "yhat": yhat_history.tolist(),
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

    xx2, y2 = obtain_vectors_inmemory(cpu_data, mem_data, power_curve)
    if xx2 is None or y2 is None:
        raise ValueError("No data found")

    x_input = xx2[0].reshape((1, pm.STEPS_IN, pm.N_FEATURES))
    y2_input = y2[0]

    yhat = model.predict(x_input, verbose=0)

    log.info("Prediction finished")
    log.info(f"Expected power consumption: {y2_input}")
    log.info(f"Predicted power consumption: {yhat}")

    return {"y2": y2_input, "yhat": yhat}
