import logging as log
import os

import numpy as np
from matplotlib import pyplot as plt

from src import parameters as pm
from src.support.log import initialize_log


def save_prediction(yhat, y2):
    # Dump the prediction to a file
    os.makedirs(pm.LOG_FOLDER + "/pred", exist_ok=True)
    np.savetxt(pm.LOG_FOLDER + "/pred/prediction.csv", yhat, delimiter=",")
    np.savetxt(pm.LOG_FOLDER + "/pred/actual.csv", y2, delimiter=",")
    np.save(pm.LOG_FOLDER + "/pred/yhat_history.npy", yhat)
    np.save(pm.LOG_FOLDER + "/pred/y2.npy", y2)


def plot_prediction(yhat, y2, columns, start=0, end=None):
    # shape of yhat: (n, steps_out, n_features)
    # plot a graph for each feature
    if end is None:
        end = yhat.shape[0]

    prediction = yhat[start:end, :].flatten()
    truth = y2[start:end, :].flatten()

    plt.figure(figsize=(15, 8))
    # First, plot what was before
    plt.plot(prediction, label='prediction', linestyle='-.', alpha=.7, color='r')
    plt.plot(truth, label='actual', linestyle='-', alpha=.5, color='b')
    if columns is not None:
        for j in columns:
            plt.axvline(x=j, linestyle='--', alpha=.3, color='g')
    plt.legend()
    plt.xlabel("Predicted day")
    plt.ylabel("Power (kWh)")
    plt.title("Prediction vs actual usage of the following day")
    plt.savefig(pm.LOG_FOLDER + f"/prediction-{start}-{end}.png")
    plt.close()

    # fill with color
    # plt.fill_between(
    #     np.arange(start, end),
    #     yhat[start:end, 1],
    #     yhat[start:end,  2],
    #     color='r',
    #     alpha=.15
    # )
    #
    # plt.fill_between(
    #     np.arange(start, end),
    #     truth[start:end, 1],
    #     truth[start:end, 2],
    #     color='b',
    #     alpha=.15
    # )


def plot_history(history):
    # list all data in history
    log.info("Available keys: " + str(history.history.keys()))

    for key in history.history.keys():
        if "val" not in key:
            continue
        plt.figure(figsize=(15, 8))
        plt.plot(history.history[key.replace("val_", "")])
        plt.plot(history.history[key])
        plt.yscale('log')
        plt.title('Loss and validation loss over epochs')
        plt.ylabel("Loss")
        plt.xlabel('Epoch')
        plt.legend(['train', 'validate'], loc='upper left')
        plt.savefig(pm.LOG_FOLDER + "/" + key.replace("val_", "") + ".png")
        plt.close()

    # New plot. We plot y as a line, while for the predictions,
    # each data point is an estimate for the subsequent pm.YWINDOW data points.
    # We plot the lower and upper bounds as a shaded area.

    #     plt.figure(figsize=(20, 10))
    #     plt.plot(yhat_history[:, 0, 0], label='target')
    #     plt.plot(y2[:, 1], label='actual')
    #     plt.fill_between(
    #         range(len(yhat_history[:, 0, 0])),
    #         yhat_history[:, 0, 1],
    #         yhat_history[:, 0, 2],
    #         alpha=0.5,
    #         label='prediction interval'
    #     )
    #     plt.legend()
    #     plt.savefig(pm.LOG_FOLDER + "/prediction.png")


def plot_splitter():
    file = input("Enter the folder name: ")
    history = np.load(file + "/pred/yhat_history.npy")
    truth = np.load(file + "/pred/y2.npy")

    initialize_log("INFO", "plot")
    for i in range(0, len(history), 500):
        plot_prediction(history, truth, columns=None, start=i, end=i + 500)

    log.info("Done!")


if __name__ == "__main__":
    plot_splitter()
