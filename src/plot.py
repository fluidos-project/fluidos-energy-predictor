import logging as log
import os

import numpy as np
from matplotlib import pyplot as plt

import parameters as pm
from support.log import initialize_log


def save_prediction(yhat, y2, diff):
    # Dump the prediction to a file
    os.makedirs(pm.LOG_FOLDER + "/pred", exist_ok=True)
    with open(pm.LOG_FOLDER + "/pred/prediction.csv", "w") as f:
        for i in range(yhat.shape[0]):
            f.write(",".join([str(x) for x in yhat[i]]) + "\n")
    with open(pm.LOG_FOLDER + "/pred/actual.csv", "w") as f:
        for i in range(y2.shape[0]):
            f.write(",".join([str(x) for x in y2[i]]) + "\n")
    # with open(pm.LOG_FOLDER + "/pred/diff.csv", "w") as f:
    #     for i in range(diff.shape[0]):
    #         f.write(",".join([str(x) for x in diff]) + "\n")

    np.save(pm.LOG_FOLDER + "/pred/yhat_history.npy", yhat)
    np.save(pm.LOG_FOLDER + "/pred/y2.npy", y2)


def plot_prediction(yhat, y2, columns, start=0, end=None):
    # shape of yhat: (n, steps_out, n_features)
    # plot a graph for each feature
    if end is None:
        end = yhat.shape[0]

    for run in range(start, end):
        for feature in range(yhat.shape[3]):
            plt.figure(figsize=(15, 8))
            plt.plot(
                yhat[run, :, :, feature][0],
                label="prediction",
                linestyle="-.",
                alpha=0.7,
                color="r",
            )
            plt.plot(
                y2[run, :, :, feature][0],
                label="actual",
                linestyle="-",
                alpha=0.5,
                color="b",
            )
            if columns is not None:
                for j in columns:
                    plt.axvline(x=j, linestyle="--", alpha=0.3, color="g")

            plt.legend()
            plt.xlabel("Time")
            plt.ylabel(f"Usage - feature {feature}")
            plt.title("Prediction vs actual usage")
            plt.savefig(pm.LOG_FOLDER + f"/prediction-{run}-f{feature}.png")
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
        plt.yscale("log")
        plt.title("Loss and validation loss over epochs")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["train", "validate"], loc="upper left")
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
