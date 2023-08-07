import argparse
import logging as log
import os
from datetime import datetime

import tensorflow as tf

import src.model as modelmd
import src.parameters as pm
from src.data import obtain_vectors, fetch_datasets, fetch_power_curve
from src.support.log import initialize_log
from src.plot import plot_history


def ask_model_name(models: list[str]) -> str:
    if len(models) > 0:
        print(f"Available models:")
        for i in range(len(models)):
            model_name = models[i]
            if os.path.isfile(pm.MODEL_FOLDER + "/" + model_name + "/model.h5"):
                print(f"\t{i + 1}) {model_name} (trained)")
            else:
                print(f"\t{i + 1}) {model_name} (status unknown)")
        print()
    print(f"Model name:"
          f"\n\t- '' or 'default' for default model ({pm.DEFAULT_MODEL})"
          f"\n\t- 'dt' for datetime"
          f"\n\t- [name] for name" + (f"\n\t- [number] for a specific model in list" if len(models) > 0 else ""))
    while True:
        name = input("Insert model name: ")
        match name:
            case "" | "default":
                name = pm.DEFAULT_MODEL
                break
            case "dt":
                name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                break
            case _:
                try:
                    name = models[int(name) - 1]
                    break
                except:
                    pass

    return name


def ask_decision(model_name: str, train_data: list[list[float]]) -> tuple[str, bool]:
    action = "n"
    print(f"Chosen model name: {model_name}")
    if os.path.exists(pm.MODEL_FOLDER + "/" + model_name + "/model.h5"):
        new_model = False
        print("Model found. What would you like to do? [r]etrain / [c]ontinue / [e]xit")
        while True:
            response = input("r/c/e: ")
            match response:
                case "r":
                    action = "t"
                    break
                case "c":
                    action = "n"
                    break
                case "e":
                    print("Exiting...")
                    exit(0)
                case _:
                    print("Invalid input")
                    continue

    else:
        new_model = True
        print("Model not found. What would you like to do? [t]rain / [s]earch hyperparameters / [e]xit")
        while True:
            response = input("t/s/e: ")
            match response:
                case "t":
                    action = "t"
                    break
                case "s":
                    from hp import search_params
                    search_params.search_hp(train_data)
                    exit(0)
                case "e":
                    print("Exiting...")
                    exit(0)
                case _:
                    print("Invalid input")
                    continue

    return action, new_model


def main():
    parser = argparse.ArgumentParser(description='FLUIDOS WP6 T6.3 Model POC')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model name (if unspecified, will be prompted)')
    parser.add_argument('--curve', '-c', type=str, default=None,
                        help='Power curve file (if unspecified, will be chosen randomly)')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                        help='Number of epochs (if unspecified, will be prompted)')
    parser.add_argument('--action', '-a', type=str, default=None,
                        help='Action to perform (train, search hyperparameters, test)')
    parser.add_argument('--machine', '-M', type=str, default=None,
                        help='GCD machine files to use (if unspecified, will be chosen randomly)')
    args = parser.parse_args()

    if args.model is not None:
        model_name = args.model
    else:
        model_name = None

    if args.action is not None:
        action = args.action
    else:
        action = None

    initialize_log("INFO")

    # Block GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    os.makedirs(pm.LOG_FOLDER, exist_ok=True)
    os.makedirs(pm.MODEL_FOLDER, exist_ok=True)
    os.makedirs(pm.GCD_FOLDER, exist_ok=True)
    os.makedirs(pm.CACHE_FOLDER, exist_ok=True)

    train_data, test_data = fetch_datasets(args.machine, banlist_file=pm.BANLIST_FILE)
    power_curve = fetch_power_curve(args.curve)

    models = sorted([i.split(".")[0] for i in os.listdir(pm.MODEL_FOLDER)])

    if model_name is None:
        model_name = ask_model_name(models)
    if action is None:
        action, new_model = ask_decision(model_name, train_data)
    else:
        if model_name in models:
            new_model = False
        else:
            new_model = True

    if action not in ["t", "n"]:
        raise EnvironmentError("Something went wrong. Decision is not t or n.")

    log.info(f"Model name: {model_name}; Retrain: {action}; New model: {new_model}")
    pm.MODEL_FOLDER = os.path.join(pm.MODEL_FOLDER, model_name)
    os.makedirs(pm.MODEL_FOLDER, exist_ok=True)

    # Load model eventually
    if not new_model:
        model = tf.keras.models.load_model(pm.MODEL_FOLDER + "/model.h5")
    else:
        model = modelmd.new_model()

    print(model.summary())

    # Train
    if action == "t":
        epochs = 0
        while True:
            try:
                epochs = input("Epochs: ")
                epochs = float(epochs)

                if int(epochs) != epochs:
                    print("Ah yes, training for a non-integer amount of epochs. That's a good idea.")
                    continue

                epochs = int(epochs)

                if epochs <= 0:
                    print("Ah yes, training for 0 epochs. That's a good idea. Congratulations, you broke math.")
                    continue

                break
            except ValueError:
                print("Ah yes, training for NaN epochs. That's a good idea.")
                continue

        xx, y = obtain_vectors(train_data, power_curve=power_curve)

        log.info(f"Training data shape: {xx.shape} -> {y.shape}")

        cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pm.PATIENCE),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=200, verbose=1, mode='min', min_lr=1e-6),
              tf.keras.callbacks.BackupAndRestore(pm.MODEL_FOLDER + "/backup"),
              tf.keras.callbacks.ModelCheckpoint(pm.MODEL_FOLDER + "/model.h5", save_best_only=True,
                                                 save_weights_only=False, monitor='val_loss', mode='min')]

        log.info(f"Training model for {epochs} epochs")
        history = model.fit(xx, y, epochs=epochs, verbose=1, validation_split=pm.SPLIT, shuffle=True, callbacks=cb)

        tf.keras.models.save_model(model, pm.MODEL_FOLDER + "/last_model.h5")

        log.info("Saved model to disk")
        plot_history(history)

        # restore best model
        model = tf.keras.models.load_model(pm.MODEL_FOLDER + "/model.h5")

    # Predict
    results = modelmd.predict(model, test_data, power_curve)
    log.info(f"Predictions: {results}")


if __name__ == '__main__':
    main()
