import json
import logging as log
import os
import random

import numpy as np

from src.support import dt
from src import parameters as pm
from src.support.log import tqdm_wrapper



def fetch_power_curve(file: str) -> list[np.ndarray]:
    spec_folder = pm.SPEC_FOLDER

    if os.path.exists(f"{spec_folder}/.DS_Store"):
        os.remove(f"{spec_folder}/.DS_Store")

    if file is None or file == "":
        available_files = [i for i in os.listdir(spec_folder) if "json" in i]
        if len(available_files) == 0:
            raise FileNotFoundError(f"No files found in {spec_folder}")
        file = random.choice(available_files)

    if not file.endswith(".json"):
        file = f"{file}.json"

    if os.path.exists(f"{spec_folder}/{file}"):
        log.info(f"Loading power curve from {spec_folder}/{file}")
        with open(f"{spec_folder}/{file}", "r") as f:
            data = json.load(f)
            cpu_data = data["data"]["Power"]
            cpu_data.append(cpu_data[-1] / 2)
            cpu_data = cpu_data[::-1]
            cpu_data = np.array(cpu_data)

            mem_data = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 2.4, 3.2, 4.8, 9.6, 18.4]
            mem_data = np.array(mem_data)

            if "memory" in data:
                installed_memory = data["memory"]
            else:
                log.warning("No memory data found in power curve file, using default values (based on 8GB)")
                installed_memory = 8

            mem_data *= installed_memory / 8

            if len(cpu_data) != len(mem_data):
                raise ValueError("CPU and memory data have different lengths")
            if len(cpu_data) != 11:
                raise ValueError("CPU and memory data have unexpected length, must be 11")

            log.info(f"CPU energy consumption: {list(cpu_data)}")
            log.info(f"Memory energy consumption: {list(mem_data)}")
            return [cpu_data, mem_data]
    else:
        raise FileNotFoundError(f"File {file}.json not found in {spec_folder}")



def fetch_datasets(folder: str, banlist_file: str = None) -> list[np.ndarray]:
    gcd_folder = pm.GCD_FOLDER

    banlist = []
    if os.path.exists(banlist_file):
        with open(banlist_file, "r") as f:
            for line in f:
                banlist.append(line.strip() + ".csv")
    else:
        with open("banlist", "w") as f:
            f.write("")

    if os.path.exists(f"{gcd_folder}/.DS_Store"):
        os.remove(f"{gcd_folder}/.DS_Store")

    available_folders = [i for i in os.listdir(gcd_folder) if "json" not in i and "npy" not in i and "cache" not in i]
    if len(available_folders) == 0:
        raise EnvironmentError(f"No folders found in {gcd_folder}. Please run obtain some datasets first.")

    if folder is None or folder == "":
        chosen_folder = random.choices(available_folders, k=1)[0]
    else:
        chosen_folder = folder

    files = [i for i in os.listdir(os.path.join(f"{gcd_folder}", chosen_folder)) if "seq" in i]
    files = sorted([i for i in files], key=lambda x: int(x.split("_")[-1].split("-")[0]))

    total_size = pm.TEST_FILE_AMOUNT + pm.TRAIN_FILE_AMOUNT
    # select a random 4-file chunk that is adjacent
    starting_index = random.randint(0, len(files) - total_size)
    files = files[starting_index:starting_index + total_size]

    train_data = [os.path.join(chosen_folder, i) for i in files[:pm.TRAIN_FILE_AMOUNT]]
    test_data = [os.path.join(chosen_folder, i) for i in files[pm.TRAIN_FILE_AMOUNT:]]

    log.info(f"Verifying test files ({len(test_data)} required)...")
    for file in test_data:
        if file in banlist:
            raise EnvironmentError(f"File {file} is banned. Please refrain from using it ever again.")
        if not os.path.exists(os.path.join(pm.GCD_FOLDER, file)):
            raise EnvironmentError(f"File {file} does not exist.")

    log.info(f"Training files: {str(train_data)}")

    log.info(f"Verifying train files ({len(train_data)} required)...")
    for file in train_data:
        if file in banlist:
            raise EnvironmentError(f"File {file} is banned. Please refrain from using it ever again.")
        if not os.path.exists(os.path.join(pm.GCD_FOLDER, file)):
            raise EnvironmentError(f"File {file} does not exist.")

    if len(train_data) == 0 or len(test_data) == 0:
        raise EnvironmentError(
            "Something went wrong. You need to specify at least one file for training and one for testing.")

    log.info(f"Testing files: {str(test_data)}")

    return train_data, test_data


def load_data(file: str) -> tuple[np.ndarray, np.ndarray]:
    npycpu = os.path.join(pm.CACHE_FOLDER, file.split("/")[-1].replace(".csv", "_cpu.npy"))
    npymem = os.path.join(pm.CACHE_FOLDER, file.split("/")[-1].replace(".csv", "_mem.npy"))
    file = os.path.join(pm.GCD_FOLDER, file)

    if os.path.exists(npycpu) and os.path.exists(npymem):
        log.info(f"Reading cached data from {npycpu} and {npymem}")
        return np.load(npycpu), np.load(npymem)
    elif os.path.exists(npycpu) or os.path.exists(npymem):
        raise Exception(f"Only one of {npycpu} and {npymem} exists")

    log.info(f"Reading data from {file}")

    ret_cpu = []
    ret_mem = []
    # ret_ts = []
    with open(file, 'r') as f:
        # data be like
        # timestamp,container_id,task_id,node_id,cpu,mem,idk
        # 38154,69b..,47f4...,K8ad..., 0.052437,0.3073234,300000

        for line in tqdm_wrapper(f):
            if 'timestamp' in line:
                continue

            fts, _, _, _, cpu, mem, *_ = line.split(',')

            # depending on the file, the timestamp can be in different positions
            # timestamp be like: filename:12341513
            if ":" in fts:
                _, timestamp = fts.split(':')
            else:
                timestamp = fts

            try:
                # ret_ts.append(int(timestamp))
                ret_cpu.append(float(cpu))
                ret_mem.append(float(mem))
            except ValueError as e:
                log.warning(f"Error while parsing {line}: {e}")
                pass

    # ret_ts = np.array(ret_ts)
    ret_cpu = np.array(ret_cpu)
    ret_mem = np.array(ret_mem)

    np.save(npycpu, ret_cpu)
    np.save(npymem, ret_mem)

    return ret_cpu, ret_mem


# def trans_forward(scaler: MinMaxScaler, arr):
#     if len(arr.shape) == 1:
#         arr = arr.reshape(-1, 1)
#     return scaler.transform(arr)


# def trans_back(scaler: MinMaxScaler, arr):
#     if len(arr.shape) == 1:
#         arr = arr.reshape(-1, 1)
#     return scaler.inverse_transform(arr)


def get_predicted_power(param: np.ndarray, power_curve: list[np.ndarray]) -> float:
    cpu_curve, mem_curve = power_curve

    cpu_usage = param[:, 0]
    mem_usage = param[:, 1]

    # transform average power over 5 minutes to actual consumption (Wh)
    cpu_power = np.interp(cpu_usage, np.arange(len(cpu_curve)), cpu_curve) * pm.GRANULARITY / 60
    mem_power = np.interp(mem_usage, np.arange(len(mem_curve)), mem_curve) * pm.GRANULARITY / 60

    return np.sum(cpu_power + mem_power) * pm.OVERHEAD


def split_sequence(sequence: np.ndarray,
                   past_len: int,
                   future_len: int,
                   steps_out: int,
                   filename: str,
                   power_curve: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    # split the sequence into samples s.t.
    # X = [past_len, N_FEATURES]
    # y = [steps_out, 1]
    # where steps_out is calculated using get_power_from_sequence, and requires future_len samples
    # to be computed

    seq_len = len(sequence) - future_len - past_len + 1
    try:
        xx, y = np.ndarray(shape=(seq_len, past_len, pm.N_FEATURES)), np.ndarray(shape=(seq_len, steps_out))
    except ValueError as e:
        log.warning(f"Error while splitting sequence (might be too short?): {e}")
        return None, None

    simple_filename = filename.split("/")[-1].replace(".csv", "")
    desc = ",".join([str(a) for a in [past_len, future_len, steps_out, simple_filename, len(sequence)]])

    seq_filename_npy = pm.CACHE_FOLDER + '/' + "samples_" + desc + ".npy"
    seq_filename_y_npy = pm.CACHE_FOLDER + '/' + "samples_" + desc + "_y.npy"
    seq_filename = pm.GCD_FOLDER + '/' + "samples_" + desc

    if os.path.exists(seq_filename_npy) and os.path.exists(seq_filename_y_npy):
        log.info(f"Using cached sequences for {seq_filename}...")
        return np.load(seq_filename_npy), np.load(seq_filename_y_npy)
    elif os.path.exists(seq_filename_npy) or os.path.exists(seq_filename_y_npy):
        raise Exception(f"Only one of {seq_filename_npy} or {seq_filename_y_npy} exists!")
    else:
        log.info(f"Generating sequences for {seq_filename}...")
        for i in tqdm_wrapper(range(seq_len)):
            end_ix = i + past_len

            seq_x = sequence[i:end_ix]
            seq_y = get_predicted_power(sequence[end_ix:end_ix + future_len], power_curve)

            xx[i] = seq_x
            y[i] = seq_y

        np.save(seq_filename_npy, xx)
        np.save(seq_filename_y_npy, y)

        return xx, y


def obtain_vectors(data_file: str | list[str],
                   power_curve: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(data_file, list):
        xx, y = np.ndarray(shape=(0, pm.STEPS_IN, pm.N_FEATURES)), np.ndarray(shape=(0, pm.STEPS_OUT))
        for file in data_file:
            xx_, y_ = obtain_vectors(file, power_curve)
            if xx_ is None or y_ is None:
                continue
            xx = np.concatenate((xx, xx_), axis=0)
            y = np.concatenate((y, y_), axis=0)
        return xx, y

    cpu, mem = load_data(data_file)
    dataset = []
    for series in range(len(cpu)):
        dataset.append([cpu[series], mem[series]])
    dataset = np.array(dataset)
    if len(dataset.shape) == 1:
        dataset = dataset.reshape(-1, 1)

    future_len = pm.STEPS_IN // dt.WEEK_IN_DAYS
    xx, y = split_sequence(sequence=dataset,
                           past_len=pm.STEPS_IN,
                           future_len=future_len,
                           steps_out=pm.STEPS_OUT,
                           filename=data_file,
                           power_curve=power_curve)
    # log.debug("Working with", xx.shape, " ", y.shape, "samples")

    return xx, y
