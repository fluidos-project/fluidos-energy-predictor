import json
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import src.parameters as pm
import src.support.dt as dt
from src.support.log import tqdm_wrapper

if __name__ == "__main__":
    granularity = pm.GRANULARITY
    # offset = pm.OFFSET
    gcd_folder = pm.GCD_FOLDER

    folder_start = f"{gcd_folder}/json"

    av = os.listdir(folder_start)
    av.sort()
    for i in range(len(av)):
        print(f"\t{i}: {av[i]}")

    try:
        name_start = av[int(input("Select file to parse: "))]
    except ValueError | IndexError:
        print("Invalid input, exiting...")
        exit(1)

    name_target = name_start.replace(".json", "").replace("_sorted", "")
    folder_target = f"{gcd_folder}/{name_target}"

    os.makedirs(folder_target, exist_ok=True)

    labels = ['start_time', 'end_time', 'machine_id', 'cpus', 'memory']

    newdata = np.array([])
    newdata = newdata.reshape(0, len(labels))
    with open(f'{folder_start}/{name_start}') as f:
        data = json.load(f)
        if len(data) == 0:
            print("No data in file, exiting...")
            exit(1)
        elif len(data) == 1:
            data = data[0]

        i = 0
        total_items = len(data)

        data = sorted(
            data,
            key=lambda q: int(q['start_time'])
        )

        print(f"Reading data from {folder_start}/{name_start}")

        for item in tqdm_wrapper(data):
            start = int(item['start_time'])
            end = int(item['end_time'])

            if len(newdata) > 1 and (
                    start == end  # same time
                    or end < start  # end before start
                    or start == newdata[-1][0]  # same start time as previous
                    or end == newdata[-1][1]  # same end time as previous
            ):
                continue

            mid = int(item['machine_id'])
            if "random_sample_usage" in item:
                cpu = float(item['random_sample_usage']['cpus'])
            else:
                cpu = float(item['cpus'])

            if "average_usage" in item:
                mem = float(item['average_usage']['memory'])
            else:
                mem = float(item['memory'])
            # make newdata an array of arrays
            newdata = np.append(newdata, np.array([[start, end, mid, cpu, mem]]), axis=0)

    # print(newdata)
    print("Length of newdata:", len(newdata))
    print(f"Each data point is 30 sec. Total time: {len(newdata) / 2 / dt.WEEK_IN_MINUTES} weeks")

    # Merge data points so that it becomes one every MINUTES minutes
    newnewdata = np.array([])
    newnewdata = newnewdata.reshape(0, 5)
    for i in tqdm_wrapper(range(0, len(newdata), granularity * 2)):
        if i + granularity * 2 > len(newdata):
            break
        start, end, mid, cpu, mem = newdata[i]
        for j in range(1, granularity * 2):
            _, _, _, cpu_, mem_ = newdata[i + j]
            cpu += cpu_
            mem += mem_
        cpu /= granularity * 2
        mem /= granularity * 2
        newnewdata = np.append(newnewdata, np.array([[start, end, mid, cpu, mem]]), axis=0)
    newdata = newnewdata

    print("Length of newdata:", len(newdata))
    print(f"Each data point is {granularity} min. Total time: {len(newdata) * granularity / dt.WEEK_IN_MINUTES} weeks")

    print("Computing moving average...")

    wsize = 2
    # smoothing with moving average
    for _ in range(1):
        for i in tqdm_wrapper(range(0, len(newdata))):
            if i < wsize:
                newdata[i][3] = np.mean(newdata[:i + wsize, 3])
                newdata[i][4] = np.mean(newdata[:i + wsize, 4])
            elif i >= len(newdata) - wsize:
                newdata[i][3] = np.mean(newdata[i - wsize:, 3])
                newdata[i][4] = np.mean(newdata[i - wsize:, 4])
            else:
                newdata[i][3] = np.mean(newdata[i - wsize:i + wsize, 3])
                newdata[i][4] = np.mean(newdata[i - wsize:i + wsize, 4])

    # Scale cpu and mem between 0 and 1
    scaler = MinMaxScaler()
    newdata[:, 3] = scaler.fit_transform(newdata[:, 3].reshape(-1, 1)).reshape(-1)
    newdata[:, 4] = scaler.fit_transform(newdata[:, 4].reshape(-1, 1)).reshape(-1)

    print("Writing data to ", folder_target, name_target)

    # split into
    with open(f'{folder_target}/{name_target}_all.csv', 'w') as f:
        f.write("timestamp,empty1,empty2,machine_id,cpu,mem,empty3\n")
        i = 1
        for item in newdata:
            start, end, mid, cpu, mem = item
            f.write(f"{i},,,{mid},{cpu},{mem},{start},{end}\n")
            i += 1

    # split into sequences of 8 days which overlap by 1 day
    interval = dt.DAY_IN_MINUTES // granularity
    days = 8
    for i in tqdm_wrapper(range(0, len(newdata), interval)):
        target = i + days * interval
        print(f"Splitting from t={i} to t={target}")
        with open(f'{folder_target}/{name_target}_seq_{i}-to-{target}.csv', 'w') as f:
            f.write("timestamp,empty1,empty2,machine_id,cpu,mem,empty3\n")
            j = 1
            for item in newdata[i:target]:
                start, end, mid, cpu, mem = item
                f.write(f"{j},,,{mid},{cpu},{mem},{start},{end}\n")
                j += 1

    os.makedirs(f"{folder_target}/plot", exist_ok=True)

    print("Plotting data...")
    # plot a week at time
    interval = dt.WEEK_IN_MINUTES // granularity
    for i in tqdm_wrapper(range(0, len(newdata), interval)):
        dd = newdata[i:i + interval]
        x = [item[0] for item in dd]
        y = [item[4] for item in dd]
        plt.figure(figsize=(30, 10))
        plt.plot(x, y)
        plt.savefig(f"{folder_target}/plot/{name_target}_plot_week_{i // interval}.png")
        plt.close()
