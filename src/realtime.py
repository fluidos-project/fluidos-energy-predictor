import argparse
import datetime
import json
import logging as log
import os
import re

from flask import Flask
import numpy as np
import tensorflow as tf
from prometheus_api_client import PrometheusConnect

import parameters as pm
import model as modelmd
from main import ask_model_name
from support.log import initialize_log

CPU_QUERY = (
    '100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[15m])) * 100)'
)
MEM_QUERY = "avg by(instance) (100 * (1 - node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes))"


# noinspection PyUnresolvedReferences
def main():
    parser = argparse.ArgumentParser(description="FLUIDOS WP6 T6.3 Model RealTime PoC")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name (if unspecified, will be prompted)",
    )
    telemetry_or_debug = parser.add_mutually_exclusive_group(required=True)
    telemetry_or_debug.add_argument(
        "--telemetry",
        "-t",
        type=str,
        default=None,
        help="FLUIDOS Telemetry endpoint (i.e. http://localhost:46405/metrics)",
    )
    # alternative: debug, cpufile, memfile, all three required together
    telemetry_or_debug.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Debug mode: use CPU and memory data from files",
    )
    parser.add_argument(
        "--cpufile",
        type=str,
        default=None,
        help="CPU data file (required in debug mode)",
    )
    parser.add_argument(
        "--memfile",
        type=str,
        default=None,
        help="Memory data file (required in debug mode)",
    )
    parser.add_argument(
        "--output_port",
        "-p",
        type=int,
        default=5000,
        help="Output port for the Flask server",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate data to the required length if it is longer, aggressively delete keys if data is shorter",
    )
    args = parser.parse_args()

    if args.debug and (args.cpufile is None or args.memfile is None):
        raise ValueError("CPU and memory files are required in debug mode")
    elif not args.debug and (args.cpufile is not None or args.memfile is not None):
        raise ValueError("CPU and memory files are only allowed in debug mode")

    initialize_log("INFO")

    if not os.path.exists(pm.MODEL_FOLDER):
        raise FileNotFoundError("Model folder not found. Please train a model first.")

    models = sorted([i.split(".")[0] for i in os.listdir(pm.MODEL_FOLDER)])

    if args.model is not None:
        model_name = args.model
    else:
        model_name = ask_model_name(models)
        if model_name is None:
            raise ValueError("No model selected")
        if not os.path.exists(os.path.join(pm.MODEL_FOLDER, model_name, "model.keras")):
            raise FileNotFoundError(
                f"Model {model_name} not found, please train it first"
            )

    local_model_folder = os.path.join(pm.MODEL_FOLDER, model_name)
    required_steps = pm.STEPS_IN

    # Check if power_curve.json exists
    if not os.path.exists(
        local_model_folder + "/power_curve_cpu.csv"
    ) or not os.path.exists(local_model_folder + "/power_curve_mem.csv"):
        raise FileNotFoundError("Power curve not found. Please train the model first.")

    # [{'metric': {'instance': '10.244.0.5:9100'}, 'values': [[1713776805, '64.11895885416666'],
    # [1713777705, '21.543016759776506'], [1713778605, '22.515642458100558'], [1713779505, '18.079050279329593'],
    # [1713780405, '16.179050279329616'], [1713781305, '15.587709497206674'], [1713782205, '16.765642458100544'],
    # [1713783105, '21.551955307262574'], [1713784005, '17.9030726256983'], [1713784905, '35.19358572699116'],
    # [1713785805, '28.654469273743004'], [1713786705, '22.819553072625723']]},
    # {'metric': {'instance': '10.244.1.9:9100'}, 'values': [[1713776805, '64.17084830833332'],
    # [1713777705, '21.52709497206702'], [1713778605, '22.528770949720666'], [1713779505, '18.076256983240242'],
    # [1713780405, '16.175139664804476'], [1713781305, '15.589385474860322'], [1713782205, '16.769273743016754'],
    # ...
    if not args.debug:
        if args.telemetry is None:
            raise ValueError(
                "Telemetry endpoint must be specified. Example: http://localhost:46405/"
            )
        else:
            if not re.match(r"https?://.*:[0-9]{4,5}/", args.telemetry):
                raise ValueError(
                    "Invalid telemetry endpoint. Example: http://localhost:46405/"
                )

        # Start pulling data from the telemetry endpoint
        # we want CPU and memory usage
        prom = PrometheusConnect(url=args.telemetry, disable_ssl=True)

        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(weeks=2)
        cpu_data = prom.custom_query_range(
            CPU_QUERY, start_time=start_time, end_time=end_time, step="15m"
        )
        mem_data = prom.custom_query_range(
            MEM_QUERY, start_time=start_time, end_time=end_time, step="15m"
        )
    else:
        # we assume data structured like a prometheus query
        with open(args.cpufile) as f:
            cpu_data = json.load(f)
        with open(args.memfile) as f:
            mem_data = json.load(f)

    if len(cpu_data) == 0:
        raise ValueError("No CPU data found")

    if len(mem_data) == 0:
        raise ValueError("No memory data found")

    if len(cpu_data) != len(cpu_data):
        raise ValueError("CPU and memory data do not match")

    ts = {}
    for i in range(len(cpu_data)):
        instance = cpu_data[i]["metric"]["instance"]
        log.info("Processing instance " + instance)
        if instance not in ts:
            ts[instance] = {}
        if len(cpu_data[i]["values"]) != len(mem_data[i]["values"]):
            log.error(
                f"CPU and memory data length do not match ({len(cpu_data[i]['values'])} for CPU vs {len(mem_data[i]['values'])} for memory)"
            )
            log.info("For your convenience, here is the intersection of the timestamps")
            cpu_ts = {int(i[0]) for i in cpu_data[i]["values"]}
            mem_ts = {int(i[0]) for i in mem_data[i]["values"]}
            log.info(f"Timestamps in CPU but not in memory: {cpu_ts - mem_ts}")
            log.info(f"Timestamps in memory but not in CPU: {mem_ts - cpu_ts}")
            truncation = False
            if not args.truncate:
                print("Do you wish to truncate the data?")
                print("1. Yes")
                print("2. No")
                choice = input("Choice: ")
                if choice == "1":
                    truncation = True
                else:
                    truncation = False
            else:
                truncation = True

            if truncation:
                # intersect timestamps
                cpu_ts = {int(i[0]) for i in cpu_data[i]["values"]}
                mem_ts = {int(i[0]) for i in mem_data[i]["values"]}
                common_ts = cpu_ts.intersection(mem_ts)
                cpu_data[i]["values"] = [
                    i for i in cpu_data[i]["values"] if int(i[0]) in common_ts
                ]
                mem_data[i]["values"] = [
                    i for i in mem_data[i]["values"] if int(i[0]) in common_ts
                ]
                # Sanity check
                if len(cpu_data[i]["values"]) != len(mem_data[i]["values"]):
                    raise ValueError(
                        "Data length mismatch even after truncation. Please check the data."
                    )
                else:
                    log.info("Data truncated successfully.")
            else:
                log.error("We cannot proceed. Adios!")
                exit(1)

        if len(cpu_data[i]["values"]) == 0:
            raise ValueError("No data found for instance " + instance)
        if len(cpu_data[i]["values"]) != required_steps:
            log.error(
                f"Data length for instance {instance} is wrong. Required {required_steps} steps, got {len(cpu_data[i]['values'])}"
            )
            if len(cpu_data[i]["values"]) < required_steps:
                raise ValueError(
                    f"Data length for instance {instance} is too short. Required {required_steps} steps, got {len(cpu_data[i]['values'])}"
                )
            else:
                truncation = False
                if not args.truncate:
                    print("Do you wish to truncate the data?")
                    print("1. Yes")
                    print("2. No")
                    choice = input("Choice: ")
                    if choice == "1":
                        truncation = True
                    else:
                        truncation = False
                else:
                    truncation = True

                if truncation:
                    cpu_data[i]["values"] = cpu_data[i]["values"][:required_steps]
                    mem_data[i]["values"] = mem_data[i]["values"][:required_steps]
                else:
                    log.error("We cannot proceed. Adios!")
                    exit(1)

        for j in range(len(cpu_data[i]["values"])):
            timestamp = int(cpu_data[i]["values"][j][0])
            if timestamp not in ts[instance]:
                ts[instance][timestamp] = {}
            ts[instance][timestamp] = {
                "cpu": float(cpu_data[i]["values"][j][1]),
                "mem": float(mem_data[i]["values"][j][1]),
            }

    metrics = {}
    for node in ts:
        log.info(f"Node {node}")
        timedata = ts[node]

        model = tf.keras.models.load_model(local_model_folder + "/model.keras")

        power_curve_cpu = np.loadtxt(
            local_model_folder + "/power_curve_cpu.csv", delimiter=","
        )
        power_curve_mem = np.loadtxt(
            local_model_folder + "/power_curve_mem.csv", delimiter=","
        )
        power_curve = [power_curve_cpu, power_curve_mem]

        results = modelmd.predict_inmemory(
            model,
            timedata,
            power_curve,
        )
        metrics[node] = results.tolist()

    # Finally expose the data with a Flask server
    # We will expose the data in the following format:
    # {"node1": 10, "node2": 20, "node3": 30}
    app = Flask(__name__)

    @app.route("/data")
    def data():
        return metrics

    app.run(port=args.output_port)


if __name__ == "__main__":
    main()
