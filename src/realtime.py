import argparse
import datetime
import os
import parameters as pm
import re
import logging as log
from support.log import initialize_log
import src.model as modelmd
import json

from prometheus_api_client import PrometheusConnect, MetricsList, Metric
from prometheus_api_client.utils import parse_datetime
from flask import Flask
from main import ask_model_name

CPU_QUERY = '100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[15m])) * 100)'
MEM_QUERY = "avg by(instance) (100 * (1 - node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes))"


def main():
    parser = argparse.ArgumentParser(description="FLUIDOS WP6 T6.3 Model RealTime PoC")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name (if unspecified, will be prompted)",
    )
    parser.add_argument(
        "--telemetry",
        "-t",
        type=str,
        default=None,
        help="FLUIDOS Telemetry endpoint (i.e. http://localhost:46405/metrics)",
    )
    parser.add_argument(
        "--output_port",
        "-p",
        type=int,
        default=5000,
        help="Output port for the Flask server",
    )
    args = parser.parse_args()

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

    # Check if power_curve.json exists
    if not os.path.exists(local_model_folder + "/power_curve.json"):
        raise FileNotFoundError(
            "power_curve.json not found. Please train the model first."
        )

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
    cpu_data = prom.custom_query_range(CPU_QUERY,
        start_time=start_time,
        end_time=end_time,
        step="15m"
    )
    mem_data = prom.custom_query_range(MEM_QUERY,
        start_time=start_time,
        end_time=end_time,
        step="15m"
    )

    # [{'metric': {'instance': '10.244.0.5:9100'}, 'values': [[1713776805, '64.11895885416666'], [1713777705, '21.543016759776506'], [1713778605, '22.515642458100558'], [1713779505, '18.079050279329593'], [1713780405, '16.179050279329616'], [1713781305, '15.587709497206674'], [1713782205, '16.765642458100544'], [1713783105, '21.551955307262574'], [1713784005, '17.9030726256983'], [1713784905, '35.19358572699116'], [1713785805, '28.654469273743004'], [1713786705, '22.819553072625723']]}, {'metric': {'instance': '10.244.1.9:9100'}, 'values': [[1713776805, '64.17084830833332'], [1713777705, '21.52709497206702'], [1713778605, '22.528770949720666'], [1713779505, '18.076256983240242'], [1713780405, '16.175139664804476'], [1713781305, '15.589385474860322'], [1713782205, '16.769273743016754'], [1713783105, '21.55474860335194'], [1713784005, '17.95418994413403'], [1713784905, '35.19469273743016'], [1713785805, '28.684916201117332'], [1713786705, '22.83268156424579']]}, {'metric': {'instance': 'opentelemetrycollector.monitoring.svc.cluster.local:8090'}, 'values': [[1713776805, '64.43809564102564'], [1713777705, '21.096648044692728'], [1713778605, '25.10782122905026'], [1713779505, '20.633240223463687'], [1713780405, '18.8413407821229'], [1713781305, '18.243016759776538'], [1713782205, '16.27849162011171'], [1713783105, '24.181005586592192'], [1713784005, '20.337430167597773'], [1713784905, '37.26201117318434'], [1713785805, '30.85698324022347'], [1713786705, '22.464525139664772']]}]

    if len(cpu_data) == 0:
        raise ValueError("No CPU data found")

    if len(mem_data) == 0:
        raise ValueError("No memory data found")

    if len(cpu_data) != len(cpu_data):
        raise ValueError("CPU and memory data do not match")

    ts = {}
    for i in range(len(cpu_data)):
        instance = cpu_data[i]["metric"]["instance"]
        if instance not in ts:
            ts[instance] = {}
        for j in range(len(cpu_data[i]["values"])):
            timestamp = int(cpu_data[i]["values"][j][0])
            if timestamp not in ts[instance]:
                ts[instance][timestamp] = {}
            ts[instance][timestamp] = {
                "cpu": float(cpu_data[i]["values"][j][1]),
                "mem": float(mem_data[i]["values"][j][1]),
            }
        
    for node in ts:
        log.info(f"Node {node}")
        timedata = ts[node]
        if len(timedata) < pm.STEPS_IN:
            log.error(f"Data for node {node} is incomplete. Required {pm.STEPS_IN} steps, got {len(timedata)}")
            print(timedata)
            continue

        model = tf.keras.models.load_model(local_model_folder + "/model.keras")

        results = modelmd.predict_inmemory(
            model,
            timedata,
            json.load(open(local_model_folder + "/power_curve.json")),
        )
        log.info(f"Predictions: {results.yhat}")

if __name__ == "__main__":
    main()
