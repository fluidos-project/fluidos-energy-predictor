import argparse
import datetime
import os
import parameters as pm
import re

from prometheus_api_client import PrometheusConnect, MetricsList, Metric
from prometheus_api_client.utils import parse_datetime
from flask import Flask
from main import ask_model_name

CPU_QUERY = 'cpu=100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[15s])) * 100)'
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

    if not os.path.exists(pm.MODEL_FOLDER):
        raise FileNotFoundError("Model folder not found. Please train a model first.")

    models = sorted([i.split(".")[0] for i in os.listdir(pm.MODEL_FOLDER)])

    if args.model is not None:
        model_name = args.model
    else:
        model_name = ask_model_name(models)
        if model_name is None:
            raise ValueError("No model selected")
        if not os.path.exists(os.path.join(pm.MODEL_FOLDER, model_name + ".keras")):
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
            "Telemetry endpoint must be specified. Example: http://localhost:46405/metrics"
        )
    else:
        if not re.match(r"https?://.*:[0-9]{4,5}/metrics", args.telemetry):
            raise ValueError(
                "Invalid telemetry endpoint. Example: http://localhost:46405/metrics"
            )

    # Start pulling data from the telemetry endpoint
    # we want CPU and memory usage
    prom = PrometheusConnect(url=args.telemetry, disable_ssl=True)

    start_time = parse_datetime("2w")
    end_time = parse_datetime("now")
    chunk_size = datetime.timedelta(days=1)

    cpu_data = prom.get_metric_range_data(
        CPU_QUERY, start_time=start_time, end_time=end_time, chunk_size=chunk_size
    )
    mem_data = prom.get_metric_range_data(
        MEM_QUERY, start_time=start_time, end_time=end_time, chunk_size=chunk_size
    )

    cpu_data = MetricsList(cpu_data)
    mem_data = MetricsList(mem_data)


if __name__ == "__main__":
    main()
