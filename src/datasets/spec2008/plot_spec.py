import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# {
#     "pc_vendor": "Sugon",
#     "pc_model": "A620-G30(AMD EPYC 7351)",
#     "link": "https://www.spec.org/power_ssj2008/results/res2018q3/power_ssj2008-20180716-00830.html",
#     "cpu": {
#         "model": "AMD EPYC 7351",
#         "mhz": 2400,
#         "chips": 2,
#         "cores": 32,
#         "threads": 64
#     },
#     "memory": 128,
#     "filename": "amd-epyc-7351-2400-2-32-64",
#     "data": [
#         {
#             "Performance/Target Load": 1.0,
#             "Performance/Actual Load": 0.997,
#             "Performance/ssj_ops": 2721413,
#             "Power": 278.0,
#             "Performance to Power Ratio": 9773.0
#         },
#         {


def main():
    files = os.listdir("data/spec2008_agg")

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind")

    for file in files:
        if file == ".DS_Store":
            continue

        with open("data/spec2008_agg/" + file, "r") as f:
            f = json.load(f)
            df = pd.DataFrame(f["data"])
            df.sort_values(by=["Power"], inplace=True)
            df["Power"] = df["Power"].astype(float)
            # plot x = performance/target load, y = power
            sns.lineplot(
                x="Performance/Target Load",
                y="Power",
                data=df,
                ax=ax,
                label=f["pc_model"],
                alpha=0.1,
                color="blue",
            )

    ax.set_xlabel("Performance/Target Load")
    ax.set_ylabel("Power (W)")
    ax.set_title("SPEC2008 Power Consumption")
    ax.legend_.remove()
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1000)
    plt.show()


if __name__ == "__main__":
    main()
