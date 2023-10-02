import json
import os
import re

import bs4 as bs
import numpy as np
import pandas as pd
import requests as req

from src.support.log import tqdm_wrapper

URL = "https://www.spec.org/power_ssj2008/results/power_ssj2008.html"


def main():
    # Get list of all pages
    pages = get_pages(URL)
    redownload = False

    print(f"Found {len(pages)} pages.")

    os.makedirs("data/spec2008", exist_ok=True)
    failed = []

    if redownload:
        for page in tqdm_wrapper(pages):
            try:
                fn = re.sub(r"[^a-zA-Z0-9]", "-", "-".join([str(i) for i in page["cpu"].values()])) \
                    .replace("--", "-") \
                    .lower()
                if fn[-1] == "-":
                    fn = fn[:-1]

                page["filename"] = fn
                os.makedirs(f"data/spec2008/{fn}", exist_ok=True)
                content = len(os.listdir(f"data/spec2008/{fn}"))

                data = parse_page(page["link"])
                page["data"] = pd.DataFrame(data).to_dict(orient="records")
                open(f"data/spec2008/{fn}/{content}.json", "w").write(json.dumps(page, indent=4))
            except Exception as e:
                failed.append(page)
                print("Error parsing page: " + page["pc_model"])
                print(e)
                continue

        open("data/spec2008/failed.json", "w").write(json.dumps(failed, indent=4))

    # Merge all files
    os.makedirs("data/spec2008_agg", exist_ok=True)

    for folder in os.listdir("data/spec2008"):
        if folder == ".DS_Store":
            continue

        if not os.path.isdir(f"data/spec2008/{folder}"):
            print(f"Skipping file {folder} as it is not a folder.")
            continue

        if os.path.exists(f"data/spec2008_agg/{folder}.json"):
            print(f"Skipping folder {folder} as it already exists.")
            continue

        merge(folder)


def get_pages(url):
    ret = []
    r = req.get(url)
    soup = bs.BeautifulSoup(r.text, "html.parser")
    # find div with class=.resultsTable
    table = soup.find("table")
    # find all rows
    rows = table.findAll("tr")
    for row in rows:
        # save info about the model and then the link to the html file
        cols = row.findAll("td")
        if len(cols) > 0:
            ret.append({
                "pc_vendor": cols[0].text,
                "pc_model": cols[1].text.split("\n")[0],
                "link": "https://www.spec.org/power_ssj2008/results/" + cols[1].find("a")["href"],
                "cpu": {
                    "model": cols[4].text,
                    "mhz": cols[5].text,
                    "chips": cols[6].text,
                    "cores": cols[7].text,
                    "threads": cols[8].text
                },
                "memory": cols[9].text
            })
            try:
                ret[-1]["cpu"]["mhz"] = int(ret[-1]["cpu"]["mhz"])
            except ValueError:
                ret[-1]["cpu"]["mhz"] = None

            try:
                ret[-1]["cpu"]["chips"] = int(ret[-1]["cpu"]["chips"])
            except ValueError:
                ret[-1]["cpu"]["chips"] = None

            try:
                ret[-1]["cpu"]["cores"] = int(ret[-1]["cpu"]["cores"])
            except ValueError:
                ret[-1]["cpu"]["cores"] = None

            try:
                ret[-1]["cpu"]["threads"] = int(ret[-1]["cpu"]["threads"])
            except ValueError:
                ret[-1]["cpu"]["threads"] = None

            try:
                ret[-1]["memory"] = int(ret[-1]["memory"])
            except ValueError:
                ret[-1]["memory"] = None

    return ret


def parse_page(url):
    # print("...parsing page: " + url)
    r = req.get(url)
    soup = bs.BeautifulSoup(r.text, "html.parser")
    # find div with class=.resultsTable
    div = soup.find("div", {"class": "resultsTable"})
    # find table
    table = div.find("table")

    # parse header, if a column has colspan, then append the text of the next row
    header = table.find("thead")
    header_rows = header.findAll("tr")
    labels = []
    i = 0
    header_cols = header_rows[i].findAll("th")
    for j in range(len(header_cols)):
        if "colspan" in header_cols[j].attrs:
            sub = [c.text for c in header_rows[i + 1].findAll("th")]
            for k in range(int(header_cols[j]["colspan"])):
                labels.append(header_cols[j].text + "/" + sub[k])
        else:
            labels.append(header_cols[j].text)

    # parse body
    body = table.find("tbody")
    body_rows = body.findAll("tr")
    body_rows = [[c.text for c in r.findAll("td")] for r in body_rows]
    body_rows = [r for r in body_rows if len(r) == len(labels)]

    df = pd.DataFrame(body_rows, columns=labels)

    # transform % to float
    df["Performance/Target Load"] = df["Performance/Target Load"].apply(lambda x: float(x[:-1]) / 100)
    df["Performance/Actual Load"] = df["Performance/Actual Load"].apply(lambda x: float(x[:-1]) / 100)

    # transform ops to int
    df["Performance/ssj_ops"] = df["Performance/ssj_ops"].apply(lambda x: int(x.replace(",", "")))

    # transform power to float
    df["Power"] = df["Power"].apply(lambda x: float(x.replace(",", "")))

    # transform ratio to float
    df["Performance to Power Ratio"] = df["Performance to Power Ratio"].apply(lambda x: float(x.replace(",", "")))

    pd.set_option('display.max_columns', None)

    return df


def merge(folder):
    print("Merging folder " + folder)
    data = []
    for file in os.listdir("data/spec2008/" + folder):
        if not file.endswith(".json"):
            raise Exception("Skipping file " + file + " as it is not a json file.")
        data.append(json.load(open("data/spec2008/" + folder + "/" + file, "r")))

    if len(data) == 0:
        print("No data found for folder " + folder)
        return

    # Verify that all data is the same, convert to lower case
    for key in data[0]["cpu"].keys():
        if type(data[0]["cpu"][key]) == str:
            target = re.sub(r"[^a-zA-Z0-9]", "-", data[0]["cpu"][key]).replace("--", "-").lower()
            check = [re.sub(r"[^a-zA-Z0-9]", "-", data[i]["cpu"][key]).replace("--", "-").lower() == target
                     for i in range(len(data)) if i > 0]
        else:
            check = [data[i]["cpu"][key] == data[0]["cpu"][key] for i in range(len(data)) if i > 0]

        if not all(check):
            raise Exception(f"CPU Data is not the same for folder {folder}: {check} for {key}")

    res = data[0].copy()
    res["data"] = {
        "Performance to Power Ratio": np.round([
            np.mean([data[i]["data"][j]["Performance to Power Ratio"] for i in range(len(data))])
            for j in range(len(data[0]["data"]))
        ], 3).tolist(),
        "Performance/Actual Load": np.round([
            np.mean([data[i]["data"][j]["Performance/Actual Load"] for i in range(len(data))])
            for j in range(len(data[0]["data"]))
        ], 3).tolist(),
        "Performance/Target Load": [data[0]["data"][i]["Performance/Target Load"]
                                    for i in range(len(data[0]["data"]))],
        "Performance/ssj_ops": np.round([
            np.mean([data[i]["data"][j]["Performance/ssj_ops"] for i in range(len(data))])
            for j in range(len(data[0]["data"]))
        ], 3).tolist(),
        "Power": np.round([
            np.mean([data[i]["data"][j]["Power"] for i in range(len(data))])
            for j in range(len(data[0]["data"]))
        ], 3).tolist(),
    }

    res["count"] = len(data)

    json.dump(res, open("data/spec2008_agg/" + folder + ".json", "w"), indent=4)


if __name__ == "__main__":
    main()
