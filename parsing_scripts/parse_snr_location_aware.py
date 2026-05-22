import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Geod
import folium
from folium.plugins import HeatMap

from typing import List, Dict
from sklearn.neighbors import BallTree

colors = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000"]

def parseTimeToken(token: str) -> int:
    """
    Parses something like: 12:21:17.903627 into millisecond
    """
    nums = [float(x) for x in token.strip(']').split(':')]
    float_time = int(nums[0]) * 3600 + int(nums[1]) * 60 + round(nums[2], 3)
    return int(1000 * float_time)


def parsePositionAndSNR(file_base: str):
    # Parsing SNR values
    snrs = {}
    with open(file_base + "lw1_snr.txt", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) != 4:
                continue
            time = parseTimeToken(tokens[1])
            snrs[time] = float(tokens[-1])

    # Parsing position values
    pos = {}
    with open(file_base + "vehicleOut.txt", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(',')
            time = parseTimeToken(tokens[-3].split()[-1])
            pos[time] = [float(x) for x in tokens[1:4]]
    
    start_time = min(pos.keys())
    end_time = max(pos.keys())
    # print(f"Start of flight: {start_time}")
    # print(f"End of Flight: {end_time}")
    # print(f"Flight Duration: {(end_time - start_time) / 1000.0} seconds")

    # Clipping SNR data
    offset = 0
    filtered_snr = {}
    for x in snrs:
        if start_time + offset <= x <= end_time:
            filtered_snr[x] = snrs[x]

    snr_df = pd.DataFrame()
    snr_df.insert(0, "TIME", list(filtered_snr.keys()))
    snr_df.insert(1, "SNR", list(filtered_snr.values()))
    print(min(snr_df["TIME"]), max(snr_df["TIME"]))

    pos_df = pd.DataFrame()
    pos_df.insert(0, "TIME", list(pos.keys()))
    pos_df.insert(1, "LAT", [x[1] for x in list(pos.values())])
    pos_df.insert(2, "LON", [x[0] for x in list(pos.values())])
    pos_df.insert(3, "ALT", [x[2] for x in list(pos.values())])
    print(min(pos_df["TIME"]), max(pos_df["TIME"]))

    merged_df = pd.merge_asof(snr_df, pos_df, "TIME", direction="nearest")

    # Adding distance to LW-1
    g = Geod(ellps="WGS84")
    lw1 = [35.72750947, -78.69595819]
    dist = []
    for i in range(len(merged_df)):
        _a, _b, d = g.inv(lw1[0], lw1[1], merged_df["LAT"][i], merged_df["LON"][i])
        dist.append(d)
    
    merged_df.insert(4, "DISTANCE", dist)

    # plt.ylabel("SNR (dB)")
    # plt.xlabel("Time (seconds)")
    # plt.title("SNR vs. Time for Sionna+CHEM on Lake Wheeler Trajectory")
    # # plt.plot(merged_df["TIME"], 5000 / (merged_df["DISTANCE"] ** 2))
    # plt.plot(merged_df["TIME"], merged_df["SNR"])
    # plt.show()

    # Plotting heatmap of SNR vs. Position with Folium
    m = folium.Map(location=[35.7273023, -78.6962747], zoom_start=12, tiles="Esri.WorldImagery")

    max_snr = np.max(merged_df["SNR"])
    min_snr = np.min(merged_df["SNR"])

    data = [[merged_df["LAT"][i], merged_df["LON"][i], (merged_df["SNR"][i] - min_snr) / (max_snr - min_snr)] 
            for i in range(len(merged_df))]
    # data = [[merged_df["LAT"][i], merged_df["LON"][i], merged_df["SNR"][i]] 
    #         for i in range(len(merged_df))]
    HeatMap(data, blur=1, radius=1, min_opacity=0.6, overlay=False).add_to(m)
    
    m.save("trajectory.html")
    merged_df.to_csv(file_base + "_merged.csv")
    return merged_df
    

def mergeOnPosition(prefixes: List[str], df_paths: List[str]):
    """
    Merges a list of dataframes in one based on their coordinates
    Assumes the columns "LAT", "LON" are present in each dataframe
    """

    composite = pd.read_csv(df_paths[0]).rename(
        columns={"SNR": prefixes[0] + "_SNR"}
    )

    for i in range(1, len(df_paths)):
        temp = pd.read_csv(df_paths[i])

        existing = np.deg2rad(composite[["LAT", "LON"]])
        merging = np.deg2rad(temp[["LAT", "LON"]])

        tree = BallTree(merging, metric='haversine')
        _, indices = tree.query(existing, k=1)

        composite['NEAREST_INDEX'] = indices.flatten()
        composite.insert(0, prefixes[i] + "_SNR", temp["SNR"].iloc[composite['NEAREST_INDEX']].values)

    composite.to_csv("composite_data.csv")


def plotCompositeData(prefixes: List[str], composite_df: str, bounds: List[int]):
    """
    Plots the composite dataframe, should have multiple SNR columns
    named according to the prefixes
    """

    df = pd.read_csv(composite_df)
    min_time = np.min(df["TIME"])
    df = df[bounds[0] + min_time <= df["TIME"]]
    df = df[df["TIME"] <= bounds[1] + min_time]

    for i in range(len(prefixes)):
        plt.plot((df["TIME"] - min_time - bounds[0]) / 1000, 
                 df[f"{prefixes[i]}_SNR"], 
                 color='#'+colors[i].lower(), 
                 label=prefixes[i])
    
    plt.ylabel("SNR (dB)")
    plt.xlabel("Time (seconds)")
    plt.title("SNR vs. Time for Animal Health Building Trajectory")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    # The Sionna 2 and 3 datasets are non-representative, we reference SIONNA-1 and SIONNA-4
    # 3GGP is the best baseline for the CHEM-only data
    # It doesn't seem like changing the update rate changes anything remotely

    base_paths = [
        "../3gpp_results/",
        "../sionna_results/"
    ]

    for path in base_paths:
        res = parsePositionAndSNR(path)
        print(np.unique(res["LAT"]))
        plt.plot(res["SNR"])
    plt.show()
    exit()

    # prefixes = ["3GGP", "SIONNA-STANDARD", "SIONNA-BLENDER-MAP"]
    prefixes = ["3GPP", "SIONNA"]
    # mergeOnPosition(prefixes, [f"{x}_merged.csv" for x in base_paths])
    
    plotCompositeData(prefixes, "composite_data.csv", [0, 100000000000])


