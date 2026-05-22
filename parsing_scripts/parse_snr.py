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


def parsePositionAndSNR(file_base: str, bounds):
    # Parsing SNR values
    snrs = {}
    with open(file_base + "/lw1_snr.txt", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) != 4:
                continue
            time = parseTimeToken(tokens[1])
            snrs[time] = float(tokens[-1])

    # Defined based on the specific experiment
    offset = 0
    filtered_snr = {}
    for x in snrs:
        if bounds[0] + offset <= x <= bounds[1]:
            filtered_snr[x] = snrs[x]

    snr_df = pd.DataFrame()
    snr_df.insert(0, "TIME", list(filtered_snr.keys()))
    snr_df.insert(1, "SNR", list(filtered_snr.values()))

    plt.ylabel("SNR (dB)")
    plt.xlabel("Time (seconds)")
    plt.title("SNR vs. Time for 3GGP on Lake Wheeler Trajectory")
    # plt.plot(merged_df["TIME"], 5000 / (merged_df["DISTANCE"] ** 2))
    plt.plot(snr_df["TIME"], snr_df["SNR"])
    plt.show()

    return list(filtered_snr.values())
    

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

    # base_paths = [
    #     "standard_lidar/3GGP/2026-04-30_16_13_04",
    #     "standard_lidar/CHEM-SIONNA-4/2026-05-01_10_42_45",
    #     "blender_lidar/CHEM-SIONNA-1/2026-05-01_12_55_22",
    # ]

    base_paths = [
        "../sionna_results",
        "../3gpp_results"
    ]

    bounds = [
        [36388000, 36497510],
        [37478000, 37513730]
    ]

    r1 = bounds[0][1] - bounds[0][0]
    r2 = bounds[1][1] - bounds[1][0]
    scale = r1 / r2
    print(scale)

    snrs = []
    for i in range(2):
        snrs.append(np.array(parsePositionAndSNR(base_paths[i], bounds[i])))

    old_indices = np.arange(len(snrs[1]))
    new_indices = np.linspace(0, len(snrs[1]) - 1, len(snrs[0]))

    # Interpolate to the new length
    snrs[1] = np.interp(new_indices, old_indices, snrs[1])

    plt.plot(snrs[0], color="green", label="Sionna SNR")
    plt.plot(snrs[1], color="blue", label="3GPP SNR")
    plt.legend(loc="best")
    plt.xlabel("Observation")
    plt.ylabel("SNR (dB)")
    plt.title("SNR comparison for Sionna and 3GPP")
    plt.show()


