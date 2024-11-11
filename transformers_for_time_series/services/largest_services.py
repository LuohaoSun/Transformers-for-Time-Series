import sys

sys.path.append("./transformers_for_time_series/external/LargeST")
import pandas as pd
import numpy as np

ca_df: pd.DataFrame = pd.read_hdf("./data/ca_2021_15min_resample/ca_his_2021.h5")  # type: ignore
ca_adj: np.ndarray = np.load("./data/ca_2021_15min_resample/ca_rn_adj.npy")

print(ca_df.head())
print(ca_adj.shape)
print(ca_adj)
