import sys
import copy
import json
import os
import pandas as pd
import pathlib
import glob

all_df = []

# path = pathlib.Path("/pscratch/sd/t/taufique/ipdps2024-experiments")
path = pathlib.Path("/pscratch/sd/t/taufique/")
for p in path.rglob("*"):
    if p.name.startswith("csv"):
        print(p.resolve())
        df = pd.read_csv(p.resolve(), index_col=None, header=0)
        print(df.iloc[-1])
        all_df.append(df)

merged_df = pd.concat(all_df, axis=0, ignore_index=True)
merged_df.to_csv("ipdps2024-experiments.csv", index=False)
