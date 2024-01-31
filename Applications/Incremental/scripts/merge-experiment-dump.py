import sys
import copy
import json
import os
#import pandas as pd
import pathlib
import glob
import csv

#all_df = []
headers = []
data = []

# path = pathlib.Path("/pscratch/sd/t/taufique/ipdps2024-experiments")
# path = pathlib.Path("/pscratch/sd/t/taufique/")
path = pathlib.Path("/home/mth/Data/nersc/temp/parameter-study/eukarya/")
for p in path.rglob("*"):
    if p.name.startswith("csv"):
        with open(p.resolve(), "r") as f:
            dict_reader = csv.DictReader(f)
            headers += dict_reader.fieldnames
            count = 0
            for row in dict_reader:
                data.append(row)
                count = count + 1
            print(count, "rows", "in file:", p.resolve())

headers = sorted(set(headers))
#print(headers)
#print(len(data))
print("Total rows:", len(data))

with open("ipdps2024-experiments.csv", "w") as f:
    dict_writer = csv.DictWriter(f, fieldnames=headers)
    dict_writer.writeheader()
    dict_writer.writerows(data)
