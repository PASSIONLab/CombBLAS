import os
from pathlib import Path
import json
from subprocess import Popen, PIPE
import subprocess
import time
import sys
import pandas as pd

fname = "/global/cscratch1/sd/taufique/virus/" + "vir_vs_vir_30_50length_propermm.mtx"
nsplit = 20
base = 1
scores = []
for alg in ["inc", "incfake"]:
    for step in range(1,nsplit):
        gtfile = fname + "." + str(nsplit) + "." + "full" + "." + str(step)
        algfile = fname + "." + str(nsplit) + "." + alg + "." + str(step)
        cmdlist = [
                "./fscore",
                "-base", str(base),
                "-M1", gtfile,
                "-M2", algfile
                ]
        print(" ".join(cmdlist))
        proc = subprocess.Popen(cmdlist, stdout=PIPE)
        proc.wait()
        for line in proc.stdout:
            dec_line = line.decode("utf-8")
            toks = dec_line.split()
            if (len(toks) == 3):
                if toks[0] == "F" and toks[1] == "score:":
                    fscore = float(toks[2])
                    info = {}
                    info["fname"] = fname
                    info["nsplit"] = nsplit
                    info["alg"] = alg
                    info["step"] = step
                    info["fscore"] = fscore
                    scores.append(info)
                    df = pd.DataFrame(scores)
                    df.to_csv (fname + "." + str(nsplit) + ".fscore", index = False, header=True)
                    print(info)


