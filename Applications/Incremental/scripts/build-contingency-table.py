import sys
import copy
import json
import pandas as pd
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def readClustLst(fname):
    clustLst = {}
    clustId = 0
    with open(fname, "r") as f:
        while(True):
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                tokens = line.split()
                clustLst[clustId] = list(tokens)
            clustId = clustId + 1
    return clustLst

def build_contingency_table(oldClustFname, newClustFname, incClustFname):
    oldClustLst = readClustLst(oldClustFname)
    newClustLst = readClustLst(newClustFname)
    incClustLst = readClustLst(incClustFname)

    nClustOld = len(oldClustLst.keys())
    nClustNew = len(newClustLst.keys())
    nClustInc = len(incClustLst.keys())

    print("Number of old clusters:", nClustOld)
    print("Number of new clusters:", nClustNew)
    print("Number of inc clusters:", nClustInc)

    ri_table = np.zeros((nClustOld+nClustNew, nClustInc))
    for i in range(nClustOld):
        for j in range(nClustInc):
            x = frozenset(oldClustLst[i])
            y = frozenset(incClustLst[j])
            ri_table[i,j] = len(x.intersection(y))
            print("Comparing", "old", i, "vs", "inc", j, ":", ri_table[i,j])
    for i in range(nClustNew):
        for j in range(nClustInc):
            print("Comparing", "new", i, "vs", "new", j)
            x = frozenset(newClustLst[i])
            y = frozenset(incClustLst[j])
            ri_table[nClustOld+i,j] = len(x.intersection(y))
            print("Comparing", "new", i, "vs", "inc", j, ":", ri_table[nClustOld+i,j])
    np.savetxt("ri_table_virus.csv", ri_table, delimiter=",")
    fig = plt.figure(figsize=(8.5*3, 11*3))
    gs = GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(ri_table)
    plt.savefig("ri_table_virus.pdf")

if __name__=="__main__":
    # if(len(sys.argv) < 7):
        # print("Not enough parameters present")
    # else:
        # oldClustFname = ""
        # newClustFname = ""
        # incClustFname = ""
        # i = 0;
        # while(i < len(sys.argv)):
            # if sys.argv[i] == "--old":
                # oldClustFname = sys.argv[i+1]
            # if sys.argv[i] == "--new":
                # newClustFname = sys.argv[i+1]
            # if sys.argv[i] == "--inc":
                # incClustFname = sys.argv[i+1]
            # i = i + 1
    oldClustFname = "/global/cscratch1/sd/taufique/virus-incremental/20-split/virus_30_50.20.18.full"
    newClustFname = "/global/cscratch1/sd/taufique/virus-incremental/20-split/virus_30_50.20.19.m22.full"
    incClustFname = "/global/cscratch1/sd/taufique/virus-incremental/20-split/virus_30_50.20.19.full"
    print(oldClustFname)
    print(newClustFname)
    print(incClustFname)
    build_contingency_table(oldClustFname, newClustFname, incClustFname)

