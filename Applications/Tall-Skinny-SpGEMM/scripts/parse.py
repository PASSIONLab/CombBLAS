import os, sys, argparse, logging
from scipy.io import mmread
import sys
import copy
import json
import os
import pandas as pd
import csv

if __name__=="__main__":
    data= ""
    n = None
    p = None
    d = None
    s = None
    alg = None
    comm = -1
    comp = -1
    tot = -1
    logf = None
    csvf = None
    abcast = -1
    bbcast = -1
    local_mult = -1
    layer_merge = -1
    fiber_reduct = -1
    fiber_merge = -1

    for i in range(1, len(sys.argv) ):
        if sys.argv[i] == "--data":
            data = sys.argv[i+1]
        elif sys.argv[i] == "--n":
            n = int(sys.argv[i+1])
        elif sys.argv[i] == "--p":
            p = int(sys.argv[i+1])
        elif sys.argv[i] == "--d":
            d = int(sys.argv[i+1])
        elif sys.argv[i] == "--s":
            s = float(sys.argv[i+1])
        elif sys.argv[i] == "--alg":
            alg = sys.argv[i+1]
        elif sys.argv[i] == "--logf":
            logf = sys.argv[i+1]
        elif sys.argv[i] == "--csvf":
            csvf = sys.argv[i+1]


    if alg == "petsc":
        with open(logf, "r") as lf:
            for line in lf:
                line = line.strip()
                tokens = line.split()
                if (line.startswith("IO")):
                    tot = float(tokens[3])
    elif alg == "summa2d":
        state = 0
        with open(logf, "r") as lf:
            for line in lf:
                line = line.strip()
                tokens = line.split()
                if (line.startswith("After permutation")):
                    state = 1
                elif (line.startswith("[Mult_AnXBn_Synch] Abcasttime:")):
                    if state == 1:
                        abcast = float(tokens[2])
                elif (line.startswith("[Mult_AnXBn_Synch] Bbcasttime:")):
                    if state == 1:
                        bbcast = float(tokens[2])
                elif (line.startswith("[Mult_AnXBn_Synch] LocalSpGEMMtime:")):
                    if state == 1:
                        local_mult = float(tokens[2])
                elif (line.startswith("[Mult_AnXBn_Synch] Mergetime:")):
                    if state == 1:
                        layer_merge = float(tokens[2])
                elif (line.startswith("Time taken for Mult_AnXBn_Synch:")):
                    tot = float(tokens[4])
        comm = abcast + bbcast
        comp = local_mult + layer_merge
        # tot = comm + comp
    elif alg == "summa3d":
        with open(logf, "r") as lf:
            for line in lf:
                line = line.strip()
                tokens = line.split()
                if (line.startswith("[SUMMA3D]")):
                    # if comm == -1:
                        # comm = 0
                    # if comp == -1:
                        # comp = 0
                    # print(tokens)
                    if (tokens[1] == 'Abcast_time:'):
                        # print(tokens)
                        abcast = float(tokens[2])
                    elif (tokens[1] == 'Bbcast_time:'):
                        bbcast = float(tokens[2])
                    elif (tokens[1] == 'Local_multiplication_time:'):
                        local_mult = float(tokens[2])
                    elif (tokens[1] == 'Merge_layer_time:'):
                        layer_merge = float(tokens[2])
                    elif (tokens[1] == 'Reduction time:'):
                        fiber_reduct = float(tokens[2])
                    elif (tokens[1] == 'Merge_fiber_time:'):
                        fiber_merge = float(tokens[2])
                elif (line.startswith("Time taken for Mult_AnXBn_SUMMA3D:")):
                    tot = float(tokens[4])
        comm = abcast + bbcast + fiber_reduct
        comp = local_mult + layer_merge + fiber_merge
        # tot = comm + comp
        pass
    
    out = data + "," + str(n) + "," + str(p) + "," + str(d) + "," + str(s) + "," + alg + "," + str(comm) + "," + str(comp) + "," + str(tot)+ "," + str(abcast) + "," + str(bbcast) + "," + str(local_mult) + "," + str(layer_merge) + "," + str(fiber_reduct) + "," + str(fiber_merge)
    print(out)
