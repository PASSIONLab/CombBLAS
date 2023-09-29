import sys
import copy
import json
import os
import pandas as pd
import csv

def normalize_json(data: dict) -> dict:
    new_data = dict()
    for key, value in data.items():
        if not isinstance(value, dict):
            new_data[key] = value
        else:
            for k, v in value.items():
                new_data[key + "_" + k] = v
      
    return new_data

state_start = -1
state_jombie = -2
state_ready = 0
state_subgraph_extraction = 1
state_cluster_m22 = 2
state_prep_minc = 3
state_cluster_minc = 4

if __name__=="__main__":
    # print(sys.argv)
    # if(len(sys.argv) < 4):
        # print("Usage: <python_script> <dataname> <ifilename> <ofilename>")
        # print("It will create another file with extension \"stats\", i.e. \"<filename>.stats\", containing stats collected from <filename>")
        # exit(1)
    dataname = sys.argv[1]
    ifname = sys.argv[2]
    ofname = sys.argv[3]

    current_state = state_start
    current_split = -1
    split_stat = {}
    metadata = {}
    metadata["dataname"] = dataname
    stats = []
    with open(ifname, "r") as f:
        for line in f:
            line = line.strip()
            tokens = line.split()
            # Read the metadata
            if (line.startswith("Process Grid") ):
                metadata["nproc"] = int(tokens[7]) * int(tokens[9])
                metadata["nthread"] = int(tokens[11])
                metadata["ncore"] = metadata["nproc"] * metadata["nthread"]
            elif (line.startswith("Input directory") ):
                metadata["input-directory"] = tokens[2]
            elif (line.startswith("Input file prefix") ):
                metadata["input-file-prefix"] = tokens[3]
            elif (line.startswith("Number of splits") ):
                metadata["nsplit"] = int(tokens[3])
            elif (line.startswith("Incremental clustering kick-in step") ):
                metadata["incremental-kickin-step"] = int(tokens[4])
            elif (line.startswith("Summary threshold") ):
                metadata["summary-threshold"] = int(tokens[2])
            elif (line.startswith("Selective prune threshold") ):
                metadata["selective-prune-threshold"] = int(tokens[3])
            elif (line.startswith("Output directory") ):
                metadata["output-directory"] = tokens[2]
            elif (line.startswith("Per process memory") ):
                metadata["per-process-memory"] = int(tokens[3])
            # Read per split stats
            elif (line.startswith("[Start] Split")):
                if(current_state == state_start):
                    current_split = int(tokens[2])
                    split_stat = {}
                    split_stat["split"] = current_split
                    current_state = state_ready
            elif (line.startswith("[End] Split")):
                # Read fscore for the specific file for current split
                #fscore_filename = metadata["output-directory"] + str(current_split) + ".fscore"
                fscore_filename = "/home/mth/Data/nersc/temp/parameter-study/eukarya/incremental." + \
                                  metadata["dataname"] + "." + \
                                  str(metadata["nsplit"]) + "." + \
                                  str(metadata["incremental-kickin-step"]) + "." + \
                                  str(metadata["summary-threshold"]) + "." + \
                                  str(metadata["selective-prune-threshold"]) + ".perlmutter_cpu.node_8.proc_8.thread_16" + "/" + str(current_split) + ".fscore"
                fscore_filename = metadata["output-directory"] + str(current_split) + ".fscore"
                fscore = -1.0
                if(os.path.exists(fscore_filename)):
                    with open(fscore_filename, 'rb') as f:
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                        last_line = f.readline().decode()
                        last_line = last_line.strip()
                        last_line_tokens = last_line.split()
                        print(last_line_tokens)
                        if (last_line.startswith("F score")): 
                            fscore = float(last_line_tokens[2])
                        else:
                            print("Could not read fscore from", fscore_filename);
                            pass
                else:
                    print(fscore_filename, "does not exist")
                    pass
                split_stat["fscore"] = fscore
                if(current_state == state_ready):
                    # stats.append(copy.deepcopy(split_stat));
                    stats.append(dict(split_stat))
                    current_state = state_start
            elif (line.startswith("[Start] Subgraph extraction")):
                if (current_state == state_ready):
                    current_state = state_subgraph_extraction
            elif (line.startswith("[End] Subgraph extraction")):
                if (current_state == state_subgraph_extraction):
                    current_state = state_ready
            elif (line.startswith("[Start] Clustering M22")):
                if (current_state == state_ready):
                    current_state = state_cluster_m22
            elif (line.startswith("[End] Clustering M22")):
                if (current_state == state_cluster_m22):
                    current_state = state_ready
            elif (line.startswith("[Start] Preparing Minc")):
                if (current_state == state_ready):
                    current_state = state_prep_minc
            elif (line.startswith("[End] Preparing Minc")):
                if (current_state == state_prep_minc):
                    current_state = state_ready
            elif (line.startswith("[Start] Clustering Minc") or line.startswith("[Start] Clustering Mall")):
                if (current_state == state_ready):
                    current_state = state_cluster_minc
            elif (line.startswith("[End] Clustering Minc")):
                if (current_state == state_cluster_minc):
                    current_state = state_ready
            else:
                if(current_state == state_subgraph_extraction):
                    # # Not collecting stats of subgraph extraction for now
                    # if (line.startswith("Time to extract M11")):
                        # split_stat["time_extract_m11"] = float(tokens[4])
                    # elif (line.startswith("Time to extract M12")):
                        # split_stat["time_extract_m12"] = float(tokens[4])
                    # elif (line.startswith("Time to extract M21")):
                        # split_stat["time_extract_m21"] = float(tokens[4])
                    # elif (line.startswith("Time to extract M22")):
                        # split_stat["time_extract_m22"] = float(tokens[4])
                    pass
                elif(current_state == state_cluster_m22):
                    if(line.startswith("Total MCL time")):
                        split_stat["time_cluster_m22"] = float(tokens[3])
                    elif(line.startswith("Number of clusters")):
                        split_stat["number_cluster_m22"] = int(tokens[3])
                    elif(line.startswith("Iteration") and (line.find("chaos") >= 0)):
                        # print(tokens)
                        split_stat["niter_cluster_m22"] = int(tokens[1])
                        # print(split_stat["niter_cluster_m22"])
                    pass
                elif(current_state == state_prep_minc):
                    if(line.startswith("Time to prepare Minc")):
                        split_stat["time_prep_minc"] = float(tokens[4])
                    elif(line.startswith("As a whole")):
                        split_stat["nrow_minc"] = int(tokens[3])
                        split_stat["ncol_minc"] = int(tokens[6])
                        split_stat["nnz_minc"] = int(tokens[9])
                    pass
                elif(current_state == state_cluster_minc):
                    if(line.startswith("Total MCL time")):
                        split_stat["time_cluster_minc"] = float(tokens[3])
                    elif(line.startswith("Number of clusters")):
                        split_stat["number_cluster_minc"] = int(tokens[3])
                    elif(line.startswith("Iteration") and (line.find("chaos") >= 0)):
                        split_stat["niter_cluster_minc"] = int(tokens[1])
                    pass
    
    # Write collected data to a JSON file
    data = { "metadata": metadata, "stats": stats}
    js_stats = json.dumps(data)
    # with open(ofname+".json", "w", encoding='utf-8') as f:
        # json.dump(data, f, ensure_ascii=False, indent=4)
	# csv_data = normalize_json(data)

    # Write collected data to a CSV file, for easier analysis with spreadsheets
    csv_data = copy.deepcopy(stats)
    headers = [] 
    for i in range(len(csv_data)):
        for k in metadata.keys():
            csv_data_key = "metadata"+ "_" + k
            csv_data[i][csv_data_key] = metadata[k]
        headers += csv_data[i].keys()
    headers = sorted(set(headers))

    with open(ofname, "w") as f:
        dict_writer = csv.DictWriter(f, fieldnames=headers)
        dict_writer.writeheader()
        dict_writer.writerows(csv_data)
