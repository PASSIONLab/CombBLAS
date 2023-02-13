import sys
import copy
import json

state_start = -1
state_jombie = -2
state_ready = 0
state_subgraph_extraction = 1
state_cluster_m22 = 2
state_prep_minc = 3
state_cluster_minc = 4

if __name__=="__main__":
    # print(sys.argv)
    if(len(sys.argv) < 2):
        print("Usage: <python_script> <filename>")
        print("It will create another file with extension \"stats\", i.e. \"<filename>.stats\", containing stats collected from <filename>")
        exit(1)
    fname = sys.argv[1]
    current_state = state_start
    current_split = -1
    split_stat = {}
    metadata = {}
    stats = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            tokens = line.split(" ")
            if (line.startswith("[Start] Split")):
                if(current_state == state_start):
                    current_split = int(tokens[2])
                    split_stat = {}
                    split_stat["split"] = current_split
                    current_state = state_ready
            elif (line.startswith("[Start] Processing split")):
                if(current_state == state_start):
                    current_split = int(tokens[3])
                    split_stat = {}
                    split_stat["split"] = current_split
                    current_state = state_ready
            elif (line.startswith("[End] Split")):
                if(current_state == state_ready):
                    stats.append(copy.deepcopy(split_stat));
                    current_state = state_start
            elif (line.startswith("[Start] Subgraph extraction")):
                if (current_state == state_ready):
                    current_state = state_subgraph_extraction
            elif (line.startswith("[End] Subgraph extraction")):
                if (current_state == state_subgraph_extraction):
                    current_state = state_ready
            elif (line.startswith("[Start] Clustering new")):
                if (current_state == state_ready):
                    current_state = state_cluster_m22
            elif (line.startswith("[End] Clustering new")):
                if (current_state == state_cluster_m22):
                    current_state = state_ready
            elif (line.startswith("[Start] Preparing incremental")):
                if (current_state == state_ready):
                    current_state = state_prep_minc
            elif (line.startswith("[End] Preparing incremental")):
                if (current_state == state_prep_minc):
                    current_state = state_ready
            elif (line.startswith("[Start] Clustering incremental")):
                if (current_state == state_ready):
                    current_state = state_cluster_minc
            elif (line.startswith("[End] Clustering incremental")):
                if (current_state == state_cluster_minc):
                    current_state = state_ready
            else:
                if(current_state == state_subgraph_extraction):
                    if (line.startswith("Time to extract M11")):
                        split_stat["time_extract_m11"] = float(tokens[4])
                    elif (line.startswith("Time to extract M12")):
                        split_stat["time_extract_m12"] = float(tokens[4])
                    elif (line.startswith("Time to extract M21")):
                        split_stat["time_extract_m21"] = float(tokens[4])
                    elif (line.startswith("Time to extract M22")):
                        split_stat["time_extract_m22"] = float(tokens[4])
                elif(current_state == state_cluster_m22):
                    if(line.startswith("Total MCL time")):
                        split_stat["time_cluster_m22"] = float(tokens[3])
                    elif(line.startswith("Number of clusters")):
                        split_stat["number_cluster_m22"] = int(tokens[3])
                elif(current_state == state_prep_minc):
                    if(line.startswith("Time to calculate vertex mapping")):
                        split_stat["time_vtx_map"] = float(tokens[5])
                    elif(line.startswith("Time to assign submatrices")):
                        split_stat["time_assign_submatrices"] = float(tokens[4])
                elif(current_state == state_cluster_minc):
                    if(line.startswith("Total MCL time")):
                        split_stat["time_cluster_minc"] = float(tokens[3])
                    elif(line.startswith("Number of clusters")):
                        split_stat["number_cluster_minc"] = int(tokens[3])

    js_stats = json.dumps({"stats": stats})
    with open(fname+".stats", "w", encoding='utf-8') as f:
        json.dump({"stats": stats}, f, ensure_ascii=False, indent=4)
        # f.write(js_stats)
    print(js_stats)

