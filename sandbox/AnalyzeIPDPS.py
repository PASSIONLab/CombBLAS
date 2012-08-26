import sys
import re

if len(sys.argv) < 2:
	print "what are you trying to do?"
	sys.exit()

filters = [1, 10, 25, 100]

if sys.argv[1] == "bfs":
	cores = {1: "result_ipdps_bfs_22_1.txt", 4: "result_ipdps_bfs_22_4.txt", 9: "result_ipdps_bfs_22_9.txt", 16: "result_ipdps_bfs_22_16.txt", 25: "result_ipdps_bfs_22_25.txt", 36: "result_ipdps_bfs_22_36.txt"}
	
	experiments = ["PythonSR_PythonFilter_OTF", "PythonSR_SejitsFilter_OTF", "C++SR_PythonFilter_OTF","C++SR_SejitsFilter_OTF", "SejitsSR_SejitsFilter_OTF", "C++SR_PythonFilter_Mat"]
	
	# ID will be replaced by strings from experiments array
	experiment_varieties = ["mean_IDtime", "min_IDtime", "max_IDtime"]
elif sys.argv[1] == "mis":
	cores = {1: "result_ipdps_MIS_1.txt", 4: "result_ipdps_MIS_4.txt", 9: "result_ipdps_MIS_9.txt", 16: "result_ipdps_MIS_16.txt", 25: "result_ipdps_MIS_25.txt", 36: "result_ipdps_MIS_36.txt"}
	
	experiments = ["PythonSR_PythonFilter_ER_OTF_22", "PythonSR_SejitsFilter_ER_OTF_22", "SejitsSR_SejitsFilter_ER_OTF_22"]
	
	# ID will be replaced by strings from experiments array
	experiment_varieties = ["mean_IDtime", "min_IDtime", "max_IDtime"]
else:
	print "unknown option. use bfs or mis"
	sys.exit()

varieties = []
for exp in experiments:
	for var in experiment_varieties:
		varieties.append(var.replace("ID", exp))


data = []
# parse
for (core, file) in cores.items():
	for line in open(file, 'r'):
		for var in varieties:
			if line.find(var) != -1:
				feats = line.split("\t")
				# var = feats[0]
				filter = int(float(feats[1]))
				# : = feats[2]
				time = float(feats[3])
				data.append((core, var, filter, time))

# printer
def format_table(data, group_col_idx, group_col_select_val, col_ids, row_ids, col_idx, row_idx, value_idx):
	ret = ""
	fdata = []
	for i in range(len(row_ids)):
		fdata.append([])
		fdata[i].append(row_ids[i])
		for j in range(len(col_ids)):
			fdata[i].append("-")
		
	# format the data into a grid
	for d in data:
		if d[group_col_idx] == group_col_select_val:
			row = row_ids.index(d[row_idx])
			column = col_ids.index(d[col_idx])+1
			#print d, row, column, fdata[row][column], "=>", d[value_idx]
			fdata[row][column] = d[value_idx]
				
	# print headers
	ret += "\t"
	for val in col_ids:
		ret += val+"\t"
	ret += "\n"
	# print grid
	for row in fdata:
		for val in row:
			ret += str(val) + "\t"
		ret += "\n"
	
	return ret

# print filter by filter

for filter in filters:
	print ""
	k = cores.keys()
	k.sort()
	print "filter ",filter, k
	grid = format_table(data, 2, filter, varieties, k, 1, 0, 3)
	print grid

	filestem = "gnuplot_%d"%filter
	
	gnuplot = ""
	gnuplot += 'set title "Filter %d"\n'%(filter)
	gnuplot += 'set terminal png\n'
	gnuplot += 'set output "%s.png"\n'%filestem
	gnuplot += ''
	gnuplot += 'set xrange [0:40]\n'
	gnuplot += 'set yrange [0.1:256]\n'
	gnuplot += 'set logscale y\n'
	gnuplot += "set xlabel 'number of MPI processes'\n"
	gnuplot += "set ylabel 'mean BFS time (s)'\n"
	
	xtics = ""
	cc = cores.keys()
	for i in range(len(cc)):
		if i+1 < len(cc):
			comma = ", "
		else:
			comma = ""
		xtics += "'%d' %d%s"%(cc[i], cc[i], comma)
	
	gnuplot += "set xtics (%s)\n"%xtics
	gnuplot += 'plot\\\n'
	vars_per_exp = len(experiment_varieties)
	for i in range(len(experiments)):
		exp_col_start = 1 + vars_per_exp*i + 1
		if i+1 < len(experiments):
			comma = ",\\"
		else:
			comma = ""
		gnuplot += ' "%s.dat" every ::1 using 1:%d:%d:%d title \'\' ps 0 lc rgb \'black\' with errorbars,\\\n'%(filestem, exp_col_start,exp_col_start+1,exp_col_start+2)
		gnuplot += ' "%s.dat" every ::1 using 1:%d title \'%s\' with lines%s\n'%(filestem, exp_col_start, experiments[i], comma)

	print ""
	print gnuplot
	
	f = open('%s.dat'%filestem, 'w')
	f.write(grid)
	f.close()

	f = open('%s.gp'%filestem, 'w')
	f.write(gnuplot)
	f.close()