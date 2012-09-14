import sys
import re
import os

from stats import compute_stats

filters = [1, 10, 25, 100]
errorbars = "candlesticks" # also "errorbars" or None
runtype = None

if len(sys.argv) < 2:
	print "what are you trying to do?"
	sys.exit()

runtype = sys.argv[1]
graphformat = "png"
machine = "mirasol"
algorithm = "bfs"

showIndividualIterations = False
showIndividualIterations_claim_to_be = "mean_%stime"

for arg in sys.argv[2:]:
	if arg == "hopper" or arg == "mirasol":
		machine = arg
	elif arg == "candlesticks" or arg == "errorbars" or arg == "noerror":
		errorbars = arg
	elif arg == "png" or arg == "eps":
		graphformat = arg
	elif arg == "indiv":
		showIndividualIterations = True

def getTerminalString(format):
	if format=="eps":
		return "postscript eps color"
	return format
	
######################
## setup experiment to plot

######################
## RMAT BFS
raw_combblas_files = None
if runtype == "bfs":
	if machine == "mirasol":
		cores = {1: "result_ipdps_bfs_22_1.txt", 4: "result_ipdps_bfs_22_4.txt", 9: "result_ipdps_bfs_22_9.txt", 16: "result_ipdps_bfs_22_16.txt", 25: "result_ipdps_bfs_22_25.txt", 36: "result_ipdps_bfs_22_36.txt"}
		combblas_file = "result_ipdps_bfs_22_combblas.txt"
		raw_combblas_files = {1: "ran_scale22_p1_notiming.det", 4: "ran_scale22_p4_notiming.det", 9: "ran_scale22_p9_notiming.det", 16: "ran_scale22_p16_notiming.det", 25: "ran_scale22_p25_notiming.det", 36: "ran_scale22_p36_notiming.det"}
		
		if showIndividualIterations:
			core_xrange = "0.9:64"
		else:
			core_xrange = "0.9:40"
		filtergrid_yrange = "0.1:256"
	else:
		cores = {121: "result_ipdps_bfs_25_121.txt", 256: "result_ipdps_bfs_25_256.txt", 576: "result_ipdps_bfs_25_576.txt", 1024: "result_ipdps_bfs_25_1024.txt", 2048: "result_ipdps_bfs_25_2048.txt"}
		combblas_file = "result_ipdps_bfs_25_combblas.txt"
		raw_combblas_files = {121: "combblas_bfs_25_121.txt", 256: "combblas_bfs_25_256.txt", 576: "combblas_bfs_25_576.txt", 1024: "combblas_bfs_25_1024.txt", 2048: "combblas_bfs_25_2048.txt"}

		core_xrange = "100:2500"
		filtergrid_yrange = "0.1:256"
		
		machine = "hopper"

	# combblas file format: each line is a tab-delimited tuple:
	# core count, filter percentage, min time, max time, mean time
	
	experiments = [("PythonSR_PythonFilter_OTF", "Python/Python KDT", "#FF0000"), # red (kinda light)
		("PythonSR_SejitsFilter_OTF", "Python/SEJITS KDT", "#8B0000"), # dark red
#		("C++SR_PythonFilter_OTF", "C++/Python KDT", "#90EE90"), # light green
#		("C++SR_SejitsFilter_OTF", "C++/SEJITS KDT", "#008000"), # green (but it's dark)
		("SejitsSR_SejitsFilter_OTF", "SEJITS/SEJITS KDT", "#0000FF"), # blue (but it's dark)
		("CombBLAS_OTF", "C++/C++ CombBLAS", "#DAA520")] # gold is FFD700, DAA520 is darker gold
#		("C++SR_PythonFilter_Mat", "C++/Python KDT (materialized)", "#000000")] # black
	
	# ID will be replaced by strings from experiments array
	experiment_varieties = ["mean_IDtime", "min_IDtime", "max_IDtime", "firstquartile_IDtime", "thirdquartile_IDtime"]
	
	result_type = "BFS"
	
	def parseCombBLAS(data):
		if not os.path.isfile(combblas_file):
			print "CombBLAS file not found"
			return

		for line in open(combblas_file, 'r'):
			feats = line.split("\t")
			try:
				core = int(feats[0])
				filter = int(float(feats[1]))
				min_time = float(feats[2])
				max_time = float(feats[3])
				mean_time = float(feats[4])
				data.append((core, "min_CombBLAS_OTFtime", filter, min_time))
				data.append((core, "max_CombBLAS_OTFtime", filter, max_time))
				data.append((core, "mean_CombBLAS_OTFtime", filter, mean_time))
			except ValueError:
				# there is some empty/invalid data
				print "omitting CombBLAS datapoints from this incomplete line:",line
				pass
	
	
	parseProcFiles = True
	parseRealFiles = False
	doFilterGrid = True
	doPermeabilityPlot = True
	doRealScalabilityPlot = False
	
######################
## real data BFS
elif runtype == "bfsreal":
	cores = {36: 1}
	files = {"small": "result_ipdps_bfs_small_36.txt", "medium": "result_ipdps_bfs_medium_36.txt", "large": "result_ipdps_bfs_large_36.txt", "huge": "result_ipdps_bfs_huge_36.txt"}
	raw_combblas_files = {"small": "ran_small_p36_notiming_august12.det", "medium": "ran_medium_p36_notiming_august12.det", "large": "ran_large_p36_notiming_august12.det", "huge": "ran_huge_p36_notiming_august12.det"}
	combblas_file = "result_ipdps_bfs_real_combblas.txt"
	# combblas file format: each line is a tab-delimited tuple:
	# core count, [small|medium|large|huge], min time, max time, mean time
	
	experiments = [("PythonSR_PythonFilter_OTF", "Python/Python KDT", "#FF0000"), # red (kinda light)
		("PythonSR_SejitsFilter_OTF", "Python/SEJITS KDT", "#8B0000"), # dark red
#		("C++SR_PythonFilter_OTF", "C++/Python KDT", "#90EE90"), # light green
#		("C++SR_SejitsFilter_OTF", "C++/SEJITS KDT", "#008000"), # green (but it's dark)
		("SejitsSR_SejitsFilter_OTF", "SEJITS/SEJITS KDT", "#0000FF"), # blue (but it's dark)
		("CombBLAS_OTF", "C++/C++ CombBLAS", "#DAA520")] # gold
#		("C++SR_PythonFilter_Mat", "C++/Python KDT (materialized)", "#000000")] # black
	
	# ID will be replaced by strings from experiments array
	experiment_varieties = ["mean_IDtime", "min_IDtime", "max_IDtime", "firstquartile_IDtime", "thirdquartile_IDtime"]
	
	result_type = "BFS"
	
	real_graph_sizes = ["small", "medium", "large", "huge"]
	
	def parseCombBLAS(data):
		if not os.path.isfile(combblas_file):
			print "CombBLAS file not found"
			return

		for line in open(combblas_file, 'r'):
			feats = line.split("\t")
			core = int(feats[0])
			graphsize = feats[1]
			min_time = float(feats[2])
			max_time = float(feats[3])
			mean_time = float(feats[4])
			data.append((core, "min_CombBLAS_OTFtime", graphsize, min_time))
			data.append((core, "max_CombBLAS_OTFtime", graphsize, max_time))
			data.append((core, "mean_CombBLAS_OTFtime", graphsize, mean_time))
	
	
	parseProcFiles = False
	parseRealFiles = True
	doFilterGrid = False
	doPermeabilityPlot = False
	doRealScalabilityPlot = True

######################
## Erdos-Renyi MIS
elif runtype == "mis":
	cores = {1: "result_ipdps_MIS_1.txt", 4: "result_ipdps_MIS_4.txt", 9: "result_ipdps_MIS_9.txt", 16: "result_ipdps_MIS_16.txt", 25: "result_ipdps_MIS_25.txt", 36: "result_ipdps_MIS_36.txt"}
	raw_combblas_files = {1: "mis_ran_scale_22_1p.txt", 4: "mis_ran_scale_22_4p.txt", 9: "mis_ran_scale_22_9p.txt", 16: "mis_ran_scale_22_16p.txt", 25: "mis_ran_scale_22_25p.txt", 36: "mis_ran_scale_22_36p.txt"}
	
	experiments = [("PythonSR_PythonFilter_ER_OTF_22", "Python/Python KDT", "#FF0000"), # red (kinda light)
				("PythonSR_SejitsFilter_ER_OTF_22", "Python/SEJITS KDT", "#8B0000"), # dark red
				("SejitsSR_SejitsFilter_ER_OTF_22", "SEJITS/SEJITS KDT", "#0000FF"), # blue (but it's dark)
				("CombBLAS_OTF", "C++/C++ CombBLAS", "#DAA520")] # gold
	
	# ID will be replaced by strings from experiments array
	experiment_varieties = ["mean_IDtime", "min_IDtime", "max_IDtime", "firstquartile_IDtime", "thirdquartile_IDtime"]

	result_type = "MIS"

	if showIndividualIterations:
		core_xrange = "0.9:64"
	else:
		core_xrange = "0.9:40"
	filtergrid_yrange = "0.1:256"

	def parseCombBLAS(data):
		exp = "CombBLAS_OTF"
		for (core, file) in raw_combblas_files.items():
			times = []
			if not os.path.isfile(file):
				print "file not found:",file
				continue
			for line in open(file, 'r'):
				# MIS time: 0.887308 seconds
				# Filter keeps 100 percentage of edges
				if line.find("MIS time:") != -1:
					time_s = line[(line.find(":")+1) : (line.rfind(" seconds"))].strip()
					times.append(float(time_s))
				elif line.find("Filter keeps") != -1:
					filter_s = line[len("Filter keeps ") : (line.find("percentage"))].strip()
					filter = int(filter_s)
					if showIndividualIterations:
						iteration = 1
						for t in times:
							data.append((getFunnyCore(core, iteration), showIndividualIterations_claim_to_be%(exp), filter, t))
							iteration += 1
					
					# summarize
					stats = compute_stats(times)
					
					data.append((core, "mean_%stime"%exp, filter, stats["mean"]))
					data.append((core, "min_%stime"%exp, filter, stats["min"]))
					data.append((core, "max_%stime"%exp, filter, stats["max"]))
					data.append((core, "firstquartile_%stime"%exp, filter, stats["q1"]))
					data.append((core, "thirdquartile_%stime"%exp, filter, stats["q3"]))
					times = []

	parseProcFiles = True
	parseRealFiles = False
	doFilterGrid = True
	doPermeabilityPlot = True
	doRealScalabilityPlot = False
	algorithm = "mis"
else:
	print "unknown option. use bfs or mis"
	sys.exit()

#######################
varieties = []
for exp in experiments:
	for var in experiment_varieties:
		varieties.append(var.replace("ID", exp[0]))

def isExperiment(str):
	for exp in experiments:
		if str == exp[0]:
			return True
	return False

def getFunnyCore(core, iteration):
	spread = core/2.0
	return float(core) + iteration/16.0*spread

def getFunnyGraphSize(graphsize, iteration):
	if iteration == 0:
		return graphsize
	
	return graphsize+str(iteration)

if showIndividualIterations:
	funnyCores = []
	for core in cores.keys():
		funnyCores.append(core - 0.1) # for a gap in the plot
		funnyCores.append(core)
		for i in range(1,17):
			funnyCores.append(getFunnyCore(core, i))
	
	funnyGraphSizes = []
	for g in real_graph_sizes:
		funnyGraphSizes.append(g)
		for i in range(1,20): # one extra to put a gap in the plot
			funnyGraphSizes.append(getFunnyGraphSize(g, i))
	real_graph_sizes = funnyGraphSizes
	

#######################
## data structure
data = []
# data[i][0] == core count
# data[i][1] == experiment variety
# data[i][2] == filter permeability
# data[i][3] == time

######################
## parse

# parse CombBLAS
def parseCombBLASIterations(file):
	ret = []
	times = []
	for line in open(file, 'r'):
		# BFS time: 0.887308 seconds
		# Filter keeps 100 percentage of edges
		if line.find("BFS time:") != -1:
			time_s = line[(line.find(":")+1) : (line.rfind(" seconds"))].strip()
			times.append(float(time_s))
		elif line.find("Filter keeps") != -1 and len(times) > 0: # the filter bit is printed twice
			filter_s = line[len("Filter keeps ") : (line.find("percentage"))].strip()
			filter = int(filter_s)
			ret.append((times, filter))
			times = []

	return ret

if raw_combblas_files is not None and algorithm != "mis":
	exp = "CombBLAS_OTF"

	for (first, file) in raw_combblas_files.items():
		if not os.path.isfile(file):
			print "file not found:",file
			continue

		if parseProcFiles:
			core = first
		elif parseRealFiles:
			core = cores.keys()[0]
			graphsize = first

		parsed = parseCombBLASIterations(file)
		for run in parsed:
			times = run[0]
			if parseProcFiles:
				filter = run[1]
			else:
				filter = graphsize
			
			if showIndividualIterations:
				iteration = 1
				for t in times:
					if parseProcFiles:
						filter_val = filter
						core_val = getFunnyCore(core, iteration)
					else:
						filter_val = getFunnyGraphSize(graphsize, iteration)
						core_val = core
						print filter_val, "ccblas"
					#data.append((d[0], showIndividualIterations_claim_to_be%(exp), filter_val, d[3]))

					data.append((core_val, showIndividualIterations_claim_to_be%(exp), filter_val, t))
					iteration += 1
			
			# summarize
			stats = compute_stats(times)
			
			data.append((core, "mean_%stime"%exp, filter, stats["mean"]))
			data.append((core, "min_%stime"%exp, filter, stats["min"]))
			data.append((core, "max_%stime"%exp, filter, stats["max"]))
			data.append((core, "firstquartile_%stime"%exp, filter, stats["q1"]))
			data.append((core, "thirdquartile_%stime"%exp, filter, stats["q3"]))
else:
	parseCombBLAS(data)

#############
# parse KDT data
# get detailed data about each individual BFS iteration
# parse individual BFS or MIS iterations then calculate stats
if parseProcFiles:
	fileItems = cores.items()
elif parseRealFiles:
	fileItems = files.items()
	
for (first, file) in fileItems:
	if not os.path.isfile(file):
		print "file not found:",file
		continue
	
	if parseProcFiles:
		core = first
	elif parseRealFiles:
		core = cores.keys()[0]
		graphsize = first

	iterationData = []
	iteration = 1
	for line in open(file, 'r'):
		if line.find("(result discarded)") != -1:
			continue

		#############################################
		if algorithm == "bfs":
			if line.find("iteration") != -1:
				feats = line.split("\t")
				filter = int(float(feats[0]))
				#iteration string = feats[1]
				iteration = len(iterationData)+1 # easier than parsing out the string
				time = float(feats[2])
				
				if parseProcFiles:
					iterationData.append((getFunnyCore(core, iteration), "", filter, time))
				else:
					iterationData.append((core, "", filter, time))
			elif line.find("BFS execution times") != -1:
				# found out what the previous iterations were for
				exp = line[(line.find("(")+1) : (line.find(")"))]
				if isExperiment(exp):
					# got all the data for an experiment, so summarize it
					print "found variety:", var
					times = []
					iteration = 1
					for d in iterationData:
						times.append(d[3])
						filter = d[2]
						if parseRealFiles:
							filter = graphsize # small/medium/large/huge
						if showIndividualIterations:
							if parseProcFiles:
								filter_val = filter
							else:
								filter_val = getFunnyGraphSize(filter, iteration)
							data.append((d[0], showIndividualIterations_claim_to_be%(exp), filter_val, d[3]))
							iteration += 1
							
					stats = compute_stats(times)
			
					data.append((core, "mean_%stime"%exp, filter, stats["mean"]))
					data.append((core, "min_%stime"%exp, filter, stats["min"]))
					data.append((core, "max_%stime"%exp, filter, stats["max"]))
					data.append((core, "firstquartile_%stime"%exp, filter, stats["q1"]))
					data.append((core, "thirdquartile_%stime"%exp, filter, stats["q3"]))

				iterationData = []
			elif len(line) > 1:
				iterationData = []
		#############################################
		elif algorithm == "mis": 
			if line.find("procs time:") != -1:
				feats = line.split("\t")
				var = feats[0]
				#core = feats[1]
				time = float(feats[3])
				iterationData.append((getFunnyCore(core, iteration), var, -1, time))
				iteration += 1
			if line.find("min_") != -1 and len(iterationData) > 0: # first line that has the filter amount
				feats = line.split("\t")
				filter = int(float(feats[1]))
				times = []
				for d in iterationData:
					times.append(d[3])
					exp = d[1]
					if showIndividualIterations:
						data.append((d[0], showIndividualIterations_claim_to_be%(d[1]), filter, d[3]))
				iteration = 1
				iterationData = []

				stats = compute_stats(times)
		
				data.append((core, "mean_%stime"%exp, filter, stats["mean"]))
				data.append((core, "min_%stime"%exp, filter, stats["min"]))
				data.append((core, "max_%stime"%exp, filter, stats["max"]))
				data.append((core, "firstquartile_%stime"%exp, filter, stats["q1"]))
				data.append((core, "thirdquartile_%stime"%exp, filter, stats["q3"]))
#for d in data:
#	print d			

######################
## function to determine if there's any data for a particular core count
def is_there_data_for_core_count(data, core_cnt):
	for datapoint in data:
		if datapoint[0] == core_cnt:
			return True
	return False


######################
## function to format 4D tuples into 3D table for a 2D plot
def format_table(data, group_col_idx, group_col_select_val, col_ids, row_ids, col_idx, row_idx, value_idx, row_ids_are_strings=False):
	ret = ""
	fdata = []
	
	for i in range(len(row_ids)):
		fdata.append([])
		if row_ids_are_strings:
			fdata[i].append(i)
		else:
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

######################
## Figure 15-style plot with a scalability plot for each filter percentage
if doFilterGrid:
	if showIndividualIterations:
		coreXVals = funnyCores
	else:
		coreXVals = cores.keys()
	# print filter by filter
	for filter in filters:
		print ""
		k = coreXVals
		k.sort()
		print "filter ",filter, k
		grid = format_table(data, 2, filter, varieties, k, 1, 0, 3)
		print grid
	
		filestem = "gnuplot_filtergrid_%d_%s_%s"%(filter, machine, algorithm)
		
		gnuplot = ""
		gnuplot += 'set title "Filtered %s (%d%% permeability)"\n'%(result_type, filter)
		gnuplot += 'set terminal %s\n'%(getTerminalString(graphformat))
		gnuplot += 'set output "%s.%s"\n'%(filestem, graphformat)
		gnuplot += '\n'
		gnuplot += 'set datafile missing "-"\n'
		gnuplot += '\n'
		gnuplot += 'set xrange [%s]\n'%(core_xrange)
		gnuplot += 'set yrange [%s]\n'%(filtergrid_yrange)
		gnuplot += 'set logscale y\n'
		gnuplot += 'set logscale x\n'
		gnuplot += 'set grid ytics mytics lt 1 lc rgb "#EEEEEE"\n'
		gnuplot += "set xlabel 'Number of MPI Processes'\n"
		gnuplot += "set ylabel 'Mean %s Time (seconds, log scale)'\n"%(result_type)
		
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
			if errorbars == "candlesticks":
				# candlestick data: x:box_min:whisker_min:whisker_high:box_high
				#      +0,            +1,          +2,                +3,                     +4
				# ["mean_IDtime", "min_IDtime", "max_IDtime", "firstquartile_IDtime", "thirdquartile_IDtime"]
				# 1, firstquantile, min, max, thirdquartile:  +3, +1, +2, +4
				gnuplot += ' "%s.dat" every ::1 using 1:%d:%d:%d:%d title \'\' ps 0 lt 1 lc rgb \'%s\' with candlesticks,\\\n'%(filestem, exp_col_start+3,exp_col_start+1,exp_col_start+2, exp_col_start+4, experiments[i][2])
			elif errorbars == "errorbars":
				# errorbars data: x:y:ylow:yhigh
				# 1, +0, +1, +2
				gnuplot += ' "%s.dat" every ::1 using 1:%d:%d:%d title \'\' ps 0 lt 1 lc rgb \'%s\' with errorbars,\\\n'%(filestem, exp_col_start,exp_col_start+1,exp_col_start+2, experiments[i][2])
			gnuplot += ' "%s.dat" every ::1 using 1:($%d) title \'%s\' lc rgb \'%s\' with lines%s\n'%(filestem, exp_col_start, experiments[i][1], experiments[i][2], comma)
	
		print ""
		print gnuplot
		
		f = open('%s.dat'%filestem, 'w')
		f.write(grid)
		f.close()
	
		f = open('%s.gp'%filestem, 'w')
		f.write(gnuplot)
		f.close()

######################
## permeability plot, increasing filter permeability on largest core count
if doPermeabilityPlot:	
	core_cnt = max(cores.keys())
	print "=== filter permeability ==="
	grid = format_table(data, 0, core_cnt, varieties, [1, 10, 25, 100], 1, 2, 3)
	print grid

	filestem = "gnuplot_perm_%d_%s_%s"%(core_cnt, machine, algorithm)
	
	gnuplot = ""
	gnuplot += 'set title "Effects of Filter Permeability (%d processes)"\n'%(core_cnt)
	gnuplot += 'set terminal %s\n'%(getTerminalString(graphformat))
	gnuplot += 'set output "%s.%s"\n'%(filestem, graphformat)
	gnuplot += '\n'
	gnuplot += 'set xrange [-5:105]\n'
	gnuplot += 'set yrange [0.1:32]\n'
	gnuplot += 'set logscale y\n'
	gnuplot += 'set grid ytics mytics lt 1 lc rgb "#EEEEEE"\n'
	gnuplot += "set xlabel 'Filter Permeability'\n"
	gnuplot += "set ylabel 'Mean %s Time (seconds, log scale)'\n"%(result_type)
	gnuplot += "set key right bottom\n"
	
	xtics = ""
	cc = filters
	for i in range(len(cc)):
		if i+1 < len(cc):
			comma = ", "
		else:
			comma = ""
		xtics += "'%d%%%%' %d%s"%(cc[i], cc[i], comma)
	
	gnuplot += "set xtics (%s)\n"%xtics
	gnuplot += 'plot\\\n'
	vars_per_exp = len(experiment_varieties)
	for i in range(len(experiments)):
		exp_col_start = 1 + vars_per_exp*i + 1
		if i+1 < len(experiments):
			comma = ",\\"
		else:
			comma = ""
		if errorbars == "candlesticks":
			gnuplot += ' "%s.dat" every ::1 using 1:%d:%d:%d:%d title \'\' ps 0 lt 1 lc rgb \'%s\' with candlesticks,\\\n'%(filestem, exp_col_start+3,exp_col_start+1,exp_col_start+2, exp_col_start+4, experiments[i][2])
		elif errorbars == "errorbars":
			gnuplot += ' "%s.dat" every ::1 using 1:%d:%d:%d title \'\' ps 0 lt 1 lc rgb \'%s\' with errorbars,\\\n'%(filestem, exp_col_start,exp_col_start+1,exp_col_start+2, experiments[i][2])
		gnuplot += ' "%s.dat" every ::1 using 1:%d title \'%s\' lc rgb \'%s\' with lines%s\n'%(filestem, exp_col_start, experiments[i][1], experiments[i][2], comma)

	print ""
	print gnuplot
	
	f = open('%s.dat'%filestem, 'w')
	f.write(grid)
	f.close()

	f = open('%s.gp'%filestem, 'w')
	f.write(gnuplot)
	f.close()

######################
## real data time plot
if doRealScalabilityPlot:	
	for plot_core_cnt in cores.keys():
		if not is_there_data_for_core_count(data, plot_core_cnt):
			continue

		print "=== real data ==="
		grid = format_table(data, 0, plot_core_cnt, varieties, real_graph_sizes, 1, 2, 3, row_ids_are_strings=True)
		print grid
	
		filestem = "gnuplot_real_%d_%s_%s"%(plot_core_cnt, machine, algorithm)
		
		gnuplot = ""
		gnuplot += 'set title "BFS on Twitter Data (%d processes)"\n'%(plot_core_cnt)
		gnuplot += 'set terminal %s\n'%(getTerminalString(graphformat))
		gnuplot += 'set output "%s.%s"\n'%(filestem, graphformat)
		gnuplot += ''
		if showIndividualIterations:
			gnuplot += 'set xrange [-0.5:80]\n'
		else:
			gnuplot += 'set xrange [-0.5:3.5]\n'
		gnuplot += '\n'
		gnuplot += 'set datafile missing "-"\n'
		gnuplot += '\n'
		gnuplot += 'set yrange [0.01:32]\n'
		gnuplot += 'set logscale y\n'
		gnuplot += 'set grid ytics mytics lt 1 lc rgb "#EEEEEE"\n'
		gnuplot += "set xlabel 'Twitter Input Graph'\n"
		gnuplot += "set ylabel 'Mean %s Time (seconds, log scale)'\n"%(result_type)
		gnuplot += "set key right top\n"
		
		xtics = ""
		cc = ["small", "medium", "large", "huge"]
		for i in range(len(cc)):
			if i+1 < len(cc):
				comma = ", "
			else:
				comma = ""
			xtics += "'%s' %d%s"%(cc[i], i, comma)
		
		gnuplot += "set xtics (%s)\n"%xtics

		gnuplot += 'plot\\\n'
		vars_per_exp = len(experiment_varieties)
		for i in range(len(experiments)):
			exp_col_start = 1 + vars_per_exp*i + 1
			if i+1 < len(experiments):
				comma = ",\\"
			else:
				comma = ""
			if errorbars == "candlesticks":
				gnuplot += ' "%s.dat" every ::1 using 1:%d:%d:%d:%d title \'\' ps 0 lt 1 lc rgb \'%s\' with candlesticks,\\\n'%(filestem, exp_col_start+3,exp_col_start+1,exp_col_start+2, exp_col_start+4, experiments[i][2])
			elif errorbars == "errorbars":
				gnuplot += ' "%s.dat" every ::1 using 1:%d:%d:%d title \'\' ps 0 lt 1 lc rgb \'%s\' with errorbars,\\\n'%(filestem, exp_col_start,exp_col_start+1,exp_col_start+2, experiments[i][2])
			gnuplot += ' "%s.dat" every ::1 using 1:($%d) title \'%s\' lc rgb \'%s\' with lines%s\n'%(filestem, exp_col_start, experiments[i][1], experiments[i][2], comma)
	
		print ""
		print gnuplot
		
		f = open('%s.dat'%filestem, 'w')
		f.write(grid)
		f.close()
	
		f = open('%s.gp'%filestem, 'w')
		f.write(gnuplot)
		f.close()
