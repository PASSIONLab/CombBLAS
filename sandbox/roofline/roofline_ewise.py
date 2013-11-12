import sys
import kdt
import time
from ctypes import *

#veclen = [10, 50, 100, 150, 200, 500, 1000, 1300, 1500, 1700]#, 2000, 2500, 3000, 5000, 10000, 15000]
#veclen = range(64, 4192+1, 64)
veclen = [8, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]

#repeats = [10, 25, 50, 75, 100, 200, 300, 400]#, 1000, 10000, 100000]
repeats = [1000]

kdt.set_verbosity(kdt.DEBUG)

# parse out the arguments
#parse arguments
if (len(sys.argv) < 2):
	kdt.p("Usage: python %s [whatToDoArg1 whatToDoArg2 ...]"%(sys.argv[0]))
	kdt.p("Each argument specifies one run to do, and any number of runs are allowed. (default is cp)")
	kdt.p("Each argument is a string of 2 letters.")
	kdt.p("The 1st letter specifies the data struct to use: p = Python-Defined Object, c = C++ (i.e. Obj2)")
	kdt.p("The 2nd letter specifies SEJITS use: p = pure Python, s = SEJITS")
	kdt.p("")
	kdt.p("Example:")
	kdt.p("python %s cp pp"%(sys.argv[0]))
	sys.exit(1)

if (len(sys.argv) >= 2):
	whatToDoList = sys.argv[1:]
else:
	whatToDoList = ["cp"] # Obj2, python callback

usePAPI = False
num_PAPI_counters = 1000
events_nm = []
events_nm = events_nm + ["PAPI_TOT_INS", "PAPI_L1_TCM", "PAPI_L2_TCM", "PAPI_L3_TCM"] # instructions, cache misses
#events_nm = events_nm + ["PAPI_L2_TCR", "PAPI_L3_TCR"]
#events_nm = events_nm + ["PAPI_L2_TCW", "PAPI_L3_TCW"] # cache read/writes
#events_nm = events_nm + ["PAPI_TLB_DM", "PAPI_TLB_IM", "PAPI_TLB_TL"] # TLB

if usePAPI:
	from pyPAPI.papi import *
	from ctypes import c_int, c_longlong, c_int64


	PAPI_library_init()

	events = (c_int*len(events_nm))()
	counter_array_type = c_longlong*len(events_nm)
	papi_counters = []
	papi_counters_used = -12
	papi_labels = []
	papi_timers = [0] * num_PAPI_counters
	for i in range(num_PAPI_counters):
		papi_counters.append((counter_array_type)())

	my_proc_rank = kdt._rank()
	np = kdt._nproc()
	#percents = '_'.join(map(str, map(lambda x: x/100, latestDatesToCheck)))
	event_join_str = '-'.join(events_nm)

	#if preSelectedStartingVerts is not None:
	#	preSel_s = "_start%d"%(preSelectedStartingVerts[0])
	#else:
	#	preSel_s = ""

	#papi_output_filename = "temp_papi_output_scale%d_filter%s%s_events%s_%s_np%d_p%02d.txt"%(gen_scale, percents, preSel_s, event_join_str, "-".join(whatToDoList), np, my_proc_rank)
	papi_output_file = sys.stdout #open(papi_output_filename, "w")

	for i in range(len(events_nm)):
		c = PAPI_event_name_to_code(events_nm[i])
		events[i] = c
		print "%s => %d"%(events_nm[i], c)


	def print_papi_counters_str(labels):
		global papi_counters, papi_counters_used, papi_output_file, papi_timers, papi_time_before
		ret = ""
		
		# print the header
		names = "\t".join(["TIME_usec"] + events_nm)

		#print("PRINTING PAPI")
		#print("num counters used = %d, num labels = %d"%(papi_counters_used, len(labels)))
		ret += "iter\tlabel\t%s\n"%(names)

		iter = 0
		label_i = -1
		for i in range(papi_counters_used):
			label_i += 1
			if label_i >= len(labels):
				iter += 1
				label_i -= len(labels)

			#values = "\t".join(papi_counters[i])
			values = "%d\t"%(papi_timers[i] - papi_timers[i-1])
			for j in range(len(events_nm)):
				values += str(papi_counters[i][j]) + "\t"
			ret += "%d\t%s\t%s\n"%(iter, labels[label_i], values)
		
		return ret



class TwitterEdge(Structure):
	_fields_ = [("follower", c_bool),
				("latest", c_uint64), # time_t
				("count", c_short)]
    
	@staticmethod
	def get_c(): # temporary while SEJITS doesn't parse ctypes.Structure directly
		return "typedef struct { bool follower; uint64_t latest; short count; } TwitterEdge;"

def runExperiment(func, e, name):
	global usePAPI, papi_counters_used, papi_counters, papi_labels
	
	best_ops_per_s = 0
	best_len = -1
	best_repeats = -1
	best_papi = ""
	
	for vl in veclen:
		vec = kdt.Vec(vl, element=e, sparse=False)
		resVec = kdt.Vec(vl, element=0, sparse=False)
		for r in repeats:
			begin = time.time()
			
			# PAPI
			if usePAPI:
				papi_counters_used = 0
				papi_timers[-1] = PAPI_get_real_usec()
				PAPI_start_counters(events)

			for _ in range(r):
				resVec.eWiseApply(vec, func)

			# PAPI
			if usePAPI:
				papi_timers[papi_counters_used] = PAPI_get_real_usec()
				PAPI_stop_counters(papi_counters[papi_counters_used])
				papi_counters_used += 1
				papi_labels = ["roofline ewise"]

			t = time.time() - begin
			ops = vl*r / t
			if best_ops_per_s < ops:
				best_ops_per_s = ops
				best_len = vl
				best_repeats = r
				if usePAPI:
					best_papi = print_papi_counters_str(papi_labels)
				record = "new record"
			else:
				record = ""
			
			kdt.p("len \t%5d\t, repeat\t%3d\ttimes:\t%e\tOp/s (took %f seconds) %s"%(vl, r, ops, t, record))
	
	p = kdt._nproc()
	kdt.p("BEST %s on\t%3d\tprocs: len\t%5d\t repeat\t%3d\ttimes:\t%e\tOp/s"%(name, p, best_len, best_repeats, best_ops_per_s))
	kdt.p(best_papi)


for whatToDo in whatToDoList:
	# determine data structure to use
	if whatToDo[0] == 'p':
		prototype_element = TwitterEdge();
		kdt.PDO_enable(True)
	elif whatToDo[0] == 'c':
		kdt.PDO_enable(False)
		prototype_element = kdt.Obj2()
	else:
		raise ValueError,"Invalid data structure specified in whatToDo %s"%whatToDo

	# determine SEJITS or pure Python
	if whatToDo[1] == 'p':
		kdt.SEJITS_enable(False)
	elif whatToDo[1] == 's':
		kdt.SEJITS_enable(True)
	else:
		raise ValueError,"Invalid SEJITS or pure Python specified in whatToDo %s"%whatToDo


	class twitterMul_eWise(kdt.Callback):
		def __init__(self, dflt):
			self.dflt = dflt
			self.filterUpperValue = 946684800
		
		def __call__(self, parent, e):
			if (e.follower == 1 and e.count > 0 and e.latest > self.filterUpperValue):
				return 0
			else:
				return parent


	kdt.p("running on Multiply %s:"%(whatToDo))
	runExperiment(twitterMul_eWise(1.0), prototype_element, "multiply %s"%(whatToDo))

