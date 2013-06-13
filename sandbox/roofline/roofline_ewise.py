import sys
import kdt
import time
from ctypes import *

#veclen = [10, 50, 100, 150, 200, 500, 1000, 1300, 1500, 1700]#, 2000, 2500, 3000, 5000, 10000, 15000]
#veclen = range(64, 4192+1, 64)
veclen = [8, 1024]

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

class TwitterEdge(Structure):
	_fields_ = [("follower", c_bool),
				("latest", c_uint64), # time_t
				("count", c_short)]
    
	@staticmethod
	def get_c(): # temporary while SEJITS doesn't parse ctypes.Structure directly
		return "typedef struct { bool follower; uint64_t latest; short count; } TwitterEdge;"

def runExperiment(func, e, name):
	best_ops_per_s = 0
	best_len = -1
	best_repeats = -1
	
	for vl in veclen:
		vec = kdt.Vec(vl, element=e, sparse=False)
		resVec = kdt.Vec(vl, element=0, sparse=False)
		for r in repeats:
			begin = time.time()
			
			for _ in range(r):
				resVec.eWiseApply(vec, func)

			t = time.time() - begin
			ops = vl*r / t
			if best_ops_per_s < ops:
				best_ops_per_s = ops
				best_len = vl
				best_repeats = r
				record = "new record"
			else:
				record = ""
			
			kdt.p("len \t%5d\t, repeat\t%3d\ttimes:\t%f\tOp/s (took %f seconds) %s"%(vl, r, ops, t, record))

	
	p = kdt._nproc()
	kdt.p("BEST %s on\t%3d\tprocs: len\t%5d\t repeat\t%3d\ttimes:\t%f\tOp/s"%(name, p, best_len, best_repeats, best_ops_per_s))


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

