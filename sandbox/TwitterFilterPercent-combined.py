import time
time_very_beginning = time.time()

import sys
import os
import math
import random
import kdt
import kdt.pyCombBLAS as pcb
from stats import splitthousands, printstats

# Adam: these imports seem unneeded, because they're imported again when they're used.
# commenting out so the script works on systems without SEJITS
#from pcb_predicate import *
#from pcb_predicate_sm import *

kdt.PDO_enable(False)

#parse arguments
if (len(sys.argv) < 2):
	kdt.p("Usage: python %s twittergraph.mtx [whatToDoArg1 whatToDoArg2 ...]"%(sys.argv[0]))
	kdt.p("The 1st argument is either a datafile or an integer which is the scale for RMAT generation.")
	kdt.p("")
	kdt.p("The next arguments specify what runs to do. Each argument specifies one run to do, and any number of runs are allowed. (default is cpo)")
	kdt.p("Each argument is a string of 3 letters.")
	kdt.p("The 1st letter specifies the semiring to use: p = pure Python, c = C++ (i.e. built-in), s = SEJITS (Python semiring run through SEJITS)")
	kdt.p("The 2nd letter specifies the filter type: p = pure Python, s = SEJITS")
	kdt.p("The 3rd letter specifes materialization: o = On-The-Fly, m = materialize")
	kdt.p("")
	kdt.p("Examples:")
	kdt.p("python %s filter_debug.mtx ppo cpo"%(sys.argv[0]))
	kdt.p("python %s 14"%(sys.argv[0]))
	sys.exit()

datasource = "file"
inmatrixfile = sys.argv[1]
gen_scale = 10

# report results of keep_starts runs, where each run traverses at least keep_min_edges edges.
# total maximum of runs is nstarts.
nstarts = 512
keep_starts = 16
keep_min_edges = 100

# figure out what to do
if (len(sys.argv) >= 3):
	whatToDoList = sys.argv[2:]
else:
	whatToDoList = ("cpo") # Python/Python OTF, C++/Python OTF

# this function is used for generation.
# obj is the object that needs to be filled in
# bin is a throwaway value.
# http://docs.python.org/library/random.html

def Twitter_obj_randomizer_runs(obj, bin):
	if random.randrange(0, 2) > 0:
		obj.count = 1
		obj.latest = random.randrange(1244592000, 1246406400)
	else:
		obj.count = 0
	obj.follower = 0
	return obj

# used for a percentage-based filtering scheme
def Twitter_obj_randomizer(obj, bin):
	obj.count = 1
	obj.latest = random.randrange(0, 10000)
	obj.follower = 0
	return obj


# 2009-06-10 0:0:0 == 1244592000
# 2009-06-13 0:0:0 == 1244851200
# 2009-07-01 0:0:0 == 1246406400

filterUpperValue = None
def twitterEdgeFilter(e):
	#print e.latest, e
	return e.count > 0 and e.latest < filterUpperValue

#	return e.count > 0 and e.latest > 946684800 and e.latest < 1249084800
#	return e.follower == 0



# determine where the data is supposed to come from
if os.path.isfile(inmatrixfile):
	datasource = "file"
else:
	try:
		gen_scale = int(inmatrixfile)
		datasource = "generate"
	except ValueError:
		datasource = "unknown"

# get the data
if datasource == "file":
	# load
	kdt.p("--Reading network from %s"%inmatrixfile)
	before = time.time()
	G = kdt.DiGraph.load(inmatrixfile, eelement=kdt.Obj2())
	kdt.p("Read in %fs. Read \t%d\t vertices and \t%d\t edges."%(time.time()-before, G.nvert(), G.nedge()))
	
	# optimize the graph
	kdt.p("--Deleting isolated vertices and randomly permuting matrix for load balance")
	before = time.time()
	G.delIsolatedVerts(True)
	kdt.p("Done in %fs."%(time.time()-before))
elif datasource == "generate":
	#type1 = kdt.DiGraph.generateRMAT(scale, element=1.0, edgeFactor=7, delIsolated=False, initiator=[0.60, 0.19, 0.16, 0.05])
	kdt.p("--Generating a plain RMAT graph of scale %d"%(gen_scale))
	before = time.time()
	binrmat = kdt.DiGraph.generateRMAT(gen_scale, element=1.0, delIsolated=True)
	kdt.p("Generated graph in %fs: \t%d\t vertices and \t%d\t edges."%(time.time()-before, binrmat.nvert(), binrmat.nedge()))

	kdt.p("--Converting binary RMAT to twitter object")
	G = kdt.DiGraph(nv=binrmat.nvert(), element=kdt.Obj2())
	G.e.eWiseApply(binrmat.e, op=Twitter_obj_randomizer, allowANulls=True, inPlace=True)
	kdt.p("Converted in %fs. G has %d vertices and %d edges."%(time.time()-before, G.nvert(), G.nedge()))
	kdt.p(G)
	
else:
	kdt.p("unknown data source. Does your file exist or did you specify an integer generation scale? quitting.")
	sys.exit()

kdt.p("--calculating degrees on original graph")
before = time.time()
origDegrees = G.degree()
kdt.p("Calculated in %fs."%(time.time()-before))

sejits_filter = None

class SemiringTypeToUse:
	PYTHON = 0
	CPP = 1
	SEJITS = 2
	
	@staticmethod
	def get_string(value):
		return {SemiringTypeToUse.PYTHON: "PythonSR",
		 SemiringTypeToUse.CPP: "C++SR",
		 SemiringTypeToUse.SEJITS: "SejitsSR"}[value]
 
def run(SR_to_use, use_SEJITS_Filter, materialize):
	global G, nstarts, origDegrees, filterUpperValue, sejits_filter
	runStarts = nstarts
	filterPercent = filterUpperValue/100.0
	
	G.addEFilter(twitterEdgeFilter)
	materializeTime = 0
	if materialize:
		kdt.p("--Materializing the filter")
		before = time.time()
		G.e.materializeFilter()
		materializeTime = time.time()-before
		kdt.p("Materialized %f in\t%f\ts."%(filterPercent, materializeTime))
		kdt.p("%f\t: \t%d\t edges survived the filter."%(filterPercent, G.nedge()))

	
	kdt.p("--Generating starting verts")
	before = time.time()
	degrees = G.degree()
	
	deg3verts = (degrees > 2).findInds()
	if len(deg3verts) == 0:
		# this is mainly for filter_debug.txt
		deg3verts = (degrees > 0).findInds()
	if len(deg3verts) == 0:
		starts = []
	else:
		deg3verts.randPerm()
		if runStarts > len(deg3verts):
			runStarts = len(deg3verts)
		starts = deg3verts[kdt.Vec.range(runStarts)]
	kdt.p("Generated starting verts in %fs."%(time.time()-before))

	kdt.p("--Doing BFS")
	if use_SEJITS_Filter:
		G.delEFilter(twitterEdgeFilter)
		G.addEFilter(sejits_filter)
	K2elapsed = [];
	K2edges = [];
	K2TEPS = [];
	K2ORIGTEPS = []
	K2MATTEPS = []
	
	i = 0
	for start in starts:
		start = int(start)
		
		# figure out which Semiring to use
		if SR_to_use == SemiringTypeToUse.PYTHON:
			PythonSR = True
			SEJITSSR = False
		if SR_to_use == SemiringTypeToUse.CPP:
			PythonSR = False
			SEJITSSR = False
		if SR_to_use == SemiringTypeToUse.SEJITS:
			PythonSR = True
			SEJITSSR = True
		
		before = time.time()
		# the actual BFS
		parents = G.bfsTree(start, usePythonSemiring=PythonSR, SEJITS_Python_SR=SEJITSSR)
		itertime = time.time() - before
		
		## // Aydin's code for finding number of edges:
		## FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
		## parentsp.Apply(set<int64_t>(1));
		## // we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
		## int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
		
		#import kdt.pyCombBLAS as pcb
		#parentsp_pcb = parents._dpv.Find(pcb.bind2nd(pcb.greater(), -1))
		#parentsp_pcb.Apply(pcb.set(1))
		#print "number of discovered verts: ",parentsp_pcb.getnee()," total: ",len(parents)
		#nedges = pcb.EWiseMult(parentsp_pcb, degrees._dpv, False, 0).Reduce(pcb.plus())
		
		# Compute the number of edges traversed by adding up each discovered vertex's degree.
		# The degree vector was computed before the reverse edges were added, hence the TEPS score
		# only includes edges present in the original graph.
		
		# This computation overwrites the parents vector, but it's not used again so it's ok.
		def TEPSupdate(p, deg):
			if p == -1:
				return 0
			else:
				return deg
		nedges = parents.eWiseApply(degrees, TEPSupdate).reduce(kdt.op_add) 
		if materialize:
			nOrigEdges = 0
		else:
			nOrigEdges = parents.eWiseApply(origDegrees, TEPSupdate).reduce(kdt.op_add) 
		
		##nedges2 = len((parents[origI] != -1).find())
		##if kdt.master():
		##	if (nedges != nedges2):
		##		print "edge counts differ! ewisemult method: %d, find() method: %d"%(nedges, nedges2)
		
		ndiscVerts = parents.count(lambda x: x != -1)
		if nedges >= keep_min_edges:
			K2elapsed.append(itertime)
			K2edges.append(nedges)
			K2TEPS.append(nedges/itertime)
			K2ORIGTEPS.append(nOrigEdges/itertime)
			K2MATTEPS.append(nedges/(itertime+materializeTime))
			discardedString = ""
		else:
			discardedString = "(result discarded)"
		
		i += 1
		# print result for this iteration
		kdt.p("%f\t: iteration %2d: start=%8d, BFS took \t%f\ts, covered \t%d\t edges, discovered \t%d\t verts, TEPS incl. filtered edges=\t%s\t, TEPS=\t%s\t %s"%(filterPercent, i, start, (itertime), nedges, ndiscVerts, splitthousands(nOrigEdges/itertime),splitthousands(nedges/itertime), discardedString))
		if len(K2edges) >= keep_starts:
			break
	
	# print results summary
	if kdt.master():
		if materialize:
			Mat = "Mat"
		else:
			Mat = "OTF"
		if use_SEJITS_Filter:
			SF = "SejitsFilter"
		else:
			SF = "PythonFilter"
		
		labeling = SemiringTypeToUse.get_string(SR_to_use)+"_"+SF+"_"+Mat
		
		print "\nBFS execution times (%s)"%(labeling)
		printstats(K2elapsed, "%stime\t%f\t"%(labeling, filterPercent), False, True, True)
		
		print "\nnumber of edges traversed %s"%(Mat)
		printstats(K2edges, "%snedge\t%f\t"%(labeling, filterPercent), False, True, True)
		
		print "\nTEPS (%s)"%(labeling)
		printstats(K2TEPS, "%s_TEPS\t%f\t"%(labeling, filterPercent), True, True)

		if not materialize:
			print "\nTEPS including filtered edges (%s)"%(labeling)
			printstats(K2ORIGTEPS, "IncFiltered_%s_TEPS\t%f\t"%(labeling, filterPercent), True, True)
		else:
			print "\nTEPS including materialization time (%s)"%(labeling)
			printstats(K2MATTEPS, "PlusMatTime_%s_TEPS\t%f\t"%(labeling, filterPercent), True, True)
	
	if use_SEJITS_Filter:
		G.delEFilter(sejits_filter)
	else:
		G.delEFilter(twitterEdgeFilter)

#for p in (0, 0.5, 1, 2, 5, 10, 20, 30, 40, 60, 100):
for p in (1, 10, 25, 100):
	filterUpperValue = int(p*100)

	for whatToDo in whatToDoList:
		# determine the semiring type to use
		if whatToDo[0] == 'p':
			SR_to_Use = SemiringTypeToUse.PYTHON
		elif whatToDo[0] == 'c':
			SR_to_Use = SemiringTypeToUse.CPP
		elif whatToDo[0] == 's':
			SR_to_Use = SemiringTypeToUse.SEJITS
		else:
			raise ValueError,"Invalid semiring specified in whatToDo %s"%whatToDo
		
		# determine the filter type to use
		if whatToDo[1] == 'p':
			use_SEJITS_Filter = False
		elif whatToDo[1] == 's':
			use_SEJITS_Filter = True
		else:
			raise ValueError,"Invalid filter type specified in whatToDo %s"%whatToDo
		
		# determine OTF or Materialize
		if whatToDo[2] == 'o':
			materialize = False
		elif whatToDo[2] == 'm':
			materialize = True
		else:
			raise ValueError,"Invalid materialization flag specified in whatToDo %s"%whatToDo
		
		single_runtime_before = time.time()
		if use_SEJITS_Filter: # put here so if the system doesn't have SEJITS it won't crash
			from pcb_predicate import *
	
			class TwitterFilter(PcbUnaryPredicate):
				def __init__(self, filterUpperValue):
					self.filterUpperValue = filterUpperValue
					super(TwitterFilter, self).__init__()
				def __call__(self, e):
					if (e.count > 0 and e.latest < self.filterUpperValue):
							return True
					else:
							return False
			before = time.time()
			sejits_filter = TwitterFilter(filterUpperValue).get_predicate()
			sejits_filter_create_time = time.time()-before
			kdt.p("Created SEJITS filter for \t%d\t%% in\t%f\ts."%(p, sejits_filter_create_time))

		run(SR_to_Use, use_SEJITS_Filter, materialize)
		single_runtime = time.time() - single_runtime_before
		kdt.p("Total runtime for %s on %d%% is\t%f\ts."%(whatToDo, p, single_runtime))

kdt.p("Total runtime for everything is %f"%(time.time()-time_very_beginning))
