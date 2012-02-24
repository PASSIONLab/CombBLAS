import sys
import os
import time
import math
import random
import kdt
import kdt.pyCombBLAS as pcb
from stats import splitthousands, printstats

#parse arguments
if (len(sys.argv) < 2):
	kdt.p("Usage: python %s twittergraph.mtx [1]"%(sys.argv[0]))
	kdt.p("The 1st argument is either a datafile or an integer which is the scale for RMAT generation.")
	kdt.p("The 2nd argument determines whether or not to use a materializing filter")
	kdt.p("Examples:")
	kdt.p("python filter_debug.mtx 1")
	kdt.p("python 14")
	sys.exit()

datasource = "file"
inmatrixfile = sys.argv[1]
gen_scale = 10

# report results of keep_starts runs, where each run traverses at least keep_min_edges edges.
# total maximum of runs is nstarts.
nstarts = 512
keep_starts = 16
keep_min_edges = 100
materializeArg = False

if (len(sys.argv) >= 3):
	materializeArg = bool(int(sys.argv[2]))

# this function is used for generation.
# obj is the object that needs to be filled in
# bin is a throwaway value.
# http://docs.python.org/library/random.html
def Twitter_obj_randomizer(obj, bin):
	if random.randrange(0, 2) > 0:
		obj.count = 1
		obj.latest = random.randrange(1244592000, 1246406400)
	else:
		obj.count = 0
	obj.follower = 0
	return obj


#def twitterMul(e, f):
#	if e.count > 0 and e.latest > 946684800 and e.latest < 1249084800:
#		return f
#	else:
#		return -1


# 2009-06-10 0:0:0 == 1244592000
# 2009-06-13 0:0:0 == 1244851200
# 2009-07-01 0:0:0 == 1246406400
def twitterEdgeFilter(e):
	#print e.latest, e
	return e.count > 0 and e.latest < 1246406400

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
	kdt.p("Read in %fs. Read %d vertices and %d edges."%(time.time()-before, G.nvert(), G.nedge()))
	
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
	kdt.p("Generated in %fs: %d vertices and %d edges."%(time.time()-before, binrmat.nvert(), binrmat.nedge()))

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


def run(materialize):
	global G, nstarts, origDegrees
	
	G.addEFilter(twitterEdgeFilter)
	if materialize:
		kdt.p("--Materializing the filter")
		before = time.time()
		G.e.materializeFilter()
		kdt.p("Materialized in %fs."%(time.time()-before))
		kdt.p("%d edges survived the filter."%(G.nedge()))

	
	kdt.p("--Generating starting verts")
	before = time.time()
	degrees = G.degree()
	
	deg3verts = (degrees > 2).findInds()
	if len(deg3verts) == 0:
		# this is mainly for filter_debug.txt
		deg3verts = (degrees > 0).findInds()
	deg3verts.randPerm()
	if nstarts > len(deg3verts):
		nstarts = len(deg3verts)
	starts = deg3verts[kdt.Vec.range(nstarts)]
	kdt.p("Generated in %fs."%(time.time()-before))
	
	kdt.p("--Doing BFS")
	
	K2elapsed = [];
	K2edges = [];
	K2TEPS = [];
	K2ORIGTEPS = []
	
	i = 0
	for start in starts:
		start = int(start)
		
		before = time.time()
		# the actual BFS
		parents = G.bfsTree(start)
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
			discardedString = ""
		else:
			discardedString = "(result discarded)"
		
		i += 1
		# print result for this iteration
		kdt.p("iteration %2d: start=%8d, BFS took %10.4fs, covered %10d edges, discovered %8d verts, TEPS incl. filtered edges=%10s, TEPS=%s %s"%(i, start, (itertime), nedges, ndiscVerts, splitthousands(nOrigEdges/itertime),splitthousands(nedges/itertime), discardedString))
		if len(K2edges) >= keep_starts:
			break
	
	# print results summary
	if kdt.master():
		if materialize:
			Mat = "(materialized)"
			Mat_ = "Mat_"
		else:
			Mat = "(on-the-fly)"
			Mat_ = "OTF_"
		print "\nBFS execution times %s"%(Mat)
		printstats(K2elapsed, "%stime"%(Mat_), False)
		
		print "\nnumber of edges traversed %s"%(Mat)
		printstats(K2edges, "%snedge"%(Mat_), False)
		
		print "\nTEPS %s"%(Mat)
		printstats(K2TEPS, "%sTEPS"%(Mat_), True)
	
	G.delEFilter(twitterEdgeFilter)

run(False)
if materializeArg:
	kdt.p("\n\nRunning again using a materializing filter.")
	run(True)