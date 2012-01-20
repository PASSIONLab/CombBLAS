import sys
import time
import math
import kdt
import kdt.pyCombBLAS as pcb
from stats import splitthousands, printstats

#parse arguments
if (len(sys.argv) < 2):
	print "Usage: python %s twittergraph.mtx [1]"%(sys.argv[0])
	print "the 2nd argument determines whether or not to use a materializing filter"
	sys.exit()

inmatrixfile = sys.argv[1]
nstarts = 16
materializeArg = False

if (len(sys.argv) >= 3):
	materializeArg = bool(int(sys.argv[2]))

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

# doubleint() constructor returns -1 now
def twitterMul(e, f):
	return f

def twitterAdd(f1, f2):
	if f2 == -1:
		return f1
	return f2

def bfsTreeTwitter(self, root):
	sR = kdt.sr(twitterAdd, twitterMul, twitterEdgeFilter, None)

	parents = kdt.Vec(self.nvert(), -1, sparse=False)
	frontier = kdt.Vec(self.nvert(), sparse=True)
	parents[root] = root
	frontier[root] = root
	while frontier.nnn() > 0:
		frontier.spRange()
		self.e.SpMV(frontier, semiring=sR, inPlace=True)
		
		# remove already discovered vertices from the frontier.
		frontier.eWiseApply(parents, op=(lambda f,p: f), doOp=(lambda f,p: f != -1 and p == -1), inPlace=True)
		# update the parents
		parents[frontier] = frontier

	return parents
kdt.DiGraph.bfsTreeTwitter = bfsTreeTwitter

# load
kdt.p("Reading network from %s"%inmatrixfile)
before = time.time()
G = kdt.DiGraph.load(inmatrixfile, eelement=kdt.Obj2())
kdt.p("Read in %fs. Read %d vertices and %d edges."%(time.time()-before, G.nvert(), G.nedge()))

#kdt.p(G)
#G.addEFilter(twitterEdgeFilter)

# optimize the graph
kdt.p("deleting isolated vertices and randomly permuting matrix for load balance")
before = time.time()
G.delIsolatedVerts(True)
kdt.p("Done in %fs."%(time.time()-before))

kdt.p("calculating degrees on original graph")
before = time.time()
origDegrees = G.degree()
kdt.p("Calculated in %fs."%(time.time()-before))


def run(materialize):
	global G, nstarts, origDegrees
	
	G.addEFilter(twitterEdgeFilter)
	if materialize:
		kdt.p("Materializing the filter")
		before = time.time()
		G.e.materializeFilter()
		kdt.p("Materialized in %fs."%(time.time()-before))
		kdt.p("%d edges survived the filter."%(G.nedge()))

	
	kdt.p("Generating starting verts")
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
	
	kdt.p("Doing BFS")
	
	K2elapsed = [];
	K2edges = [];
	K2TEPS = [];
	K2ORIGTEPS = []
	
	i = 0
	for start in starts:
		start = int(start)
		
		before = time.time()
		# the actual BFS
		parents = G.bfsTreeTwitter(start)
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
		
		K2elapsed.append(itertime)
		K2edges.append(nedges)
		K2TEPS.append(nedges/itertime)
		K2ORIGTEPS.append(nOrigEdges/itertime)
		ndiscVerts = parents.count(lambda x: x != -1)
		
		i += 1
		# print result for this iteration
		kdt.p("iteration %2d: start=%8d, BFS took %10.4fs, covered %10d edges, discovered %8d verts, TEPS incl. filtered edges=%10s, TEPS=%s"%(i, start, (itertime), nedges, ndiscVerts, splitthousands(nOrigEdges/itertime),splitthousands(nedges/itertime)))
	
	# print results summary
	if kdt.master():
		print "\nBFS execution times"
		printstats(K2elapsed, "time", False)
		
		print "\nnumber of edges traversed"
		printstats(K2edges, "nedge", False)
		
		print "\nTEPS"
		printstats(K2TEPS, "TEPS", True)
	
	G.delEFilter(twitterEdgeFilter)

run(False)
if materializeArg:
	kdt.p("\n\nRunning again using a materializing filter.")
	run(True)