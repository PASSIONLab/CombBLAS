"""
The Graph500 module implements the Graph500 benchmark (v1.1), which includes
kernels 1 (graph construction) and 2 (breadth-first search).  In addition to
constructing the graph as specified, the module implements all 5 validation
steps in the spec. See www.graph500.org/Specifications.html for more detail.  

The variables in this script that will commonly be changed are:
	scale:  The logarithm base 2 of the number of vertices in the 
	    resulting graph.
	nstarts:  The number of times to create a BFS tree from a random
	    root vertex.

The edge factor, whose default value is 16, is not easily changeable.
"""
import time
import math
import sys
import getopt
import kdt
from stats import splitthousands, printstats

scale = 15
nstarts = 64
file = ""

doValidation = False
useEWise = False

def usage():
	print "Graph500.py [-sSCALE] [-nNUM_STARTS] [-fFILE]"
	print "SCALE refers to the size of the generated RMAT graph G. G will have 2^SCALE vertices and edge factor 16. Default scale is 15."
	print "NUM_STARTS is the number of randomly chosen starting vertices. The Graph500 spec requires 64 starts."
	print "FILE is a MatrixMarket .mtx file with graph to use. Graph should be directed and symmetric"
	print "Default is: python Graph500.py -s15 -n64"

try:
	opts, args = getopt.getopt(sys.argv[1:], "hs:n:f:ve", ["help", "scale=", "nstarts=", "file=", "validate","ewise"])
except getopt.GetoptError, err:
	# print help information and exit:
	print str(err) # will print something like "option -a not recognized"
	usage()
	sys.exit(2)
output = None
verbose = False
for o, a in opts:
	if o in ("-h", "--help"):
		usage()
		sys.exit()
	elif o in ("-s", "--scale"):
		scale = int(a)
	elif o in ("-n", "--nstarts"):
		nstarts = int(a)
	elif o in ("-f", "--file"):
		file = a
	elif o in ("-v", "--validate"):
		doValidation = True
	elif o in ("-e", "--ewise"):
		useEWise = True
	else:
		assert False, "unhandled option"


def k2Validate(G, start, parents):
	good = True
	
	(valid, levels) = G.isBfsTree(start, parents)
	#	isBfsTree implements Graph500 tests 1 and 2 
	if not valid:
		if kdt.master():
			print "isBfsTree detected failure of Graph500 test %d" % abs(ret)
		return False

	# Spec test #3:
	# every input edge has vertices whose levels differ by no more than 1
	# Note:  don't actually have input edges, will use the edges in
	#    the resulting graph as a proxy
	[origI, origJ, ign] = G.toParVec()
	del ign
	li = levels[origI]; 
	lj = levels[origJ]
	del origI
	if not ((abs(li-lj) <= 1) | ((li==-1) & (lj==-1))).all():
		if kdt.master():
			print "At least one graph edge has endpoints whose levels differ by more than one and is in the BFS tree"
			print li, lj
		good = False

	# Spec test #4:
	# the BFS tree spans a connected component's vertices (== all edges 
	# either have both endpoints in the tree or not in the tree, or 
	# source is not in tree and destination is the root)
	neither_in = (li == -1) & (lj == -1)
	both_in = (li > -1) & (lj > -1)
	out2root = (li == -1) & (origJ == start)
	del origJ
	if not (neither_in | both_in | out2root).all():
		if kdt.master():
			print "The tree does not span exactly the connected component, root=%d" % start
			#print levels, neither_in, both_in, out2root, (neither_in | both_in | out2root)
		good = False
	del both_in, out2root

	# Spec test #5:
	# a vertex and its parent are joined by an edge of the original graph
	respects = abs(li-lj) <= 1
	if not (neither_in | respects).all():
		if kdt.master():
			print "At least one vertex and its parent are not joined by an original edge"
		good = False

	return good

########################
# Test BFS with EWise()

def bfsTreeEWise(G, root, sym=False):
	import kdt.pyCombBLAS as pcb
	
	if not sym:
		G.T()
	parents = pcb.pyDenseParVec(G.nvert(), -1)
	frontier = pcb.pySpParVec(G.nvert())
	parents[root] = root
	frontier[root] = root
	
	def iterop(vals):
		#print "visiting ",vals[2]
		# vals[0] = frontier value
		# vals[1] = parents value
		# vals[2] = index
		if (vals[1] == -1):
			# discovered new vertex. Update parents and set frontier's value to its index
			vals[1] = vals[0]
			vals[0] = vals[2]
		else:
			# vertex already discovered. Remove it from the frontier
			vals[0] = None
	
	ops = 0;
	while frontier.getnee() > 0:
		G._spm.SpMV_SelMax_inplace(frontier)
		#opss = time.time()
		#pcb.EWise(iterop, [pcb.EWise_OnlyNZ(frontier), parents, pcb.EWise_Index()])
		pcb.Graph500VectorOps(frontier, parents)
		#ops = ops + (time.time() - opss)
	
	#print "time: %fs"%(ops)
	
	if not sym:
		G.T()
	return kdt.ParVec.toParVec(parents)

###########################



if len(file) == 0:
	if kdt.master():
		print "Generating a Graph500 RMAT graph with 2^%d vertices..."%(scale)
	G = kdt.DiGraph()
	G.toBool()
	
	#deg3verts = kdt.ParVec(1);
	K1elapsed, degrees = G.genGraph500Edges(scale)
	#G.save("testgraph.mtx")
	if kdt.master():
		print "Generation took %fs."%(K1elapsed)

	if nstarts > G.nvert():
		nstarts = G.nvert()
	#	indices of vertices with degree > 2

	#deg3verts = (G.degree() > 2).findInds()
	deg3verts = (degrees > 2).findInds()
	deg3verts.randPerm()
	starts = deg3verts[kdt.ParVec.range(nstarts)]

else:
	if kdt.master():
		print 'Loading %s'%(file)
	before = time.time()
	G = kdt.DiGraph.load(file)
	K1elapsed = time.time() - before
	
	degrees = G.degree()

	deg3verts = (degrees > 2).findInds()
	deg3verts.randPerm()
	starts = deg3verts[kdt.ParVec.range(nstarts)]
	G.toBool()

if False:
	if kdt.master():
		print 'Using 2D torus graph generator'
	G = kdt.DiGraph.twoDTorus(2**(scale/2))
	K1elapsed = 0.00005
	starts = kdt.ParVec.range(nstarts)
	degrees = G.degree()



#[origI, ign, ign2] = G.toParVec()
#del ign, ign2

K2elapsed = [];
K2edges = [];
K2TEPS = [];

i = 0
for start in starts:
	start = int(start)
	before = time.time()
	
	# the actual BFS
	if (useEWise):
		parents = bfsTreeEWise(G, start, sym=True)
	else:
		parents = G.bfsTree(start, sym=True)
	
	itertime = time.time() - before
	
	# // Aydin's code for finding number of edges:
	# FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
	# parentsp.Apply(set<int64_t>(1));
	# // we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
	# int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
	import kdt.pyCombBLAS as pcb
	parentsp_pcb = parents._dpv.Find(pcb.bind2nd(pcb.greater(), -1))
	parentsp_pcb.Apply(pcb.set(1))
	#print "number of discovered verts: ",parentsp_pcb.getnee()," total: ",len(parents)
	nedges = pcb.EWiseMult(parentsp_pcb, degrees._dpv, False, 0).Reduce(pcb.plus())
	
	#nedges2 = len((parents[origI] != -1).find())
	#if kdt.master():
	#	if (nedges != nedges2):
	#		print "edge counts differ! ewisemult method: %d, find() method: %d"%(nedges, nedges2)
	
	K2elapsed.append(itertime)
	K2edges.append(nedges)
	K2TEPS.append(nedges/itertime)
	
	i += 1
	if (doValidation):
		verifyInitTime = time.time()
		verifyResult = "succeeded"
		if not k2Validate(G, start, parents):
			verifyResult = "FAILED"
		verifyTime = time.time() - verifyInitTime
	else:
		verifyTime = 0
		verifyResult = "not done (use -v switch)"

	if kdt.master():
		print "iteration %d: start=%d, BFS took %fs, verification took %fs and %s, TEPS=%s"%(i, start, (itertime), verifyTime, verifyResult, splitthousands(nedges/itertime))

if kdt.master():
	print 'Graph500 benchmark run for scale = %2i' % scale
	print 'Kernel 1 time = %8.4f seconds' % K1elapsed
	#print 'Kernel 2 time = %8.4f seconds' % K2elapsed
	#print '                    %8.4f seconds for each of %i starts' % (K2elapsed/nstarts, nstarts)
	#print 'Kernel 2 TEPS = %7.4e' % (K2edges/K2elapsed)
	
	print "\nKernel 2 BFS execution times"
	printstats(K2elapsed, "time", False)
	
	print "\nKernel 2 number of edges traversed"
	printstats(K2edges, "nedge", False)
	
	print "\nKernel 2 TEPS"
	printstats(K2TEPS, "TEPS", True)

