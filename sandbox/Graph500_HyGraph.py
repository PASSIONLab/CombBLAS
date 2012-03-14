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
nstarts = 5
file = ""

def usage():
	print "Graph500 [-sSCALE] [-nNUM_STARTS] [-fFILE]"
	print "SCALE refers to the size of the generated RMAT graph G. G will have 2^SCALE vertices and edge factor 16. Default scale is 15."
	print "NUM_STARTS is the number of randomly chosen starting vertices. The Graph500 spec requires 64 starts."
	print "FILE is a MatrixMarket .mtx file with graph to use. Graph should be directed and symmetric"
	print "Default is: python Graph500.py -s15 -n64"

try:
	opts, args = getopt.getopt(sys.argv[1:], "hs:n:f:", ["help", "scale=", "nstarts=", "file="])
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
	else:
		assert False, "unhandled option"


def k2Validate(G, start, parents):
	good = True
	
	[valid, levels] = G.isBfsTree(start, parents)
	#	isBfsTree implements Graph500 tests 1 and 2 
	if not valid:
		if kdt.master():
			print "isBfsTree detected failure of Graph500 test %d" % abs(ret)
		return False

	# Spec test #3:
	# every input edge has vertices whose levels differ by no more than 1
	edgeMax = kdt.SpParVec.toSpParVec(G._spm.SpMV_SelMax(levels.toSpParVecAll()._spv))
	edgeMin = -kdt.SpParVec.toSpParVec(G._spm.SpMV_SelMax((-levels).toSpParVecAll()._spv))
	if ((edgeMax-edgeMin) > 1).any():
		if kdt.master():
			print "At least one graph edge has endpoints whose levels differ by more than one"
		good = False

	# Spec test #4:
	# the BFS tree spans a connected component's vertices (== all edges 
	# either have both endpoints in the tree or not in the tree, or 
	# source is not in tree and destination is the root)

	# set not-in-tree vertices' levels to -2
	import pyCombBLAS as pcb
	levels._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(),-1), pcb.set(-2), pcb.identity()))
	edgeMax = kdt.SpParVec.toSpParVec(G._spm.SpMV_SelMax(levels.toSpParVecAll()._spv))
	edgeMin = -kdt.SpParVec.toSpParVec(G._spm.SpMV_SelMax((-levels).toSpParVecAll()._spv))
	if ((edgeMax-edgeMin) > 1).any():
		if kdt.master():
			print "The tree does not span exactly the connected component, root=%d"
		good = False

	# Spec test #5:
	# a vertex and its parent are joined by an edge of the original graph,
	# except for the root, which has no parent in the tree
	Gnv = G.nvert(); Gne = G.nedge()
	[Gi, Gj, ign] = G.toParVec()
	del ign
	# non-root tree vertices == NRT Vs
	NRTVs = (levels!=-2) & (parents!=kdt.ParVec.range(Gnv))
	nNRTVs = NRTVs.nnz()
	TGi = kdt.ParVec.range(nNRTVs)
	TGj1 = kdt.ParVec.range(Gnv)[NRTVs]
	TGj2 = parents[NRTVs]
	M = max(Gne, Gnv)
	#FIX:  really should use SpParMats here, as don't need spm and spmT	
	tmpG1 = kdt.HyGraph(TGi, TGj1, 1, M, Gnv)
	tmpG2 = kdt.HyGraph(TGi, TGj2, 1, M, Gnv)
	tmpG1._spm  += tmpG2._spm
	tmpG1._spmT += tmpG2._spmT
	del tmpG2
	tmpG3 = kdt.HyGraph(Gi, Gj, 1, M, Gnv)
	tmpG4 = kdt.DiGraph()
	tmpG4._spm = tmpG1._spm.SpMM(tmpG3._spmT)  #!?  not tmp3._spmT ?
	maxIncid = tmpG4.max(kdt.DiGraph.Out)[kdt.ParVec.range(Gnv) < nNRTVs]
	if (maxIncid != 2).any():
		if kdt.master():
			print "At least one vertex and its parent are not joined by an original edge"
		good = False

	return good


if len(file) == 0:
	raise SystemExit, "No generation of Graph500 HyGraph for now; must use file"
	if kdt.master():
		print "Generating a Graph500 RMAT graph with 2^%d vertices..."%(scale)
	G = kdt.HyGraph()
	K1elapsed = G.genGraph500Edges(scale)
	#G.save("testgraph.mtx")
	if kdt.master():
		print "Generation took %fs."%(K1elapsed)

else:
	if kdt.master():
		print 'Loading %s'%(file)
	G = kdt.HyGraph.load(file)
	K1elapsed = 0.0


if False:
	raise SystemExit, "No use of synthetic HyGraph generator for now; must use file"
	if kdt.master():
		print 'Using 2D torus graph generator'
	G = kdt.DiGraph.twoDTorus(2**(scale/2))
	K1elapsed = 0.00005
	starts = kdt.ParVec.range(nstarts)


#	indices of vertices with degree > 0
deg3verts = (G.degree() > 0).findInds()
if len(deg3verts) == 0:
	raise SystemExit, 'No vertices with degree greater than 0'
if nstarts > len(deg3verts):
	nstarts = len(deg3verts)
deg3verts.randPerm()
starts = deg3verts[kdt.ParVec.range(nstarts)]

#G.toBool()		# not now;  SpMM not defined for Bool pySpParMats
G.set(1);		# set all values to 1

[origI, ign, ign2] = G.toParVec()
del ign, ign2

K2elapsed = [];
K2edges = [];
K2TEPS = [];

i = 0
for start in starts:
	start = int(start)
	# FIX
	# print "\nFIX!!  root hard-coded to 1\n"
	# start = 1
	before = time.time()
	
	# the actual BFS
	parents = G.bfsTree(start)
	
	itertime = time.time() - before
	nedges = len((parents[origI] != -1).find())
	
	K2elapsed.append(itertime)
	K2edges.append(nedges)
	K2TEPS.append(nedges/itertime)
	
	i += 1
	verifyInitTime = time.time()
	verifyResult = "succeeded"
	if not k2Validate(G, start, parents):
		verifyResult = "FAILED"
	verifyTime = time.time() - verifyInitTime

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

