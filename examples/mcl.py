import kdt
import sys
import time
from stats import splitthousands

if (len(sys.argv) < 1):
	B = kdt.DiGraph()
	B.genGraph500Edges(10)
	B._spm.Apply(kdt.pyCombBLAS.set(1))
else:
	inmatrixfile = sys.argv[1]
	if (kdt.master()):
		print "Loading matrix from",inmatrixfile
	B = kdt.DiGraph.load(inmatrixfile)

bedges = B._spm.getnee()

expansion=3
inflation = 3
prunelimit = 0.0000001
addSelfLoops=False

# nedges run
if kdt.master():
	print "Starting run to find number of edges..."
C, nedges = B._markov(addSelfLoops=addSelfLoops, expansion=expansion, inflation=inflation, prunelimit=prunelimit, retNEdges=True)

# timed run
if kdt.master():
	print "nedges=%d. Starting timed run..."%(nedges)
before = time.time()
B._markov(addSelfLoops=addSelfLoops, expansion=expansion, inflation=inflation, prunelimit=prunelimit)
time = time.time() - before


cedges = C._spm.getnee()
if kdt.master():
	print "Started with %d edges and finished with %d. Took %lfs, TEPS=%s"%(bedges, cedges, time, splitthousands(nedges/time))