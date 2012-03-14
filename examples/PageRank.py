import sys
import kdt
import time

#parse arguments
if (len(sys.argv) < 1):
	if kdt.master():
		print "Usage: python %s GRAPH.mtx"%(sys.argv[0])
	sys.exit()
	
inmatrixfile = sys.argv[1]

# load
G = kdt.DiGraph.load(inmatrixfile)
#G.ones()

if kdt.master():
	print "Graph loaded."

# compute PageRank
startTime = time.time()
pr = G.pageRank(epsilon=0.0000001)
endTime = time.time()

if kdt.master():
	print "PageRank computation took %f seconds."%(endTime-startTime)
