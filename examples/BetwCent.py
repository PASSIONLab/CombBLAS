import kdt
import time
import getopt
import sys
from stats import splitthousands

scale = 50
sample = 0.05
batchSize = -1
file = ""
BCdebug=0

class Source:
    FILE=1
    TORUS=2
    G500=3
    PRINT_USAGE=4
source = Source.PRINT_USAGE

def usage():
	print "BetwCent.py (-tTORUS_DIM | -gG500_SCALE | -fFILE) [-xSAMPLE] [-bBATCH_SIZE] [-d]"
	print "The first three arguments specify the graph source. Exactly one of the three must be used."
	print "  TORUS_DIM refers to the size of a generated Torus graph G. G will have TORUS_DIM^2 vertices."
	print "  G500_SCALE refers to the size of a generated Graph500-style RMAT graph G. G will have 2^G500_SCALE vertices."
	print "  FILE is a MatrixMarket .mtx file with graph to use. Graph should be directed, symmetric and connected (for accurate TEPS score)."
	print ""
	print "  SAMPLE refers to the fraction of vertices to use as SSSP starts. 1.0 = exact BC."
	print "  BATCH_SIZE allows manually specifying the batch size used at each iteration."
	print "  -d prints debug info."
	print ""
	print "Example:"
	print "Calculate BC on a scale 12 Graph500 RMAT with a sample size of 10%"
	print "python BetwCent.py -g12 -x0.1"

try:
	opts, args = getopt.getopt(sys.argv[1:], "ht:g:f:x:b:dD", ["help", "torus=", "g500", "file=", "sample=", "batchsize", "debug", "DEBUG"])
except getopt.GetoptError, err:
	# print help information and exit:
	if kdt.master():
		print str(err) # will print something like "option -a not recognized"
		usage()
	sys.exit(2)
output = None
verbose = False
for o, a in opts:
	if o in ("-h", "--help"):
		usage()
		sys.exit()
	elif o in ("-t", "--torus"):
		source = Source.TORUS
		scale = int(a)
	elif o in ("-g", "--g500"):
		source = Source.G500
		scale = int(a)
	elif o in ("-x", "--sample"):
		sample = float(a)
	elif o in ("-b", "--batch"):
		batchSize = int(a)
	elif o in ("-f", "--file"):
		source = Source.FILE
		file = a
	elif o in ("-d", "--debug"):
		BCdebug = 1
	elif o in ("-D", "--DEBUG"):
		BCdebug = 2
	else:
		assert False, "unhandled option"
		
		
# setup the graph
if source == Source.TORUS:
	if kdt.master():
		print "Generating a Torus graph with %d^2 vertices..."%(scale)

	G1 = kdt.DiGraph.twoDTorus(scale)
	nverts = G1.nvert()
	if kdt.master():
		print "Graph has",nverts,"vertices."
	#G1.toBool()
elif source == Source.G500:
	if kdt.master():
		print "Generating a Graph500 graph with 2^%d vertices..."%(scale)

	G1 = kdt.DiGraph()
	G1.genGraph500Edges(scale)
	nverts = G1.nvert()
	if kdt.master():
		print "Graph has",nverts,"vertices."
	#G1.toBool()
elif source == Source.FILE:
	if kdt.master():
		print 'Loading %s'%(file)
	G1 = kdt.DiGraph.load(file)
	#G1.toBool()
else:
	if kdt.master():
		usage()
	sys.exit(2)
	

# Call BC	
before = time.time();
bc, nStartVerts = G1.centrality('approxBC', sample=sample, BCdebug=BCdebug, batchSize=batchSize, retNVerts=True)
time = time.time() - before;

# Torus BC scores should all be identical.
if source == Source.TORUS and ((bc - bc[0]) > 1e-15).any():
	if kdt.master():
		print "not all vertices have same BC value"

# Report
nedges = G1._spm.getnee()*nStartVerts
TEPS = float(nedges)/time
min = bc.min()
max = bc.max()
mean = bc.mean()
std = bc.std()
if kdt.master():
	print "bc[0] = %f, min=%f, max=%f, mean=%f, std=%f" % (bc[0], min, max, mean, std)
	print "   used %d starting vertices" % nStartVerts
	print "   took %4.3f seconds" % time
	print "   TEPS = %s (assumes the graph was connected)" % splitthousands(TEPS)

