import kdt
import time
import getopt
import sys

scale = 12
sample = 0.05
file = ""
BCdebug=0

useTorus = True

def usage():
	print "BetwCent.py [-sSCALE] [-xSAMPLE] [-fFILE]"
	print "SCALE refers to the size of the generated Torus graph G. G will have 2^SCALE vertices."
	print "SAMPLE refers to the fraction of vertices to use as SSSP starts. 1.0 = exact BC."
	print "FILE is a MatrixMarket .mtx file with graph to use. Graph should be directed and symmetric"
	print "Default is: python BetwCent.py -s%d -x%f"%(scale, sample)

try:
	opts, args = getopt.getopt(sys.argv[1:], "hs:f:x:d", ["help", "scale=", "file=", "sample=", "debug"])
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
	elif o in ("-x", "--sample"):
		nstarts = int(a)
	elif o in ("-f", "--file"):
		file = a
		useTorus = False
	elif o in ("-d", "--debug"):
		BCdebug = 1
	else:
		assert False, "unhandled option"
		
		
# setup the graph
if len(file) == 0:
	if kdt.master():
		print "Generating a Torus graph with 2^%d vertices..."%(scale)

	G1 = kdt.DiGraph.twoDTorus(scale)
	#G1.toBool()
else:
	if kdt.master():
		print 'Loading %s'%(file)
	G1 = kdt.DiGraph.load(file)
	#G1.toBool()

# Call BC	
before = time.time();
bc = G1.centrality('approxBC', sample=sample, BCdebug=BCdebug)
time = time.time() - before;

# Check
if useTorus and ((bc - bc[0]) > 1e-15).any():
	if kdt.master():
		print "not all vertices have same BC value"

# Report
min = bc.min()
max = bc.max()
mean = bc.mean()
std = bc.std()
if kdt.master():
	print "bc[0] = %f, min=%f, max=%f, mean=%f, std=%f" % (bc[0], min, max, mean, std)
	print "   took %4.3f seconds" % time

