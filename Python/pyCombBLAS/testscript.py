import sys
import time
import math
import pyCombBLAS as pcb

###############################################
###########    HELPER FUNCTIONS

# makes numbers pretty
def splitthousands(s, sep=','):
	s = str(int(s))
	if (len(s) <= 3): return s  
	return splitthousands(s[:-3], sep) + sep + s[-3:]

# prints statistics about an array
def printstats(data, label, israte):
	n = len(data)
	data.sort()
	
	#min
	min = data[0]
	
	#first quantile
	t = (n+1) / 4.0
	k = int(t)
	if (t == k):
		q1 = data[k]
	else:
		q1 = 3*(data[k]/4.0) + data[k+1]/4.0;
		
	# median
	t = (n+1) / 2.0
	k = int(t)
	if (t == k):
		median = data[k]
	else:
		median = data[k]/2.0 + data[k+1]/2.0;
	
	# third quantile
	t = 3*((n+1) / 4.0);
	k = int(t)
	if (t == k):
		q3 = data[k]
	else:
		q3 = data[k]/4.0 + 3*(data[k+1]/4.0);

	#max
	max = data[n-1];
	
	#mean
	sum = 0.0
	for i in range(n-1, -1, -1):
		sum = sum + data[i]
	mean = sum/n;
	
	#standard deviation
	s = 0.0
	for k in range(n-1, -1, -1):
		tmp = data[k] - mean
		s = s + tmp*tmp
	sampleStdDev = math.sqrt(s/(n-1))

	#harmonic mean
	s = 0.0
	for k in range(0,n):
		if (data[k]):
			s = s + 1.0/data[k]
	harmonicMean = n/s
	m = s/n
		
	#harmonic sample standard deviation
	s = 0.0
	for k in range(0, n):
		if (data[k]):
			tmp = 1.0/data[k] - m;
		else:
			tmp = -m
		s = tmp*tmp
	harmonicSampleStdDev = (math.sqrt (s)/(n-1)) * harmonicMean * harmonicMean
	
	print "            min_%s: %20.17e"%(label, min);
	print "  firstquartile_%s: %20.17e"%(label, q1);
	print "         median_%s: %20.17e"%(label, median);
	print "  thirdquartile_%s: %20.17e"%(label, q3);
	print "            max_%s: %20.17e"%(label, max);
	if (israte):
		print "  harmonic_mean_%s: %20.17e"%(label, harmonicMean);
		print "harmonic_stddev_%s: %20.17e"%(label, harmonicSampleStdDev);
	else:
		print "           mean_%s: %20.17e"%(label, mean);
		print "         stddev_%s: %20.17e"%(label, sampleStdDev);

###############################################
###########    MATRIX CREATION

A = pcb.pySpParMat()
scale = 20

degrees = pcb.pyDenseParVec(4, 0);
k1time = 0.0

if len(sys.argv) == 2:
	scale = int(sys.argv[1])

if (scale < 0):
	path = "/home/alugowski/matrices/rmat_scale16.mtx";
	path = "../../CombBLAS/TESTDATA/SCALE16BTW-TRANSBOOL/input.txt";
	print "loading matrix from %s"%(path)
	A.load(path)
	A.Apply_SetTo(1)
	n = A.getnrow()
	
	colreducer = pcb.pyDenseParVec(n, 1).sparse();
	degrees = A.SpMV_PlusTimes(colreducer).dense();

else:
	if (pcb.root()):
		print "Generating RMAT with 2**%d nodes" %(scale)
	k1time = A.GenGraph500Edges(scale, degrees)
	if (pcb.root()):
		print "Generation took %lf s"%(k1time)
	

n = A.getnrow()
m = A.getncol()
nnz = A.getnnz()
edgefactor = nnz/n;
if (pcb.root()):
	print "A is %d by %d with %d nonzeros." % (n, m, nnz)


###############################################
###########    CANDIDATE SELECTION

numCands = 64
if (numCands > n):
	numCands = n

#Cands = A.FindIndsOfColsWithSumGreaterThan(4);

#numAvailableCands = Cands.length()
#if (numAvailableCands < numCands):
#	if (pcb.root()):
#		print "Not enough vertices in the graph qualify as candidates. Only %d have enough degree."%(numAvailableCands)
#	numCands = numAvailableCands

#Cands.RandPerm();
#First64 = pcb.pyDenseParVec.range(numCands, 0);
#Cands = Cands.SubsRef(First64);


Cands = A.GenGraph500Candidates(numCands)

#if (pcb.root()):
#	print "The candidates are:"
#Cands.printall()

if (pcb.root()):
	print "Starting vertices generated."

###############################################
###########    MAIN LOOP

times = [];
travEdges = [];
TEPS = [];
iterations = [];

for i in range(0, numCands):
	c = Cands.GetElement(i)

	# start the clock
	tstart = time.time()
	#------------------------- TIMED --------------------------------------------
	
	parents = pcb.pyDenseParVec(n, -1, -1)
	fringe = pcb.pySpParVec(n)
	fringe.SetElement(c, c);
	parents.SetElement(c, c);
	
	#if (pcb.root()):
	#	print "start fringe:" 
	#fringe.printall()
	
	niter = 0
	while (fringe.getnnz() > 0):
		fringe.setNumToInd()
		#print "fringe at start of iteration"
		#fringe.printall();
		A.SpMV_SelMax_inplace(fringe) #
		#print "fringe after SpMV"
		#fringe.printall();
		pcb.EWiseMult_inplacefirst(fringe, parents, True, -1)	#// clean-up vertices that already have parents 
		#print "fringe at end of iteration"
		#fringe.printall();
		#parents.ApplyMasked_SetTo(fringe, 0)
		parents.add(fringe)
		niter += 1
	
	#------------------------- TIMED --------------------------------------------
	telapsed = time.time() - tstart;

	################## REPORTING:

	#parents.printall()
	r = parents.Count_GreaterThan(-1)

	#find the number of edges we traversed. Some were traversed multiple times, but the spec
	# says the number of input edges.
	parentsSP = parents.sparse(-1);
	if (parentsSP.getnnz() != r):
		pnnz = parentsSP.getnnz()
		if (pcb.root()):
			print "oh oh oh oh noooooooo! (parents.nnz) %d != %d (parentsSP.nnz)"%(r, pnnz)
	parentsSP.Apply_SetTo(1);
	nedges = pcb.EWiseMult(parentsSP, degrees, False, 0).Reduce_sum()
	
	#s = A.SpMV_PlusTimes(parentsSP)
	#nedges = s.Reduce_sum()


	times.append(telapsed)
	iterations.append(niter)
	travEdges.append(nedges)
	TEPS.append(nedges/telapsed)

	# summarize this round	
	if (pcb.root()):
		print "Round %2d, root %-10d %15d parents (%5.2f%%),%3d iterations,%15d edges, %f s,   TEPS: %s" %((i+1), c, r, (100.0*r/n), niter, nedges, telapsed, splitthousands(nedges/telapsed))

###############################################
###########    RESULTS

if (pcb.root()):
	print "     SCALE: %d"%(scale)
	print "      nvtx: %d"%(n)
	print " num edges: %d"%(nnz)
	print "edgefactor: %d"%(edgefactor)
	print "   num BFS: %d"%(numCands)
	print "  kernel 1: %d seconds"%(numCands)
	
	print "\n   kernel 2 Times"
	printstats(times, "time", False)
	
	print "\n   kernel 2 Number of edges"
	printstats(travEdges, "nedge", False)
	
	print "\n   kernel 2 TEPS"
	printstats(TEPS, "TEPS", True)

###############################################
###########    CLEANUP

# These have to be explicitly deleted because they must release their MPI-backed data
# before finalize() calls MPI::Finalize(). Otherwise you get a crash.
del A
del parents
del fringe
del degrees



pcb.finalize()
