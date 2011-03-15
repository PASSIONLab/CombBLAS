#!

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


def k2validate(G, root, parents):

	ret = 1;	# assume valid
	nrowG = G.getnrow();

	# calculate level in the tree for each vertex; root is at level 0
	# about the same calculation as bfsTree, but tracks levels too
	parents2 = pcb.pyDenseParVec(nrowG, -1);
	fringe = pcb.pySpParVec(nrowG);
	parents2[root] = root;
	fringe[root] = root;
	levels = pcb.pyDenseParVec(nrowG, -1);
	levels[root] = 0;

	level = 1;
	while fringe.getnee() > 0:
		fringe.setNumToInd();
		G.SpMV_SelMax_inplace(fringe);
		pcb.EWiseMult_inplacefirst(fringe, parents2, True, -1);
		#fringe.printall();
		parents2.ApplyMasked(pcb.set(0), fringe);
		parents2.add(fringe);
		levels.ApplyMasked(pcb.set(level), fringe);
		level += 1;
	
	# spec test #1
	#	Not implemented
	

	# spec test #2
	#    tree edges should be between verts whose levels differ by 1
	
	#print "starting spec test#2"
	#  root = element of parents that points to itself
	##tmp1 = parents.copy()
	##tmp1 -= pcb.pyDenseParVec.range(nrowG,0)
	##root = tmp1.FindInds_NotEqual(0);
	#treeEdges = ((parents <> -1) & (parents <> root);
	tmp1 = parents.copy();
	tmp1[root] = -1;
	treeEdges = tmp1.FindInds(pcb.bind2nd(pcb.not_equal_to(), -1));
	#treeI = parents[treeEdges]
	treeI = parents.SubsRef(treeEdges);
	#treeJ = 1..nrowG[treeEdges]
	treeJ = pcb.pyDenseParVec.range(nrowG,0).SubsRef(treeEdges);
	#if any(levels[treeI]-levels[treeJ] <> -1):
	tmp1 = levels.SubsRef(treeI);
	tmp1 -= levels.SubsRef(treeJ);
	tmp2 = tmp1.FindInds(pcb.bind2nd(pcb.not_equal_to(), -1));
	if tmp2.getnee():
		print "spec test #2 failed."
		ret = -1;

	# spec test #3
	#	Not implemented

	# spec test #4
	#	Not implemented

	# spec test #5
	#	Not implemented

	

	del G, parents, parents2, fringe, levels, tmp1, tmp2, treeEdges, treeI, treeJ
	
	return ret

###############################################
###########    MATRIX CREATION

A = pcb.pySpParMatBool()
scale = 10

#degrees = pcb.pyDenseParVec(4, 0);
k1time = 0.0

if len(sys.argv) >= 2:
	scale = int(sys.argv[1])

if (scale < 0):
	if len(sys.argv) >= 3:
		path = sys.argv[2]
	else:
		print "Expecting a path to a matrix file as argument."
		sys.exit();

	print "loading matrix from %s"%(path)
	A.load(path)
	A.Apply(pcb.set(1))
	#print "converting to boolean" # already boolean
	#A = pcb.pySpParMatBool(A)
	n = A.getnrow()
	
	colreducer = pcb.pyDenseParVec(n, 1).sparse();
	degrees = A.SpMV_PlusTimes(colreducer).dense();

else:
	if (pcb.root()):
		print "Generating RMAT with 2**%d nodes" %(scale)
	k1time = A.GenGraph500Edges(scale)
	A.Apply(pcb.set(1))
	degrees = A.Reduce(pcb.pySpParMat.Column(), pcb.plus());
	if (pcb.root()):
		print "Generation took %lf s"%(k1time)
	
#A.save("/home/alugowski-ucsb/matrices/rmat%d.mtx"%(scale))
n = A.getnrow()
m = A.getncol()
nee = A.getnee()
nnz = A.getnnz()
edgefactor = nnz/n;
if (pcb.root()):
	print "A is %d by %d with %d elements (%d nonzeros)." % (n, m, nee, nnz)


###############################################
###########    CANDIDATE SELECTION
debugprint = False

numCands = 10
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


Cands = degrees.FindInds(pcb.bind2nd(pcb.greater(), 2))
Cands.RandPerm()

Firsts = pcb.pyDenseParVec.range(numCands, 0)

Cands = Cands[Firsts]
#Cands = Cands.SubsRef(Firsts)

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
	c = Cands[i]
	#try:
	#	print "c=%d"%(c)
	#except TypeError:
	#	print "------------------------------ TYPE ERROR on c! setting c=0"
	#	c = 0
	#	print "i=%d, len(Cands)=%d"%(i, len(Cands))
	#	Cands.printall()
	#	Cands.sparse().printall()
	
	# start the clock
	tstart = time.time()
	#------------------------- TIMED --------------------------------------------
	
	parents = pcb.pyDenseParVec(n, -1);
	fringe = pcb.pySpParVec(n)
	fringe[c] = c;
	parents[c] = c;
	
	if debugprint:
		if (pcb.root()):
			print "start fringe:" 
		fringe.printall()
	
	niter = 0
	while (fringe.getnee() > 0):
		if debugprint:
			print "----- on iteration %d"%(niter+1) 
		fringe.setNumToInd()
		if debugprint:
			print "fringe at start of iteration"
			fringe.printall();
		A.SpMV_SelMax_inplace(fringe) #
		if debugprint:
			print "fringe after SpMV"
			fringe.printall();
		pcb.EWiseMult_inplacefirst(fringe, parents, True, -1)	#// clean-up vertices that already have parents 
		if debugprint:
			print "fringe at end of iteration"
			fringe.printall();
		parents.ApplyMasked(pcb.set(0), fringe)
		parents.add(fringe)
		if debugprint:
			print "parents at end of iteration"
			parents.printall()
		niter += 1
	
	#------------------------- TIMED --------------------------------------------
	telapsed = time.time() - tstart;

	################## REPORTING:

	if debugprint:
		print "------------------------- resulting parents vector:"
		parents.printall()
	r = parents.Count(pcb.bind2nd(pcb.greater(), -1))

	#find the number of edges we traversed. Some were traversed multiple times, but the spec
	# says the number of input edges.
	parentsSP = parents.sparse(-1);
	if (parentsSP.getnee() != r):
		pnnz = parentsSP.getnee()
		if (pcb.root()):
			print "oh oh oh oh noooooooo! (parents.nnz) %d != %d (parentsSP.nee)"%(r, pnnz)
			print "parents: "
			parents.printall()
			print "parentsSP: "
			parentsSP.printall()
	parentsSP.Apply(pcb.set(1));
	nedges = pcb.EWiseMult(parentsSP, degrees, False, 0).Reduce(pcb.plus())
	
	#s = A.SpMV_PlusTimes(parentsSP)
	#nedges = s.Reduce(pcb.plus())

	k2Fail = False;
	if False:
		print "Not validating BFS tree"
	elif k2validate(A, c, parents) < 0:
		k2Fail = True;
		print "k2validate found errors in BFS tree"

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

	if k2Fail:
		print "***ERROR:  At least one tree failed kernel 2 validation"

###############################################
###########    CLEANUP

# These have to be explicitly deleted because they must release their MPI-backed data
# before finalize() calls MPI::Finalize(). Otherwise you get a crash.
del A
del parents, parentsSP
del fringe, Cands 
del degrees

pcb.finalize()
