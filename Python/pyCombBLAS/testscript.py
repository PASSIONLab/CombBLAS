import sys
import time
import pyCombBLAS as pcb


# makes numbers pretty
def splitthousands(s, sep=','):
	s = str(int(s))
	if (len(s) <= 3): return s  
	return splitthousands(s[:-3], sep) + sep + s[-3:]



###############################################
###########    MATRIX CREATION

A = pcb.pySpParMat()
scale = 20

if len(sys.argv) == 2:
	scale = int(sys.argv[1])

if (scale < 0):
	path = "/home/alugowski/matrices/rmat_scale16.mtx";
	path = "../../CombBLAS/TESTDATA/SCALE16BTW-TRANSBOOL/input.txt";
	print "loading matrix from %s"%(path)
	A.load(path)
	A.Apply_SetTo(1)
else:
	if (pcb.root()):
		print "Generating RMAT with 2**%d nodes" %(scale)
	A.GenGraph500Edges(scale)
	A.Apply_SetTo(1)

n = A.getnrow()
m = A.getncol()
nnz = A.getnnz()
if (pcb.root()):
	print "A is %d by %d with %d nonzeros." % (n, m, nnz)


###############################################
###########    CANDIDATE SELECTION

numCands = 64
if (numCands > n):
	numCands = n

Cands = A.FindIndsOfColsWithSumGreaterThan(4);

numAvailableCands = Cands.length()
if (numAvailableCands < numCands):
	if (pcb.root()):
		print "Not enough vertices in the graph qualify as candidates. Only %d have enough degree."%(numAvailableCands)
	numCands = numAvailableCands

Cands.RandPerm();
First64 = pcb.pyDenseParVec.range(numCands, 0);
Cands = Cands.SubsRef(First64);

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

	times.append(telapsed);
	iterations.append(niter);
	
	#find the number of edges we traversed. Some were traversed multiple times, but the spec
	# says the number of input edges.
	parentsSP = parents.sparse(-1);
	if (parentsSP.getnnz() != r):
		pnnz = parentsSP.getnnz()
		if (pcb.root()):
			print "oh oh oh oh noooooooo! (parents.nnz) %d != %d (parentsSP.nnz)"%(r, pnnz)
	parentsSP.Apply_SetTo(1);
	s = A.SpMV_PlusTimes(parentsSP)
	nedges = s.Reduce_sum()

	# summarize this round	
	if (pcb.root()):
		print "Round %2d, root %-10d %15d parents (%5.2f%%),%3d iterations,%15d edges, %f s,   TEPS: %s" %((i+1), c, r, (100.0*r/n), niter, nedges, telapsed, splitthousands(nedges/telapsed))

###############################################
###########    CLEANUP

# These have to be explicitly deleted because they must release their MPI-backed data
# before finalize() calls MPI::Finalize(). Otherwise you get a crash.
del A
del parents
del fringe

pcb.finalize()
