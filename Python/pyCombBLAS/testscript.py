import sys
import pyCombBLAS as pcb

A = pcb.pySpParMat()
scale = 16

if (scale < 0):
	path = "/home/alugowski/matrices/rmat_scale16.mtx";
	print "loading matrix from %s"%(path)
	A.load(path)
	A.Apply_SetTo(1)
else:
	A.GenGraph500Edges(scale)

print "A is %d by %d with %d nonzeros." % (A.getncol(), A.getnrow(),  A.getnnz())

n = A.getnrow()

numCands = 64
if (numCands > n):
	numCands = n

Cands = A.FindIndsOfColsWithSumGreaterThan(2);

numAvailableCands = Cands.length()
if (numAvailableCands < numCands):
	print "Not enough vertices in the graph qualify as candidates. Only %d have enough degree."%(numAvailableCands)
	numCands = numAvailableCands

Cands.RandPerm();
First64 = pcb.pyDenseParVec.range(numCands, 0);
Cands = Cands.SubsRef(First64);

if (pcb.root()):
	print "The candidates are:"
Cands.printall()


print "Starting vertices generated."

#numCands = 1;
#CandSp.SetElement(0, 7281);

for i in range(0, numCands):
	c = Cands.GetElement(i)
	if (pcb.root()):
		print "\nOn round %d with root %d..." %(i, c)

#	levels = pcb.pyDenseParVec(n, 0)
#	level = 1

	parents = pcb.pyDenseParVec(n, -1, -1)
	fringe = pcb.pySpParVec(n)
	fringe.SetElement(c, c);
	parents.SetElement(c, c);
	
	while (fringe.getnnz() > 0):
		fringe.setNumToInd()
		#print "fringe at start of iteration"
		#fringe.printall();
		fringe = A.SpMV_SelMax(fringe) #
		#print "fringe after SpMV"
		#fringe.printall();
		fringe = pcb.EWiseMult(fringe, parents, True, -1)	#// clean-up vertices that already have parents 
		#print "fringe at end of iteration"
		#fringe.printall();
		#parents.ApplyMasked_SetTo(fringe, 0)
		parents += fringe
	
	#parents.printall()
	r = parents.Count_GreaterThan(-1)
	if (pcb.root()):
		print "We have %d parents" %(r)

del A
del parents
#del levels
del fringe

pcb.finalize()
