import pyCombBLAS as pcb

A = pcb.pySpParMat()
path = "/home/alugowski/matrices/small.mtx";
print "loading matrix from %s"%(path)
A.load(path)
print "A is %d by %d with %d nonzeros." % (A.getncol(), A.getnrow(),  A.getnnz())

n = A.getnrow()

numCands = 2
if (numCands > n):
	numCands = n

#	SpParVec<int64_t,int64_t> First64, CandSp;
#	Cands.RandPerm();
#	Cands.PrintInfo("Candidates array (permuted)");
#	CandSp = Cands.Find(totality<int64_t>());
#	First64.iota(64, 1);			// NV is also 1-based
#	CandSp = CandSp(First64);		// Because SpRef expects a 1-based parameter
#	CandSp.PrintInfo("First 64 of candidates (randomly chosen) array");

Cands = A.FindIndsOfColsWithSumGreaterThan(1);

numAvailableCands = Cands.length()
if (numAvailableCands < numCands):
	print "Not enough vertices in the graph qualify as candidates. Only %d have enough degree."%(numAvailableCands)
	numCands = numAvailableCands

Cands.RandPerm();
CandSp = Cands.Find_totality();
First64 = pcb.pySpParVec.range(numCands, 1);
CandSp = CandSp.SpRef(First64);

print "And the candidates are:"
CandSp.printall()


print "Starting vertices generated."

for i in range(0, numCands):
	c = CandSp.GetElement(i)
	print "\n\nOn start %d with root %d" %(i, c)

	parents = pcb.pyDenseParVec(n, -1)
#	levels = pcb.pyDenseParVec(n, 0)
#	level = 1
	fringe = pcb.pySpParVec(n) #SpParVec<int64_t, int64_t>	// numerical values are stored 1-based
	fringe.SetElement(c, c);
	parents.SetElement(c, c);
	
	print "initial fringe"
	fringe.printall()
	print ""
	
	while (fringe.getnnz() > 0):
		print "---------"
		fringe.setNumToInd()
		print "fringe at beginning of iteration"
		fringe.printall();
		
		fringe = A.SpMV_SelMax(fringe) #	// SpMV with sparse vector
		print "fringe after SpMV"
		fringe.printall()
		print ""
		
		fringe = pcb.EWiseMult(fringe, parents, True, -1)	#// clean-up vertices that already has parents 
		print "fringe after EWiseMult with parents"
		fringe.printall()
		print ""

		parents.ApplyMasked_SetTo(fringe, 0)
		parents += fringe
		print "parents at the end of iteration"
		parents.printall()
		print ""

	
	r = parents.Count_GreaterThan(0)
	print "We have %d parents" %(r)

del A
del parents
#del levels
del fringe

pcb.finalize()
