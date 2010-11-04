import pyCombBLAS as pcb

A = pcb.pySpParMat()
print "loading matrix from /home/alugowski/matrices/rmat_scale14.mtx";
A.load("/home/alugowski/matrices/rmat_scale14.mtx");
print "A is %d by %d with %d nonzeros." % (A.getncol(), A.getnrow(),  A.getnnz())

parents = pcb.pyDenseParVec(A.getnrow(), 0) # A.getcommgrid(), (int64_t) 0);	// identity is 0 
levels = pcb.pyDenseParVec(A.getnrow(), 0)
level = 1
fringe = pcb.pySpParVec() #SpParVec<int64_t, int64_t>	// numerical values are stored 1-based
print "set element"
fringe.SetElement(5, 5);
print "starting loop"
while (fringe.getnnz() > 0):
	print "running SpMV"
	fringe = A.SpMV_SelMax(fringe) #	// SpMV with sparse vector
	print "running EWiseMult"
	fringe = pcb.EWiseMult(fringe, parents, True, 0)	#// clean-up vertices that already has parents 
	print "running add"
	parents.add(fringe)


del A
del parents
del levels
del fringe

pcb.finalize()
