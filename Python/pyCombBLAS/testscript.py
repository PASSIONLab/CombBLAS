import pyCombBLAS as pcb

A = pcb.pySpParMat()
print "loading matrix from /home/alugowski/matrices/rmat_scale14.mtx";
A.load("/home/alugowski/matrices/rmat_scale14.mtx");
print(A.nedges())

parents = pcb.pyDenseParVec() # A.getcommgrid(), (int64_t) 0);	// identity is 0 
levels = pcb.pyDenseParVec()
level = 1
fringe = pcb.pySpParVec() #SpParVec<int64_t, int64_t>	// numerical values are stored 1-based
print "set element"
fringe.SetElement(5, 5);
print "starting loop"
while (fringe.getnnz() > 0):
	fringe = A.SpMV(fringe) #	// SpMV with sparse vector
	fringe = EWiseMult(fringe, parents, true);	#// clean-up vertices that already has parents 
	parents.add(fringe);


del A

pcb.finalize()
