import pyCombBLAS as pcb

A = pcb.pySpParMat()
path = "/home/alugowski/matrices/rmat_scale16.mtx";
print "loading matrix from %s"%(path)
A.load(path)
#A.Apply_SetTo(1)
print "A is %d by %d with %d nonzeros." % (A.getncol(), A.getnrow(),  A.getnnz())

n = A.getnrow()

c = 7281

parents = pcb.pyDenseParVec(n, -1)
fringe = pcb.pySpParVec(n)
fringe.SetElement(c, c);
parents.SetElement(c, c);

if (pcb.root()):
	print "start fringe:" 
fringe.printall()

while (fringe.getnnz() > 0):
	fringe.setNumToInd()
	print "\nfringe at start of iteration"
	fringe.printall();
	fringe = A.SpMV_SelMax(fringe) #
	print "\nfringe after SpMV. This should not be empty---------"
	fringe.printall();
	fringe = pcb.EWiseMult(fringe, parents, True, -1)	#// clean-up vertices that already have parents 
	print "\nfringe at end of iteration"
	fringe.printall();
	parents.ApplyMasked_SetTo(fringe, 0)
	parents += fringe

#parents.printall()
r = parents.Count_GreaterThan(-1)
print "We have %d parents" %(r)

del A
del parents
del fringe

pcb.finalize()
