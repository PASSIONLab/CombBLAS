import pyCombBLAS as pcb

d = pcb.pyDenseParVec(16, 0)
d.SetElement(1, 1)
d.SetElement(5, 5)
d.SetElement(9, 9)
d.SetElement(13, 13)

d.printall()

print "get 1=%d, get 9=%d" %(d.GetElement(1), d.GetElement(9))

if (pcb.root()):
	print "negating..."

d.negate()
d.printall()

nnz = d.getnnz()
if (pcb.root()):
	print "nonzeros=%d" %(nnz)

del d
pcb.finalize()
