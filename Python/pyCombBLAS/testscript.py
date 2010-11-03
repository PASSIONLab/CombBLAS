import pyCombBLAS as pcb

g = pcb.pySpParMat()
print "loading matrix from /home/alugowski/matrices/rmat_scale14.mtx";
g.load("/home/alugowski/matrices/rmat_scale14.mtx");
print(g.nedges())

del g

pcb.finalize()
