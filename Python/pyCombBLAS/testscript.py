import pyCombBLAS as pcb

g = pcb.DiGraph()
g.load("/home/alugowski/matrices/rmat_scale14.mtx");
print(g.nedges())

del g

print "1"

pcb.finalize()
