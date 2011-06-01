import kdt
B = kdt.DiGraph()
B.genGraph500Edges(10)
B._spm.Apply(kdt.pyCombBLAS.set(1))
bedges = B._spm.getnee()
C = B._markov(addSelfLoops=True, inflation=2, prunelimit=0.0000001)



cedges = C._spm.getnee()
if kdt.master():
	print "Started with %d edges and finished with %d."%(bedges, cedges)