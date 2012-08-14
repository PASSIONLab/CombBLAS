import kdt
import time
import math

def printGraphStats(G):
	kdt.p("")
	kdt.p("# verts: %d"%G.nvert())
	kdt.p("# edges: %d"%G.nedge())

def verifyMIS(G, MIS):
	ok = True
	
	#def traverse(edge, sourceVert):
	#	return 1
	#def add(x, y):
	#	return 1
		
	invMIS = G.e.SpMV(MIS, semiring=kdt.sr_select2nd, inPlace=False)
		
	# sanity check
	if (invMIS.nnn() + MIS.nnn() != G.nvert()):
		kdt.p("size of MIS does not match: MIS size=%d, inverse size=%d, (sum=%d, should be number of verts=%d)"%(MIS.nnn(), invMIS.nnn(), (invMIS.nnn() + MIS.nnn()), G.nvert()))
		#print G
		#print MIS
		#print invMIS
		#print MIS.dense()
		#print invMIS.dense()
		ok = False
	
	# make sure there is no overlap
	overlap_set = invMIS.eWiseApply(MIS, op=(lambda x,y: 1), allowANulls=False, allowBNulls=False, allowIntersect=True, inPlace=False)
	if (overlap_set.nnn() != 0):
		kdt.p("MIS and invMIS overlap in %d vertices!"%(overlap_set.nnn()))
		ok=False
	
	if ok:
		kdt.p("verification succeeded")
		return ""
	else:
		return "VERIFICATION FAILED"

scale = 12

## RMAT
GRmat = kdt.DiGraph.generateRMAT(scale, edgeFactor=5, initiator=[0.3, 0.1, 0.1, 0.5], delIsolated=False)

printGraphStats(GRmat)

start = time.time()
S = GRmat.MIS()
finish = time.time()

verifyString = verifyMIS(GRmat, S)
kdt.p("MIS time (sec) RMAT scale\t%d\t(%d verts, %d edges) on \t%d\t procs:\t%f\t%s"%(scale, GRmat.nvert(), GRmat.nedge(), kdt._nproc(), (finish-start), verifyString))


## torus

torusDim = int(math.sqrt(2**scale)) # match scale of RMAT
Gtorus = kdt.DiGraph.generate2DTorus(torusDim)
printGraphStats(Gtorus)

start = time.time()
S = Gtorus.MIS()
finish = time.time()
verifyString = verifyMIS(Gtorus, S)
kdt.p("MIS time (sec) Torus scale\t%d\t(Torus dim %d, %d verts, %d edges) on \t%d\t procs:\t%f\t%s"%(scale, torusDim, Gtorus.nvert(), Gtorus.nedge(), kdt._nproc(), (finish-start), verifyString))

