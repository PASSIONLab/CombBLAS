import kdt
import time
import math

def printGraphStats(G):
	kdt.p("")
	kdt.p("# verts: %d"%G.nvert())
	kdt.p("# edges: %d"%G.nedge())

def verifyMIS(G, MIS):
	ok = True
	
	def traverse(edge, sourceVert):
		return 1
	def add(x, y):
		return 1
		
	invMIS = G.e.SpMV(MIS, semiring=kdt.sr(add, traverse), inPlace=False)
	
	# sanity check
	if (invMIS.nnn() + MIS.nnn() != G.nvert()):
		kdt.p("size of MIS does not match: MIS size=%d, inverse size=%d, (sum=%d, should be number of verts=%d)"%(MIS.nnn(), invMIS.nnn(), (invMIS.nnn() + MIS.nnn()), G.nvert()))
		print G
		print MIS
		print invMIS
		print MIS.dense()
		print invMIS.dense()
		ok = False
	
	if ok:
		kdt.p("verification succeeded")

GRmat = kdt.DiGraph.generateRMAT(4, edgeFactor=5, initiator=[0.3, 0.1, 0.1, 0.5], delIsolated=False)

printGraphStats(GRmat)

start = time.time()
S = GRmat.MIS()
finish = time.time()
kdt.p("\nMIS time (RMAT):")
kdt.p(finish-start)

verifyMIS(GRmat, S)

