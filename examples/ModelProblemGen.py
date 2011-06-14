from kdt import *
import kdt.pyCombBLAS as pcb

def getModelProbem(k):
	"""
	Create a k-by-k grid and a matching length k^2 vector b. b is formed to represent a 
	room with 0 on all sides except one side of length k with value 1.
	"""
	k = int(k)
	k2 = k*k
	
	# add the self loops
	A = DiGraph(ParVec.range(k2), ParVec.range(k2), ParVec(k2, 4), k2, k2)
	
	# add the left and right edges (same row, -1 and +1 column, -1 and +1 index)
	A += DiGraph(ParVec.range(k2-1), ParVec.range(1, k2), ParVec(k2-1, -1), k2)
	A += DiGraph(ParVec.range(1, k2), ParVec.range(k2-1), ParVec(k2-1, -1), k2)
	
	# add the up and down edges (same column, -1 and +1 column, -k and +k index)
	A += DiGraph(ParVec.range(k2-k), ParVec.range(k, k2), ParVec(k2-k, -1), k2)
	A += DiGraph(ParVec.range(k, k2), ParVec.range(k2-k), ParVec(k2-k, -1), k2)
	
	# create the right-hand side
	b = ParVec.range(k2)
	b._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.less(), k), pcb.set(1.0), pcb.set(0.0)))
	
	return A, b
