from DiGraph import DiGraph
from Vec import Vec
from Mat import Mat
from Util import *

#TODO this import should not be necessary
import kdt.pyCombBLAS as pcb

# NEEDED: update to transposed edge matrix
# NEEDED: update to new fields
# NEEDED: tests
def _cluster_markov(self, expansion=2, inflation=2, addSelfLoops=False, selfLoopWeight=1, prunelimit=0.00001, sym=False, retNEdges=False):
	"""
	Performs Markov Clustering (MCL) on self and returns a graph representing the clusters.
	"""
	
	#self is a DiGraph
	
	EPS = 0.001
	#EPS = 10**(-100)
	chaos = 1000
	
	#Check parameters
	if expansion <= 1:
		raise KeyError, 'expansion parameter must be greater than 1'
	if inflation <= 1:
		raise KeyError, 'inflation parameter must be greater than 1'
	
	A = self.copy()
	#if not sym:
		#A = A + A.Transpose() at the points where A is 0 or null
	
	#Add self loops
	N = A.nvert()
	if addSelfLoops:
		A.addSelfLoops(selfLoopWeight)
	
	#Create stochastic matrix

	#Avoid divide-by-zero error
	sums = A.sum(DiGraph.In)
	sums._apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0),
		pcb.set(1),
		pcb.identity()))
	
	A.scale( ParVec.ones(A.nvert()) / sums , dir=DiGraph.In )
	
	if retNEdges:
		nedges = 0
	
	#Iterations tally
	iterNum = 0
	
	#MCL Loop
	while chaos > EPS and iterNum < 300:
		iterNum += 1;
	
		#Expansion - A^(expansion)
		if retNEdges:
			AA = A.copy()
		for i in range(1, expansion):
			if retNEdges:
				AA._apply(pcb.set(1))
				AA = AA._SpGEMM(AA)
				nedges += AA.sum(DiGraph.In)._dpv.Reduce(pcb.plus())
			A = A._SpGEMM(A)
	
		#Inflation - Hadamard power - greater inflation parameter -> more granular results
		A._apply(pcb.bind2nd(pcb.pow(), inflation))
		
		#Re-normalize
		sums = A.sum(DiGraph.In)
		sums._apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0),
			pcb.set(1),
			pcb.identity()))

		A.scale( ParVec.ones(A.nvert()) / sums, dir=DiGraph.In)
	
		#Looping Condition:
		colssqs = A._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.bind2nd(pcb.pow(), 2))
		colmaxs = A._spm.Reduce(pcb.pySpParMat.Column(), pcb.max(), 0.0)
		chaos = ParVec.toParVec(colmaxs - colssqs).max()
		#print "chaos=",chaos

		# Pruning implementation - switch out with TopK / give option
		A._spm.Prune(pcb.bind2nd(pcb.less(), prunelimit))
		#print "number of edges remaining =", A._spm.getnee()
	
	#print "Iterations = %d" % iterNum
	
	if retNEdges:
		return A,nedges

	return A
DiGraph._cluster_markov = _cluster_markov