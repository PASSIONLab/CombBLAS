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
	
	A = self.e.copy()
	#if not sym:
		#A = A + A.Transpose() at the points where A is 0 or null
	
	#Add self loops
	N = self.nvert()
	if addSelfLoops:
		A += Mat.eye(N, element=selfLoopWeight)
	
	#Create stochastic matrix

	# get inverted sums, but avoid divide by 0
	invSums = A.sum(Mat.Column)
	def inv(x):
		if x == 0:
			return 1
		else:
			return 1/x
	invSums.apply(inv)
	A.scale( invSums , dir=Mat.Column)
	
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
				AA.apply(pcb.set(1))
				AA = AA.SpGEMM(AA, semiring=sr_plustimes)
				nedges += AA.sum(Mat.Column).reduce(op_add)
			A = A.SpGEMM(A, semiring=sr_plustimes)
	
		#Inflation - Hadamard power - greater inflation parameter -> more granular results
		A.apply((lambda x: x**inflation))
		
		#Re-normalize
		invSums = A.sum(Mat.Column)
		invSums.apply(inv)
		A.scale( invSums , dir=Mat.Column)
	
		#Looping Condition:
		colssqs = A.reduce(Mat.Column, op_add, (lambda x: x*x))
		colmaxs = A.reduce(Mat.Column, op_max, init=0.0)
		chaos = (colmaxs - colssqs).max()
		#print "chaos=",chaos

		# Pruning implementation - switch out with TopK / give option
		A._prune((lambda x: x < prunelimit))
		#print "number of edges remaining =", A._spm.getnee()
	
	#print "Iterations = %d" % iterNum
	
	if retNEdges:
		return A,nedges

	return A
DiGraph._cluster_markov = _cluster_markov