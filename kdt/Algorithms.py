import math
import time
from DiGraph import DiGraph
from Vec import Vec
from Mat import Mat

from Util import *

#TODO this import should not be necessary
import kdt.pyCombBLAS as pcb

# ==================================================================
#  "complex ops" implemented below here
# ==================================================================


#	creates a breadth-first search tree of a Graph from a starting
#	set of vertices.  Returns a 1D array with the parent vertex of 
#	each vertex in the tree; unreached vertices have parent == -1.
#	sym arg denotes whether graph is symmetric; if not, need to transpose
#
# NEEDED: update to new EWiseApply
# NEEDED: tests
def bfsTree(self, root, useOldFunc=False):
	"""
	calculates a breadth-first search tree from the edges in the
	passed DiGraph, starting from the root vertex.  "Breadth-first"
	in the sense that all vertices reachable in step i are added
	to the tree before any of the newly-reachable vertices' reachable
	vertices are explored.

	Input Arguments:
		root:  an integer denoting the root vertex for the tree
		sym:  a Boolean denoting whether the DiGraph is symmetric
			(i.e., each edge from vertex i to vertex j has a
			companion edge from j to i).  If the DiGraph is 
			symmetric, the operation is faster.  The default is 
			False.

	Input Arguments:
		parents:  a Vec instance of length equal to the number
			of vertices in the DiGraph, with each element denoting 
			the vertex number of that vertex's parent in the tree.
			The root is its own parent.  Unreachable vertices
			have a parent of -1. 

	SEE ALSO: isBfsTree 
	"""
	#mulFn = lambda x,y: y
	#addFn = lambda x,y: y
	#sR = sr(addFn, mulFn)
	sR = sr_select2nd

	parents = Vec(self.nvert(), -1, sparse=False)
	frontier = Vec(self.nvert(), sparse=True)
	parents[root] = root
	frontier[root] = root
	while frontier.nnn() > 0:
		frontier.spRange()
		self.e.SpMV(frontier, semiring=sR, inPlace=True)
		
		if useOldFunc:
			# this method uses an old CombBLAS routine.
			# it will be deprecated when acceptable performance from SEJITS is attained.
			pcb.EWiseMult_inplacefirst(frontier._v_, parents._v_, True, -1)
			parents[frontier] = frontier
		else:
			# this is the preferred method. It is a bit slower than the above due to unoptimized
			# Python callbacks in this version of KDT, but future SEJITS integration should remove
			# that penalty and the above method will be deprecated.
			
			# remove already discovered vertices from the frontier.
			frontier.eWiseApply(parents, op=(lambda f,p: f), doOp=(lambda f,p: p == -1), inPlace=True)
			# update the parents
			parents[frontier] = frontier

	return parents
DiGraph.bfsTree = bfsTree

	# returns tuples with elements
	# 0:  True/False of whether it is a BFS tree or not
	# 1:  levels of each vertex in the tree (root is 0, -1 if not reached)
# NEEDED: update to transposed edge matrix
# NEEDED: update to new fields
# NEEDED: tests
def isBfsTree(self, root, parents, sym=False):
	"""
	validates that a breadth-first search tree in the style created
	by bfsTree is correct.

	Input Arguments:
		root:  an integer denoting the root vertex for the tree
		parents:  a ParVec instance of length equal to the number
			vertices in the DiGraph, with each element denoting 
			the vertex number of that vertex's parent in the tree.
			The root is its own parent.  Vertices unreachable
			from the root have a parent of -1. 
		sym:  a scalar Boolean denoting whether the DiGraph is 
			symmetric (i.e., each edge from vertex i to vertex j
			has a companion edge from j to i).  If the DiGraph 
			is symmetric, the operation is faster.  The default 
			is False.
	
	Output Arguments:
		ret:  a 2-element tuple.  The first element is an integer,
			whose value is 1 if the graph is a BFS tree and whose
			value is the negative of the first test below that failed,
			if one of them failed.  If the graph is a BFS tree,
			the second element of the tuple is a ParVec of length 
			equal to the number of vertices in the DiGraph, with 
			each element denoting the level in the tree at which 
			the vertex resides.  The root resides in level 0, its
			direct neighbors in level 1, and so forth.  Unreachable vertices have a
			level value of -1.  If the graph is not a BFS tree (one
			of the tests failed), the second element of the
			tuple is None.
	
	Tests:
		The tests implement some of the Graph500 (www.graph500.org) 
		specification. (Some of the Graph500 tests also depend on 
		input edges.)
		1:  The tree does not contain cycles,  that every vertex
			with a parent is in the tree, and that the root is
			not the destination of any tree edge.
		2:  Tree edges are between vertices whose levels differ 
			by exactly 1.

	SEE ALSO: bfsTree 
	"""
	raise NotImplementedError,"isBfsTree not updated to working on transposed matrices yet."
	
	ret = 1		# assume valid
	nvertG = self.nvert()


	# spec test #1
	# Confirm that the tree is a tree;  i.e., that it does not
	# have any cycles (visited more than once while building
	# the tree) and that every vertex with a parent is
	# in the tree. 

	# build a new graph from just tree edges
	tmp2 = parents != ParVec.range(nvertG)
	treeEdges = (parents != -1) & tmp2
	treeI = parents[treeEdges.findInds()]
	treeJ = ParVec.range(nvertG)[treeEdges.findInds()]
	del tmp2, treeEdges
	# root cannot be destination of any tree edge
	if (treeJ == root).any():
		return (-1, None)
	# note treeJ/TreeI reversed, so builtGT is transpose, as
	#   needed by SpMV
	builtGT = DiGraph(treeJ, treeI, 1, nvertG)
	visited = ParVec.zeros(nvertG)
	visited[root] = 1
	frontier = SpParVec(nvertG)
	frontier[root] = root
	cycle = False
	multiparents = False
	while frontier.nnn() > 0 and not cycle and not multiparents:
		frontier.spOnes()
		newfrontier = SpParVec.toSpParVec(builtGT._spm.SpMV_PlusTimes(frontier._spv))
		if visited[newfrontier.toParVec().findInds()].any():
			cycle = True
			break
		if (newfrontier > 1).any():
			multiparents = True
			break
		frontier = newfrontier
		visited[frontier] = 1
	if cycle or multiparents:
		return (-1, None)
	del visited, builtGT
	
	# spec test #2
	#    tree edges should be between verts whose levels differ by 1
	
	# calculate level in the tree for each vertex; root is at level 0
	# about the same calculation as bfsTree, but tracks levels too
	if not sym:
		self.reverseEdges()
	parents2 = ParVec.zeros(nvertG) - 1
	parents2[root] = root
	frontier = SpParVec(nvertG)
	frontier[root] = root
	levels = ParVec.zeros(nvertG) - 1
	levels[root] = 0

	level = 1
	while frontier.nnn() > 0:
		frontier.spRange()
		#ToDo:  create PCB graph-level op
		self._spm.SpMV_SelMax_inplace(frontier._spv)
		#ToDo:  create PCB graph-level op
		pcb.EWiseMult_inplacefirst(frontier._spv, parents2._dpv, True, -1)
		parents2[frontier] = frontier
		levels[frontier] = level
		level += 1
	if not sym:
		self.reverseEdges()
	del parents2
	if (levels[treeI]-levels[treeJ] != -1).any():
		return (-2, None)

	return (ret, levels)
DiGraph.isBfsTree = isBfsTree

# returns a Boolean vector of which vertices are neighbors
def neighbors(self, source, nhop=1):
	"""
	calculates, for the given DiGraph instance and starting vertices,
	the vertices that are neighbors of the starting vertices (i.e.,
	reachable within nhop hops in the graph).

	Input Arguments:
		self:  a DiGraph instance
		source:  a Boolean Vec with True (1) in the positions
			of the starting vertices.  
		nhop:  a scalar integer denoting the number of hops to 
			use in the calculation. The default is 1.
		sym:  a scalar Boolean denoting whether the DiGraph is 
			symmetric (i.e., each edge from vertex i to vertex j
			has a companion edge from j to i).  If the DiGraph 
			is symmetric, the operation is faster.  The default 
			is False.

	Output Arguments:
		ret:  a ParVec of length equal to the number of vertices
			in the DiGraph, with a True (1) in each position for
			which the corresponding vertex is a neighbor.

			Note:  vertices from the start vector may appear in
			the return value.

	SEE ALSO:  pathsHop
	"""

	dest = Vec(self.nvert(), element=0.0, sparse=False)
	frontier = Vec(self.nvert(), sparse=True)
	frontier[source] = 1
	for i in range(nhop):
		frontier.spRange()
		self.e.SpMV(frontier, semiring=sr_select2nd, inPlace=True)
		# remove already discovered vertices from the frontier.
		frontier.eWiseApply(dest, op=(lambda f,p: f), doOp=(lambda f,p: p == 0), inPlace=True)
		dest[frontier] = 1

	return dest

DiGraph.neighbors = neighbors

# returns:
#   - source:  a vector of the source vertex for each new vertex
#   - dest:  a Boolean vector of the new vertices
#ToDo:  nhop argument?
def pathsHop(self, source, sym=False):
	"""
	calculates, for the given DiGraph instance and starting vertices,
	which can be viewed as the frontier of a set of paths, the vertices
	that are reachable by traversing one graph edge from one of the 
	starting vertices.  The paths are kept distinct, as only one path
	will extend to a given vertex.

	Input Arguments:
		self:  a DiGraph instance
		source:  a Boolean Vec of length nverts with True (1) in the positions
			of the starting vertices.  

	Output Arguments:
		ret:  a dense Vec of length equal to the number of vertices
			in the DiGraph.  The value of each element of the Vec 
			with a value other than -1 denotes the starting vertex
			whose path extended to the corresponding vertex.  In
			the case of multiple paths potentially extending to
			a single vertex, the chosen source is arbitrary. 

	SEE ALSO:  neighbors
	"""

	ret = Vec(self.nvert(), element=-1, sparse=False)
	frontier = source.find()
	frontier.spRange()
	self.e.SpMV(frontier, semiring=sr_select2nd, inPlace=True)

	ret[frontier] = frontier

	return ret
DiGraph.pathsHop = pathsHop

def normalizeEdgeWeights(self, dir=DiGraph.Out):
	"""
	Normalize the outward edge weights of each vertex such
	that for Vertex v, each outward edge weight is
	1/outdegree(v).
	"""
	if self.isObj():
		raise NotImplementedError, "Cannot normalize object weights yet."

	degscale = self.e.reduce(dir, (lambda x,y: x+y), init=0)
	degscale.apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0), pcb.identity(), pcb.bind1st(pcb.divides(), 1)))			
	self.e.scale(degscale, op=op_mul, dir=dir)
DiGraph.normalizeEdgeWeights = normalizeEdgeWeights

def pageRank(self, epsilon = 0.1, dampingFactor = 0.85, iterations=1000000):
	"""
	Compute the PageRank of vertices in the graph.

	The PageRank algorithm computes the percentage of time
	that a "random surfer" spends at each vertex in the
	graph. If the random surfer is at Vertex v, she will
	take one of two actions:
		1) She will hop to another vertex to which Vertex
				   v has an outward edge. Self loops are ignored.
		2) She will become "bored" and randomly hop to any
				   vertex in the graph. This action is taken with
				   probability (1 - dampingFactor).

	When the surfer is visiting a vertex that is a sink
	(i.e., has no outward edges), she hops to any vertex
	in the graph with probability one.

	Optional argument epsilon controls the stopping
	condition. Iteration stops when the 1-norm of the
	difference in two successive result vectors is less
	than epsilon.

	Optional parameter dampingFactor alters the results
	and speed of convergence, and in the model described
	above dampingFactor is the percentage of time that the
	random surfer hops to an adjacent vertex (rather than
	hopping to a random vertex in the graph).

	See "The PageRank Citation Ranking: Bringing Order to
	the Web" by Page, Brin, Motwani, and Winograd, 1998
	(http://ilpubs.stanford.edu:8090/422/) for more
	information.
	"""

	# We don't want to modify the user's graph.
	G = self.copy(element=1.0)
	
	nvert = G.nvert()

	# Remove self loops.
	G.removeSelfLoops()
	
	# Handle sink nodes (nodes with no outgoing edges) by
	# connecting them to all other nodes.
	sinkV = G.degree(DiGraph.Out)
	sinkV.apply(pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(0), pcb.set(1./nvert)))
	
	# Normalize edge weights such that for each vertex,
	# each outgoing edge weight is equal to 1/(number of
	# outgoing edges).
	G.normalizeEdgeWeights(DiGraph.Out)

	# PageRank loop.
	delta = 1
	dv1 = Vec(nvert, 1./nvert, sparse=False)
	v1 = dv1.sparse()
	prevV = Vec(nvert, sparse=True)
	onesVec = Vec.ones(nvert, sparse=True)
	dampingVec = onesVec * ((1 - dampingFactor)/nvert)
	while delta > epsilon and iterations > 0:
		prevV = v1.copy()
		v2 = G.e.SpMV(v1, semiring=sr_plustimes)

		# Compute the inner product of sinkV and v1.
		sinkContribV = sinkV.eWiseApply(v1, op_mul, inPlace=False)
		sinkContrib = sinkContribV.reduce(op_add)
		
		v1 = v2 + (onesVec*sinkContrib)
		v1 = v1*dampingFactor + dampingVec
		delta = (v1 - prevV).reduce(op_add, op_abs)
		iterations -= 1
	return v1.dense()
DiGraph.pageRank = pageRank
	
def centrality(self, alg, **kwargs):
	"""
	calculates the centrality of each vertex in the DiGraph instance,
	where 'alg' can be one of 
		'exactBC':  exact betweenness centrality
		'approxBC':  approximate betweenness centrality

	Each algorithm may have algorithm-specific arguments as follows:
		'exactBC':  
			normalize=True:  normalizes the values by dividing by 
					(nVert-1)*(nVert-2)
		'approxBC':
		sample=0.05:  the fraction of the vertices to use as sources 
			and destinations;  sample=1.0 is the same as exactBC
			normalize=True:  normalizes the values by multiplying by 
			nVerts / (nVertsCalculated * (nVerts-1) * (nVerts-2))
	The return value is a ParVec with length equal to the number of
	vertices in the DiGraph, with each element of the ParVec containing
	the centrality value of the vertex.
	"""
	# TODO: make this look up the function named '_centrality_($arg)' 
	# instead of hard coding the name in.
	if alg=='exactBC':
		#cent = DiGraph._centrality_approxBC(self, sample=1.0, **kwargs)
		cent = DiGraph._centrality_exactBC(self, **kwargs)
	elif alg=='approxBC':
		cent = DiGraph._centrality_approxBC(self, **kwargs)
	elif alg=='kBC':
		raise NotImplementedError, "k-betweenness centrality unimplemented"
	elif alg=='degree':
		raise NotImplementedError, "degree centrality unimplemented"
	else:
		raise KeyError, "unknown centrality algorithm (%s)" % alg

	return cent
DiGraph.centrality = centrality

def _centrality_approxBC(self, sample=0.05, normalize=True, nProcs=pcb._nprocs(), memFract=0.1, BCdebug=0, batchSize=-1, retNVerts=False):
	"""
	calculates the approximate or exact (with sample=1.0) betweenness
	centrality of the input DiGraph instance.  _approxBC is an internal
	method of the user-visible centrality method, and as such is
	subject to change without notice.  Currently the following expert
	argument is supported:
		- memFract:  the fraction of node memory that will be considered
		available for a single strip in the strip-mining
		algorithm.  Fractions that lead to paging will likely
		deliver atrocious performance.  The default is 0.1.
	
	This function uses Brandes' algorithm.
	"""
	if True:
		A = self.e.copy(element=1.0)
		BC_SR = sr_plustimes
	else:
		# no copy, but slower because of Python semiring
		A = self.e
		A.spOnes()
		def sel(x, y):
			return y
		def plus(x, y):
			return x+y
		BC_SR = sr(plus, sel)
	
	#A.transpose() # Adam: why?
	#A.spOnes()
	N = self.nvert()

	# initialize final BC value vector
	bc = Vec(N, sparse=False, element=0)

	# create the batches
	nVertToCalc = int(math.ceil(N * sample))
	nVertToCalc = min(nVertToCalc, N)
	
	# batchSize = #rows/cols that will fit in memory simultaneously.
	# bcu has a value in every element, even though it's literally
	# a sparse matrix (DiGraph).  So batchsize is calculated as
	#   nrow = memory size / (memory/row)
	#   memory size (in edges)
	#        = 2GB * 0.1 (other vars) / 18 (bytes/edge) * nProcs
	#   memory/row (in edges)
	#        = N
	physMemPCore = 2e9; bytesPEdge = 18
	if (batchSize < 0):
		batchSize = int(2e9 * memFract / bytesPEdge * nProcs / N)
	batchSize = min(nVertToCalc, batchSize)
	
	nBatches = int(math.ceil(float(nVertToCalc) / float(batchSize)))
	nPossBatches = int(math.ceil(float(N) / float(batchSize)))
	
	# sources for the batches
	# the i-th batch is defined as randVerts[ startVs[i] to (startVs[i]+numV[i]) ]
	randVerts = Vec.range(N)
	randVerts.randPerm()
	
	if (batchSize >= nVertToCalc):
		startVs = [0]
		endVs = [nVertToCalc]
		numVs = [nVertToCalc]
	else: #sample == 1.0:
		startVs = range(0,nVertToCalc,batchSize)
		endVs = range(batchSize, nVertToCalc, batchSize)
		endVs.append(nVertToCalc)
		numVs = [y-x for [x,y] in zip(startVs,endVs)]

	if BCdebug>0:
		p("batchSz=%d, nBatches=%d, nPossBatches=%d" % (batchSize, nBatches, nPossBatches))
	if BCdebug>1:
		p("summary of batches:")
		p(("startVs:",startVs))
		p(("  numVs:", numVs))
	
	# main loop.
	# iterate over each batch and update the BC values as we go along
	for [startV, numV] in zip(startVs, numVs):
		
		# get the batch
		startV = int(startV); numV = int(numV)
		if BCdebug>0:
			p("startV=%d, numV=%d" % (startV, numV))
		bfs = []		
		batchRange = Vec.range(startV, startV+numV)
		if BCdebug>0:
			p(("batchrange",batchRange))
		batch = randVerts[batchRange]
		if BCdebug>1:
			p(("batch=",batch))
		curSize = len(batch)

		nsp = Mat(batch, Vec.range(curSize), 1, curSize, N)
		#next:  frontier should be Vs; indexing must be impl to support that; seems should be a collxn of spVs, hence a SpParMat
		frontier = A[Vec.range(N), batch]

		# main batched SSSP solve (using a BFS. Same algorithm as in bfsTree(), except instead of a single starting vertex we have numV of them)
		depth = 0
		while frontier.nnn() > 0:
			if BCdebug>1:
				before = time.time()
	
			depth = depth+1
			if BCdebug>1 and depth>1:
				nspne = tmp.nnn(); tmpne = tmp.nnn(); frontierne = frontier.nnn()
				p("BC: in while: depth=%d, nsp.nedge()=%d, tmp.nedge()=%d, frontier.nedge()=%d" % (depth, nspne, tmpne, frontierne))
			nsp += frontier
			tmp = frontier.copy()
			tmp.spOnes()
			bfs.append(tmp) # save each BFS frontier
			frontier = A.SpGEMM(frontier, semiring=BC_SR)

			# prune new-frontier to new verts only
			#frontier = tmp._mulNot(nsp)
			frontier.eWiseApply(nsp, op=(lambda f,n: f), allowBNulls=True, allowIntersect=False, inPlace=True)
			
			if BCdebug>1:
				p("    %f seconds" % (time.time()-before))

		# compute the BC update for all vertices except the sources
		bcu = Mat.ones(curSize,N)
		for depth in range(depth-1,0,-1):
			# compute the weights to be applied based on the child values
			w = bfs[depth] / nsp 
			w *= bcu

			# Apply the child value weights and sum them up over the parents
			# then apply the weights based on parent values
			w = A.SpGEMM(w, semiring=BC_SR)
			w *= bfs[depth-1]
			w *= nsp
			bcu += w

		# update the bc with the bc update
		bc = bc + bcu.sum(Mat.Row)

	# subtract off the additional values added in by precomputation
	bc = bc - nVertToCalc
	
	# normalize, if desired
	if normalize:
		nVertSampled = sum(numVs)
		bc = bc * (float(N)/float(nVertSampled*(N-1)*(N-2)))
	
	# return
	if retNVerts:
		return bc,nVertSampled
	else:
		return bc
DiGraph._centrality_approxBC = _centrality_approxBC

def _centrality_exactBC(self, **kwargs):
	"""
	Computes exact betweenness centrality. This is an alias for
	approxBC(sample=1.0).
	"""
	return DiGraph._centrality_approxBC(self, sample=1.0, **kwargs)
DiGraph._centrality_exactBC = _centrality_exactBC

# NEEDED: tests
def cluster(self, alg, **kwargs):
#		ToDo:  Normalize option?
	"""
	cluster a DiGraph using algorithm `alg`.
	Output Arguments:
		ret: a dense Vec of length equal to the number of vertices
			in the DiGraph. The value of each element of the Vec 
			denotes a cluster root for that vertex.
	"""
	if alg=='Markov' or alg=='markov':
		A = DiGraph._MCL(self, **kwargs)
		#A.save("problemMat.mtx")
		#A = Mat.load("problemMat.mtx")
		G = DiGraph(edges=A)
		clus = G.connComp()
		return clus, G

	elif alg=='agglomerative':
		return _cluster_agglomerative(self, **kwargs)

	elif alg=='kNN' or alg=='knn':
		raise NotImplementedError, "k-nearest neighbors clustering not implemented"

	else:
		raise KeyError, "unknown clustering algorithm (%s)" % alg

	return clus
DiGraph.cluster = cluster

def connComp(self):
	"""
	Finds the connected components of the graph by BFS.
	Output Arguments:
		ret:  a dense Vec of length equal to the number of vertices
			in the DiGraph.  The value of each element of the Vec 
			denotes a cluster root for that vertex.
	"""
	
	# TODO: use a boolean matrix
	# we want a symmetric matrix with self loops
	n = self.nvert()
	G = self.e.copy(element=1.0)
	G_T = G.copy()
	G_T.transpose()
	G += G_T
	G += Mat.eye(n, n, element=1.0)
	G.apply(op_set(1))
	
	# the semiring we want to use
	mulFn = lambda x,y: y           # use the value from the vector
	addFn = lambda x,y: max(x, y)   # out of all incomming edges, use the max
	selectMax = sr(addFn, mulFn)

	roots = Vec.range(n, sparse=False)
	frontier = roots.sparse()
	
	while frontier.nnn() > 0:
		frontier = G.SpMV(frontier, semiring=selectMax)
		
		# prune the frontier of vertices that have not changed
		frontier.eWiseApply(roots, op=(lambda f,r: f), doOp=(lambda f,r: f != r), inPlace=True)
		
		# update the roots
		roots[frontier] = frontier

	return roots
	
DiGraph.connComp = connComp

def _MCL(self, expansion=2, inflation=2, addSelfLoops=False, selfLoopWeight=1, prunelimit=0.00001, sym=False, retNEdges=False):
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
				AA.apply(op_set(1))
				AA.SpGEMM(AA, semiring=sr_plustimes, inPlace=True)
				nedges += AA.sum(Mat.Column).reduce(op_add)
			#A = A.SpGEMM(A, semiring=sr_plustimes)
			A.SpGEMM(A, semiring=sr_plustimes, inPlace=True)
	
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
DiGraph._MCL = _MCL

def _cluster_agglomerative(self, roots):
	"""
	build len(roots) clusters. Each vertex in the graph is added
	to the cluster closest to it. "closest" means shortest path.
	"""
	raise NotImplementedError, "broken."
	A = self.e.copy()
	
	t = A.copy()
	t.transpose()
	A += t

	# we need 0-weight self loops
	A.removeMainDiagonal()
	A += Mat.eye(self.nvert(), element=0.0)
	
	k = len(roots)
	n = self.nvert()

	#print "G:",A
	
	expandAdd = lambda x,y: min(x,y)
	expandMul = lambda x,y: x+y
	expandSR = sr(expandAdd, expandMul)
	
	# clusters start with just the roots
	nullRoots = Mat(roots, Vec.range(k), A._identity_, k, self.nvert())
	frontier = nullRoots
	clusters = Vec.range(self.nvert())
	#print "initial frontier:",frontier
	
	# expand each cluster using BFS. A discovered vertex v is part of
	# a root r's cluster if the length of a path from r to v is less than
	# a path from any other root to v.
	delta = 1
	itercount = 0
	while delta > 0: #and itercount < 30:
		# save the current frontier for comparison
		lastFrontier = frontier
		
		# expand out from the current clusters
		frontier = A.SpGEMM(frontier, semiring=expandSR)
		#print "new frontier:",frontier
		
		# if a vertex was discovered from two different clusters,
		# keep only the one with the shorter path
		mins = frontier.reduce(Mat.Row, (lambda x,y: min(x,y)), init=1000000)
		#print "mins:",mins
		boolFrontier = frontier.copy()
		boolFrontier.scale(mins, op=(lambda x,y: x == y), dir=Mat.Row)
		#print "bool frontier:",boolFrontier
		boolFrontier._prune(lambda x: round(x) != 1)
		#print "pruned bool frontier:",boolFrontier
		frontier = frontier.eWiseApply(boolFrontier, (lambda f,bf: f))
		#print "pruned frontier:",frontier
		
		# prune the frontier of vertices that have not changed
		# required EWise allowANulls not implemented in CombBLAS yet.
		#print "last frontier:",lastFrontier
		#frontier = lastFrontier.eWiseApply(frontier, (lambda l,f: f), doOp=(lambda l,f: l != f), allowANulls=True)
		#print "newly discovered frontier:",frontier
		
		# update the clusters
		# collapse the cluster mat into a vector
		updateFrontier = frontier.copy()
		updateFrontier.scale(roots, op=(lambda f,r: r), dir=Mat.Column)
		#print "updateFrontier:",updateFrontier
		update = updateFrontier.reduce(Mat.Row, (lambda v,s: min(v,s)), init=1.8e302)
		#print "update:  ",update
		delta = clusters.eWiseApply(update, (lambda c,u: int(c) != int(u) and u != 1.8e302), inPlace=False).reduce(op_add)
		clusters.eWiseApply(update, (lambda c,u: u), doOp=(lambda c,u: int(c) != int(u) and u != 1.8e302), inPlace=True)
		
		#delta = frontier.nnn()
		#print "delta=",delta
		#print "clusters:",clusters
		itercount += 1
		#print "finished iteration",itercount,"-----------------------------------------------------------------"
		#return clusters
		
	return clusters
DiGraph._cluster_agglomerative = _cluster_agglomerative

def _findClusterModularity(self, C):
	return 3
DiGraph._findClusterModularity = _findClusterModularity
