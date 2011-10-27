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
def bfsTree(self, root, useOldFunc=True):
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
		parents:  a ParVec instance of length equal to the number
			of vertices in the DiGraph, with each element denoting 
			the vertex number of that vertex's parent in the tree.
			The root is its own parent.  Unreachable vertices
			have a parent of -1. 

	SEE ALSO: isBfsTree 
	"""
	#ToDo:  doesn't handle filters on a doubleint DiGraph
	if not self.isObj() and self._hasFilter(self):
		raise NotImplementedError, 'DiGraph(element=default) with filters not supported'
	if self.isObj():
		#tmpG = self.copy()._toDiGraph()
		matrix = self.copy(element=0).e
	else:
		matrix = self.e

	if not matrix.isObj():
		sR = sr_select2nd
	else:
		mulFn = lambda x,y: x._SR_second_(y)
		addFn = lambda x,y: x._SR_max_(y)
		sR = sr(addFn, mulFn)

	parents = Vec(self.nvert(), -1, sparse=False)
	fringe = Vec(self.nvert(), sparse=True)
	parents[root] = root
	fringe[root] = root
	while fringe.nnn() > 0:
		fringe.spRange()
		matrix.SpMV(fringe, semiring=sR, inPlace=True)
		
		if useOldFunc:
			# this method uses an old CombBLAS routine.
			# it will be deprecated when acceptable performance from SEJITS is attained.
			pcb.EWiseMult_inplacefirst(fringe._v_, parents._v_, True, -1)
			parents[fringe] = fringe
		else:
			# this is the preferred method. It is a bit slower than the above due to unoptimized
			# Python callbacks in this version of KDT, but future SEJITS integration should remove
			# that penalty and the above method will be deprecated.
			
			# remove already discovered vertices from fringe.
			fringe.eWiseApply(parents, op=(lambda f,p: f), doOp=(lambda f,p: p == -1), inPlace=True)
			# update the parents
			parents[fringe] = fringe

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
	fringe = SpParVec(nvertG)
	fringe[root] = root
	cycle = False
	multiparents = False
	while fringe.nnn() > 0 and not cycle and not multiparents:
		fringe.spOnes()
		newfringe = SpParVec.toSpParVec(builtGT._spm.SpMV_PlusTimes(fringe._spv))
		if visited[newfringe.toParVec().findInds()].any():
			cycle = True
			break
		if (newfringe > 1).any():
			multiparents = True
			break
		fringe = newfringe
		visited[fringe] = 1
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
	fringe = SpParVec(nvertG)
	fringe[root] = root
	levels = ParVec.zeros(nvertG) - 1
	levels[root] = 0

	level = 1
	while fringe.nnn() > 0:
		fringe.spRange()
		#ToDo:  create PCB graph-level op
		self._spm.SpMV_SelMax_inplace(fringe._spv)
		#ToDo:  create PCB graph-level op
		pcb.EWiseMult_inplacefirst(fringe._spv, parents2._dpv, True, -1)
		parents2[fringe] = fringe
		levels[fringe] = level
		level += 1
	if not sym:
		self.reverseEdges()
	del parents2
	if (levels[treeI]-levels[treeJ] != -1).any():
		return (-2, None)

	return (ret, levels)
DiGraph.isBfsTree = isBfsTree

# returns a Boolean vector of which vertices are neighbors
# NEEDED: tests
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
	fringe = SpParVec(self.nvert(), sparse=True)
	fringe[source] = 1
	for i in range(nhop):
		self.e.SpMV(fringe, sr=sr_select2nd, inPlace=True)
		dest[fringe] = 1

	return dest

DiGraph.neighbors = neighbors

# returns:
#   - source:  a vector of the source vertex for each new vertex
#   - dest:  a Boolean vector of the new vertices
#ToDo:  nhop argument?
# NEEDED: update to transposed edge matrix
# NEEDED: update to new fields
# NEEDED: tests
def pathsHop(self, source, sym=False):
	"""
	calculates, for the given DiGraph instance and starting vertices,
	which can be viewed as the fringe of a set of paths, the vertices
	that are reachable by traversing one graph edge from one of the 
	starting vertices.  The paths are kept distinct, as only one path
	will extend to a given vertex.

	Input Arguments:
		self:  a DiGraph instance
		source:  a Boolean Vec with True (1) in the positions
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

	ret = Vec(self2.nvert(), element=-1, sparse=False)
	fringe = source.find()
	fringe.spRange()
	self.e.SpMV(fringe, sr=sr_select2nd, inPlace=True)

	ret[fringe] = fringe

	return ret
DiGraph.pathsHop = pathsHop

# NEEDED: tests
def normalizeEdgeWeights(self, dir=DiGraph.Out):
	"""
	Normalize the outward edge weights of each vertex such
	that for Vertex v, each outward edge weight is
	1/outdegree(v).
	"""
	if self.isObj():
		raise NotImplementedError, "Cannot normalize object weights yet."

	degscale = self.degree(dir)
	degscale.apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0), pcb.identity(), pcb.bind1st(pcb.divides(), 1)))			
	self.scale(degscale, dir, op_mul)
DiGraph.normalizeEdgeWeights = normalizeEdgeWeights

# NEEDED: make sure normalization is done correctly
# NEEDED: update to new fields
# NEEDED: tests
def pageRank(self, epsilon = 0.1, dampingFactor = 0.85):
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

	sinkV = G.degree(DiGraph.In)
	sinkV.apply(pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(0), pcb.set(1./nvert)))

	# Normalize edge weights such that for each vertex,
	# each outgoing edge weight is equal to 1/(number of
	# outgoing edges).
	print "DEBUG: make sure PageRank normalization is done in the correct direction"
	G.normalizeEdgeWeights(DiGraph.In)

	# PageRank loop.
	delta = 1
	dv1 = Vec(nvert, 1./nvert, sparse=False)
	v1 = dv1.sparse()
	prevV = Vec(nvert, sparse=True)
	onesVec = Vec.ones(nvert, sparse=True)
	dampingVec = onesVec * ((1 - dampingFactor)/nvert)
	while delta > epsilon:
		prevV = v1.copy()
		v2 = G.e.SpMV(v1, sr=sr_plustimes)

		# Compute the inner product of sinkV and v1.
		sinkContribV = sinkV.EWiseApply(v1, op_mul, inPlace=False)
		sinkContrib = sinkContribV.reduce(op_add)
		
		v1 = v2 + (onesVec*sinkContrib)
		v1 = v1*dampingFactor + dampingVec
		delta = (v1 - prevV).reduce(op_add, op_abs)
	return v1
DiGraph.pageRank = pageRank
	
# NEEDED: update to transposed edge matrix
# NEEDED: update to new fields
# NEEDED: tests
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

# NEEDED: update to transposed edge matrix
# NEEDED: update to new fields
# NEEDED: tests
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
	"""
	A = self.copy()
	Anv = A.nvert()
	if BCdebug>0 and master():
		print "in _approxBC, A.nvert=%d, nproc=%d" % (Anv, nProcs)

	if BCdebug>1 and master():
		print "Apply(set(1))"
	#FIX:  should not overwrite input
	self.ones()
	#Aint = self.ones()	# not needed;  Gs only int for now
	if BCdebug>1 and master():
		print "spm.getnrow and col()"
	N = A.nvert()
	if BCdebug>1 and master():
		print "densevec(%d, 0)"%N
	bc = ParVec(N)
	if BCdebug>1 and master():
		print "getnrow()"
	nVertToCalc = int(math.ceil(self.nvert() * sample))
	nVertToCalc = min(nVertToCalc, self.nvert())
	
	# batchSize = #rows/cols that will fit in memory simultaneously.
	# bcu has a value in every element, even though it's literally
	# a sparse matrix (DiGraph).  So batchsize is calculated as
	#   nrow = memory size / (memory/row)
	#   memory size (in edges)
	#        = 2GB * 0.1 (other vars) / 18 (bytes/edge) * nProcs
	#   memory/row (in edges)
	#        = self.nvert()
	physMemPCore = 2e9; bytesPEdge = 18
	if (batchSize < 0):
		batchSize = int(2e9 * memFract / bytesPEdge * nProcs / N)
	batchSize = min(nVertToCalc, batchSize)
	
	nBatches = int(math.ceil(float(nVertToCalc) / float(batchSize)))
	nPossBatches = int(math.ceil(float(N) / float(batchSize)))
	
	# sources for the batches
	# the i-th batch is defined as randVerts[ startVs[i] to (startVs[i]+numV[i]) ]
	randVerts = ParVec.range(Anv)
	
	if master():
		print "NOTE! SKIPPING RANDPERM()! starting vertices will be sequential."
	#randVerts.randPerm()
	
	if (batchSize >= nVertToCalc):
		startVs = [0]
		endVs = [nVertToCalc]
		numVs = [nVertToCalc]
	else: #elif sample == 1.0:
		startVs = range(0,nVertToCalc,batchSize)
		endVs = range(batchSize, nVertToCalc, batchSize)
		endVs.append(nVertToCalc)
		numVs = [y-x for [x,y] in zip(startVs,endVs)]
	if False: #else:
		if BCdebug>1 and master():
			print "densevec iota(0, %d) (i think in that order)"%nPossBatches
		perm = ParVec.range(nPossBatches)
		if BCdebug>1 and master():
			print "densevec randperm()"
		perm.randPerm()
		#   ideally, could use following 2 lines, but may have
		#   only 1-2 batches, which makes index vector look
		#   like a Boolean, which doesn't work right
		#startVs = ParVec.range(nBatches)[perm[ParVec.range(nBatches]]
		#numVs = [min(x+batchSize,N)-x for x in startVs]
		if BCdebug>1 and master():
			print "densevec iota(0, %d) (i think in that order)"%nPossBatches
		tmpRange = ParVec.range(nPossBatches)
		if BCdebug>1 and master():
			print "densevec(%d, 0)"%nBatches
		startVs = ParVec.zeros(nBatches)
		if BCdebug>1 and master():
			print "densevec(%d, 0)"%nBatches
		numVs = ParVec.zeros(nBatches)
		for i in range(nBatches):
			if BCdebug>1 and master():
				print "dense vec SubsRef, GetElement and SetElement"
			startVs[i] = tmpRange[perm[i]]*batchSize
			if BCdebug>1 and master():
				print "dense vec GetElement and SetElement"
			numVs[i] = min(startVs[i]+batchSize,N)-startVs[i]

	if BCdebug>0 and master():
		print "batchSz=%d, nBatches=%d, nPossBatches=%d" % (batchSize, nBatches, nPossBatches)
	if BCdebug>1 and master():
		print "summary of batches:"
		print "startVs:",startVs
		print "  numVs:", numVs
	for [startV, numV] in zip(startVs, numVs):
		startV = int(startV); numV = int(numV)
		if BCdebug>0 and master():
			print "startV=%d, numV=%d" % (startV, numV)
		bfs = []		
		batchRange = ParVec.range(startV, startV+numV)
		batch = randVerts[batchRange]
		curSize = len(batch)
		#next:  nsp is really a SpParMat
		nsp = DiGraph(ParVec.range(curSize), batch, 1, curSize, N)
		#next:  fringe should be Vs; indexing must be impl to support that; seems should be a collxn of spVs, hence a SpParMat
		fringe = A[batch,ParVec.range(N)]
		depth = 0
		while fringe.nedge() > 0:
			before = time.time()
			depth = depth+1
			if BCdebug>1 and depth>1:
				nspne = tmp.nedge(); tmpne = tmp.nedge(); fringene = fringe.nedge()
				if master():
					print "BC: in while: depth=%d, nsp.nedge()=%d, tmp.nedge()=%d, fringe.nedge()=%d" % (depth, nspne, tmpne, fringene)
			nsp += fringe
			tmp = fringe.copy()
			tmp.ones()
			bfs.append(tmp)
			#next:  changes how???
			tmp = fringe._SpGEMM(A)
			if BCdebug>1:
				#nspsum = nsp.sum(DiGraph.Out).sum() 
				#fringesum = fringe.sum(DiGraph.Out).sum()
				#tmpsum = tmp.sum(DiGraph.Out).sum()
				if master():
					#print depth, nspsum, fringesum, tmpsum
					pass
			# prune new-fringe to new verts
			fringe = tmp.mulNot(nsp)
			if BCdebug>1 and master():
				print "    %f seconds" % (time.time()-before)

		bcu = DiGraph.fullyConnected(curSize,N)
		# compute the bc update for all vertices except the sources
		for depth in range(depth-1,0,-1):
			# compute the weights to be applied based on the child values
			w = bfs[depth] / nsp 
			w *= bcu
			if BCdebug>2:
				tmptmp = w.sum(DiGraph.Out).sum()
				if master():
					print tmptmp
			# Apply the child value weights and sum them up over the parents
			# then apply the weights based on parent values
			w._T()
			w = A._SpGEMM(w)
			w._T()
			w *= bfs[depth-1]
			w *= nsp
			bcu += w

		# update the bc with the bc update
		if BCdebug>2:
			tmptmp = bcu.sum(DiGraph.Out).sum()
			if master():
				print tmptmp
		bc = bc + bcu.sum(DiGraph.In)	# column sums

	# subtract off the additional values added in by precomputation
	bc = bc - nVertToCalc
	if normalize:
		nVertSampled = sum(numVs)
		bc = bc * (float(N)/float(nVertSampled*(N-1)*(N-2)))
	
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
		G = DiGraph._cluster_markov(self, **kwargs)
		clus = G.connComp()
		return clus, G

	elif alg=='kNN' or alg=='knn':
		raise NotImplementedError, "k-nearest neighbors clustering not implemented"

	else:
		raise KeyError, "unknown clustering algorithm (%s)" % alg

	return clus
DiGraph.cluster = cluster

# NEEDED: tests
def connComp(self):
	"""
	Finds the connected components of the graph by BFS.
	Output Arguments:
		ret:  a dense Vec of length equal to the number of vertices
			in the DiGraph.  The value of each element of the Vec 
			denotes a cluster root for that vertex.
	"""
	
	# we want a symmetric matrix with self loops
	n = self.nvert()
	G = self.e.copy(element=1.0)
	G += G.copy().transpose()
	G += Mat.eye(n, 1.0)
	G.apply(op_set(1))
	
	# Future: use a dense accumulator and a sparse frontier to take advantage
	# of vertices that are found in the correct component and will not be
	# reshuffled.
	#component = ParVec.range(G.nvert())
	#frontier = component.toSpParVec()
	frontier = Vec.range(n, sparse=False)
	
	delta = 1
	while delta > 0:
		last_frontier = frontier
		frontier = G.SpMV(frontier, sr=sr_select2nd)
		
		deltaV = frontier.eWiseApply(last_frontier, op=(lambda f, l: int(f != l)), inPlace=False)
		delta = deltaV.reduce(op_add)
	
	return frontier.dense()
DiGraph.connComp = connComp

# markov clustering temporarily moved to MCL.py
