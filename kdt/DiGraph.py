import math
import time
import Graph as gr
from Vec import Vec
from Mat import Mat
from Util import *
from Util import master

import kdt.pyCombBLAS as pcb

class DiGraph(gr.Graph):

	# NOTE: these have been transposed.
	# Traversing a matrix row will give the vertex's incomming edges.
	# Traversing a matrix column will give the vertex's outgoing edges.
	In  = Mat.Row
	Out = Mat.Column
	All = Mat.All

	# NOTE:  for any vertex, out-edges are in the column and in-edges
	#	are in the row
	# NEEDED: tests
	def __init__(self, sourceV=None, destV=None, valueV=None, nv=None, element=0, edges=None, vertices=None):
		"""
		FIX:  doc
		creates a new DiGraph instance.  Can be called in one of the 
		following forms:

	DiGraph():  creates a DiGraph instance with no vertices or edges.  Useful as input for genGraph500Edges.

	DiGraph(sourceV, destV, weightV, n)
	DiGraph(sourceV, destV, weightV, n, m)
		create a DiGraph Instance with edges with source represented by 
		each element of sourceV and destination represented by each 
		element of destV with weight represented by each element of 
		weightV.  In the 4-argument form, the resulting DiGraph will 
		have n out- and in-vertices.  In the 5-argument form, the 
		resulting DiGraph will have n out-vertices and m in-vertices.

		Input Arguments:
			sourceV:  a ParVec containing integers denoting the 
			    source vertex of each edge.
			destV:  a ParVec containing integers denoting the 
			    destination vertex of each edge.
			weightV:  a ParVec containing double-precision floating-
			    point numbers denoting the weight of each edge.
			n:  an integer scalar denoting the number of out-vertices 
			    (and also in-vertices in the 4-argument case).
			m:  an integer scalar denoting the number of in-vertices.

		Output Argument:  
			ret:  a DiGraph instance

		Note:  If two or more edges have the same source and destination
		vertices, their weights are summed in the output DiGraph instance.

		SEE ALSO:  toParVec
	def __init__(self, sourceV=None, destV=None, valueV=None, nv=None, element=0):
		"""
		if edges is not None:
			self.e = edges
		else:
			if destV is None or sourceV is None:
				sourceV = Vec(0, element=0, sparse=False)
				destV = Vec(0, element=0, sparse=False)
			if valueV is None:
				valueV = Vec(len(sourceV), element=element, sparse=False)
			self.e = Mat(i=destV, j=sourceV, v=valueV, n=nv, element=element)  # AL: swapped
		
		if vertices is not None:
			self.v = vertices
		else:
			self.v = Vec(length=self.nvert(), element=0.0, sparse=False)

#	# This has to add both edges and vectors
#	def __add__(self, other):


	# NEEDED: tests
	def __repr__(self):
		ret = "edge Mat: " + self.e.__repr__() + "\nvertex attribute Vec: " + self.v.__repr__()
		return ret
	

	#in-place, so no return value
	# NEEDED: modify to make sense in graph context, not just matrix context
	# NEEDED: update to new fields
	# NEEDED: tests
	def ___apply(self, op, other=None, notB=False):
		"""
		applies the given operator to every edge in the DiGraph

		Input Argument:
			self:  a DiGraph instance, modified in place.
			op:  a Python or pyCombBLAS function

		Output Argument:  
			None.

		"""
		if other is None:
			if not isinstance(op, pcb.UnaryFunction):
				self.e.Apply(pcb.unary(op))
			else:
				self.e.Apply(op)
			return
		else:
			if not isinstance(op, pcb.BinaryFunction):
				self.e = pcb.EWiseApply(self.e, other.e, pcb.binary(op), notB)
			else:
				self.e = pcb.EWiseApply(self.e, other.e, op, notB)
			return

	# NEEDED: modify to make sense in graph context, not just matrix context
	# NEEDED: update to new fields
	# NEEDED: tests
	def ___eWiseApply(self, other, op, allowANulls, allowBNulls, noWrap=False):
		"""
		ToDo:  write doc
		"""
		if hasattr(self, '_eFilter_') or hasattr(other, '_eFilter_'):
			class tmpB:
				if hasattr(self,'_eFilter_') and len(self._eFilter_) > 0:
					selfEFLen = len(self._eFilter_)
					eFilter1 = self._eFilter_
				else:
					selfEFLen = 0
				if hasattr(other,'_eFilter_') and len(other._eFilter_) > 0:
					otherEFLen = len(other._eFilter_)
					eFilter2 = other._eFilter_
				else:
					otherEFLen = 0
				@staticmethod
				def fn(x, y):
					for i in range(tmpB.selfEFLen):
						if not tmpB.eFilter1[i](x):
							x = type(self._identity_)()
							break
					for i in range(tmpB.otherEFLen):
						if not tmpB.eFilter2[i](y):
							y = type(other._identity_)()
							break
					return op(x, y)
			superOp = tmpB().fn
		else:
			superOp = op
		if noWrap:
			if isinstance(other, (float, int, long)):
				m = pcb.EWiseApply(self.e, other   ,  superOp)
			else:
				m = pcb.EWiseApply(self.e, other.e, superOp)
		else:
			if isinstance(other, (float, int, long)):
				m = pcb.EWiseApply(self.e, other   ,  pcb.binaryObj(superOp))
			else:
				m = pcb.EWiseApply(self.e, other.e, pcb.binaryObj(superOp))
		ret = self._toDiGraph(m)
		return ret

	# NEEDED: tests
	@staticmethod
	def _hasFilter(self):
		try:
			ret = (hasattr(self,'_eFilter_') and len(self._eFilter_)>0) # ToDo: or (hasattr(self,'vAttrib') and self.vAttrib._hasFilter(self.vAttrib)) 
		except AttributeError:
			ret = False
		return ret

	def isObj(self):
		return isinstance(self.e, (pcb.pySpParMatObj1, pcb.pySpParMatObj2))
	
	def isBool(self):
		return isinstance(self._identity_, bool)

	# in-place, so no return value
	# NEEDED: tests
	def addEFilter(self, filter):
		"""
		adds an edge filter to the DiGraph instance.  

		SEE ALSO:
			delEFilter  
		"""
		self.e.addFilter(filter)

	def addVFilter(self, filter):
		"""
		adds a vertex filter to the DiGraph instance.  

		SEE ALSO:
			delVFilter  
		"""
		self.v.addFilter(filter)
		
	def contract(self, groups=None, clusterParents=None):
		"""
		contracts all vertices that are like-numbered in the groups
		argument into single vertices, removing all edges between
		vertices in the same group and retaining edges where any
		vertex in a group has an edge to a vertex in another group.
		The result DiGraph will have as many vertices as the maximum
		value in groups.

		Input Arguments:
			self:  a DiGraph instance.
			
			Specify exactly one of the following:
			groups:  a Vec denoting into which group (result
				vertex) each input vertex should be placed
			clusterParents: a ParVec denoting which vertex an input vertex
			    should be collapsed into.

		Output Argument:
			ret:  a DiGraph instance
		"""
		#ToDo:  implement weighting properly
		n = self.nvert()

		if type(self.nvert()) == tuple:
			raise NotImplementedError, 'only implemented for square graphs'
		if groups is None and clusterParents is None:
			raise KeyError, 'groups or collapseInto must be specified'

		if groups is not None and len(groups) != n:
			raise KeyError, 'len(groups) does not match self.nvert()'
		if groups is not None and (groups.min < 0 or groups.max() >= len(groups)):
			raise KeyError, 'at least one groups value negative or greater than len(groups)-1'
		if clusterParents is not None and len(clusterParents) != n:
			raise KeyError, 'len(clusterParents) does not match self.nvert()'
		if clusterParents is not None and (clusterParents.min < 0 or clusterParents.max() >= n):
			raise KeyError, 'at least one groups value negative or greater than len(clusterParents)-1'
			
		if clusterParents is not None:
			# convert to groups
			groups = DiGraph.convClusterParentToGroup(clusterParents)
		
		nvRes = int(groups.max()+1)
		origVtx = Vec.range(n)
		# lhrMat == left-/right-hand-side matrix
		lrhMat = Mat(groups, origVtx, Vec.ones(n), n, nvRes)
		tmpMat = lrhMat.SpGEMM(self.e, semiring=sr_plustimes)
		lrhMat.transpose()
		res = tmpMat.SpGEMM(lrhMat, semiring=sr_plustimes)
		return DiGraph(edges=res);
	
	@staticmethod
	def convClusterParentToGroup(clusterParents, retInvPerm = False):
		"""
		converts a component list composed of parent vertices into
		a component list composed of cluster numbers.
		For k clusters, each cluster number will be in the range [0,k).

		Input Arguments:
			clusterParents:  a vector where each element specifies which vertex
			    is its cluster parent. It is assumed that each cluster has only
			    one parent vertex.
			retInvPerm:  if True, says to return a permutation going from group
			    number to original parent vertex.

		Output Argument:
			ret:  a vector of the same length as clusterParents where the elements
			    are in the range [0,k) and correspond to cluster group numbers.
			    If retInvPerm is True, the return is a tuple which also contains
			    a permutation from group number to original parent vertex.
		"""
		
		n = len(clusterParents)

		# Count the number of elements in each parent's component to identify the parents
		countM = Mat(Vec.range(n), clusterParents, Vec.ones(n), n)  # AL: swapped
		counts = countM.reduce(Mat.Column, op_add)
		del countM
		
		# sort to put them all at the front
		sorted = counts.copy()
		sorted.apply(op_negate)
		perm = sorted.sort()

		# find inverse of sort permutation so that [1,2,3,4...] can be put back into the
		# original parent locations
		invM = Mat(perm, Vec.range(n), Vec.ones(n), n)  # AL: swapped
		invPerm = invM.SpMV(Vec.range(n, sparse=True), semiring=sr_plustimes).dense() # SpMV with dense vector is broken at the moment
		del invM
		
		# Find group number for each parent vertex
		groupNum = invPerm
		
		# Broadcast group number to all vertices in cluster
		broadcastM = Mat(Vec.range(n), clusterParents, Vec.ones(n), n)  # AL: swapped
		ret = broadcastM.SpMV(groupNum.sparse(), sr_plustimes).dense() # SpMV with dense vector is broken at the moment
		
		if retInvPerm:
			return ret, perm 
		else:
			return ret 
	
	# NEEDED: tests
	@staticmethod
	def convMaskToIndices(mask):
		verts = mask.findInds(lambda x: x == 1)
		return verts

	def copy(self, element=None):
		"""
		creates a deep copy of a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance.

		Output Argument:
			ret:  a DiGraph instance containing a copy of the input.
		"""
		ret = DiGraph(edges=self.e.copy(), vertices=self.v.copy())
		return ret

	def degree(self, dir=Out):
		"""
		calculates the degrees of the appropriate edges of each vertex of 
		the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges to count, with choices being
			    DiGraph.Out (default), DiGraph.In or DiGraph.All.

		Output Argument:
			ret:  a dense Vec instance with each element containing the
			    degree of the corresponding vertex.

		SEE ALSO:  sum 
		"""
		if dir == DiGraph.In or dir == DiGraph.Out:
			return self.e.reduce(dir, (lambda x,y: x+y), uniOp=(lambda x: 1), init=0)
		elif dir == DiGraph.All:
			return self.degree(DiGraph.In) + self.degree(DiGraph.Out)
		else:
			raise KeyError, 'Invalid edge-direction'
		
	# NEEDED: update to new fields
	# NEEDED: tests
	# in-place, so no return value
	def delEFilter(self, filter=None):
		"""
		deletes an edge filter from the DiGraph instance.  

		Input Arguments:
			self:  a DiGraph instance
			filter:  a Python function, which can be either a function
			    previously added to this DiGraph instance by a call to
			    addEFilter or None, which signals the deletion of all
			    vertex filters.

		SEE ALSO:
			addEFilter  
		"""
		self.e.delFilter(filter)

	# in-place, so no return value
	def removeSelfLoops(self):
		"""
		removes all edges whose source and destination are the same
		vertex, in-place in a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance, modified in-place.

		"""
		if self.nvert() > 0:
			self.e.removeMainDiagonal()
		return
	
	def delIsolatedVerts(self, loadBalance=True):
		"""
		removes all vertices with no incoming our outgoing edges. Optionally

		Input Argument:
			self:  a DiGraph instance, modified in-place.
		"""
		
		deg = self.degree(DiGraph.All)
		nonisov = deg.findInds(lambda x: x > 0)
		if loadBalance:
			nonisov.randPerm()
		
		self.e[nonisov, nonisov, True]
		self.v = self.v[nonisov]

	def addSelfLoops(self, selfLoopAttr=1):
		"""
		adds edges whose source and destination are the same
		vertex, in-place in a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance, modified in-place.
			selfLoopAttr: the value to put on each self loop

		"""
		if self.nvert() > 0:
			self.e += Mat.eye(self.nvert(), self.nvert(), element=selfLoopAttr)
		return

	@staticmethod
	def fullyConnected(n, element=1.0):
		"""
		creates edges in a DiGraph instance that connects each vertex
		directly to every other vertex.

		Input Arguments:
			n:  an integer scalar denoting the number of vertices in
			    the graph.
			element: the edge attribute.

		Output Argument:
			ret:  a DiGraph instance with directed edges from each
			    vertex to every other vertex. 
		"""
		
		edges = Mat.ones(n=n, element=element)
		return DiGraph(edges=edges)
	
	@staticmethod
	def generate2DTorus(nnodes):
		"""
		constructs a DiGraph instance with the connectivity pattern of a 2D
		torus;  i.e., each vertex has edges to its north, west, south, and
		east neighbors, where the neighbor may be wrapped around to the
		other side of the torus.  

		Input Parameter:
			nnodes:  an integer scalar that denotes the number of nodes
			    on each axis of the 2D torus.  The resulting DiGraph
			    instance will have nnodes**2 vertices.

		Output Parameter:
			ret:  a DiGraph instance with nnodes**2 vertices and edges
			    in the pattern of a 2D torus. 
		"""
		n = nnodes
		N = n*n
		rows = Vec.range(N)
		main = rows.copy() # main diagonal, self loops
		p1 = main.copy()
		p1.apply(lambda x: (x+1)%N) # main diagonal moved right 1, connect to east neighbor
		pn = main.copy()
		pn.apply(lambda x: (x+(n-1))%N) # main diagonal moved right (n-1), connect to south neighbor
		mn = main.copy()
		mn.apply(lambda x: (x-(n-1))%N) # main diagonal moved left (n-1), connect to north neighbor
		m1 = main.copy()
		m1.apply(lambda x: (x-1)%N) # main diagonal moved left 1, connect to west neighbor
		
		ret = Mat(rows, p1, 1, N)
		ret += Mat(rows, m1, 1, N)
		ret += Mat(rows, pn, 1, N)
		ret += Mat(rows, mn, 1, N)
		#ret += Mat(rows, main, 1, N) # would include self loops
		
		return DiGraph(edges=ret)

	@staticmethod
	def generateSelfLoops(n, selfLoopAttr=1):
		"""
		creates edges in a DiGraph instance where each vertex
		has exactly one edge connecting to itself.

		Input Arguments:
			n:  an integer scalar denoting the number of vertices in
			    the graph.
			selfLoopAttr: the value to put on the self loop.

		Output Argument:
			ret:  a DiGraph instance with directed edges from each
			    vertex to itself. 
		"""
		return DiGraph(edges=Mat.eye(n, n, element=selfLoopAttr))
		
	@staticmethod
	def generateRMAT(scale, edgeFactor=16, initiator=[.57, .19, .19, .05], delIsolated=True, retKernel1Time = False, element=True):
		"""
		creates edges in a DiGraph instance that meets the Graph500 
		specification.  The graph is symmetric. (See www.graph500.org 
		for details.)

		Input Arguments:
			scale:  an integer scalar representing the logarithm base
			    2 of the number of vertices in the resulting DiGraph.
			edgeFactor: an integer specifying the average degree.
			    Graph500 value: 16
			initiator: an array of 4 floating point numbers that must
			    sum to 1. Specifies the probabilities of hitting each
			    quadrant of the Kroenecker product. More lopsided values
			    result in fewer vertices having more edges.
			    Graph500 value: [a,b,c,d] = [.57, .19, .19, .05]
			delIsolated: whether or not to remove isolated vertices.
			retKernel1Time: whether or not to also return the Graph500
			    Kernel 1 (graph construction time).
			    
		Output Argument:
			
			ret:  a double-precision floating-point scalar denoting
			    the amount of time to taken to convert the created edges into
			    the DiGraph instance.  This equals the value of Kernel 1
			    of the Graph500 benchmark. Timing the entire generateRMAT
			    call would also time the edge generation, which is not part
			    of Kernel 1.
			    Degrees of all vertices.
		"""
		if not isinstance(scale, (int, float, long)):
			raise KeyError, "scale must be an integer!"
			
		edges, degrees, k1time = Mat.generateRMAT(int(scale), fillFactor=edgeFactor, initiator=initiator, delIsolated=delIsolated, element=element)
		
		ret = DiGraph(edges=edges, vertices=degrees)
				
		if retKernel1Time:
			return (ret, k1time)
		else:
			return ret

	# NEEDED: tests
	@staticmethod
	def load(fname, eelement=0.0, par_IO=False):
		"""
		loads the contents of the file named fname (in the Coordinate Format 
		of the Matrix Market Exchange Format) into a DiGraph instance.

		Input Argument:
			fname:  a filename from which the DiGraph data will be loaded.
		Output Argument:
			ret:  a DiGraph instance containing the graph represented
			    by the file's contents.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The load
		method makes this conversion while reading the data and creating
		the graph.

		SEE ALSO:  save, UFget
		"""
		mat = Mat.load(fname, element=eelement, par_IO=par_IO)
		return DiGraph(edges=mat, vertices=None)
	
	# NEEDED: tests
	def save(self, fname):
		"""
		saves the contents of the passed DiGraph instance to a file named
		fname in the Coordinate Format of the Matrix Market Exchange Format.

		Input Arguments:
			self:  a DiGraph instance
			fname:  a filename to which the DiGraph data will be saved.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The save
		method makes this conversion while writing the data.

		SEE ALSO:  load, UFget
		"""
		self.e.save(fname)
		return

	def nedge(self, vpart=None):
		"""
		returns the number of edges in (each partition of the vertices
		of) the DiGraph instance, including edges with zero weight.

		Input Arguments:
			self:  a DiGraph instance
			vpart: an optional argument; a ParVec instance denoting a 
			    partition of the vertices of the graph.  The value of
			    each element of vpart denotes the partition in which
			    the corresponding vertex resides.  Note that this is
			    the same format as the return value from cluster().

		Output Arguments:
			ret:  the number of edges in the DiGraph (if vpart not
			    specified) or a ParVec instance with each element
			    denoting the number of edges in the corresponding
			    partition (if vpart specified).

		SEE ALSO:  npart, contract
		"""
		nv = self.nvert()
		if vpart is None:
			return self.e.nnn()
		else:
			if self.nvert() != len(vpart):
				raise KeyError,'vpart must be same length as number of vertices in DiGraph instance'
			retLen = int(vpart.max())+1
		if self.nvert() == 0:
			if retLen == 1:
				return 0
			else:
				return Vec.zeros(retLen)
		if retLen == 1:
			ret = self.e.nnn()
		else:
			selfcopy = self.e.copy()
			selfcopy.spOnes(1)
			C = Mat(vpart,Vec.range(nv), Vec.ones(nv), retLen, nv)
			tmpMat = C.SpGEMM(selfcopy, sr_plustimes)
			C.transpose()
			tmpMat = tmpMat.SpGEMM(C, sr_plustimes)
			#HACK!!
			#ret = tmpMat[diag]
			ret = Vec(retLen)
			for i in range(retLen):
				ret[i] = int(tmpMat[i].sum()[0])
		return ret
			

	# NEEDED: tests
	#FIX:  good idea to have this return an int or a tuple?
	def nvert(self, vpart=None):
		"""
		ToDo:  fix docstring for vertex-partition angle

		returns the number of vertices in the given DiGraph instance.

		Input Argument:
			self:  a DiGraph instance.

		Output Argument:
			ret:  if the DiGraph was created with the same number of
			    vertices potentially having in- and out-edges, the
			    return value is a scalar integer of that number.  If
			    the DiGraph was created with different numbers of
			    vertices potentially having in- and out-edges, the
			    return value is a tuple of length 2, with the first
			    (second) element being the number of vertices potentially 
			    having out-(in-)edges.

		SEE ALSO:  nedge, degree
		"""
		nrow = self.e.nrow()
		ncol = self.e.ncol()
		if nrow!=ncol:
			return (nrow, ncol)
		if vpart is None:
			return nrow
		if len(vpart) != nrow:
			raise KeyError, 'vertex partition length not equal to number of vertices'
		ret = vpart.hist()
		return ret


	#in-place, so no return value
	def spOnes(self):
		"""
		sets every edge in the graph to the value 1.

		Input Argument:
			self:  a DiGraph instance, modified in place.

		Output Argument:
			None.

		SEE ALSO:  set
		"""
		self.e.spOnes()

	#in-place, so no return value
	def reverseEdges(self):
		"""
		reverses the direction of each edge of a DiGraph instance in-place,
		switching its source and destination vertices.

		Input Argument:
			self:  a DiGraph instance, modified in-place.
		"""
		self.e.transpose()

	# NEEDED: update to new fields
	# NEEDED: tests
	#in-place, so no return value
	def _set(self, value):
		"""
		sets every edge in the graph to the given value.

		Input Arguments:
			self:  a DiGraph instance, modified in place.
			value:  a scalar integer or double-precision floating-
			    point value.

		Output Argument:
			None.

		SEE ALSO:  spOnes
		"""
		if isinstance(self._identity_, (float, int, long)):
			self._apply(pcb.set(value))
		else:
			raise NotImplementedError, 'not for Obj DiGraphs yet' 
		return

	def subgraph(self, ndx1=None, ndx2=None, mask=None):
		"""
		creates a new DiGraph instance consisting of only designated vertices 
		of the input graph and their indicent edges.

		Input Arguments:
			self:  a DiGraph instance
			ndx1:  a Vec of consecutive vertex
			    numbers to be included in the subgraph along with edges
			    starting from these vertices.
			ndx2:  an optional argument; if specified, a Vec of consecutive
			    vertex numbers to be included in the subgraph along with
			    any edges ending at these vertices.
			mask:  a length nverts Vec with True elements for vertices
			    that should be kept and False elements for vertices to
			    be discarded.
			 
		Output Argument:
			ret:  a DiGraph instance composed of the selected vertices
			    and their incident edges.

		SEE ALSO:  DiGraph.__getitem__
		"""
		if ndx1 is None and mask is None:
			raise KeyError, 'Either indices or a mask must be provided'
		
		if ndx1 is None:
			# convert mask to indices
			ndx1 = mask.findInds(lambda x: x == 1)
			ndx2 = None
		
		if ndx2 is None:
			ndx2 = ndx1
		retE = self.e[ndx1, ndx2]
		return DiGraph(edges=retE, vertices=self.v.copy())

	# NEEDED: modify to make sense in graph context, not just matrix context
	# NEEDED: tests
	def _sum(self, dir=Out):
		"""
		adds the weights of the appropriate edges of each vertex of the
		passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges to sum, with choices being
			    DiGraph.Out (default) or DiGraph.In.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    sum of the weights of the corresponding vertex.

		SEE ALSO:  degree 
		"""
		if dir != DiGraph.In and dir != DiGraph.Out:
			raise KeyError, 'Invalid edge-direction'
		ret = self.e.reduce(dir, op_add)
		return ret

	# NEEDED: tests
	def toBool(self):
		"""
		converts the DiGraph instance in-place such that each edge has only
		a Boolean (True) value, thereby consuming less space and making
		some operations faster.

		Input Argument:
			self:  a DiGraph instance that is overwritten by the method

		Output Argument:
			None.
		"""
		self.e.toBool()
		self._identity_ = True

	# NEEDED: tests
	# old name: def _toDiGraph(self, pcbMat=0):
	def _toScalar(self):
		"""
		converts a DiGraph whose element is an ObjX to an element
		of a 64-bit container.  Currently not customizable with how
		the conversion is done; value of 1 is used.
		"""

		self.e.toScalar()
