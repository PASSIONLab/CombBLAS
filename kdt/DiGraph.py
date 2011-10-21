import math
import Graph as gr
from Graph import master
from Vec import Vec, DeVec, SpVec
from Mat import Mat
import kdt.pyCombBLAS as pcb

import time

class DiGraph(gr.Graph):

	# NEEDED: Reverse these so we don't have to do a transpose all the time
	In  = Mat.Column
	Out = Mat.Row

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
		if matrix is not None:
			self.e = matrix
		else:
			self.e = Mat(sourceV=sourceV, destV=destV, valueV=valueV, nv=nv, element=element)
		
		if vertices is not None:
			self.v = vertices
		else:
			self.v = Vec(length=self.nvert(), element=0.0)

#	# This has to add both edges and vectors
#	def __add__(self, other):


	# NEEDED: tests
	def __repr__(self):
		return e.__repr__()
	

	#in-place, so no return value
	# NEEDED: modify to make sense in graph context, not just matrix context
	# NEEDED: update to new fields
	# NEEDED: tests
	def apply(self, op, other=None, notB=False):
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
				self._m_.Apply(pcb.unary(op))
			else:
				self._m_.Apply(op)
			return
		else:
			if not isinstance(op, pcb.BinaryFunction):
				self._m_ = pcb.EWiseApply(self._m_, other._m_, pcb.binary(op), notB)
			else:
				self._m_ = pcb.EWiseApply(self._m_, other._m_, op, notB)
			return

	# NEEDED: modify to make sense in graph context, not just matrix context
	# NEEDED: update to new fields
	# NEEDED: tests
	def eWiseApply(self, other, op, allowANulls, allowBNulls, noWrap=False):
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
				m = pcb.EWiseApply(self._m_, other   ,  superOp)
			else:
				m = pcb.EWiseApply(self._m_, other._m_, superOp)
		else:
			if isinstance(other, (float, int, long)):
				m = pcb.EWiseApply(self._m_, other   ,  pcb.binaryObj(superOp))
			else:
				m = pcb.EWiseApply(self._m_, other._m_, pcb.binaryObj(superOp))
		ret = self._toDiGraph(m)
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	@staticmethod
	def _hasFilter(self):
		try:
			ret = (hasattr(self,'_eFilter_') and len(self._eFilter_)>0) # ToDo: or (hasattr(self,'vAttrib') and self.vAttrib._hasFilter(self.vAttrib)) 
		except AttributeError:
			ret = False
		return ret

	@staticmethod
	# NEEDED: update to new fields
	# NEEDED: tests
	def isObj(self):
		return not isinstance(self._identity_, (float, int, long, bool))
		#try:
		#	ret = hasattr(self,'_elementIsObject') and self._elementIsObject
		#except AttributeError:
		#	ret = False
		#return ret

	#FIX:  put in a common place
	op_add = pcb.plus()
	op_sub = pcb.minus()
	op_mul = pcb.multiplies()
	op_div = pcb.divides()
	op_mod = pcb.modulus()
	op_fmod = pcb.fmod()
	op_pow = pcb.pow()
	op_max  = pcb.max()
	op_min = pcb.min()
	op_bitAnd = pcb.bitwise_and()
	op_bitOr = pcb.bitwise_or()
	op_bitXor = pcb.bitwise_xor()
	op_and = pcb.logical_and()
	op_or = pcb.logical_or()
	op_xor = pcb.logical_xor()
	op_eq = pcb.equal_to()
	op_ne = pcb.not_equal_to()
	op_gt = pcb.greater()
	op_lt = pcb.less()
	op_ge = pcb.greater_equal()
	op_le = pcb.less_equal()

	# in-place, so no return value
	# NEEDED: update to new fields
	# NEEDED: tests
	def addEFilter(self, filter):
		"""
		adds a vertex filter to the DiGraph instance.  

		A vertex filter is a Python function that is applied elementally
		to each vertex in the DiGraph, with a Boolean True return value
		causing the vertex to be considered and a False return value
		causing it not to be considered.

		Vertex filters are additive, in that each vertex must pass all
		filters to be considered.  All vertex filters are executed before
		a vertex is considered in a computation.
#FIX:  how is an argument passed to the function?

		Input Arguments:
			self:  a DiGraph instance
			filter:  a Python function

		SEE ALSO:
			delEFilter  
		"""
		if hasattr(self, '_eFilter_'):
			self._eFilter_.append(filter)
		else:
			self._eFilter_ = [filter]
		return
		
	# NEEDED: update to new fields
	# NEEDED: tests
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
			groups:  a ParVec denoting into which group (result
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
		origVtx = ParVec.range(n)
		# lhrMat == left-/right-hand-side matrix
		lrhMat = DiGraph(groups, origVtx, ParVec.ones(n), nvRes, n)
		tmpMat = lrhMat._SpGEMM(self)
		lrhMat._T()
		res = tmpMat._SpGEMM(lrhMat)
		return res;
	
	# NEEDED: update to new fields
	# NEEDED: tests
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
			ret:  a vertex of the same length as clusterParents where the elements
			    are in the range [0,k) and correspond to cluster group numbers.
			    If retInvPerm is True, the return is a tuple which also contains
			    a permutation going from group number to original parent vertex.
		"""
		
		n = len(clusterParents)
		
		# Count the number of elements in each parent's component to identify the parents
		countM = DiGraph(clusterParents, ParVec.range(n), ParVec.ones(n), n)
		counts = countM._spm.Reduce(pcb.pySpParMat.Row(), pcb.plus())
		del countM
		
		# sort to put them all at the front
		sorted = counts.copy()
		sorted.Apply(pcb.negate())
		perm = sorted.Sort()
		sorted.Apply(pcb.negate())
		
		# find inverse of sort permutation so that [1,2,3,4...] can be put back into the
		# original parent locations
		invM = DiGraph(ParVec.toParVec(perm), ParVec.range(n), ParVec.ones(n), n)
		invPerm = invM._spm.SpMV(SpParVec.range(n)._spv, pcb.TimesPlusSemiring()).dense()
		del invM
		
		# Find group number for each parent vertex
		groupNum = invPerm
		
		# Broadcast group number to all vertices in cluster
		broadcastM = DiGraph(ParVec.range(n), clusterParents, ParVec.ones(n), n)
		ret = broadcastM._spm.SpMV(groupNum.sparse(), pcb.TimesPlusSemiring()).dense()
		
		if retInvPerm:
			return ParVec.toParVec(ret), ParVec.toParVec(perm)
		else:
			return ParVec.toParVec(ret)
	
	# NEEDED: update to new fields
	# NEEDED: tests
	@staticmethod
	def convMaskToIndices(mask):
		verts = mask._dpv.FindInds(pcb.bind2nd(pcb.equal_to(), 1))
		return ParVec.toParVec(verts)

	# NEEDED: update to new fields
	# NEEDED: tests
	def copy(self, element=None):
		"""
		creates a deep copy of a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance.

		Output Argument:
			ret:  a DiGraph instance containing a copy of the input.
		"""
		ret = DiGraph(element=self._identity_)
		ret._m_ = self._m_.copy()
		if hasattr(self,'_eFilter_'):
			if type(self.nvert()) is tuple:
				raise NotImplementedError, 'only square DiGraphs for now'
			class tmpU:
				_eFilter_ = self._eFilter_
				@staticmethod
				def fn(x):
					for i in range(len(tmpU._eFilter_)):
						if not tmpU._eFilter_[i](x):
							return type(self._identity_)()
					return x
			tmpInstance = tmpU()
			ret._m_.Apply(pcb.unaryObj(tmpInstance.fn))
			ret._m_.Prune(pcb.unaryObjPred(lambda x: x.prune()))
		if element is not None and type(self._identity_) is not type(element):
			if not isinstance(element, (float, int, long)):
				# because EWiseApply(pySpParMat,pySpParMatObj)
				#   applies only where the first argument has
				#   non-nulls;  the only way I know to avoid
				#   is to use the result of 
				#   pySpParMat(pySpParMatObj), which
				#   only works for converting to doubleints
				raise NotImplementedError, 'can only convert to long for now'
			tmp = DiGraph(None,None,None,self.nvert(),element=element)
			# FIX: remove following 2 lines when EWiseApply works 
			#   as noted above 
			tmpMat = pcb.pySpParMat(self._m_)
			tmp._m_ = tmpMat
			def func(x, y): 
				#ToDo:  assumes that at least x or y is an ObjX
				if isinstance(x,(float,int,long)):
					ret = y.coerce(x, False)
				else:
					ret = x.coerce(y, True)
				return ret
			tmp2 = tmp._eWiseApply(ret, func, True, True)
			ret = tmp2
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	def degree(self, dir=Out):
		"""
		calculates the degrees of the appropriate edges of each vertex of 
		the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges to count, with choices being
			    DiGraph.Out (default) or DiGraph.In.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    degree of the weights of the corresponding vertex.

		SEE ALSO:  sum 
		"""
		if dir != DiGraph.In and dir != DiGraph.Out:
			raise KeyError, 'Invalid edge-direction'
		if isinstance(self._identity_, (float, int, long)):
			ret = self._reduce(dir, pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
		else:
			tmp = self._reduce(dir, pcb.binaryObj(lambda x,y: x.count(y)))
			ret = tmp.copy(element=0)
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	# in-place, so no return value
	def delEFilter(self, filter=None):
		"""
		deletes a vertex filter from the DiGraph instance.  

		Input Arguments:
			self:  a DiGraph instance
			filter:  a Python function, which can be either a function
			    previously added to this DiGraph instance by a call to
			    addEFilter or None, which signals the deletion of all
			    vertex filters.

		SEE ALSO:
			addEFilter  
		"""
		if not hasattr(self, '_eFilter_'):
			raise KeyError, "no edge filters previously created"
		if filter is None:
			del self._eFilter_	# remove all filters
		else:
			self._eFilter_.remove(filter)
		return

	# NEEDED: update to new fields
	# NEEDED: tests
	# in-place, so no return value
	def removeSelfLoops(self):
		"""
		removes all edges whose source and destination are the same
		vertex, in-place in a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance, modified in-place.

		"""
		if self.nvert() > 0:
			self._spm.removeSelfLoops()
		return

	# NEEDED: update to new fields
	# NEEDED: tests
	def addSelfLoops(self, selfLoopAttr=1):
		"""
		removes all edges whose source and destination are the same
		vertex, in-place in a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance, modified in-place.
			selfLoopAttr: the value to put on each self loop

		"""
		if self.nvert() > 0:
			self += DiGraph.eye(self.nvert(), selfLoopAttr=selfLoopAttr)
		return

	# NEEDED: update to new fields
	# NEEDED: tests
	@staticmethod
	def fullyConnected(n,m=None):
		"""
		creates edges in a DiGraph instance that connects each vertex
		directly to every other vertex.

		Input Arguments:
			n:  an integer scalar denoting the number of vertices in
			    the graph that may potentially have out-edges.
			m:  an optional argument, which if specified is an integer
			    scalar denoting the number of vertices in the graph
			    that may potentially have in-edges.

		Output Argument:
			ret:  a DiGraph instance with directed edges from each
			    vertex to every other vertex. 
		"""
		if m is None:
			m = n
		i = (ParVec.range(n*m) % n).floor()
		j = (ParVec.range(n*m) / n).floor()
		v = ParVec.ones(n*m)
		ret = DiGraph(i,j,v,n,m)
		return ret
	
	# NEEDED: update to new fields
	# NEEDED: tests
	@staticmethod
	def eye(n, selfLoopAttr=1):
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
		return DiGraph(ParVec.range(n),ParVec.range(n),ParVec(n, selfLoopAttr),n,n)

	# NEEDED: update to new fields
	# NEEDED: tests
	def genGraph500Edges(self, scale):
		"""
		creates edges in a DiGraph instance that meet the Graph500 
		specification.  The graph is symmetric. (See www.graph500.org 
		for details.)

		Input Arguments:
			self:  a DiGraph instance, usually with no edges
			scale:  an integer scalar representing the logarithm base
			    2 of the number of vertices in the resulting DiGraph.
			    
		Output Argument:
			ret:  a double-precision floating-point scalar denoting
			    the amount of time to converted the created edges into
			    the DiGraph instance.  This equals the value of Kernel 1
			    of the Graph500 benchmark. Timing the entire genGraph500Edges
			    call would also time the edge generation, which is not part
			    of Kernel 1.
			    Degrees of all vertices.
		"""
		degrees = pcb.pyDenseParVec(1, 1)
		elapsedTime = self._spm.GenGraph500Edges(scale, degrees)
	 	return (elapsedTime, ParVec.toParVec(degrees))

	# NEEDED: tests
	@staticmethod
	def load(fname):
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
		mat = Mat.load(fname)
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

	# NEEDED: update to new fields
	# NEEDED: tests
	def max(self, dir=Out):
		"""
		finds the maximum weights of the appropriate edges of each vertex 
		of the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges over which to find the maximum,
			    with choices being DiGraph.Out (default) or DiGraph.In.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    maximum of the weights of the corresponding vertex.

		SEE ALSO:  degree, min 
		"""
		if dir != DiGraph.In and dir != DiGraph.Out:
			raise KeyError, 'Invalid edge-direction'
		ret = self._reduce(dir, pcb.max())
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	def min(self, dir=Out):
		"""
		finds the minimum weights of the appropriate edges of each vertex 
		of the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges over which to find the minimum,
			    with choices being DiGraph.Out (default), DiGraph.In.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    minimum of the weights of the corresponding vertex.

		SEE ALSO:  degree, max 
		"""
		if dir != DiGraph.In and dir != DiGraph.Out:
			raise KeyError, 'Invalid edge-direction'
		ret = self._reduce(dir, pcb.min())
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	def mulNot(self, other):
		"""
		multiplies corresponding edge weights of two DiGraph instances,
		taking the logical not of the second argument before doing the 
		multiplication.  In effect, each nonzero edge of the second
		argument deletes its corresponding edge of the first argument.

		Input Arguments:
			self:  a DiGraph instance
			other:  another DiGraph instance

		Output arguments:
			ret:  a DiGraph instance 
		"""
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		else:
			ret = self.copy()
			ret._apply(pcb.multiplies(), other, True)
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
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
			return self._m_.getnee()
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
			ret = self._m_.getnee()
		else:
			selfcopy = self.copy()
			selfcopy.set(1)
			C = DiGraph(vpart,Vec.range(nv), Vec.ones(nv), retLen, nv)
			tmpMat = C._SpGEMM(selfcopy)
			C._T()
			tmpMat = tmpMat._SpGEMM(C)
			#HACK!!
			#ret = tmpMat[diag]
			ret = Vec(retLen)
			for i in range(retLen):
				ret[i] = int(tmpMat[i].sum()[0])
		return ret
			

	# NEEDED: update to new fields
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
		nrow = self._m_.getnrow()
		ncol = self._m_.getncol()
		if nrow!=ncol:
			return (nrow, ncol)
		if vpart is None:
			return nrow
		if len(vpart) != nrow:
			raise KeyError, 'vertex partition length not equal to number of vertices'
		ret = vpart.hist()
		return ret


	##in-place, so no return value
	#def ones(self):
	#	"""
	#	sets every edge in the graph to the value 1.

	#	Input Argument:
	#		self:  a DiGraph instance, modified in place.

	#	Output Argument:
	#		None.

	#	SEE ALSO:  set
	#	"""
	#	self._spm.Apply(pcb.set(1))
	#	return

	# NEEDED: tests
	#in-place, so no return value
	def reverseEdges(self):
		"""
		reverses the direction of each edge of a DiGraph instance in-place,
		switching its source and destination vertices.

		Input Argument:
			self:  a DiGraph instance, modified in-place.
		"""
		self.e.Transpose()

	# NEEDED: update to new fields
	# NEEDED: tests
	#in-place, so no return value
	def scale(self, other, dir=Out):
		"""
		multiplies the weights of the appropriate edges of each vertex of
		the passed DiGraph instance in-place by a vertex-specific scale 
		factor.

		Input Arguments:
			self:  a DiGraph instance, modified in-place
			other: a ParVec whose elements are used
			dir:  a direction of edges to scale, with choices being
			    DiGraph.Out (default) or DiGraph.In.

		Output Argument:
			None.

		SEE ALSO:  * (DiGraph.__mul__), mulNot
		"""
		if not isinstance(other,gr.ParVec):
			raise KeyError, 'Invalid type for scale vector'
		selfnv = self.nvert()
		if type(selfnv) == tuple:
			[selfnv1, selfnv2] = selfnv
		else:
			selfnv1 = selfnv; selfnv2 = selfnv
		if dir == DiGraph.In:
			if selfnv2 != len(other):
				raise IndexError, 'graph.nvert()[1] != len(scale)'
			self._spm.DimWiseApply(pcb.pySpParMat.Column(), other._dpv, pcb.multiplies())
		elif dir == DiGraph.Out:
			if selfnv1 != len(other):
				raise IndexError, 'graph.nvert()[0] != len(scale)'
			self._spm.DimWiseApply(pcb.pySpParMat.Row(), other._dpv, pcb.multiplies())
		else:
			raise KeyError, 'Invalid edge direction'
		return

	# NEEDED: update to new fields
	# NEEDED: tests
	#in-place, so no return value
	def set(self, value):
		"""
		sets every edge in the graph to the given value.

		Input Arguments:
			self:  a DiGraph instance, modified in place.
			value:  a scalar integer or double-precision floating-
			    point value.

		Output Argument:
			None.

		SEE ALSO:  ones
		"""
		if isinstance(self._identity_, (float, int, long)):
			self._apply(pcb.set(value))
		else:
			raise NotImplementedError, 'not for Obj DiGraphs yet' 
		return

	# NEEDED: update to new fields
	# NEEDED: tests
	def subgraph(self, ndx1=None, ndx2=None, mask=None):
		"""
		creates a new DiGraph instance consisting of only designated vertices 
		of the input graph and their indicent edges.

		Input Arguments:
			self:  a DiGraph instance
			ndx1:  an integer scalar or a ParVec of consecutive vertex
			    numbers to be included in the subgraph along with edges
			    starting from these vertices.
			ndx2:  an optional argument; if specified, is an integer
			    scalar or a ParVec of consecutive vertex numbers to
			    be included in the subgraph along with any edges ending
			    at these vertices.
			mask:  a length nverts ParVec with True elements for vertices
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
			verts = mask._dpv.FindInds(pcb.bind2nd(pcb.equal_to(), 1))
			ndx1 = ParVec.toParVec(verts)
			ndx2 = None
		
		if ndx2 is None:
			ndx2 = ndx1
		ret = self[ndx1, ndx2]
		return ret

	# NEEDED: modify to make sense in graph context, not just matrix context
	# NEEDED: update to new fields
	# NEEDED: tests
	def sum(self, dir=Out):
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
		ret = self._reduce(dir, pcb.plus())
		return ret

	# ADAM: removed on purpose to point out places that need to be updated with reversed edges.
	# NEEDED: delete entirely
	#_T = reverseEdges

	# NEEDED: update to new fields
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
		if hasattr(self,'_m_'):
			ret = DiGraph(nv=self.nvert(),element=False)
			tmpM = self._m_ 	# shallow copy
			if self.isObj(self):
				tmpM = pcb.pySpParMat(tmpM)
			tmpM = pcb.pySpParMatBool(tmpM)
			ret._m_ = tmpM
			return ret
		else:
			raise TypeError, 'DiGraph has no contents'

	# NEEDED: Use better name. DiGraphs can have ObjX fields as well.
	# NEEDED: update to new fields
	# NEEDED: tests
	def _toDiGraph(self, pcbMat=0):
		"""
		converts a DiGraph whose element is an ObjX to an element
		of a 64-bit container.  Currently not customizable with how
		the conversion is done; the value of the weight is used.
		"""
		#ToDo:  currently assumes but does not check that the
		#   output element-type is float/int/long

		#if not isinstance(self._identity_, (pcb.Obj1, pcb.Obj2)):
		#	raise NotImplementedError, 'source must be Obj'
		#if not isinstance(pcbMat[0,0], (float, int, long)):
		#	raise NotImplementedError, 'result must be 64-bit element'
		if hasattr(self,'_m_'):
			ret = DiGraph(nv=self.nvert(),element=0)
			tmpM = pcbMat 	# shallow copy
			if self.isObj(self):
				pcbMat = pcb.pySpParMat(pcbMat)
			ret._m_ = pcbMat
			ret._identity_ = 0
			return ret
		else:
			raise TypeError, 'DiGraph has no contents'

	# NEEDED: update to new fields
	# NEEDED: tests
	def toVec(self):
		"""
		decomposes a DiGraph instance to 3 ParVec instances, with each
		element of the first ParVec denoting the source vertex of an edge,
		the corresponding element of the second ParVec denoting the 
		destination vertex of the edge, and the corresponding element of
		the third ParVec denoting the value or weight of the edge.

		Input Argument:
			self:  a DiGraph instance

		Output Argument:
			ret:  a 3-element tuple with ParVec instances denoting the
			    source vertex, destination vertex, and weight, respectively.

		SEE ALSO:  DiGraph 
		"""
		ne = self.nedge()
		if ne != 0:
			reti = Vec(ne)
			retj = Vec(ne)
			retv = Vec(ne, element=self._identity_)
			self._m_.Find(reti._v_, retj._v_, retv._v_)
		else:
			reti = Vec(0)
			retj = Vec(0)
			retv = Vec(0)
		#ToDo:  return nvert() of original graph, too
		return (reti, retj, retv)

	# NEEDED: update to new fields
	# NEEDED: tests
	@staticmethod
	def twoDTorus(n):
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
		N = n*n
		nvec =   ((ParVec.range(N*4)%N) / n).floor()	 # [0,0,0,...., n-1,n-1,n-1]
		nvecil = ((ParVec.range(N*4)%N) % n).floor()	 # [0,1,...,n-1,0,1,...,n-2,n-1]
		north = gr.Graph._sub2ind((n,n),(nvecil-1) % n,nvec)	
		south = gr.Graph._sub2ind((n,n),(nvecil+1) % n,nvec)
		west = gr.Graph._sub2ind((n,n),nvecil, (nvec-1) % n)
		east = gr.Graph._sub2ind((n,n),nvecil, (nvec+1) % n)
		Ndx = ParVec.range(N*4)
		northNdx = Ndx < N
		southNdx = (Ndx >= N) & (Ndx < 2*N)
		westNdx = (Ndx >= 2*N) & (Ndx < 3*N)
		eastNdx = Ndx >= 3*N
		col = ParVec.zeros(N*4)
		col[northNdx] = north
		col[southNdx] = south
		col[westNdx] = west
		col[eastNdx] = east
		row = ParVec.range(N*4) % N
		ret = DiGraph(row, col, 1, N)
		return ret

