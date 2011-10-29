import math
from Graph import master
from Vec import Vec
from Util import *
from Util import _op_make_unary
from Util import _op_make_unary_pred
from Util import _op_make_binary
from Util import _op_make_binary_pred

import kdt.pyCombBLAS as pcb

import time

class Mat:
	Column  = pcb.pySpParMat.Column()
	Row = pcb.pySpParMat.Row()

	# NOTE:  for any vertex, out-edges are in the column and in-edges
	#	are in the row
	def __init__(self, sourceV=None, destV=None, valueV=None, nv=None, element=0):
		"""
		FIX:  doc
		creates a new Mat instance.  Can be called in one of the 
		following forms:

	Mat():  creates an empty Mat instance with elements.  Useful as input for genGraph500Edges.

	Mat(sourceV, destV, weightV, n)
	Mat(sourceV, destV, weightV, n, m)
		create a Mat Instance with edges with source represented by 
		each element of sourceV and destination represented by each 
		element of destV with weight represented by each element of 
		weightV.  In the 4-argument form, the resulting Mat will 
		have n out- and in-vertices.  In the 5-argument form, the 
		resulting Mat will have n out-vertices and m in-vertices.

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
			ret:  a Mat instance

		Note:  If two or more edges have the same source and destination
		vertices, their weights are summed in the output Mat instance.

		SEE ALSO:  toParVec
	def __init__(self, sourceV=None, destV=None, valueV=None, nv=None, element=0):
		"""
		if sourceV is None:
			if nv is not None: #create a Mat with an underlying pySpParMat* of the right size with no nonnulls
				nullVec = pcb.pyDenseParVec(0,0)
			if isinstance(element, (float, int, long)):
				if nv is None:
					self._m_ = pcb.pySpParMat()
				else:
					self._m_ = pcb.pySpParMat(nv,nv,nullVec,nullVec, nullVec)
			elif isinstance(element, bool):
				if nv is None:
					self._m_ = pcb.pySpParMatBool()
				else:
					self._m_ = pcb.pySpParMatBool(nv,nv,nullVec,nullVec, nullVec)
			elif isinstance(element, pcb.Obj1):
				if nv is None:
					self._m_ = pcb.pySpParMatObj1()
				else:
					self._m_ = pcb.pySpParMatObj1(nv,nv,nullVec,nullVec, nullVec)
			elif isinstance(element, pcb.Obj2):
				if nv is None:
					self._m_ = pcb.pySpParMatObj2()
				else:
					self._m_ = pcb.pySpParMatObj2(nv,nv,nullVec,nullVec, nullVec)
			self._identity_ = element
		elif sourceV is not None and destV is not None and nv is not None:
			i = sourceV
			j = destV
			v = valueV
			if type(v) == tuple and isinstance(element,(float,int,long)):
				raise NotImplementedError, 'tuple valueV only valid for Obj element'
			if len(i) != len(j):
				raise KeyError, 'source and destination vectors must be same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = Vec(len(i), v)
#			if i.max() > nv-1:
#				raise KeyError, 'at least one first index greater than #vertices'
#			if j.max() > nv-1:
#				raise KeyError, 'at least one second index greater than #vertices'
			
			if i.isObj() or j.isObj():
				raise ValueError, "sourceV and destV cannot be objects!"
			if i.isSparse() or j.isSparse():
				raise ValueError, "sourceV and destV cannot be sparse!"
			
			if isinstance(v, Vec):
				element = v._identity_
				if v.isSparse():
					raise ValueError, "valueV cannot be sparse!"
			else:
				element = v
			
			if isinstance(element, (float, int, long)):
				self._m_ = pcb.pySpParMat(nv,nv,i._v_,j._v_,v._v_)
			elif isinstance(element, bool):
				self._m_ = pcb.pySpParMatBool(nv,nv,i._v_,j._v_,v._v_)
			elif isinstance(element, pcb.Obj1):
				self._m_ = pcb.pySpParMatObj1(nv,nv,i._v_,j._v_,v._v_)
			elif isinstance(element, pcb.Obj2):
				self._m_ = pcb.pySpParMatObj2(nv,nv,i._v_,j._v_,v._v_)
			self._identity_ = Mat._getExampleElement(self._m_)
		else:
			raise ValueError, "Incomplete arguments to Mat()"
	
	@staticmethod
	def _getExampleElement(pcbMat):
		if isinstance(pcbMat, pcb.pySpParMat):
			return 0.0
		if isinstance(pcbMat, pcb.pySpParMatBool):
			return True
		if isinstance(pcbMat, pcb.pySpParMatObj1):
			return pcb.Obj1()
		if isinstance(pcbMat, pcb.pySpParMatObj2):
			return pcb.Obj2()
		raise NotImplementedError, 'Unknown vector type!'

	@staticmethod
	def _toMat(pcbMat):
		ret = Mat()
		ret._m_ = pcbMat
		ret._identity_ = Mat._getExampleElement(pcbMat)
		return ret

	def copy(self, element=None):
		"""
		creates a deep copy of a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance.

		Output Argument:
			ret:  a DiGraph instance containing a copy of the input.
		"""

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
			ret = Mat()
			ret._m_.Apply(pcb.unaryObj(tmpInstance.fn))
			ret._m_.Prune(pcb.unaryObjPred(lambda x: x.prune()))
		else:
			ret = Mat._toMat(self._m_.copy())
		
		# TODO: integrate filter/copy and element conversion
		# so they are not a separate steps that make two copies of the matrix.
		if element is not None and type(self._identity_) is not type(element):
			#if not isinstance(element, (bool, float, int, long)):
				# because EWiseApply(pySpParMat,pySpParMatObj)
				#   applies only where the first argument has
				#   non-nulls;  the only way I know to avoid
				#   is to use the result of 
				#   pySpParMat(pySpParMatObj), which
				#   only works for converting to doubleints
			#	raise NotImplementedError, 'can only convert to long for now'
			
			if isinstance(element, bool):
				ret = Mat._toMat(pcb.pySpParMatBool(ret._m_))
			elif isinstance(element, (float, int, long)):		
				# FIX: remove following 2 lines when EWiseApply works 
				#   as noted above 
				ret = Mat._toMat(pcb.pySpParMat(ret._m_))
				
				if self.isObj():
					def func(x, y): 
						#ToDo:  assumes that at least x or y is an ObjX
						if isinstance(x,(float,int,long)):
							ret = y.coerce(x, False)
						else:
							ret = x.coerce(y, True)
						return ret
					ret = ret.eWiseApply(ret, func, True, True)
			else: # object matrix
				raise NotImplementedError, 'can only convert to long for now'
				
		return ret
	
	def toBool(self, inPlace=True):
		"""
		converts the Mat instance in-place such that each element only has
		a Boolean (True) value, thereby consuming less space and making
		some operations faster.

		Input Argument:
			self:  a Mat instance that is overwritten by the method

		Output Argument:
			None.
		"""
		
		if inPlace:
			if not self.isBool():
				self._m_ = pcb.pySpParMatBool(self._m_)
				self._identity_ = Mat._getExampleElement(self._m_)
		else:
			return Mat._toMat(pcb.pySpParMatBool(self._m_))

	def toScalar(self, inPlace=True):
		"""
		converts the Mat instance in-place such that each element only has
		a scalar (64-bit) value.

		Input Argument:
			self:  a Mat instance that is overwritten by the method

		Output Argument:
			None.
		"""
		
		if inPlace:
			if not self.isScalar():
				self._m_ = pcb.pySpParMat(self._m_)
				self._identity_ = Mat._getExampleElement(self._m_)
		else:
			return Mat._toMat(pcb.pySpParMat(self._m_))

	def getnrow(self):
		return self._m_.getnrow()
		
	def getncol(self):
		return self._m_.getncol()
	
	def getnee(self):
		"""
		returns the number of existing elements in this matrix.
		"""
		if self._hasFilter():
			raise NotImplementedError, "this operation does not support filters yet."
			
		return self._m_.getnee()
	
	@staticmethod
	def generateRMAT(scale, edgeFactor=16, initiator=[.57, .19, .19, .05], delIsolated=True, boolean=True):
		"""
		generates a Kroenecker product matrix using the Graph500 RMAT graph generator.
		
		Output Argument:
			ret: a tuple: Mat, Vec, time.
				Mat is the matrix itself,
				Vec is a vector containing the degrees of each vertex in the original graph
				time is the time for the Graph500 Kernel 1.
		"""
		degrees = Vec(0, element=1.0, sparse=False)
		if boolean:
			matrix = pcb.pySpParMatBool()
		else:
			matrix = pcb.pySpParMat()
		kernel1Time = matrix.GenGraph500Edges(scale, degrees._v_, scale, delIsolated, initiator[0], initiator[1], initiator[2], initiator[3])
		return Mat._toMat(matrix), degrees, kernel1Time

	@staticmethod
	def eye(n, element=1.0):
		"""
		creates an identity matrix. The resulting matrix is n-by-n
		with `element` on the main diagonal.
		In other words, ret[i,i] = element for 0 < i < n.

		Input Arguments:
			n:  an integer scalar denoting the dimensions of the matrix
			element: the value to put on the main diagonal elements.

		Output Argument:
			ret:  an identity matrix. 
		"""
		return Mat(Vec.range(n),Vec.range(n),Vec(n, element),n)
	
	def removeMainDiagonal(self):
		"""
		removes all elements whose row and column index are equal.
		Operation is in-place.

		Input Argument:
			self:  a Mat instance, modified in-place.

		"""
		self._m_.removeSelfLoops()


	# NEEDED: update to new fields
	# NEEDED: tests
	def __add__(self, other):
		"""
		adds corresponding edges of two Mat instances together,
		resulting in edges in the result only where an edge exists in at
		least one of the input Mat instances.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			raise NotImplementedError
		elif self.getnrow() != other.getnrow() or self.getncol() != other.getncol():
			raise IndexError, 'Matrices must have matching dimensions'
		elif isinstance(other, Mat):
			ret = self.copy()
			ret._m_ += other._spm
			#ret._apply(pcb.plus(), other);  # only adds if both mats have nonnull elems!!
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	def __div__(self, other):
		"""
		divides corresponding edges of two Mat instances together,
		resulting in edges in the result only where edges exist in both
		input Mat instances.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.apply(pcb.bind2nd(pcb.divides(),other))
		elif self.getnrow() != other.getnrow() or self.getncol() != other.getncol():
			raise IndexError, 'Matrices must have matching dimensions'
		elif isinstance(other,Mat):
			ret = self.copy()
			ret.apply(pcb.divides(), other)
		else:
			raise NotImplementedError
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	def __getitem__(self, key):
		"""
		FIX:  fix documentation

		implements indexing on the right-hand side of an assignment.
		Usually accessed through the "[]" syntax.

		Input Arguments:
			self:  a Mat instance
			key:  one of the following forms:
			    - a non-tuple denoting the key for both dimensions
			    - a tuple of length 2, with the first element denoting
			        the key for the first dimension and the second 
			        element denoting for the second dimension.
			    Each key denotes the out-/in-vertices to be addressed,
			    and may be one of the following:
				- an integer scalar
				- the ":" slice denoting all vertices, represented
				  as slice(None,None,None)
				- a ParVec object containing a contiguous range
				  of monotonically increasing integers 
		
		Output Argument:
			ret:  a Mat instance, containing the indicated vertices
			    and their incident edges from the input Mat.

		SEE ALSO:  subgraph
		"""
		#ToDo:  accept slices for key0/key1 besides ParVecs
		if type(key)==tuple:
			if len(key)==1:
				[key0] = key; key1 = -1
			elif len(key)==2:
				[key0, key1] = key
			else:
				raise KeyError, 'Too many indices'
		else:
			key0 = key;  key1 = key
		if type(key0) == int or type(key0) == long or type(key0) == float:
			tmp = ParVec(1)
			tmp[0] = key0
			key0 = tmp
		if type(key1) == int or type(key0) == long or type(key0) == float:
			tmp = ParVec(1)
			tmp[0] = key1
			key1 = tmp
		#if type(key0)==slice and key0==slice(None,None,None):
		#	key0mn = 0; 
		#	key0tmp = self.nvert()
		#	if type(key0tmp) == tuple:
		#		key0mx = key0tmp[0] - 1
		#	else:
		#		key0mx = key0tmp - 1
		#if type(key1)==slice and key1==slice(None,None,None):
		#	key1mn = 0 
		#	key1tmp = self.nvert()
		#	if type(key1tmp) == tuple:
		#		key1mx = key1tmp[1] - 1
		#	else:
		#		key1mx = key1tmp - 1
		
		ret = Mat()
		ret._spm = self._spm.SubsRef(key0._dpv, key1._dpv)
		return ret

	# NEEDED: update to new fields
	# NEEDED: tests
	def __iadd__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			raise NotImplementedError
		elif self.getnrow() != other.getnrow() or self.getncol() != other.getncol():
			raise IndexError, 'Matrices must have matching dimensions'
		elif isinstance(other, Mat):
			#self._apply(pcb.plus(), other)
			self._m_ += other._spm
		return self

	# NEEDED: update to new fields
	# NEEDED: tests
	def __imul__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self.apply(pcb.bind2nd(pcb.multiplies(),other))
		elif self.getnrow() != other.getnrow() or self.getncol() != other.getncol():
			raise IndexError, 'Matrices must have matching dimensions'
		elif isinstance(other,Mat):
			self.apply(pcb.multiplies(), other)
		else:
			raise NotImplementedError
		return self

	# NEEDED: tests
	def __mul__(self, other):
		"""
		multiplies corresponding edges of two Mat instances together,
		resulting in edges in the result only where edges exist in both
		input Mat instances.

		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.apply(pcb.bind2nd(pcb.multiplies(),other))
		elif self.getnrow() != other.getnrow() or self.getncol() != other.getncol():
			raise IndexError, 'Matrices must have matching dimensions'
		elif isinstance(other,Mat):
			ret = self.copy()
			ret.apply(pcb.multiplies(), other)
		else:
			raise NotImplementedError
		return ret

	# NEEDED: tests
	def __neg__(self):
		ret = self.copy()
		ret.apply(pcb.negate())
		return ret

	# NEEDED: tests
	#ToDo:  put in method to modify _REPR_MAX
	_REPR_MAX = 100
	def __repr__(self):
		if self.getnee() == 0:
			return 'Empty Mat object'
		#if self.getnee() == 1:
		#	[i, j, v] = self.toVec()
		#	if len(v) > 0:
		#		return "%d %f" % (v[0], v[0])
		#	else:
		#		return "%d %f" % (0, 0.0)
		else:
			[i, j, v] = self.toVec()
			if len(i) < self._REPR_MAX:
				return "" + i + j + v
		return ' '

	# NEEDED: tests
	#in-place, so no return value
	def apply(self, op, other=None, notB=False):
		"""
		applies the given operator to every edge in the Mat

		Input Argument:
			self:  a Mat instance, modified in place.
			op:  a Python or pyCombBLAS function

		Output Argument:  
			None.

		"""
		if self._hasFilter():
			raise NotImplementedError, "this operation does not support filters yet."

		if other is None:
			if not isinstance(op, pcb.UnaryFunction):
				self._m_.Apply(pcb.unaryObj(op))
			else:
				self._m_.Apply(op)
			return
		else:
			if not isinstance(op, pcb.BinaryFunction):
				self._m_ = pcb.EWiseApply(self._m_, other._m_, pcb.binaryObj(op), notB)
			else:
				self._m_ = pcb.EWiseApply(self._m_, other._m_, op, notB)
			return

	# NEEDED: tests
	def eWiseApply(self, other, op, allowANulls, allowBNulls, inPlace=False):
		"""
		ToDo:  write doc
		"""
		
		if not isinstance(other, Mat):
			raise NotImplementedError, "eWiseApply with scalars not implemented yet"
			# use apply()
			return
		
		if self._hasFilter() or other._hasFilter():
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

		if inPlace:
			self._m_ = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(superOp))
		else:
			m = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(superOp))
			ret = self._toMat(m)
			return ret
		
	# NEEDED: update to new fields
	# NEEDED: tests
	#in-place, so no return value
	def scale(self, other, op=op_mul, dir=Column):
		"""
		multiplies the weights of the appropriate edges of each vertex of
		the passed DiGraph instance in-place by a vertex-specific scale 
		factor.

		Input Arguments:
			self:  a DiGraph instance, modified in-place
			other: a Vec whose elements are used
			dir:  a direction of edges to scale, with choices being
			    Mat.Column (default) or Mat.Row.

		Output Argument:
			None.

		SEE ALSO:  * (DiGraph.__mul__), mulNot
		"""
		if not isinstance(other, Vec):
			raise KeyError, 'Invalid type for scale vector'

		if dir == Mat.Column:
			if self.getncol() != len(other):
				raise IndexError, 'ncol != len(vec)'
		elif dir == Mat.Row:
			if self.getnrow() != len(other):
				raise IndexError, 'nrow != len(vec)'
		else:
			raise KeyError, 'Invalid edge direction'

		self._m_.DimWiseApply(dir, other.dense()._v_, _op_make_binary(op))
		return

	# TODO: make a _keep() which reverses the predicate
	def _prune(self, pred):
		"""
		returns a new matrix that only contains the elements e for which
		pred(e) == false.
		"""
		return Mat._toMat(self._m_.Prune(_op_make_unary_pred(pred)))

	def _hasFilter(self):
		try:
			ret = (hasattr(self,'_eFilter_') and len(self._eFilter_)>0) # ToDo: or (hasattr(self,'vAttrib') and self.vAttrib._hasFilter(self.vAttrib)) 
		except AttributeError:
			ret = False
		return ret

	def isObj(self):
		return not isinstance(self._identity_, (float, int, long, bool))
		#try:
		#	ret = hasattr(self,'_elementIsObject') and self._elementIsObject
		#except AttributeError:
		#	ret = False
		#return ret

	def isBool(self):
		return isinstance(self._m_, (pcb.pySpParMatBool))

	def isScalar(self):
		return isinstance(self._m_, (pcb.pySpParMat))

	# NEEDED: tests
	def reduce(self, dir, op, unOp=None, init=None):
		"""
		ToDo:  write doc
		NOTE:  need to doc clearly that the 2nd arg to the reduction
		fn is the sum;  the first is the current addend and the second
		is the running sum
		"""
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'unknown direction'
		
		if init is None:
			init = self._identity_
		
		if type(init) is not type(self._identity_) and not isinstance(init, (float, int, long)):
			raise NotImplementedError, "Reduce output type must either match the matrix type or be float."
		
		if self._hasFilter():
			class tmpB:
				_eFilter_ = self._eFilter_
				@staticmethod
				def fn(x, y):
					for i in range(len(tmpB._eFilter_)):
						if not tmpB._eFilter_[i](x):
							#x = type(self._identity_)()
							return y # no contribution; return existing 'sum'
							#break
					return op(x, y)
			tmpInstance = tmpB()
			superOp = pcb.binaryObj(tmpInstance.fn)
			#self._v_.Apply(pcb.unaryObj(tmpInstance.fn))
		else:
			superOp = _op_make_binary(op)

		ret = Vec(element=init, sparse=False)
		self._m_.Reduce(dir, ret._v_, superOp, _op_make_unary(unOp), init)
		return ret

	# NEEDED: tests
	# possibly in-place;  if so, no return value
	def SpMV(self, other, semiring, inPlace=False):
		"""
		FIX:  add doc
		inPlace -> no return value
		"""
		# check input
		if not isinstance(other, Vec):
			raise KeyError, "SpMV needs a Vec"
		if len(other) != self.getncol():
			raise KeyError, "Dimension mismatch in SpMV. The number of elements of the vector must equal the number of columns of the matrix."
		
		if self._hasFilter() or other._hasFilter():
			raise NotImplementedError, "this operation does not support filters yet"

		# the operation itself
		if inPlace:
			self._m_.SpMV_inplace(other._v_, semiring)
			return other
		else:
			return Vec._toVec(self._m_.SpMV(other._v_, semiring))
		
		# Adam:
		# Why is the rest so complicated?
		
		#ToDo:  is code for if/else cases actually different?
		if isinstance(self._identity_, (float, int, long, bool)) and isinstance(other._identity_, (float, int, long)):
			if isinstance(self._identity_, bool):
				#HACK OF HACKS!
				self._m_.SpMV_SelMax_inplace(other._v_)
				return
			if semiring is None:
				tSR = pcb.TimesPlusSemiring()
			else:  
				tSR = semiring
			if not inPlace:
				ret = Vec()
				ret._v_ = self._m_.SpMV(other._v_, tSR)
				return ret
			else:
				self._m_.SpMV_inplace(other._v_, tSR)
				return
		else:
			if semiring is None:
				tSR = pcb.TimesPlusSemiring()
			else:
				tSR = semiring
			if not inPlace:
				ret = Vec()
				ret._v_ = self._m_.SpMV(other._v_, tSR)
				return ret
			else:
				self._m_.SpMV_inplace(other._v_, tSR)
				return
	spMV = SpMV

	# NEEDED: update to new fields
	# NEEDED: tests
	def SpGEMM(self, other):
		"""
		"multiplies" two Mat instances together as though each was
		represented by a sparse matrix, with rows representing out-edges
		and columns representing in-edges.
		"""
		selfnv = self.nvert()
		if type(selfnv) == tuple:
			[selfnv1, selfnv2] = selfnv
		else:
			selfnv1 = selfnv; selfnv2 = selfnv
		othernv = other.nvert()
		if type(othernv) == tuple:
			[othernv1, othernv2] = othernv
		else:
			othernv1 = othernv; othernv2 = othernv
		if selfnv2 != othernv1:
			raise ValueError, '#in-vertices of first graph not equal to #out-vertices of the second graph '
		ret = Mat()
		ret._m_ = self._m_.SpGEMM(other._m_)
		return ret
	spGEMM = SpGEMM
	
	def transpose(self):
		"""
		performs an in-place transpose of this Mat instance
		"""
		self._m_.Transpose()
	
	# in-place, so no return value
	def addEFilter(self, filter):
		"""
		adds a vertex filter to the Mat instance.  

		A vertex filter is a Python function that is applied elementally
		to each vertex in the Mat, with a Boolean True return value
		causing the vertex to be considered and a False return value
		causing it not to be considered.

		Vertex filters are additive, in that each vertex must pass all
		filters to be considered.  All vertex filters are executed before
		a vertex is considered in a computation.
#FIX:  how is an argument passed to the function?

		Input Arguments:
			self:  a Mat instance
			filter:  a Python function

		SEE ALSO:
			delEFilter  
		"""
		if hasattr(self, '_eFilter_'):
			self._eFilter_.append(filter)
		else:
			self._eFilter_ = [filter]
		return
		
	# NEEDED: support for filters
	@staticmethod
	def load(fname, element=0.0):
		"""
		loads the contents of the file named fname (in the Coordinate Format 
		of the Matrix Market Exchange Format) into a Mat instance.

		Input Argument:
			fname:  a filename from which the matrix data will be loaded.
		Output Argument:
			ret:  a Mat instance containing the graph represented
			    by the file's contents.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The load
		method makes this conversion while reading the data and creating
		the graph.

		SEE ALSO:  save, UFget
		"""
		# Verify file exists.
		# TODO: make pcb load throw an error if the file cannot be opened.
		file = open(fname, 'r')
		file.close()
		
		#FIX:  crashes if any out-of-bound indices in file; easy to
		#      fall into with file being 1-based and Py being 0-based
		ret = Mat(element=element)
		ret._m_.load(fname)
		return ret

	# NEEDED:  support for filters
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
		self._m_.save(fname)
		return
