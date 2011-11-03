import math
#from Graph import master
from Vec import Vec
from Util import *
from Util import _op_make_unary
from Util import _op_make_unary_pred
from Util import _op_make_binary
from Util import _op_make_binaryObj
from Util import _op_make_binary_pred

import kdt.pyCombBLAS as pcb

import time

class Mat:
	Column  = pcb.pySpParMat.Column()
	Row = pcb.pySpParMat.Row()

	# NOTE:  for any vertex, out-edges are in the column and in-edges
	#	are in the row
	def __init__(self, i=None, j=None, v=None, n=None, m=None, element=0):
		"""
		FIX:  doc
		creates a new Mat instance.  Can be called in one of the 
		following forms:

	Mat():  creates an empty Mat instance with no elements.

	Mat(i, j, v, n)
	Mat(i, j, v, n, m)
		create a Mat instance with n columns, m rows, and initialized
		by the tuples i,j,v. 
		edges with source represented by 
		each element of sourceV and destination represented by each 
		element of destV with weight represented by each element of 
		weightV.  In the 4-argument form, the resulting Mat will 
		have n out- and in-vertices.  In the 5-argument form, the 
		resulting Mat will have n out-vertices and m in-vertices.

		Input Arguments:
			i:  a Vec containing integers denoting the 
			    (0-based) row of each element.
			i:  a Vec containing integers denoting the 
			    (0-based) column of each element.
			v:  a Vec containing the value of each element.
			n:  an integer scalar denoting the number of columns
			    (out-vertices for an adjacency matrix) 
			    (and also rows in the 4-argument case).
			m:  an integer scalar denoting the number of rows
			    (in-vertices for an adjacency matrix)

		Output Argument:  
			ret:  a Mat instance

		Note:  If two or more elements (tuples) have the same i and j
		values then their values are summed.

		SEE ALSO:  toVec
		"""
		if m is None and n is not None:
			m = n
		if i is None:
			if n is not None: #create a Mat with an underlying pySpParMat* of the right size with no nonnulls
				nullVec = pcb.pyDenseParVec(0,0)
			if isinstance(element, (float, int, long)):
				if n is None:
					self._m_ = pcb.pySpParMat()
				else:
					self._m_ = pcb.pySpParMat(m,n,nullVec,nullVec, nullVec)
			elif isinstance(element, bool):
				if n is None:
					self._m_ = pcb.pySpParMatBool()
				else:
					self._m_ = pcb.pySpParMatBool(m,n,nullVec,nullVec, nullVec)
			elif isinstance(element, pcb.Obj1):
				if n is None:
					self._m_ = pcb.pySpParMatObj1()
				else:
					self._m_ = pcb.pySpParMatObj1(m,n,nullVec,nullVec, nullVec)
			elif isinstance(element, pcb.Obj2):
				if n is None:
					self._m_ = pcb.pySpParMatObj2()
				else:
					self._m_ = pcb.pySpParMatObj2(m,n,nullVec,nullVec, nullVec)
			self._identity_ = element
		elif i is not None and j is not None and n is not None:
			#j = sourceV
			#i = destV
			#v = valueV
			if type(v) == tuple and isinstance(element,(float,int,long)):
				raise NotImplementedError, 'tuple valueV only valid for Obj element'
			if len(i) != len(j):
				raise KeyError, 'source and destination vectors must be same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = Vec(len(i), v, sparse=False)
#			if i.max() > n-1:
#				raise KeyError, 'at least one first index greater than #vertices'
#			if j.max() > n-1:
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
				self._m_ = pcb.pySpParMat(m,n,i._v_,j._v_,v._v_)
			elif isinstance(element, bool):
				self._m_ = pcb.pySpParMatBool(m,n,i._v_,j._v_,v._v_)
			elif isinstance(element, pcb.Obj1):
				self._m_ = pcb.pySpParMatObj1(m,n,i._v_,j._v_,v._v_)
			elif isinstance(element, pcb.Obj2):
				self._m_ = pcb.pySpParMatObj2(m,n,i._v_,j._v_,v._v_)
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

	def toVec(self):
		"""
		decomposes a DiGraph instance to 3 Vec instances, with each
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
		ne = self.getnnn()
		if ne != 0:
			reti = Vec(ne, element=0, sparse=False)
			retj = Vec(ne, element=0, sparse=False)
			retv = Vec(ne, element=self._identity_, sparse=False)
			self._m_.Find(reti._v_, retj._v_, retv._v_)
		else:
			reti = Vec(0)
			retj = Vec(0)
			retv = Vec(0)
		#ToDo:  return nvert() of original graph, too
		return (reti, retj, retv)

	def getnrow(self):
		return self._m_.getnrow()
		
	def getncol(self):
		return self._m_.getncol()
	
	def getnnn(self):
		"""
		returns the number of existing elements in this matrix.
		"""
		if self._hasFilter():
			raise NotImplementedError, "this operation does not support filters yet."
			
		return self._m_.getnee()
		
	# NEEDED: tests
	#ToDo:  put in method to modify _REPR_MAX
	_REPR_MAX = 200
	def __repr__(self):
		if self.getnnn() == 0:
			return 'Empty Mat object'
		#if self.getnnn() == 1:
		#	[i, j, v] = self.toVec()
		#	if len(v) > 0:
		#		return "%d %f" % (v[0], v[0])
		#	else:
		#		return "%d %f" % (0, 0.0)
		else:
			[i, j, v] = self.toVec()
			ret = "" + str(self.getnrow()) + "-by-" + str(self.getncol()) + " (row-by-col) Mat with " + str(self.getnnn()) + " elements.\n"
			if self.getncol() < 20:
				# pretty print a nice matrix
				
				# make empty 2D array, I'm sure there's a more proper way to initialize it
				mat = []
				for rowc in range(self.getnrow()):
					r = []
					for colc in range(self.getncol()):
						r.append(" - ")
					mat.append(r)
				
				# manually fill the array with matrix values
				for count in range(len(i)):
					mat[int(i[count])][int(j[count])] = str(v[count])
				
				# print row by row
				for rowc in range(self.getnrow()):
					for colc in range(self.getncol()):
						ret += mat[rowc][colc] + " "
					ret += "\n"
				
				return ret
			elif len(i) < self._REPR_MAX:
				print ret
				print "i (row index): ", repr(i)
				print "j (col index): ", repr(j)
				print "v (value)    : ", repr(v)
				#return ret + repr(i) + repr(j) + repr(v)
				return ""
			else:
				return ret + "Too many elements to print."
		return ' '

	# NEEDED: support for filters
	@staticmethod
	def load(fname, element=0.0):
		"""
		loads the contents of the file named fname (in the Coordinate Format 
		of the Matrix Market Exchange Format) into a Mat instance.

		Input Argument:
			fname:  a filename from which the matrix data will be loaded.
		Output Argument:
			ret:  a Mat instance containing the matrix represented
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
		saves the contents of the passed Mat instance to a file named
		fname in the Coordinate Format of the Matrix Market Exchange Format.

		Input Arguments:
			self:  a DiGraph instance
			fname:  a filename to which the matrix data will be saved.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The save
		method makes this conversion while writing the data.

		SEE ALSO:  load, UFget
		"""
		self._m_.save(fname)
		return

###########################
### Generators
###########################

	@staticmethod
	def generateRMAT(scale, edgeFactor=16, initiator=[.57, .19, .19, .05], delIsolated=True, element=True):
		"""
		generates a Kroenecker product matrix using the Graph500 RMAT graph generator.
		
		Output Argument:
			ret: a tuple: Mat, Vec, time.
				Mat is the matrix itself,
				Vec is a vector containing the degrees of each vertex in the original graph
				time is the time for the Graph500 Kernel 1.
		"""
		degrees = Vec(0, element=1.0, sparse=False)
		matrix = Mat(element=element)
		if matrix.isObj():
			raise NotImplementedError,"RMAT generation to supported on object matrices yet."
		kernel1Time = matrix._m_.GenGraph500Edges(scale, degrees._v_, scale, delIsolated, initiator[0], initiator[1], initiator[2], initiator[3])
		return matrix, degrees, kernel1Time

	@staticmethod
	def eye(n, m=None, element=1.0):
		"""
		creates an identity matrix. The resulting matrix is n-by-n
		with `element` on the main diagonal.
		In other words, ret[i,i] = element for 0 < i < n.

		Input Arguments:
			n:  number of columns of the matrix
			m:  number of rows of the matrix
			element: the value to put on the main diagonal elements.

		Output Argument:
			ret:  an identity matrix. 
		"""
		if m is None:
			m = n
		nnz = min(n, m)
		return Mat(Vec.range(nnz),Vec.range(nnz),Vec(nnz, element, sparse=False),n, m)

	@staticmethod
	def full(n,m=None, element=1.0):
		"""
		creates an `m`-by-`n` matrix with all elements set to `element`.

		Input Arguments:
			n:  an integer specifying the number of rows in the matrix.
			m:  an integer specifying the number of columns in the matrix.
			    If omitted, it defaults to the same value as `n`.

		Output Argument:
			ret:  an m-by-n full Mat instance. 
		"""
		
		if m is None:
			m = n
		i = (Vec.range(n*m) % n).floor()
		j = (Vec.range(n*m) / n).floor()
		v = Vec.ones(n*m)
		ret = Mat(i,j,v,n,m)
		return ret

##########################
### Filtering Methods
##########################

	def _hasFilter(self):
		try:
			ret = (hasattr(self,'_eFilter_') and len(self._eFilter_)>0) # ToDo: or (hasattr(self,'vAttrib') and self.vAttrib._hasFilter(self.vAttrib)) 
		except AttributeError:
			ret = False
		return ret

	# in-place, so no return value
	def addFilter(self, filter):
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

##########################
### Basic Methods
##########################

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
	def eWiseApply(self, other, op, allowANulls=False, allowBNulls=False, doOp=None, inPlace=False):
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
		
		if doOp is not None:
			# new version
			if inPlace:
				self._m_ = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(superOp), _op_make_binary_pred(doOp), allowANulls, allowBNulls, self._identity_, other._identity_)
			else:
				m = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(superOp), _op_make_binary_pred(doOp), allowANulls, allowBNulls, self._identity_, other._identity_)
				ret = Mat._toMat(m)
				return ret

		# old version, will be removed
		if inPlace:
			self._m_ = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(superOp))
		else:
			m = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(superOp))
			ret = self._toMat(m)
			return ret

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
			tmp = Vec(1)
			tmp[0] = key0
			key0 = tmp
		if type(key1) == int or type(key0) == long or type(key0) == float:
			tmp = Vec(1)
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
		
		ret = Mat._toMat(self._m_.SubsRef(key0._v_, key1._v_))
		return ret
		
	# TODO: make a _keep() which reverses the predicate
	def _prune(self, pred):
		"""
		only keep elements for which pred(e) == false.
		"""
		self._m_.Prune(_op_make_unary_pred(pred))

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
			superOp = _op_make_binaryObj(op)

		ret = Vec(element=init, sparse=False)
		self._m_.Reduce(dir, ret._v_, superOp, _op_make_unary(unOp), init)
		return ret

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
		if self.isBool():
			raise NotImplementedError, 'scale not implemented on boolean matrices do to C++ template irregularities.'
		
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

	# NEEDED: tests
	def SpGEMM(self, other, semiring):
		"""
		"multiplies" two Mat instances together as though each was
		represented by a sparse matrix, with rows representing in-edges
		and columns representing out-edges.
		"""
		
		# check input
		if not isinstance(other, Mat):
			raise ValueError, "SpGEMM needs a Mat"
		if self.getncol() != other.getnrow():
			raise ValueError, "Dimension mismatch in SpGEMM: %d != %d"%(self.getncol(),other.getnrow())
		
		ret = Mat._toMat(self._m_.SpGEMM(other._m_, semiring))
		return ret
	spGEMM = SpGEMM

	# possibly in-place;  if so, no return value
	def SpMV(self, other, semiring, inPlace=False):
		"""
		FIX:  add doc
		inPlace -> no return value
		"""
		# check input
		if not isinstance(other, Vec):
			raise ValueError, "SpMV needs a Vec"
		if len(other) != self.getncol():
			raise ValueError, "Dimension mismatch in SpMV. The number of elements of the vector must equal the number of columns of the matrix."
		
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

	def transpose(self):
		"""
		performs an in-place transpose of this Mat instance
		"""
		self._m_.Transpose()
	


##########################
### Operations
##########################

	# NEEDED: Handle the init properly (i.e. don't use 0, will break a test)
	def max(self, dir=Column):
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
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
		ret = self.reduce(dir, op_max, init=self._identity_)
		return ret

	# NEEDED: Handle the init properly (i.e. don't use 0, will break a test)
	def min(self, dir=Column):
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
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
		ret = self.reduce(dir, op_min, init=self._identity_)
		return ret

	def ones(self):
		"""
		makes every element to 1.
		"""
		if self.isObj():
			raise NotImplementedError, "at the moment only float matrices are supported."
		
		self.apply(op_set(1))
		
	def removeMainDiagonal(self):
		"""
		removes all elements whose row and column index are equal.
		Operation is in-place.

		Input Argument:
			self:  a Mat instance, modified in-place.

		"""
		self._m_.removeSelfLoops()

	
	def sum(self, dir=Column):
		"""
		adds the weights of the elements along a row or column.

		Input Arguments:
			self:  a Mat instance
			dir:  a direction of elements to sum, with choices being
			    Mat.Column (default) or Mat.Row.

		Output Argument:
			ret:  a Vec instance with each element containing the
			    sum of the weights of the corresponding row/column.

		SEE ALSO:  degree 
		"""
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
		ret = self.reduce(dir, op_add, init=self._identity_)
		return ret
	


##########################
### Arithmetic Operations
##########################

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
			ret._m_ += other._m_
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
	def __iadd__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			raise NotImplementedError
		elif self.getnrow() != other.getnrow() or self.getncol() != other.getncol():
			raise IndexError, 'Matrices must have matching dimensions'
		elif isinstance(other, Mat):
			#self._apply(pcb.plus(), other)
			self._m_ += other._m_
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
	
	# NEEDED: This needs to be generalized
	# NEEDED: update to new fields
	# NEEDED: tests
	def _mulNot(self, other):
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
		if self.getnrow() != other.getnrow() or self.getncol() != other.getncol():
			raise IndexError, 'Matrix dimensions must match'
		else:
			ret = self.copy()
			ret.apply(pcb.multiplies(), other, True)
		return ret

	# NEEDED: tests
	def __neg__(self):
		ret = self.copy()
		ret.apply(pcb.negate())
		return ret
			