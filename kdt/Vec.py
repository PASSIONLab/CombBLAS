import math
import kdt.pyCombBLAS as pcb
import feedback
import UFget as uf
import Mat as Mat
import ctypes
from Util import *
from Util import _op_make_unary
from Util import _op_make_unary_pred
from Util import _op_make_binary
from Util import _op_make_binaryObj
from Util import _op_make_binary_pred
from Util import _op_is_wrapped
from Util import _makePythonOp
from Util import _opStruct_int
from Util import _opStruct_float
from Util import _PDO_to_CPP
from Util import _CPP_to_PDO
from Util import _typeWrapInfo

#	naming convention:
#	names that start with a single underscore and have no final underscore
#		are functions
#	names that start and end with a single underscore are fields

class Vec(object):

	def __init__(self, length=0, element=0.0, sparse=True, _leaveEmpty=False):
		if _leaveEmpty:
			return

		if length < 0:
			raise ValueError, "length must be positive"

		typeInfo = _typeWrapInfo(type(element))
		
		if issubclass(typeInfo._getStorageType(), (float, int, long)):
			if sparse:
				self._v_ = pcb.pySpParVec(length)
			else:
				self._v_ = pcb.pyDenseParVec(length, element)
			self._identity_ = 0.0
		elif issubclass(typeInfo._getStorageType(), pcb.Obj1):
			if sparse:
				self._v_ = pcb.pySpParVecObj1(length)
			else:
				self._v_ = pcb.pyDenseParVecObj1(length, _PDO_to_CPP(element, pcb.Obj1))
			#self._identity_ = pcb.Obj1()
			self._identity_ = typeInfo._getElementType()()
		elif issubclass(typeInfo._getStorageType(), pcb.Obj2):
			if sparse:
				self._v_ = pcb.pySpParVecObj2(length)
			else:
				self._v_ = pcb.pyDenseParVecObj2(length, _PDO_to_CPP(element, pcb.Obj2))
			#self._identity_ = pcb.Obj2()
			self._identity_ = typeInfo._getElementType()()
		#elif isinstance(element, ctypes.Structure):
			# Python-defined object type
		#	elType = type(element)
		#	if ctypes.sizeof(elType) <= pcb.Obj1.capacity():
		#		if sparse:
		#			self._v_ = pcb.pySpParVecObj1(length)
		#		else:
		#			self._v_ = pcb.pyDenseParVecObj1(length, _PDO_to_CPP(element, pcb.Obj1))
		#		self._identity_ = elType()
		#	else:
		#		raise TypeError, "Cannot fit object into any available sizes. Largest possible object is %d bytes."%(kdt.Obj1.capacity())
		else:
			raise TypeError, "don't know type %s"%(type(element))
	
	@staticmethod
	def _buildSparse(vlen, ind, val):
		if len(ind) != len(val):
			raise KeyError, "index and value vectors must have equal length!"
		if ind.nnn() != len(ind) or val.nnn() != len(val):
			raise KeyError, "index and value vectors must be full!"
		
		ind = ind.dense()
		val = val.dense()
		
		# AL: This method works in a roundabout way and is very inefficient.
		# It does this because that's the only way to do it at the moment.
		
		# use a temporary matrix and SpMV to create this because that's the only
		# set of implemented CombBLAS routines that can result in what we want.
		M = Mat(i=ind, j=Vec(len(ind), sparse=False), v=val, n=vlen, m=vlen)
		ret = Vec(vlen, element=val._identity_, sparse=False)
		ret = ret.sparse() # gets us a sparse but full vector
		
		def mul(m,v):
			return m
		def add(x,y):
			raise ValueError,"duplicate coordinates"
		M.SpMV(ret, semiring=sr(add, mul), inPlace=True)
		
		return ret
	
	def _stealFrom(self, other):
		self._v_ = other._v_
		self._identity_ = other._identity_
	
	def _getStorageType(self):
		if isinstance(self._v_, (pcb.pyDenseParVec, pcb.pySpParVec)):
			return float
		if isinstance(self._v_, (pcb.pyDenseParVecObj1, pcb.pySpParVecObj1)):
			return pcb.Obj1
		if isinstance(self._v_, (pcb.pyDenseParVecObj2, pcb.pySpParVecObj2)):
			return pcb.Obj2
		raise NotImplementedError, 'Unknown vector type!'
	
	def _getElementType(self):
		if isinstance(self._identity_, (float, int, bool)):
			return float
		else:
			return type(self._identity_)

	@staticmethod
	def _getExampleElement(pcbVec):
		if isinstance(pcbVec, (pcb.pyDenseParVec, pcb.pySpParVec)):
			return 0.0
		if isinstance(pcbVec, (pcb.pyDenseParVecObj1, pcb.pySpParVecObj1)):
			return pcb.Obj1()
		if isinstance(pcbVec, (pcb.pyDenseParVecObj2, pcb.pySpParVecObj2)):
			return pcb.Obj2()
		raise NotImplementedError, 'Unknown vector type!'
	
	@staticmethod
	def _isPCBVecSparse(pcbVec):
		if isinstance(pcbVec, (pcb.pyDenseParVec, pcb.pyDenseParVecObj1, pcb.pyDenseParVecObj2)):
			return False
		if isinstance(pcbVec, (pcb.pySpParVec, pcb.pySpParVecObj1, pcb.pySpParVecObj2)):
			return True
		raise NotImplementedError, 'Unknown vector type!'
	
	@staticmethod
	def _toVec(pcbVec, identity):
		ret = Vec(_leaveEmpty=True)
		ret._v_ = pcbVec
		ret._identity_ = identity  #Vec._getExampleElement(pcbVec)
		return ret

	# NEEDED: type change needs to be updated		
	# ToDo:  add a doFilter=True arg at some point?
	def copy(self, element=None, materializeFilter=True):
		"""
		creates a deep copy of the input argument.
		FIX:  doc 'element' arg that converts element of result
		"""
		ret = Vec() #Vec(element=self._identity_, sparse=self.isSparse())
		ret._identity_ = self._identity_
		if element is not None and type(self._identity_) is not type(element):
			# changing elemental type. This will also implement a hard filter.
			if not materializeFilter:
				raise KeyError, "cannot copy a filter copy if changing type. Specify materializeFilter=True"
				
			ret = Vec(len(self), element=element, sparse=self.isSparse())
			def coercefunc(x, y): 
				#ToDo:  assumes that at least x or y is an ObjX
				if isinstance(x,(float,int,long)):
					ret = y.coerce(x, False)
				else:
					ret = x.coerce(y, True)
				return ret
			#raise NotImplementedError, "add coerce call to copy()"
			ret.eWiseApply(self, coercefunc, allowANulls=True, inPlace=True)
			
		elif self._hasFilter():
			if materializeFilter:
				# materialize the filter
				ret = Vec(len(self), element=self._identity_, sparse=self.isSparse())
				# eWiseApply already has filtering implemented, so piggy-back on it.
				# inPlace ensures that eWiseApply doesn't try to make a copy (leading
				# to an infinite loop)
				ret.eWiseApply(self, (lambda r, s: s), allowANulls=True, inPlace=True)
			else:
				# copy all the data as is and copy the filters
				import copy
				ret._v_ = self._v_.copy()
				ret._filter_ = copy.copy(self._filter_)
		else:
			ret._v_ = self._v_.copy()
		
		return ret
			
	@staticmethod
	def load(fname, element=0.0, sparse=True):
		"""
		loads a vector from a file. 
		"""
		ret = Vec(0, element=element, sparse=sparse)
		ret._v_.load(fname)
		return ret
	
	def save(self, fname):
		"""
		saves this vector to a file.
		"""
		if self._hasFilter():
			f = self.copy(materializeFilter=True)
			f._v_.save(fname)
		else:
			self._v_.save(fname)

	def dense(self):	
		"""
		converts a sparse Vec instance into a dense instance of the same
		length with the non-null elements of the sparse instance placed 
		in their corresonding positions in the dense instance.
		If the Vec instance is already dense, self is returned.
		"""
		if self.isDense():
			return self
		else:
			return Vec._toVec(self._v_.dense(), self._identity_)

	# TODO: accept a predicate that defines the sparsity. Use eWiseApply to implement.
	def sparse(self):
		"""
		converts a dense Vec instance into a sparse instance which
		contains all the elements but are stored in a sparse structure.
		If the Vec instance is already sparse, self is returned.
		"""
		if self.isSparse():
			return self
		else:
			return Vec._toVec(self._v_.sparse(), self._identity_)

#########################
### Methods
#########################

	# in-place, so no return value
	def apply(self, op):
		"""
		applies an operator to every element in the vector.
		For every element x in the vector, the following is performed:
		x = op(x)

		Input Arguments:
			self:  a Vec instance, modified in-place
			op:  the operation to perform.
		"""
		if self._hasFilter():
			op = _makePythonOp(op)
			if self.isSparse():
				op = FilterHelper.getFilteredUniOpOrSelf(self, op)
			else:
				op = FilterHelper.getFilteredUniOpOrOpVal(self, op, self._identity_)
		
		self._v_.Apply(_op_make_unary(op, self, self))
		return
		
	# in-place, so no return value
	# NEEDED: filters
	def applyInd(self, op):
		"""
		applies an operator to every element in the vector.
		For every element x at position i in the vector, the following is performed:
		x = op(x, i)

		Input Arguments:
			self:  a Vec instance, modified in-place
			op:  the operation to perform.
		"""
		
		if self._hasFilter():
			class tmpB:
				selfVFLen = len(self._filter_)
				vFilter1 = self._filter_
				@staticmethod
				def fn(x, y):
					for i in range(tmpB.selfVFLen):
						if not tmpB.vFilter1[i](x):
							x = type(self._identity_)()
							return x
							#break
					return op(x, y)
			superOp = tmpB().fn
		else:
			superOp = op

		self._v_.ApplyInd(_op_make_binary(superOp, self, _opStruct_float(), self))
	
	def __len__(self):
		"""
		returns the length (the maximum number of potential nonnull elements
		that could exist) of a Vec instance.
		"""
		return len(self._v_)

	# NEEDED: filters
	def __delitem__(self, key):
		if isinstance(other, (float, int, long)):
			del self._v_[key]
		else:
			del self._v_[key._v_];	
		return

	# NEEDED: filters
	def __getitem__(self, key):
		"""
		performs indexing of a Vec instance on the right-hand side
		of an equation.  The following forms are supported:
	scalar = vec[integer scalar]
	vec = vec[non-boolean vec]

		The first form takes as the index an integer scalar and returns
		the corresponding element of the Vec instance.  This form is
		for convenience and is not highly performing.

		The second form takes as the index a non-Boolean Vec instance
		and returns an Vec instance of the same length with the 
		elements of the result corresponding to the nonnull values of the
		index set to the values of the base SpParVec instance. 
		"""
		if isinstance(key, (int, long, float)):
			if key < 0 or key > len(self)-1:
				raise IndexError
			
			try:
				ret = _CPP_to_PDO(self._v_[key], type(self._identity_))
			except:
				# not found in a sparse vector
				ret = None
			
			if ret is not None and self._hasFilter() and not FilterHelper.getFilterPred(self)(ret):
				if self.isSparse():
					ret = None
				else:
					ret = self._identity_

			return ret
		else:
			if self._hasFilter() or key._hasFilter():
				raise NotImplementedError,"filtered __getitem__(Vec)"
			return Vec._toVec(self._v_[key._v_], self._identity_)

	#ToDo:  implement find/findInds when problem of any zero elements
	#         in the sparse vector getting stripped out is solved
	#ToDO:  simplfy to avoid dense() when pySpParVec.Find available
	def find(self, pred=None):
		"""
		returns the elements of a Vec for which a predicate returns True.

		Input Argument:
			self:  a Vec instance
			pred:  the predicate to check with. Default checks for non-zero.

		Output Argument:
			ret:  a Vec instance

		SEE ALSO:  findInds
		"""
		if self.isSparse():
			raise NotImplementedError, "find not implemented on sparse vectors yet."

		# provide a default predicate		
		if pred is None:
			if Vec.isObj(self):
				pred = lambda x: True
			else:
				pred = lambda x: x != 0
			
		if self._hasFilter():
			pred = FilterHelper.getFilteredUniOpOrVal(self, pred, False)

		retV = self._v_.Find(_op_make_unary_pred(pred, self))
		ret = Vec._toVec(retV, self._identity_)
		return ret


	#ToDO:  simplfy to avoid dense() when pySpParVec.FindInds available
	def findInds(self, pred=None):
		"""
		returns the indices of elements of a Vec for which a predicate returns True.

		Input Argument:
			self:  a Vec instance
			pred:  the predicate to check with. Default checks for non-zero.

		Output Argument:
			ret:  a Vec instance

		SEE ALSO:  find
		"""
		if self.isSparse():
			raise NotImplementedError, "findInds not implemented on sparse vectors yet."

		# provide a default predicate		
		if pred is None:
			if Vec.isObj(self):
				pred = lambda x: True
			else:
				pred = lambda x: x != 0
			
		if self._hasFilter():
			pred = FilterHelper.getFilteredUniOpOrVal(self, pred, False)

		ret = self._v_.FindInds(_op_make_unary_pred(pred, self))
		return Vec._toVec(ret, 0.0)

	def isDense(self):
		"""
		returns true if this Vec represents a dense vector, false otherwise.
		"""
		return isinstance(self._v_, (pcb.pyDenseParVec, pcb.pyDenseParVecObj1, pcb.pyDenseParVecObj2))
	
	def isObj(self):
		return not isinstance(self._identity_, (float, int, long))

	def isSparse(self):
		"""
		returns true if this Vec represents a sparse vector, false otherwise.
		"""
		return isinstance(self._v_, (pcb.pySpParVec, pcb.pySpParVecObj1, pcb.pySpParVecObj2))
	
	@staticmethod
	def ones(sz, element=1.0, sparse=False):
		"""
		creates a Vec instance of the specified size whose elements
		are all nonnull with the value 1.
		"""
		if sparse:
			return Vec.ones(sz, element=element, sparse=False).sparse()
		
		if isinstance(element, (float, int, long)):
			element = 1.0
		elif isinstance(element, pcb.Obj1):
			element = pcb.Obj1().set(1)
		elif isinstance(element, pcb.Obj2):
			element = pcb.Obj2().set(1)
		else:
			raise TypeError
		
		ret = Vec(sz, element, sparse=False)
		return ret
	
	@staticmethod
	def zeros(sz, element=0.0, sparse=False):
		"""
		creates a Vec instance of the specified size whose elements
		are all nonnull with the value 0.
		"""
		
		if sparse:
			return Vec.zeros(sz, element=element, sparse=False).sparse()
		
		if isinstance(element, (float, int, long)):
			element = 0.0
		elif isinstance(element, pcb.Obj1):
			element = pcb.Obj1().set(0)
		elif isinstance(element, pcb.Obj2):
			element = pcb.Obj2().set(0)
		else:
			raise TypeError
		
		ret = Vec(sz, element, sparse=False)
		return ret

	def printAll(self):
		"""
		prints all elements of a Vec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		p(str(self))

	def randPerm(self):
		"""
		randomly permutes all elements of the vector. Currently only
		supports dense vectors.
		"""
		if self.isDense():
			if len(self) == 0:
				return # otherwise some MPI implementations can crash
			self._v_.RandPerm()
		else:
			raise NotImplementedError, "Sparse vectors do not support RandPerm."
	
	@staticmethod
	def range(arg1, stop=None, element=0, sparse=False):
		"""
		FIX:  update doc
		creates a SpParVec instance with consecutive integer values.

	range(stop)
	range(start, stop)
		The first form creates a SpParVec instance of length stop whose
		values are all nonnull, starting at 0 and stopping at stop-1.
		The second form creates a SpParVec instance of length stop-start
		whose values are all nonnull, starting at start and stopping at
		stop-1.
		"""
		if stop is None:
			start = 0
			stop = arg1
		else:
			start = arg1
		if start > stop:
			raise ValueError, "start > stop"
		ret = Vec(stop-start, element=element)
		if not ret.isObj():
			if sparse:
				ret._v_ = pcb.pySpParVec.range(stop-start,start)
			else:
				ret._v_ = pcb.pyDenseParVec.range(stop-start,start)
		else:
			#HACK:  serial set is not practical for large sizes
			Obj1 = pcb.Obj1()
			for i in range(stop-start):
				Obj1.weight = start + i
				ret[i] = Obj1
		return ret

	def reduce(self, op, uniOp=None, init=None):
		"""
		accumulates vector elements.
		
		Input Arguments:
			self:  a Vec instance
			op:  a binary function. Its first argument is the result of application
				of uniOp to an element. Its second argument is the sum accumulated
				so far. Having incorporated the uniOp of another element into the sum,
				the function returns this new value of the sum. Example:
				lambda convertedElement, sum: sum + convertedElement
			uniOp:  a unary function that converts a matrix element before it is passed
				to op function as its first argument. In its simplest form, uniOp may
				return its input, i.e. lambda element: element.
			init:  the value to which the accumulated sum is initialized.

		Output Arguments:
			ret: the final accumulation value, or init if the vector has no elements.
		"""
		if init is None:
			init = self._identity_

		if self._hasFilter():
			if uniOp is None:
				uniOp = (lambda x: x)
			
			uniOp = _makePythonOp(uniOp)
			uniOp = FilterHelper.getFilteredUniOpOrVal(self, uniOp, init)
			uniOp = pcb.unaryObj(uniOp)
		
		if self.isObj():
			if type(init) is not type(self._identity_):# and not isinstance(init, (float, int, long)):
				raise NotImplementedError, "at the moment the result of reduce must have the same type as the Vec itself or a scalar."
		else:
			if not isinstance(init, (float, int, long)):
				raise NotImplementedError, "at the moment the result of reduce must have the same type as the Vec itself."
		
		# get the return type
		uniRetType = _typeWrapInfo(type(init))
		
		# cannot mix and match new and old reduce versions, so until we can totally get rid of
		# non-object BinaryFunction and UnaryFunction we have to support both.
		# In the future only the else clause of this if will be kept.
		if (isinstance(op, pcb.BinaryFunction) or isinstance(uniOp, pcb.UnaryFunction)) and not (isinstance(op, pcb.BinaryFunctionObj) or isinstance(uniOp, pcb.UnaryFunctionObj)):
			if init is not None and init is not self._identity_:
				raise ValueError, "you called the old reduce by using a built-in function, but this old version does not support the init attribute. Use a Python function instead of a builtin."
			ret = self._v_.Reduce(_op_make_binary(op, uniRetType, uniRetType, uniRetType), _op_make_unary(uniOp, self, uniRetType))
		else:
			ret = self._v_.Reduce(_op_make_binaryObj(op, uniRetType, uniRetType, uniRetType), _op_make_unary(uniOp, self, uniRetType), _PDO_to_CPP(init, uniRetType._getStorageType()))
		return _CPP_to_PDO(ret, uniRetType._getElementType())
	
	def count(self, pred=None):
		"""
		returns the number of elements for which `pred` is true.
		
		SEE ALSO: find, findInds
		"""
		if pred is None:
			pred = lambda x: bool(x)
		
		return self.reduce((lambda x,y: x+y), uniOp=pred, init=0.0)

	_REPR_MAX = 30;
	_REPR_WARN = 0
	def __repr__(self):
		"""
		returns a string representation of this Vec instance.

		SEE ALSO:  printAll
		"""
		if not hasattr(self,'_v_'):
			return "Vec with no _v_"

		ret = "length=%d, "%len(self)
		if self.isSparse():
			ret += "sparse, ["
		else:
			ret += "dense, ["
			
		if len(self) > self._REPR_MAX:
			ret += " *too many to print* ]"
			return ret
			
		if self.isSparse():
			first = True
			for i in range(len(self._v_)):
				val = self[i]
				if val is not None:
					if not first:
						ret += ", "
					else:
						first = False
					
					ret += "%d => %s"%(i, str(val))
		else:
			first = True
			for i in range(len(self._v_)):
				if not first:
					ret += ", "
				else:
					first = False
				
				ret += str(self[i])

		ret += "]"
		return ret
	
	def __setitem__(self, key, value):
		"""
		performs assignment of a Vec instance on the left-hand side of
		an equation.  The following forms are supported.
	vec[integer scalar] = scalar
	vec[Boolean vec] = scalar
	vec[Boolean vec] = vec
	vec[non-Boolean vec] = scalar
	vec[non-Boolean vec] = vec

		For the first form, the element of the Vec instance indicated 
		by the index to the is set to the scalar value.

		For the second form, the elements of the Vec instance
		corresponding to the True elements of the index ParVec instance
		are set to the scalar value.

		For the third form, the elements of the Vec instance 
		corresponding to the True elements of the index ParVec instance
		are set to the corresponding element of the value ParVec instance.

		For the fourth form, the elements of the Vec instance
		corresponding to the nonnull elements of the index Vec
		instance are set to the scalar value.

		For the fifth form, the elements of the Vec instance 
		corresponding to the nonnull elements of the index Vec
		instance are set to the corresponding value of the value
		Vec instance.  Note that the base, key, and value Vec
		instances must all be of the same length, though the base may
		have a different number of nonzeros from the key and value. 
		"""
		if isinstance(key, (float, int, long)):
			if key < 0 or key > len(self)-1:
				raise IndexError, "key %d is out of range length of vector is %d"%(key, len(self))
			self._v_[key] = _PDO_to_CPP(value, self._getStorageType())
		elif isinstance(key,Vec):
			if isinstance(value,Vec):
				pass
			elif type(value) == float or type(value) == long or type(value) == int:
				if key.isDense():
					value = Vec(len(key), element=value, sparse=False)
				else:
					valV = Vec(len(key), sparse=True)
					valV.eWiseApply(key, lambda v,k: value, allowANulls=True, inPlace=True)
					value = valV
			else:
				raise KeyError, 'Unknown value type'
				
			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
				raise IndexError, 'Key and Value must be same length as Vec'
			
			if self.isDense():
				self._v_[key.sparse()._v_] = value.sparse()._v_
			else:
				self._v_[key.dense()._v_] = value.dense()._v_
		elif type(key) == str and key == 'nonnull':
			self.apply(op_set(_PDO_to_CPP(value, self._getStorageType())))
		else:
			raise KeyError, 'Unknown key type'
		return
		
################################
#### Filter management
################################

	# in-place, so no return value
	def addFilter(self, filter):
		"""
		adds a filter to the Vec instance.  

		A filter is a Python predicate function that is applied elementally
		to each element in the Vec whenever an operation is performed on the
		Vec. If `filter(x)` returns a Boolean True then the element will be
		considered, otherwise it will not be considered.

		Filters are additive, in that each element must pass all
		filters added to the Vec to be considered. 

		Input Arguments:
			filter:  a Python predicate function

		SEE ALSO:
			delFilter  
		"""
		if hasattr(self, '_filter_'):
			self._filter_.append(filter)
		else:
			self._filter_ = [filter]
		return

	# in-place, so no return value
	def delFilter(self, filter=None):
		"""
		deletes a filter from the Vec instance.  

		Input Arguments:
			filter:  either a Python predicate function which has
			    been previoiusly added to this instance by a call to
			    addFilter or None, which signals the deletion of all
			    filters.

		SEE ALSO:
			addFilter  
		"""
		if not hasattr(self, '_filter_'):
			raise KeyError, "no filters previously created"
		if filter is None:
			del self._filter_	# remove all filters
		else:
			self._filter_.remove(filter)
			if len(self._filter_) == 0:
				del self._filter_
		return

	def _hasFilter(self):
		try:
			ret = hasattr(self,'_filter_') and len(self._filter_)>0
		except AttributeError:
			ret = False
		return ret
	
	def _hasMaterializedFilter(self):
		try:
			ret = hasattr(self,'_materialized')
		except AttributeError:
			ret = False
		return ret
	
	def _updateMaterializedFilter(self):
		self._materialized = self.copy()
	
	def _dirty(self):
		if self._hasMaterializedFilter():
			del self._materialized
			self._updateMaterializedFilter()
	
################################
#### EWiseApply
################################

	
	def eWiseApply(self, other, op, allowANulls=False, allowBNulls=False, doOp=None, inPlace=False, predicate=False, allowIntersect=True, ANull=None, BNull=None):
		"""
		applies a binary operation to corresponding elements of two vectors.
		The operation may be optionally performed in-place.
		This function is equivalent to:
		for all i:
		    if doOp(self[i], other[i]):
		        ret[i] = op(self[i], other[i])
		
		Since KDT vectors might be sparse, there are multiple ways to handle different
		sparsity patterns. This is for cases where one vector has an element at
		position i but the other does not or vice versa. The allowANulls,
		allowBNulls, and allowIntersect parameters are used to control what happens
		in each portion of the sparsity Venn diagram.

		Input Arguments:
			self:  a Vec instance representing the first vector
			other:  a Vec instance representing the second vector
			op:  a binary operation accepting two elements and returning an element
			allowANulls:  If True and self does not have a value at a position but
				other does, still perform the operation using a default value.
				If False, do not perform the operation at any position where self
				does not have a value.
			allowBNulls:  If True and other does not have a value at a position but
				self does, still perform the operation using a default value.
				If False, do not perform the operation at any position where other
				does not have a value.
			doOp:  a binary predicate accepting two corresponding elements of two
				matrices and returning True if they should be processed and False
				otherwise.
			inPlace:  indicates whether to perform the operation in-place storing
				the result in self or to create a new vector.
			allowIntersect:  indicates whether or not the operation should be
			    performed if both self and copy have a value at a position.
			ANull: Value to pass to op and doOp if allowANulls is True and self has a null element.
			BNull: Value to pass to op and doOp if allowBNulls is True and other has a null element.
			predicate:  Not Supported Yet

		SEE ALSO: Mat.eWiseApply
		"""
		
		# elementwise operation with a regular object
		if not isinstance(other, Vec):
			if allowANulls:
				raise NotImplementedError, "eWiseApply with a scalar requires allowANulls=False."
			
			if doOp is not None:
				# note: maybe can be implemented using a filter, i.e. predicate=doOp
				raise NotImplementedError, "eWiseApply with a scalar does not handle doOp yet."

			if inPlace:
				ret = self
			else:
				ret = self.copy()
			
			func = lambda x: op(x, other)
			ret.apply(func)
			return ret
		
		if len(self) != len(other):
			raise IndexError, "vectors must be of the same length. len(self)==%d != len(other)==%d"%(len(self), len(other))
		
		# not allowing intersection means nothing needs to be done if either of the vectors is dense
		if not allowIntersect and (not self.isSparse() or not other.isSparse):
			if inPlace:
				return
			else:
				return Vec(len(self), init=self._identity_)
		
		# wrap the ops
		if predicate:
			op = _op_make_binary_pred(op, self, other)
		else:
			op = _op_make_binaryObj(op, self, other, self)
		doOp = _op_make_binary_pred(doOp, self, other)
		selfFilter = _op_make_unary_pred(FilterHelper.getFilterPred(self), self)
		otherFilter = _op_make_unary_pred(FilterHelper.getFilterPred(other), other)
		
		if ANull is None:
			ANull = self._identity_
		if BNull is None:
			BNull = other._identity_
		ANull = _PDO_to_CPP(ANull, self._getStorageType())
		BNull = _PDO_to_CPP(BNull, self._getStorageType())

		# there are 4 possible permutations of dense and sparse vectors,
		# and each one can be either inplace or not.
		if self.isSparse():
			if other.isSparse(): # sparse, sparse
				if inPlace:
					ret = pcb.EWiseApply(self._v_, other._v_, op, doOp, allowANulls, allowBNulls, ANull, BNull, allowIntersect, selfFilter, otherFilter)
					ret = Vec._toVec(ret, self._identity_)
					self._stealFrom(ret)
				else:
					ret = pcb.EWiseApply(self._v_, other._v_, op, doOp, allowANulls, allowBNulls, ANull, BNull, allowIntersect, selfFilter, otherFilter)
					return Vec._toVec(ret, self._identity_)
			else: # sparse, dense
				if inPlace:
					ret = pcb.EWiseApply(self._v_, other._v_, op, doOp, allowANulls, ANull, selfFilter, otherFilter)
					ret = Vec._toVec(ret, self._identity_)
					self._stealFrom(ret)
				else:
					ret = pcb.EWiseApply(self._v_, other._v_, op, doOp, allowANulls, ANull, selfFilter, otherFilter)
					return Vec._toVec(ret, self._identity_)
		else:
			if other.isSparse(): # dense, sparse
				if inPlace:
					self._v_.EWiseApply(other._v_, op, doOp, allowBNulls, BNull, selfFilter, otherFilter)
				else:
					ret = self.copy()
					ret._v_.EWiseApply(other._v_, op, doOp, allowBNulls, BNull, selfFilter, otherFilter)
					return ret
			else: # dense, dense
				if inPlace:
					self._v_.EWiseApply(other._v_, op, doOp, selfFilter, otherFilter)
				else:
					ret = self.copy()
					ret._v_.EWiseApply(other._v_, op, doOp, selfFilter, otherFilter)
					return ret

#################################################
##  Operators
#################################################

	# NEEDED: update docstring
	def __abs__(self):
		"""
		"""
		
		ret = self.copy()
		if not Vec.isObj(self):
			f = op_abs
		else:
			f = lambda x: abs(x)
		ret.apply(f)
		return ret

	def argmax(self):
		m = self.max()
		ret = int(self.findInds(lambda x: x == m)[0])
		return ret

	def argmin(self):
		m = self.min()
		ret = int(self.findInds(lambda x: x == m)[0])
		return ret
	
	def _ewise_bin_op_worker(self, other, func, intOnly=False, predicate=False):
		"""
		is an internal function used to implement elementwise arithmetic operators.
		"""
		funcUse = func
		if intOnly:
			# if other is a floating point, make it an int
			if isinstance(other, (float, int, long)):
				other = int(other)
			
			# if self is a float vector, add conversions to int. Object vectors are assumed to handle
			# conversion themselves.
			if not self.isObj():
				if isinstance(other, Vec) and not other.isObj():
					funcUse = lambda x, y: func(int(x), int(y))
				else:
					funcUse = lambda x, y: func(int(x), y)

		if not isinstance(other, Vec):
			# if other is a scalar, then only apply it to the nonnull elements of self.
			return self.eWiseApply(other, funcUse, allowANulls=False, allowBNulls=False, inPlace=False, predicate=predicate)
		else:
			return self.eWiseApply(other, funcUse, allowANulls=True, allowBNulls=True, inPlace=False, predicate=predicate)
	
	def __add__(self, other):
		"""
		adds the corresponding elements of two Vec instances into the
		result Vec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		ToDo:  elucidate combinations, overloading, etc.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x + other))


	def __and__(self, other):
		"""
		performs a bitwise And between the corresponding elements of two
		Vec instances into the result Vec instance.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x & other), intOnly=True)

	# NEEDED: update docstring
	def __div__(self, other):
		"""
		divides each element of the first argument (a Vec instance),
		by either a scalar or the corresonding element of the second 
		Vec instance, with a non-null element where either of the 
		two input vectors was nonnull.
		
		Note:  ZeroDivisionException will be raised if any element of 
		the second argument is zero.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x / other))

	# NEEDED: update docstring
	def __eq__(self, other):
		"""
		calculates the element-wise Boolean equality of the first argument with the second argument 

		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x == other), predicate=True)

	# NEEDED: update docstring
	def __ge__(self, other):
		"""
		#FIX: doc
		calculates the elementwise Boolean greater-than-or-equal relationship of the first argument with the second argument 

		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x >= other), predicate=True)

	# NEEDED: update docstring
	def __gt__(self, other):
		"""
		calculates the elementwise Boolean greater-than relationship of the first argument with the second argument 
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x > other), predicate=True)

	# NEEDED: update docstring
	#def __iadd__(self, other):
	#	"""
	#	adds the corresponding elements of two SpParVec instances into the
	#	result SpParVec instance, with a nonnull element where either of
	#	the two input vectors was nonnull.
	#	"""
	#	return self._ewise_bin_op_worker(other, (lambda x, other: x + other))
		
	def __invert__(self):
		"""
		bitwise inverts each nonnull element of the Vec instance.
		"""
		ret = self.copy()
		if isinstance(self._identity_, (float, int, long)):
			func = lambda x: int(x).__invert__()
		else:
			func = lambda x: x.__invert__()
		ret.apply(func)
		return ret

	# NEEDED: update docstring
	#def __isub__(self, other):
	#	"""
	#	subtracts the corresponding elements of the second argument (a
	#	scalar or a SpParVec instance) from the first argument (a SpParVec
	#	instance), with a nonnull element where either of the two input 
	#	arguments was nonnull.
	#	"""
	#	return self._ewise_bin_op_worker(other, (lambda x, other: x - other))
		
	# NEEDED: update docstring
	def __le__(self, other):
		"""
		calculates the elementwise Boolean less-than-or-equal relationship of the first argument with the second argument 
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x <= other), predicate=True)

	# NEEDED: update docstring
	def __lt__(self, other):
		"""
		calculates the elementwise Boolean less-than relationship of the first argument with the second argument 

		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x < other), predicate=True)

	# NEEDED: update docstring
	def __mod__(self, other):
		"""
		calculates the modulus of each element of the first argument by the
		second argument (a scalar or a Vec instance), with a nonnull
		element where the input Vec argument(s) were nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x % other))

	# NEEDED: update docstring
	def __mul__(self, other):
		"""
		multiplies each element of the first argument by the second argument 
		(a scalar or a Vec instance), with a nonnull element where 
		the input Vec argument(s) were nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x * other))


	# NEEDED: update docstring
	def __ne__(self, other):
		"""
		calculates the Boolean not-equal relationship of the first argument with the second argument.
		other can be a Vec or a scalar.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x != other), predicate=True)

	def __neg__(self):
		"""
		negates each nonnull element of the passed Vec instance.
		"""
		ret = self.copy()
		func = lambda x: -x
		ret.apply(func)
		return ret


	# NEEDED: update docstring
	def __or__(self, other):
		"""
		performs a bitwise Or between the corresponding elements of two
		Vec instances.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x | other), intOnly=True)

	# NEEDED: update docstring
	def __sub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a Vec instance) from the first argument (a Vec
		instance), with a nonnull element where the input Vec argument(s)
		are nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x - other))

	# NEEDED: update docstring
	def __xor__(self, other):
		"""
		performs a bitwise Xor between the corresponding elements of two
		Vec instances.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x ^ other))
	

	def all(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		Vec instance are True (nonzero), and False otherwise.
		"""
		if isinstance(self._identity_, (float, int, long)):
			#ret = self.reduce(op_and, pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0)))
			#ret = (ret != 0)
			ret = self.reduce((lambda x,y: bool(x) and bool(y)), uniOp=(lambda x: bool(x)), init=1)
			ret = bool(ret)
			#print "all on vector:",self," result:",ret
			return ret
		else:
			tmp = self.copy()
			# only because have to set tmp[0]
					# because of element0 snafu
			identity = pcb.Obj1()
			identity.weight = tmp[0].weight
			#FIX: "=  bool(...)"?
			identity.category = 99
			tmp[0] = identity
			func = lambda x, other: x.all(other)
			ret = tmp.reduce(pcb.binaryObj(func)).weight > 0
			return ret

	def allCloseToInt(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		Vec instance have values within epsilon of an integer,
		and False otherwise.
		"""
		if self.nnn() == 0:
			return True;
		eps = info.eps()
		ret = (((self % 1.0) < eps) | (((-(self%1.0))+1.0)< eps)).all()
		return ret

	def any(self):
		"""
		returns a Boolean True if any of the nonnull elements of the
		Vec instance is True (nonzero), and False otherwise.
		"""
		tmp = self.copy()
		# only because have to set tmp[0]
					# because of element0 snafu
		if isinstance(self._identity_, (float, int, long)):
			return self.reduce(op_or, pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0)))
		else:
			identity = pcb.Obj1()
			identity.weight = tmp[0].weight
			#FIX: "=  bool(...)"?
			identity.category = 99
			tmp[0] = identity
			func = lambda x, other: x.any(other)
			ret = tmp.reduce(func).weight > 0
			return ret
	
	def floor(self):
		from math import floor
		ret = self.copy()
		ret.apply(lambda x: floor(x))
		return ret

	def isBool(self):
		"""
		returns a Boolean scalar denoting whether all elements of the input 
		Vec instance are equal to either True (1) or False (0).
		"""
		if self.nnn() == 0:
			return True
		eps = info.eps()
		
		c = self.count(lambda x: (abs(x) < eps) or (abs(x-1.0) < eps) )
		return c == self.nnn()
		
		#ret = ((abs(self) < eps) | (abs(self-1.0) < eps)).all()
		#return ret

	# NEEDED: update docstring
	def logicalAnd(self, other):
		"""
		performs a logical And between the corresponding elements of two
		Vec instances.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: bool(x) and bool(other)))

	def logicalNot(self):
		ret = self.copy()
		ret.apply(op_not)
		return ret
		
	# NEEDED: update docstring
	def logicalOr(self, other):
		"""
		performs a logical Or between the corresponding elements of two
		Vec instances.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: bool(x) or bool(other)))

	# NEEDED: update docstring
	def logicalXor(self, other):
		"""
		performs a logical XOr between the corresponding elements of two
		Vec instances.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: bool(x) != bool(other)))

	def max(self, initNegInf=None):
		"""
		returns the maximum value of the nonnull elements in the Vec 
		instance.
		"""
		if not self.isObj():
			if initNegInf is None:
				initNegInf = -1.8e308
			ret = self.reduce((lambda x,y: max(x, y)), init=initNegInf)
		else:
			if initNegInf is None:
				raise KeyError,"please provide an initNegInf argument which specifies a smallest possible value, i.e. something that acts like negative infinity."
			#func = lambda x, other: x.max(other)
			func = lambda x, other: max(x, other)
			ret = self.reduce(func, init=initNegInf)
		return ret

	def mean(self):
		"""
		calculates the mean (average) of a Vec instance, returning
		a scalar.
		"""
		return self.sum() / len(self)

	def min(self, initInf=None):
		"""
		returns the minimum value of the nonnull elements in the Vec 
		instance.
		"""
		if not self.isObj():
			if initInf is None:
				initInf = 1.8e308
			ret = self.reduce((lambda x,y: min(x, y)), init=initInf)
		else:
			if initInf is None:
				raise KeyError,"please provide an initInf argument which specifies a largest possible value, i.e. something that acts like infinity."
			#func = lambda x, other: x.min(other)
			func = lambda x, other: min(x, other)
			ret = self.reduce(func, init=initInf)
		return ret

	def nn(self):
		"""
		returns the number of nulls (non-existent entries) in the 
		Vec instance.

		Note:  for x a Vec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nnn, len
		"""
		if self.isDense():
			return 0
		return len(self) - self.nnn()

	def nnn(self):
		"""
		returns the number of non-nulls (existent entries) in the
		Vec instance.
	
		Note:  for Vec instance x, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nn, len
		"""
		if self.isDense():
			return len(self)

		if not self._hasFilter():
			return self._v_.getnee()
		
		# temporary hack while reduce can't do type changing:
		cp = self.copy(element=1.0)
		ret = cp.reduce((lambda x,y: x+y), uniOp=(lambda x: 1), init=0.0)
		return ret
		
		# ideal code:
		ret = self.reduce(op_add, uniOp=(lambda x: 1), init=0.0)
		return ret

	def norm(self, order=2):
		"""
		calculates the norm of a Vec instance, where the order of the
		norm is used as follows:
			pow((sum(abs(x)**order)),1/order)
		The order must be a scalar greater than zero.
		"""
		if order <= 0:
			raise ValueError, 'Order must be positive'
		else:
			#ret = self._reduce(pcb.plus(),pcb.compose1(pcb.bind2nd(pcb.pow(), order), pcb.abs()))
			ret = self.reduce(lambda x,y: x+y, uniOp=(lambda x: abs(x)**order))
			ret = pow(ret, 1.0/order)
			return ret

	#in-place, so no return value
	def set(self, value):
		"""
		sets every non-null value in the Vec instance to value, in-place.
		"""
		def f(x):
			x = value
			return x
		self.apply(f)
		return

	# in-place, so no return value
	# NEEDED: update docstring
	# NEEDED: filters
	def sort(self):
		"""
		sorts the non-null values in the passed Vec instance in-place
		in ascending order and return the permutation used.

		Input Arguments:
			self:  a Vec instance.

		Output Argument:
			ret: the permutation used to perform the sort. self is also
			     sorted.
		"""
		if self._hasFilter():
			raise NotImplementedError, "filtered sort not implemented"
			
		return Vec._toVec(self._v_.Sort(), self._identity_)

	# NEEDED: update docstring
	def sorted(self):
		"""
		returns a new Vec instance with the sorted values (in ascending
		order) from the input Vec instance and a Vec permutation 
		vector.

		Input Arguments:
			self:  a Vec instance.

		Output Argument:
			ret:  a tuple containing as its first element a Vec 
			    instance of the same length and same number and position
			    of non-nulls containing the sorted non-null values and
			    as its second element the indices of the sorted values
			    in the input vector.

		See Also:  sort
		"""
		if self._hasFilter():
			raise NotImplementedError, "filtered sort not implemented"

		ret1 = self.copy();
		ret2 = ret1.sort()
		return (ret1, ret2)

	#in-place, so no return value
	def spOnes(self):
		"""
		sets every non-null value in the Vec instance to 1, in-place.
		"""
		if not Vec.isObj(self):
			self.apply(lambda x: 1)
		else:
			self.apply(lambda x: x.set(1))
		return

	#in-place, so no return value
	def spRange(self):
		"""
		sets every non-null value in the Vec instance to its position
		(offset) in the vector, in-place.
		"""

		if not self.isObj() and self.isSparse() and not self._hasFilter():
			self._v_.setNumToInd()
		else:
			if self.isObj():
				func = lambda x,y: x.set(y)
			else:
				func = lambda x,y: y
			self.applyInd(func)
		return
	
	def std(self):
		"""
		calculates the standard deviation of a Vec instance, returning
		a scalar.  Calculated as sqrt((self-self.mean()).sum)
		"""
		mean = self.mean();
		diff = self - mean
		ret = math.sqrt((diff*diff).sum()/len(self))
		return ret 

	def sum(self):
		"""
		returns the sum of all the non-null values in the Vec instance.
		"""
		return self.reduce(lambda x,y: x+y)
		
	#in-place, so no return value
	# NEEDED: handle objects
	def toBool(self):
		"""
		converts the Vec instance, in-place, into Boolean
		values (1.0 for True, 0.0 for False) according to whether the
		initial values are nonzero or zero, respectively.
		"""
		def f(x):
			x.weight = bool(x.weight)
			return x
		self.apply(f)
		return

	# NEEDED: make it work (bugs in CombBLAS/PCB implementation)
	def topK(self, k):
		"""
		returns the largest k non-null values in the passed Vec instance.

		Input Arguments:
			self:  a Vec instance.
			k:  a scalar integer denoting how many values to return.

		Output Argument:
			ret:  a Vec instance of length k containing the k largest
			    values from the input vector, in ascending order.
		"""
		raise NotImplementedError, "TopK does not work properly"
		
		if isinstance(self._identity_, (float, int, long)):
			ret = Vec(0)
			ret._v_ = self._v_.TopK(k)
		else:
			raise NotImplementedError
		return ret
		
	def hist(self):
		"""
		finds a histogram
		ToDo:  write docstring
		"""
		if self.isObj():
			raise NotImplementedError, "histogram not yet supported on objects"
		#if not self.allCloseToInt():
		#	raise KeyError, 'input values must be all integer'
		selfInt = self.copy()
		selfInt.apply(lambda x: round(x)) #self.round()
		rngV = Vec.range(len(self))
		oneV = Vec.ones(len(self))
		selfMax = int(self.max())
		tmpMat = Mat.Mat(selfInt, rngV, oneV, selfMax+1, len(self)) # AL: swapped
		ret = tmpMat.reduce(Mat.Mat.Row, op_add)
		return ret	
