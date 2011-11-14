import math
import kdt.pyCombBLAS as pcb
import feedback
import UFget as uf
import Mat as Mat
from Util import *
from Util import _op_make_unary
from Util import _op_make_unary_pred
from Util import _op_make_binary
from Util import _op_make_binaryObj
from Util import _op_make_binary_pred
from Util import _op_is_wrapped
from Util import _makePythonOp

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

		if isinstance(element, (float, int, long)):
			if sparse:
				self._v_ = pcb.pySpParVec(length)
			else:
				self._v_ = pcb.pyDenseParVec(length, element)
			self._identity_ = 0.0
		elif isinstance(element, pcb.Obj1):
			if sparse:
				self._v_ = pcb.pySpParVecObj1(length)
			else:
				self._v_ = pcb.pyDenseParVecObj1(length, element)
			self._identity_ = pcb.Obj1()
		elif isinstance(element, pcb.Obj2):
			if sparse:
				self._v_ = pcb.pySpParVecObj2(length)
			else:
				self._v_ = pcb.pyDenseParVecObj2(length, element)
			self._identity_ = pcb.Obj2()
		else:
			raise TypeError, "don't know type %s"%(type(element))
	
	def _stealFrom(self, other):
		self._v_ = other._v_
		self._identity_ = other._identity_
	
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
	def _toVec(pcbVec):
		ret = Vec(_leaveEmpty=True)
		ret._v_ = pcbVec
		ret._identity_ = Vec._getExampleElement(pcbVec)
		return ret

	# NEEDED: type change needs to be updated		
	def copy(self, element=None, hardFilter=True):
		"""
		creates a deep copy of the input argument.
		FIX:  doc 'element' arg that converts element of result
		ToDo:  add a doFilter=True arg at some point?
		"""
		ret = Vec() #Vec(element=self._identity_, sparse=self.isSparse())
		ret._identity_ = self._identity_
		if element is not None and type(self._identity_) is not type(element):
			# changing elemental type. This will also implement a hard filter.
			if not hardFilter:
				raise KeyError, "cannot do a soft filter copy if changing type. Specify hardFilter=True"
				
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
			if hardFilter:
				# materialize the filter
				ret = Vec(len(self), element=self._identity_, sparse=self.isSparse())
				# eWiseApply already has filtering implemented, so piggy-back on it
				# inPlace ensures that eWiseApply doesn't try to make a copy, leading
				# to an infinite loop
				ret.eWiseApply(self, (lambda r, s: s), allowANulls=True, inPlace=True)
			else:
				# copy all the data as is and copy the filters
				import copy
				ret._v_ = self._v_.copy()
				ret._filter_ = copy.copy(self._filter_)
		else:
			ret._v_ = self._v_.copy()
		
		return ret
		
		# filter the new vector; note generic issue of distinguishing
		#   zero from null
		if self._hasFilter():
			class tmpU:
				_filter_ = self._filter_
				@staticmethod
				def fn(x):
					for i in range(len(tmpU._filter_)):
						if not tmpU._filter_[i](x):
							return type(self._identity_)()
					return x
			tmpInstance = tmpU()
			ret._v_.Apply(pcb.unaryObj(tmpInstance.fn))
			pass
		if element is not None and type(self._identity_) is not type(element):
			tmp = Vec(len(self), element=element, sparse=self.isSparse())
			def func(x, y): 
				#ToDo:  assumes that at least x or y is an ObjX
				if isinstance(x,(float,int,long)):
					ret = y.coerce(x, False)
				else:
					ret = x.coerce(y, True)
				return ret
			tmp2 = tmp.eWiseApply(ret, func, True, True)
			ret = tmp2
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
			f = self.copy(hardFilter=True)
			f._v_.save(fname)
		else:
			self._v_.save(fname)

	# NEEDED: filters
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
			return Vec._toVec(self._v_.dense())

	# TODO: have it accept a predicate that defines the sparsity. Use eWiseApply to implement.
	# NEEDED: filters
	def sparse(self):
		"""
		converts a dense Vec instance into a sparse instance which
		contains all the elements but are stored in a sparse structure.
		If the Vec instance is already sparse, self is returned.
		"""
		if self.isSparse():
			return self
		else:
			return Vec._toVec(self._v_.sparse())

#########################
### Methods
#########################

	# in-place, so no return value
	def apply(self, op):
		"""
		ToDo:  write doc;  note pcb built-ins cannot be used as filters.
		"""
		if self._hasFilter():
			op = _makePythonOp(op)
			#if self.isSparse():
			op = FilterHelper.getFilteredUniOpOrSelf(self, op)
			#else:
			#	op = FilterHelper.getFilteredUniOpOrOpVal(self, op, self._identity_)
		
		self._v_.Apply(_op_make_unary(op))
		return
		
	# in-place, so no return value
	# NEEDED: filters
	def applyInd(self, op):
		"""
		ToDo:  write doc;  note pcb built-ins cannot be used as filters.
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

		self._v_.ApplyInd(_op_make_binary(superOp))
	
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
		performs indexing of a SpParVec instance on the right-hand side
		of an equation.  The following forms are supported:
	scalar = spparvec[integer scalar]
	spparvec = spparvec[non-boolean parvec]

		The first form takes as the index an integer scalar and returns
		the corresponding element of the SpParVec instance.  This form is
		for convenience and is not highly performing.

		The second form takes as the index a non-Boolean SpParVec instance
		and returns an SpParVec instance of the same length with the 
		elements of the result corresponding to the nonnull values of the
		index set to the values of the base SpParVec instance. 
		"""
		if isinstance(key, (int, long, float)):
			if key < 0 or key > len(self)-1:
				raise IndexError
			
			ret = self._v_[key]
			if self._hasFilter() and not FilterHelper.getFilterPred(self)(ret):
				ret = self._identity_

			return ret
		else:
			if self._hasFilter() or key._hasFilter():
				raise NotImplementedError,"filtered __getitem__(Vec)"
			return Vec._toVec(self._v_[key._v_])

	#ToDo:  implement find/findInds when problem of any zero elements
	#         in the sparse vector getting stripped out is solved
	#ToDO:  simplfy to avoid dense() when pySpParVec.Find available
	def find(self, pred=None):
		"""
		returns the elements of a Boolean Vec instance that are both
		nonnull and nonzero.

		Input Argument:
			self:  a Vec instance

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

		ret = self._v_.Find(_op_make_unary_pred(pred))
		return Vec._toVec(ret)


	#ToDO:  simplfy to avoid dense() when pySpParVec.FindInds available
	def findInds(self, pred=None):
		"""
		returns the indices of the elements of a Boolean Vec instance
		that are both nonnull and nonzero.

		Input Argument:
			self:  a Vec instance

		Output Argument:
			ret:  a Vec instance of length equal to the number of
			    nonnull and nonzero elements in self

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

		ret = self._v_.FindInds(_op_make_unary_pred(pred))
		return Vec._toVec(ret)

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
	
	# NEEDED: update with proper init
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
			element = pcb.Obj1().spOnes()
		elif isinstance(element, pcb.Obj2):
			element = pcb.Obj2().spOnes()
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
			element = pcb.Obj1().spZeros()
		elif isinstance(element, pcb.Obj2):
			element = pcb.Obj2().spZeros()
		else:
			raise TypeError
		
		ret = Vec(sz, element, sparse=False)
		return ret

	# NEEDED: filters
	def printAll(self):
		"""
		prints all elements of a Vec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		self._v_.printall()
		return ' '

	def randPerm(self):
		"""
		randomly permutes all elements of the vector. Currently only
		supports dense vectors.
		"""
		if self.isDense():
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
		ToDo:  write doc
			return is a scalar
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
		
		# cannot mix and match new and old reduce versions, so until we can totally get rid of
		# non-object BinaryFunction and UnaryFunction we have to support both.
		# In the future only the else clause of this if will be kept.
		if (isinstance(op, pcb.BinaryFunction) or isinstance(uniOp, pcb.UnaryFunction)) and not (isinstance(op, pcb.BinaryFunctionObj) or isinstance(uniOp, pcb.UnaryFunctionObj)):
			if init is not None and init is not self._identity_:
				raise ValueError, "you called the old reduce by using a built-in function, but this old version does not support the init attribute. Use a Python function instead of a builtin."
			ret = self._v_.Reduce(_op_make_binary(op), _op_make_unary(uniOp))
		else:
			ret = self._v_.Reduce(_op_make_binaryObj(op), _op_make_unary(uniOp), init)
		return ret
	
	def count(self, pred=None):
		"""
		returns the number of elements for which `pred` is true.
		"""
		if pred is None:
			pred = lambda x: bool(x)
		
		return self.reduce((lambda x,y: x+y), uniOp=pred, init=0.0)

	_REPR_MAX = 30;
	_REPR_WARN = 0
	# NEEDED: filters
	def __repr__(self):
		"""
		prints the first N elements of the SpParVec instance, where N
		is roughly equal to the value of self._REPR_MAX.

		SEE ALSO:  printAll
		"""
		if hasattr(self,'_v_'):
			self._v_.printall()
		else:
			return "Vec with no _v_"
		return ' '
		#TODO:  limit amount of printout?
		nPrinted = 0
		i = 0
		while i < len(self) and nPrinted < self._REPR_MAX:
			#HACK check for nonnull
			#ToDo: return string instead of printing here
			print "__repr__ loop,", self[i]
			if self[i].weight > info.eps or self[i].category!=0:
				print self[i]
				nPrinted += 1
			i += 1
		if i < len(self)-1 and master():
			print "Limiting print-out to first %d elements" % self._REPR_MAX
		return ' '
	
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
			if key > len(self)-1:
				raise IndexError, "key %d is out of range length of vector is %d"%(key, len(self))
			self._v_[key] = value
		elif isinstance(key,Vec) and key.isDense():
			if not key.isBool():
				raise KeyError, 'only Boolean Vec indexing of Vecs supported'
			if isinstance(value,Vec):
				pass
			elif type(value) == float or type(value) == long or type(value) == int:
				value = Vec(len(key),value)
			else:
				raise KeyError, 'Unknown value type'
			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
				raise IndexError, 'Key and Value must be same length as Vec'
			self._v_[key.sparse()._v_] = value.sparse()._v_
		elif isinstance(key,Vec) and key.isSparse():
			#FIX:  get isBool() working
			#if key.isBool():
			#	raise KeyError, 'Boolean SpVec indexing of SpParVecs not supported'
			if isinstance(value,Vec):
				pass
			elif isinstance(value,SpVec):
				value = value.toDeVec()
			elif type(value) == float or type(value) == long or type(value) == int:
				tmp = value
				value = key.copy()
				value.set(tmp)
				value = value.toDeVec()
			else:
				raise KeyError, 'Unknown value type'
			#key = key.toDeVec()
			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
				raise IndexError, 'Key and Value must be same length as SpVec'
			self._v_[key._v_] = value._v_
		elif type(key) == str and key == 'nonnull':
			self.apply(pcb.set(value))
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
	
	
################################
#### EWiseApply
################################


# Old filtering code left here for reference:
#
#		if hasattr(self, '_filter_') or hasattr(other, '_filter_'):
#			class tmpB:
#				if hasattr(self,'_filter_') and len(self._filter_) > 0:
#					selfVFLen = len(self._filter_)
#					vFilter1 = self._filter_
#				else:
#					selfVFLen = 0
#				if hasattr(other,'_filter_') and len(other._filter_) > 0:
#					otherVFLen = len(other._filter_)
#					vFilter2 = other._filter_
#				else:
#					otherVFLen = 0
#				@staticmethod
#				def fn(x, y):
#					for i in range(tmpB.selfVFLen):
#						if not tmpB.vFilter1[i](x):
#							x = type(self._identity_)()
#							break
#					for i in range(tmpB.otherVFLen):
#						if not tmpB.vFilter2[i](y):
#							y = type(other._identity_)()
#							break
#					return op(x, y)
#			superOp = tmpB().fn
#		else:
#			superOp = op

	# NOTE: this function is specific because pyCombBLAS calling
	#  sequences are different for EWiseApply on sparse/dense vectors
	def _sparse_sparse_eWiseApply(self, other, op, doOp, allowANulls, allowBNulls, ANull, BNull, predicate=False):
		"""
		ToDo:  write doc
		"""
		superOp, doOp = FilterHelper.getEWiseFilteredOps(self, other, op, doOp, allowANulls, allowBNulls, ANull, BNull)
		
		if predicate:
			superOp = _op_make_binary_pred(superOp)
		else:
			superOp = _op_make_binaryObj(superOp)
		
		v = pcb.EWiseApply(self._v_, other._v_, superOp, _op_make_binary_pred(doOp), allowANulls, allowBNulls, ANull, BNull)
		ret = Vec._toVec(v)
		return ret

	# NOTE: this function is specific because pyCombBLAS calling
	#  sequences are different for EWiseApply on sparse/dense vectors
	def _sparse_dense_eWiseApply(self, other, op, doOp, allowANulls, ANull, predicate=False):
		"""
		ToDo:  write doc
		"""
		superOp, doOp = FilterHelper.getEWiseFilteredOps(self, other, op, doOp, allowANulls, True, ANull, other._identity_)

		if predicate:
			superOp = _op_make_binary_pred(superOp)
		else:
			superOp = _op_make_binaryObj(superOp)

		v = pcb.EWiseApply(self._v_, other._v_, superOp, _op_make_binary_pred(doOp), allowANulls, ANull)
		ret = Vec._toVec(v)
		return ret

	# NOTE: this function is specific because pyCombBLAS calling
	#  sequences are different for EWiseApply on sparse/dense vectors
	def _dense_sparse_eWiseApply_inPlace(self, other, op, doOp, allowBNulls, BNull, predicate=False):
		"""
		ToDo:  write doc
		"""
		superOp, doOp = FilterHelper.getEWiseFilteredOps(self, other, op, doOp, True, allowBNulls, self._identity_, BNull)
			
		if predicate:
			superOp = _op_make_binary_pred(superOp)
		else:
			superOp = _op_make_binaryObj(superOp)
			
		self._v_.EWiseApply(other._v_, superOp, _op_make_binary_pred(doOp), allowBNulls, BNull)
	
	# NOTE: this function is specific because pyCombBLAS calling
	#  sequences are different for EWiseApply on sparse/dense vectors
	def _dense_dense_eWiseApply_inPlace(self, other, op, doOp, predicate=False):
		"""
		ToDo:  write doc
		in-place operation
		"""
		superOp, doOp = FilterHelper.getEWiseFilteredOps(self, other, op, doOp, True, True, self._identity_, other._identity_)

		if predicate:
			superOp = _op_make_binary_pred(superOp)
		else:
			superOp = _op_make_binaryObj(superOp)
		self._v_.EWiseApply(other._v_, superOp, _op_make_binary_pred(doOp))
	
	# NEEDED: filters
	def eWiseApply(self, other, op, allowANulls=False, allowBNulls=False, doOp=None, inPlace=False, predicate=False):
		"""
		Performs an element-wise operation between the two vectors.
		if inPlace is true the result is stored in self.
		if inPlace is false, the result is returned in a new vector.
		
		doOp is a predicate that determines whether or not a op is performed on each pair of values.
		i.e.:
			if doOp(self[i], other[i])
				op(self[i], other[i])
		"""
		
		# elementwise operation with a regular object
		if not isinstance(other, Vec):
			if allowANulls:
				raise NotImplementedError, "eWiseApply with a scalar requires allowANulls=True for now."
			
			if doOp is not None:
				# note: can be implemented using a filter, i.e. predicate=doOp
				raise NotImplementedError, "eWiseApply with a scalar does not handle doOp for now."

			if inPlace:
				ret = self
			else:
				ret = self.copy()
			
			func = lambda x: op(x, other)
			ret.apply(func)
			return ret
		
		if len(self) != len(other):
			raise IndexError, "vectors must be of the same length. len(self)==%d != len(other)==%d"%(len(self), len(other))
		
		# there are 4 possible permutations of dense and sparse vectors,
		# and each one can be either inplace or not.
		if self.isSparse():
			if other.isSparse():
				if inPlace:
					ret = self._sparse_sparse_eWiseApply(other, op, doOp, allowANulls=allowANulls, allowBNulls=allowBNulls, ANull=self._identity_, BNull=other._identity_, predicate=predicate)
					self._stealFrom(ret)
				else:
					return self._sparse_sparse_eWiseApply(other, op, doOp, allowANulls=allowANulls, allowBNulls=allowBNulls, ANull=self._identity_, BNull=other._identity_, predicate=predicate)
			else: # sparse, dense
				if inPlace:
					ret = self._sparse_dense_eWiseApply(other, op, doOp, allowANulls=allowANulls, ANull=self._identity_, predicate=predicate)
					self._stealFrom(ret)
				else:
					return self._sparse_dense_eWiseApply(other, op, doOp, allowANulls=allowANulls, ANull=self._identity_, predicate=predicate)
		else: # dense
			if other.isSparse():
				if inPlace:
					self._dense_sparse_eWiseApply_inPlace(other, op, doOp, allowBNulls=allowBNulls, BNull=other._identity_, predicate=predicate)
				else:
					ret = self.copy()
					ret._dense_sparse_eWiseApply_inPlace(other, op, doOp, allowBNulls=allowBNulls, BNull=other._identity_, predicate=predicate)
					return ret
			else: # dense, dense
				if inPlace:
					self._dense_dense_eWiseApply_inPlace(other, op, doOp, predicate=predicate)
				else:
					ret = self.copy()
					ret._dense_dense_eWiseApply_inPlace(other, op, doOp, predicate=predicate)
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

	# NEEDED: update to eWiseApply
	# NEEDED: update docstring
	def __div__(self, other):
		"""
		divides each element of the first argument (a SpParVec instance),
		by either a scalar or the corresonding element of the second 
		SpParVec instance, with a non-null element where either of the 
		two input vectors was nonnull.
		
		Note:  ZeroDivisionException will be raised if any element of 
		the second argument is zero.

		Note:  For v0.1, the second argument may only be a scalar.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x / other))

		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__div__(other)
			ret.apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			raise NotImplementedError, 'no SpVecObj/SpVecObj yet'
		 	func = lambda x, other: x.__div__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

	# NEEDED: update docstring
	def __eq__(self, other):
		"""
		calculates the Boolean equality of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the scalar and the corresponding
		element of the first argument are equal.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and equal.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x == other), predicate=True)

	# NEEDED: update docstring
	def __ge__(self, other):
		"""
		#FIX: doc
		calculates the Boolean greater-than-or-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is greater than or equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is greater than or equal to the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x >= other), predicate=True)

	# NEEDED: update docstring
	def __gt__(self, other):
		"""
		calculates the Boolean greater-than relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is greater than the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is greater than the second.
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
		calculates the Boolean less-than-or-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is less than or equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is less than or equal to the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x <= other), predicate=True)

	# NEEDED: update docstring
	def __lt__(self, other):
		"""
		calculates the Boolean less-than relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is less than the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is less than the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x < other), predicate=True)

	# NEEDED: update docstring
	def __mod__(self, other):
		"""
		calculates the modulus of each element of the first argument by the
		second argument (a scalar or a SpParVec instance), with a nonnull
		element where the input SpParVec argument(s) were nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x % other))

	# NEEDED: update docstring
	def __mul__(self, other):
		"""
		multiplies each element of the first argument by the second argument 
		(a scalar or a SpParVec instance), with a nonnull element where 
		the input SpParVec argument(s) were nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x * other))


	# NEEDED: update docstring
	def __ne__(self, other):
		"""
		calculates the Boolean not-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is not equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is not equal to the second.
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
		performs a logical Or between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where at least one of the input vectors is True.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x | other))

	# NEEDED: update docstring
	def __sub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where the input SpParVec argument(s)
		are nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x - other))

	# NEEDED: update docstring
	def __xor__(self, other):
		"""
		performs a logical Xor between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where exactly one of the input vectors is True.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x ^ other))
	

	def all(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		Vec instance are True (nonzero), and False otherwise.
		"""
		tmp = self.copy()
		# only because have to set tmp[0]
					# because of element0 snafu
		if isinstance(self._identity_, (float, int, long)):
			return self.reduce(op_and, pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0)))
		else:
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
		SpParVec instance have values within epsilon of an integer,
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
		SpVecObj instances into the result SpVecObj instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where both of the input vectors are True.
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
		SpVecObj instances into the result SpVecObj instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where either of the input vectors is True.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: bool(x) or bool(other)))

	# NEEDED: update docstring
	def logicalXor(self, other):
		"""
		performs a logical Or between the corresponding elements of two
		SpVecObj instances into the result SpVecObj instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where either of the input vectors is True.
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
			func = lambda x, other: x.max(other)
			ret = self.reduce(pcb.binaryObj(func), init=initNegInf)
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
			func = lambda x, other: x.min(other)
			ret = self.reduce(pcb.binaryObj(func), init=initInf)
		return ret

	def nn(self):
		"""
		returns the number of nulls (non-existent entries) in the 
		SpVec instance.

		Note:  for x a SpVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nnn, nnz
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

		SEE ALSO:  nn, nnz
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
		sets every non-null value in the Vec instance to the second
		argument, in-place.
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
			raise NotImplementedError, "filtered sort"
			
		return Vec._toVec(self._v_.Sort())

	# NEEDED: update docstring
	def sorted(self):
		"""
		returns a new SpParVec instance with the sorted values (in ascending
		order) from the input SpParVec instance and a SpParVec permutation 
		vector.

		Input Arguments:
			self:  a SpParVec instance.

		Output Argument:
			ret:  a tuple containing as its first element a SpParVec 
			    instance of the same length and same number and position
			    of non-nulls containing the sorted non-null values and
			    as its second element the indices of the sorted values
			    in the input vector.

		See Also:  sort
		"""
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
			self.apply(lambda x: x.spOnes())
		return

	#in-place, so no return value
	def spRange(self):
		"""
		sets every non-null value in the SpParVec instance to its position
		(offset) in the vector, in-place.
		"""

		if not self.isObj() and self.isSparse() and not self._hasFilter():
			self._v_.setNumToInd()
		else:
			if self.isObj():
				func = lambda x,y: x.spRange(y)
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

	# NEEDED: AL: shouldn't this just be a call to reduce?
	def sum(self):
		"""
		returns the sum of all the non-null values in the Vec instance.
		"""
		return self.reduce(lambda x,y: x+y)
		
		if self.nnn() == 0:
			if isinstance(self._identity_, (float, int, long)):
				ret = 0
			elif isinstance(self._identity_, pcb.Obj1):
				ret = pcb.Obj1()
		else:
			if isinstance(self._identity_, (float, int, long)):
				ret = self.reduce(op_add)
			elif isinstance(self._identity_, (pcb.Obj1, pcb.Obj2)):
		 		func = lambda x, other: x.__iadd__(other)
				ret = self.reduce(pcb.binaryObj(func))
		return ret

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
		returns the largest k non-null values in the passed SpParVec instance.

		Input Arguments:
			self:  a SpParVec instance.
			k:  a scalar integer denoting how many values to return.

		Output Argument:
			ret:  a ParVec instance of length k containing the k largest
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