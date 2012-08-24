import pyCombBLAS as pcb

import ctypes

class info:
	@staticmethod
	def eps():
		"""
		Return IEEE floating point machine epsilon.
		The problem with this operation is that Python only provides a portable way to get this
		value in v2.6 and NumPy isn't always available. This function attempts to use whatever
		knows this value or returns a reasonable default otherwise.
		"""
		# try Python v2.6+ float_info
		try:
			from sys import float_info as fi
			return fi.epsilon
		except ImportError:
			pass
			
		# try Numpy
		try:
			import numpy as np
			return float(np.finfo(np.float).eps)
		except ImportError:
			pass
		except AttributeError:
			pass
			
		# return a reasonable value
		return 2.220446049250313e-16;

	def minInt():
		return -(2**62)

class FilterHelper:
	@staticmethod
	def getFilteredUniOpOrSelf(filteredObject, op):
		if filteredObject._hasFilter():
			class tmpU:
				filter = filteredObject._filter_
				@staticmethod
				def fn(x):
					for i in range(len(tmpU.filter)):
						if not tmpU.filter[i](x):
							return x
					return op(x)
			tmpInstance = tmpU()
			return tmpInstance.fn
		else:
			return op

	@staticmethod
	def getFilteredUniOpOrOpVal(filteredObject, op, defaultVal):
		if filteredObject._hasFilter():
			class tmpU:
				filter = filteredObject._filter_
				identity = defaultVal
				@staticmethod
				def fn(x):
					for i in range(len(tmpU.filter)):
						if not tmpU.filter[i](x):
							return op(type(tmpU.identity)())
					return op(x)
			tmpInstance = tmpU()
			return tmpInstance.fn
		else:
			return op

	@staticmethod
	def getFilteredUniOpOrVal(filteredObject, op, defaultVal):
		if filteredObject._hasFilter():
			class tmpU:
				filter = filteredObject._filter_
				identity = defaultVal
				@staticmethod
				def fn(x):
					for i in range(len(tmpU.filter)):
						if not tmpU.filter[i](x):
							#print "returning identity"
							return tmpU.identity
					#print "x=",x,"returning op(x)=",op(x)
					return op(x)
			tmpInstance = tmpU()
			return tmpInstance.fn
		else:
			return op

	@staticmethod
	def getFilterPred(filteredObject):
		if filteredObject._hasFilter():
			if len(filteredObject._filter_) == 1:
				# only one filter, so pass the single predicate along
				return filteredObject._filter_[0]
			else:
				# multiple filters, so create a shim that calls each one
				# successively, supporting shortcuts.
				class FilterStacker:
					def __init__(self, f):
						self.filters = f
	
					def __call__(self, x):
						for i in range(len(self.filters)):
							if not self.filters[i](x):
								return False
						return True
				return FilterStacker(filteredObject._filter_)
		else:
			return None
			
def master():
	"""
	Return Boolean value denoting whether calling process is the 
	master process or a slave process in a parallel program.
	"""
	return pcb.root()

def _barrier():
	"""
	Synchronizes multiple processes. Corresponds to an MPI barrier.
	"""
	pcb._barrier();

def _broadcast(obj):
	"""
	Broadcasts a Python object from the root process to others.
	If no serializers are available, the function broadcasts
	a string representation of an object. Currently, the function
	is limited to broadcasting the objects whose string
	representation is less than 1024 bytes.

	Input Arguments:
		obj - an object to broadcast. This argument is important
			only in the root process, the broadcaster. It may have
			any value in a receiver process.

	Output Arguments:
		The function returns a copy of the broadcasted object in
		every process, including the root.
	"""
	serializer = None;
	try:
		import cPickle as serializer;
	except ImportError:
		try:
			import pickle as serializer;
		except ImportError:
			if(not isinstance(obj, str)):
				raise "Unable to broadcast a non-string object due to "\
					  "the absence of both cPickle and pickle serializers.";

	isRoot = master();
	toSend = None;

	if(isRoot):
		if serializer != None:
			toSend = serializer.dumps(obj);
		else:
			toSend = str(obj);

	received = pcb._broadcast(toSend);

	if(isRoot):
		return obj;
	else:
		if serializer == None:
			result = received;
		else:
			result = serializer.loads(received);
		return result;

def p(s):
	"""
	printer helper
	"""
	s = str(s)
	#if master():
	#	print s
	pcb.prnt(s)
	pcb.prnt("\n")

def _nproc():
	"""
	Return the number of processors available.
	"""
	return pcb._nprocs()

def _rank():
	"""
	Return the number of the processor executing this function (MPI rank).
	"""
	return pcb._rank()

def version():
	"""
	Return KDT version number, as a string.
	"""
	return "0.2.x"

def revision():
	"""
	Return KDT revision number, as a string.
	"""
	return "r7xx"

def sr(addFn, mulFn, leftFilter=None, rightFilter=None):
	ret = pcb.SemiringObj(addFn, mulFn, leftFilter, rightFilter)
	ret.origAddFn = addFn
	ret.origMulFn = mulFn
	#ret.origLeftFilter = leftFilter # overrriden by setFilters()
	#ret.origRightFilter = rightFilter # overrriden by setFilters()
	return ret


NONE = 0
INFO = 1
DEBUG = 2
verbosity = NONE

####
#
# KDT-level wrapping of fast functions which only work on floating point values.
# If a unary or binary operation is composed of these functions then it will
# run at C++ speed when available and default back to a Python implementation
# when not available. When SEJITS integration is complete these will all be
# deprecated in favor of direct Python code.
#
####

# built-in semirings
sr_select2nd = pcb.SecondMaxSemiringObj()
sr_plustimes = pcb.TimesPlusSemiringObj()

def times(x, y):
	return x*y
def plus(x, y):
	return x+y
def select2nd(x, y):
	return y
py_sr_select2nd = sr(select2nd, select2nd)
py_sr_plustimes = sr(plus, times)
		
# built-in operations that only work on floating point scalars
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

op_id = pcb.identity()
# safemultinv()
op_abs = pcb.abs()
op_negate = pcb.negate()
op_bitNot = pcb.bitwise_not()
op_not = pcb.logical_not()
#totality()

class _complex_op:
	def __init__(self, op, name):
		self.pcb_op = op
		self.name = name
	

def op_set(val):
	return pcb.set(val)
	#ret = _complex_op(pcb.set(val), "set")
	#ret.val = val
	#return ret

def op_IfThenElse(predicate, runTrue, runFalse):
	return pcb.ifthenelse(predicate, runTrue, runFalse)
	#ret = _complex_op(pcb.ifthenelse(predicate, runTrue, runFalse), "ifthenelse")
	#ret.predicate = predicate
	#ret.runTrue = runTrue
	#ret.runFalse = runFalse
	#return ret

def op_bind1st(op, val):
	return pcb.bind1st(op, val)

def op_bind2nd(op, val):
	return pcb.bind2nd(op, val)

def op_compose1(f, g):
	return pcb.compose1(f, g)

def op_compose2(f, g1, g2):
	return pcb.compose2(f, g1, g2)

def op_not1(op):
	return pcb.not1(op)

def op_not2(op):
	return pcb.not1(op)

def _op_builtin_pyfunc(op):
	# binary functions
	if op == op_add:
		return lambda x, y: (x + y)
	if op == op_sub:
		return lambda x, y: (x - y)
	if op == op_mul:
		return lambda x, y: (x * y)
	if op == op_div:
		return lambda x, y: (x / y)
	if op == op_mod:
		return lambda x, y: (x % y)
	if op == op_fmod:
		raise NotImplementedError, 'fmod Python expression not implemented' 
		#return lambda x, y: (x % y)
	if op == op_pow:
		return lambda x, y: (x**y)
	if op == op_max:
		return lambda x, y: max(x, y)
	if op == op_min:
		return lambda x, y: min(x, y)
	if op == op_bitAnd:
		return lambda x, y: (x & y)
	if op == op_bitOr:
		return lambda x, y: (x | y)
	if op == op_bitXor:
		return lambda x, y: (x ^ y)
	if op == op_and:
		return lambda x, y: (x and y)
	if op == op_or:
		return lambda x, y: (x or y)
	if op == op_xor:
		return lambda x, y: (bool(x) != bool(y))
	if op == op_eq:
		return lambda x, y: (x == y)
	if op == op_ne:
		return lambda x, y: (x != y)
	if op == op_gt:
		return lambda x, y: (x > y)
	if op == op_lt:
		return lambda x, y: (x < y)
	if op == op_ge:
		return lambda x, y: (x >= y)
	if op == op_le:
		return lambda x, y: (x <= y)
	
	# unary functions
	if op == op_id:
		return lambda x: (x)
	if op == op_abs:
		return lambda x: abs(x)
	if op == op_negate:
		return lambda x: (-x)
	if op == op_bitNot:
		return lambda x: (~x)
	if op == op_not:
		return lambda x: (not x)
	
	raise NotImplementedError, 'Unable to convert functor to Python expression.'


## Python-defined object helpers

# a flag to enable or disable Python-Defined Objects.
# if disabled, the Obj->PDO conversion functions become simple passthroughs.
PDO_enabled = True

def PDO_enable(tf):
	global PDO_enabled
	PDO_enabled = tf

# take a pure ctypes structure and put it into a pyCombBLAS object
def _coerceToInternal(value, storageType):
	if not PDO_enabled:
		return value

	if isinstance(value, storageType) or (isinstance(value, (bool, int, float)) and issubclass(storageType, float)):
		return value
	else:
		if not issubclass(type(value), ctypes.Structure):
			raise ValueError,"coercion is meant to work on Python-defined types. You provided %s => %s"%(str(type(value)), str(storageType))
		
		# create the pyCombBLAS obj to store the object
		ret = storageType()
		ctypesObj = type(value).from_address(ret.getDataPtrLong())
		# copy the object into its new location
		ctypes.memmove(ctypes.byref(ctypesObj), ctypes.byref(value), ctypes.sizeof(value))
		#done
		return ret

# take a pyCombBLAS object and turn it into a ctypes object
def _coerceToExternal(value, extType):
	if not PDO_enabled:
		return value

	#print "_coerceToExternal types:",type(value),extType
	if isinstance(value, extType) or (isinstance(value, (bool, int, float)) and issubclass(extType, (bool, int, float))):
		return value
	else:
		if not issubclass(extType, ctypes.Structure):
			raise ValueError,"coercion is meant to work on Python-defined types. You provided %s => %s"%(str(type(value)), str(extType))
		
		ret = extType.from_address(value.getDataPtrLong())
		ret.referenceToMemoryObj = value # keep a reference to the storage object so it doesn't get garbage collected
		return ret
		
# shim to make a CombBLAS object act like a ctypes-defined object
class _python_def_shim_unary:
	def __init__(self, callback, ctypesClass, retStorageType):
		self.callback = callback
		self.ctypesClass = ctypesClass
		self.retStorageType = retStorageType
	
	def __call__(self, arg1):
		result = self.callback(_coerceToExternal(arg1, self.ctypesClass))
		if self.retStorageType is not None:
			# a value stored back in the structure
			return _coerceToInternal(result, self.retStorageType)
		else:
			# a POD type, eg for predicates
			return result

class _python_def_shim_binary:
	def __init__(self, callback, ctypesClass1, ctypesClass2, retStorageType):
		self.callback = callback
		self.ctypesClass1 = ctypesClass1
		self.ctypesClass2 = ctypesClass2
		self.retStorageType = retStorageType
	
	def __call__(self, arg1, arg2):
		#print "in binop callback. types of args:",type(arg1), type(arg2), " ctypesTypes:",self.ctypesClass1, self.ctypesClass2, self.retStorageType
		result = self.callback(_coerceToExternal(arg1, self.ctypesClass1),
			_coerceToExternal(arg2, self.ctypesClass2))
			
		if self.retStorageType is not None:
			# a value stored back in the structure
			return _coerceToInternal(result, self.retStorageType)
		else:
			# a POD type, eg for predicates
			return result

## helper functions to transform Python callbacks into pyCombBLAS functor objects

# Wrap a Python unary callback into pyCombBLAS's UnaryFunctionObj,
# or, if already wrapped, return the existing wrapper.
def _op_make_unary(op, opStruct, opStructRet=None):
	if op is None:
		return None
	if issubclass(opStruct._getElementType(), ctypes.Structure):
		if opStructRet is None:
			opStructRet = opStruct
		op = _python_def_shim_unary(op, opStruct._getElementType(), opStructRet._getStorageType())

	if isinstance(op, (pcb.UnaryFunction, pcb.UnaryFunctionObj)):
		return op
	return pcb.unaryObj(op)

def _op_make_unary_pred(op, opStruct, opStructRet=None):
	if op is None:
		return None
	if issubclass(opStruct._getElementType(), ctypes.Structure):
		op = _python_def_shim_unary(op, opStruct._getElementType(), None)

	if isinstance(op, (pcb.UnaryFunction, pcb.UnaryPredicateObj)):
		return op
	return pcb.unaryObjPred(op)

# Wrap a Python binary callback into pyCombBLAS's BinaryFunctionObj,
# or, if already wrapped, return the existing wrapper.
def _op_make_binary(op, opStruct1, opStruct2, opStructRet):
	if op is None:
		return None
	if issubclass(opStruct1._getElementType(), ctypes.Structure) or issubclass(opStruct2._getElementType(), ctypes.Structure):
		op = _python_def_shim_binary(op, opStruct1._getElementType(), opStruct2._getElementType(), opStructRet._getStorageType())

	if isinstance(op, (pcb.BinaryFunction, pcb.BinaryFunctionObj)):
		return op
	return pcb.binaryObj(op)

# same as above, but will convert a BinaryFunction into a
# BinaryFunctionObj, if necessary.
def _op_make_binaryObj(op, opStruct1, opStruct2, opStructRet):
	if op is None:
		return None
	if issubclass(opStruct1._getElementType(), ctypes.Structure) or issubclass(opStruct2._getElementType(), ctypes.Structure):
		op = _python_def_shim_binary(op, opStruct1._getElementType(), opStruct2._getElementType(), opStructRet._getStorageType())

	if isinstance(op, (pcb.BinaryFunctionObj)):
		return op
	if isinstance(op, (pcb.BinaryFunction)):
		return pcb.binaryObj(_op_builtin_pyfunc(op))
	return pcb.binaryObj(op)

def _op_make_binary_pred(op, opStruct1, opStruct2, opStructRet=None):
	if op is None:
		return None
	if issubclass(opStruct1._getElementType(), ctypes.Structure) or issubclass(opStruct2._getElementType(), ctypes.Structure):
		op = _python_def_shim_binary(op, opStruct1._getElementType(), opStruct2._getElementType(), None)

	if isinstance(op, (pcb.BinaryFunction, pcb.BinaryPredicateObj)):
		return op
	return pcb.binaryObjPred(op)

def _sr_addTypes(inSR, opStruct1, opStruct2, opStructRet):
	if issubclass(opStruct1._getElementType(), ctypes.Structure) or issubclass(opStruct2._getElementType(), ctypes.Structure):
		mul = inSR.origMulFn
		add = inSR.origAddFn
		mul = _python_def_shim_binary(mul, opStruct1._getElementType(), opStruct2._getElementType(), opStructRet._getStorageType())
		add = _python_def_shim_binary(add, opStruct2._getElementType(), opStruct2._getElementType(), opStructRet._getStorageType())
		outSR = sr(add, mul)
		return outSR
	else:
		return inSR
		
# more type management helpers
class _typeWrapInfo:
	def __init__(self, externalType):
		self.externalType = externalType;

		if issubclass(externalType, ctypes.Structure):
			if ctypes.sizeof(externalType) <= pcb.Obj1.capacity():
				self.storageType = pcb.Obj1
			else:
				raise TypeError, "Cannot fit object into any available storage sizes. Largest supported object size is %d bytes, yours is %d."%(kdt.Obj1.capacity(), ctypes.sizeof(externalType))
		else:
			self.storageType = externalType

	def _getStorageType(self):
		return self.storageType
	
	def _getElementType(self):
		return self.externalType

# stub classes to use built-in types with the above functions
# ints
class _opStruct_int:
	def _getStorageType(self):
		return int
	
	def _getElementType(self):
		return int
# floats
class _opStruct_float:
	def _getStorageType(self):
		return float
	
	def _getElementType(self):
		return float


def _op_is_wrapped(op):
	return isinstance(op, (pcb.UnaryFunction, pcb.UnaryFunctionObj, pcb.UnaryPredicateObj, pcb.BinaryFunction, pcb.BinaryFunctionObj, pcb.BinaryPredicateObj))

# given a built-in routine (like op_add), return a Python equivalent.
# this is used to filter the built-in routines, because the filters
# operate on Python, not C++ code.
def _makePythonOp(op):
	if isinstance(op, (pcb.UnaryFunction, pcb.BinaryFunction)):
		return _op_builtin_pyfunc(op)
	elif isinstance(op, (pcb.UnaryFunctionObj, pcb.BinaryFunctionObj)):
		raise NotImplementedError, 'Unable to convert OBJ functor back to Python expression.'
	elif op == sr_plustimes:
		return py_sr_plustimes
	elif op == sr_select2nd:
		return py_sr_select2nd
	else:
		return op

def _sr_get_python_add(SR):
	if SR == sr_plustimes:
		# use an actual function instead of a lambda to make SEJITS handle it easier
		def plus(x, y):
			return x+y
		return plus
	else:
		return SR.getAddCallback()

def _sr_get_python_mul(SR):
	if SR == sr_plustimes:
		# use an actual function instead of a lambda to make SEJITS handle it easier
		def mul(x, y):
			return x*y
		return mul
	else:
		return SR.getMulCallback()
